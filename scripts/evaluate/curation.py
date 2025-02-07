import logging
from typing import *

import hkkang_utils.misc as misc_utils
import hydra
import torch
import tqdm
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import ReLLamaLightningModule
from src.tokenizer import ReLlamaTokenizer
from src.utils import check_argument

logger = logging.getLogger("EvaluateLambada")


def check_arguments(cfg: DictConfig) -> DictConfig:
    check_argument(
        cfg,
        name="ckpt_path",
        arg_type=str,
        help="Path to the checkpoint",
        is_requried=True,
    )
    return cfg


def process_text_with_stride(
    text: str, tokenizer, max_length: int = 2048, stride: int = 1024
) -> List[Dict[str, torch.Tensor]]:
    """Process text with striding window to handle long sequences."""
    # Tokenize the entire text first
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"][0]

    # If the sequence is shorter than max_length, just return it as is
    if len(input_ids) <= max_length:
        return [
            tokenizer(
                text, return_tensors="pt", padding="max_length", max_length=max_length
            )
        ]

    # Process text with stride
    chunks = []
    for start_idx in range(0, len(input_ids), stride):
        end_idx = min(start_idx + max_length, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]

        # Create attention mask
        attention_mask = torch.ones_like(chunk_ids)

        # Pad if necessary
        if len(chunk_ids) < max_length:
            padding_length = max_length - len(chunk_ids)
            chunk_ids = torch.cat(
                [chunk_ids, torch.full((padding_length,), tokenizer.pad_token_id)]
            )
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length)])

        chunks.append(
            {
                "input_ids": chunk_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            }
        )

    return chunks


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    1. On Curation Corpus we concatenate the article, the "TL;DR:" string,
    and the summary, but only evaluate the bpb on the summary.
    2. Evaluate with a sequence length of 2048 tokens,
    but use a stride of 1024 within documents to mitigate boundary effects
    """
    # Check the arguments
    cfg = check_arguments(cfg)

    # Load the dataset
    dataset = load_dataset(cfg.eval.dataset_name)["train"]

    # Load the pre-trained model and tokenizer
    device = "cuda"
    use_rellama = False
    if use_rellama:
        model = ReLLamaLightningModule.load_from_checkpoint(cfg.ckpt_path).to(device)
        tokenizer = ReLlamaTokenizer.from_pretrained(cfg.model.base_name)
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Process each document with stride
    running_loss = 0
    running_tokens = 0

    with torch.no_grad():
        for doc_idx in tqdm.tqdm(range(len(dataset)), desc="Processing documents"):
            # Concatenate article and summary
            full_text = f"{dataset[doc_idx]['article_content']}\nTL;DR: {dataset[doc_idx]['summary']}"

            # Find the position of TL;DR in the text
            tldr_pos = full_text.find("\nTL;DR: ")

            # Process text with stride
            chunks = process_text_with_stride(
                full_text, tokenizer, max_length=2048, stride=1024
            )

            # Process each chunk
            for chunk in chunks:
                input_ids = chunk["input_ids"].to(device)
                attention_mask = chunk["attention_mask"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )

                # Find the position of TL;DR in this chunk
                tldr_token_ids = tokenizer("TL;DR:", return_tensors="pt")[
                    "input_ids"
                ].to(device)
                tldr_pos_in_chunk = None

                for i in range(len(input_ids[0]) - len(tldr_token_ids[0]) + 1):
                    if torch.equal(
                        input_ids[0, i : i + len(tldr_token_ids[0])], tldr_token_ids[0]
                    ):
                        tldr_pos_in_chunk = i + len(tldr_token_ids[0])
                        break

                if tldr_pos_in_chunk is not None:
                    # Calculate loss only on the summary portion
                    logits = outputs.logits[:, tldr_pos_in_chunk:-1, :]
                    target_ids = input_ids[:, tldr_pos_in_chunk + 1 :]

                    # Calculate cross entropy loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        target_ids.reshape(-1),
                        reduction="sum",
                        ignore_index=tokenizer.pad_token_id,
                    )

                    # Count only non-padding tokens
                    num_tokens = (target_ids != tokenizer.pad_token_id).sum().item()

                    running_loss += loss.item()
                    running_tokens += num_tokens

    # Calculate final bits per byte
    avg_loss = running_loss / running_tokens
    bits_per_byte = avg_loss / torch.log(torch.tensor(2.0))

    logger.info(f"Evaluation Results:")
    logger.info(f"Total tokens evaluated: {running_tokens}")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Bits per byte: {bits_per_byte:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()
    main()
