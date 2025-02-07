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


def find_the_last_non_pad_idx(
    input_ids_batch: torch.Tensor,
    attention_masks_batch: torch.Tensor,
    pad_token_id: int,
) -> List[int]:
    last_non_pad_indices: List[int] = []
    for i in range(input_ids_batch.shape[0]):
        # Find the index of pad_token_id in input_ids[i]
        pad_idx = (input_ids_batch[i] == pad_token_id).nonzero().tolist()
        last_non_pad_idx = (
            pad_idx[0][0] - 1 if pad_idx else input_ids_batch.shape[1] - 1
        )
        # Check by the attention mask
        assert (
            attention_masks_batch[i, last_non_pad_idx] == 1
        ), "The last non-padding token is a padding token"
        last_non_pad_indices.append(last_non_pad_idx)
    return last_non_pad_indices


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Check the arguments
    cfg = check_arguments(cfg)

    # Load the dataset
    dataset = load_dataset(cfg.eval.dataset_name)["test"]

    # Load the pre-trained model and tokenizer
    device = "cuda"
    use_rellama = False
    if use_rellama:
        model = ReLLamaLightningModule.load_from_checkpoint(cfg.ckpt_path).to(device)
        tokenizer = ReLlamaTokenizer.from_pretrained(cfg.model.base_name)
    else:
        # Use GPT-small
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], return_tensors="pt", padding=True),
        batched=True,
        batch_size=cfg.eval.batch_size,
        remove_columns=["domain", "text"],
    )

    # Evaluate the model
    all_accuracies: List[float] = []
    with torch.no_grad():
        for start_idx in tqdm.tqdm(
            range(0, len(tokenized_dataset), cfg.eval.batch_size),
            desc="Evaluating",
            total=len(tokenized_dataset) // cfg.eval.batch_size,
        ):
            end_idx = min(start_idx + cfg.eval.batch_size, len(tokenized_dataset))
            input_ids: torch.Tensor = torch.tensor(
                tokenized_dataset["input_ids"][start_idx:end_idx]
            ).to(model.device)
            attention_masks: torch.Tensor = torch.tensor(
                tokenized_dataset["attention_mask"][start_idx:end_idx]
            ).to(model.device)

            # Get the predictions
            predictions = model(
                input_ids=input_ids, attention_mask=attention_masks, return_dict=True
            )

            # Get the find the last non-padding token id for each sample
            last_non_pad_indices = find_the_last_non_pad_idx(
                input_ids, attention_masks, tokenizer.pad_token_id
            )
            # Get the predictions for the last non-padding token
            batch_indices = torch.arange(len(last_non_pad_indices), device=model.device)
            predictions = predictions["logits"][batch_indices, last_non_pad_indices]
            # Get the labels for the last non pad tokens using tensor indexing
            labels = input_ids[batch_indices, last_non_pad_indices]
            # Evaluate the results
            predicted_tokens = predictions.argmax(dim=-1)
            accuracy = (predicted_tokens == labels).float()

            # Save the accuracy
            all_accuracies.extend(accuracy.tolist())
            debug = False
            if debug:
                for i in range(len(last_non_pad_indices)):
                    label_token = tokenizer.convert_ids_to_tokens(labels[i : i + 1])
                    predicted_token = tokenizer.convert_ids_to_tokens(
                        predicted_tokens[i : i + 1]
                    )
                    # Print only up to the last non-padding token
                    logger.info(
                        f"Last Few Words: {tokenizer.decode(tokenized_dataset['input_ids'][start_idx + i][:last_non_pad_indices[i] + 1])}"
                    )
                    logger.info(f"Label: {label_token}")
                    logger.info(f"Predicted: {predicted_token}")
    logger.info(
        f"Accuracy: {sum(all_accuracies) / len(all_accuracies)} ({len(all_accuracies)} samples)"
    )
    return None


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    misc_utils.load_dotenv()

    main()
