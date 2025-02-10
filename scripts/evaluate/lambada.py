import logging
from typing import *
import copy
import hkkang_utils.misc as misc_utils
import hydra
import torch
import tqdm
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BasicTokenizer

from scripts.evaluate.utils import normalize_quotes
from src.model import ReLLamaLightningModule
from src.tokenizer import ReLlamaTokenizer
from src.utils import check_argument
from scripts.evaluate.utils import STOPWORDS_FROM_GPT2

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


def split_text_into_context_and_last_word(
    line: str, tokenizer: BasicTokenizer
) -> Dict[str, str]:
    line = line.strip()
    toks = tokenizer.tokenize(line)
    length_of_word = len(toks[-1])
    assert length_of_word > 0, f"The last word is empty: {toks[-1]}"
    return {"context": line[:-length_of_word].strip(), "last_word": toks[-1]}


def predict(
    token_ids: List[List[int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    steps_to_predict: int = 6,
    beam_width: int = 128,
) -> List[str]:
    """Give continuation of the line with at most max_predictions BPE tokens. Returns line extended with predictions of
    the model."""
    bsize = len(token_ids)
    # Convert the token ids to a tensor
    current_input_token_ids = torch.tensor(token_ids).to(model.device)
    # Get the predictions
    all_token_ids: List[List[int]] = copy.deepcopy(token_ids)
    states = None
    for _ in range(steps_to_predict):
        outputs = model(current_input_token_ids, past_key_values=states)
        logits = outputs.logits  # Get logits from the outputs
        states = outputs.past_key_values  # Get the state from outputs

        # Get the top k candidates
        _, line_encoded_candidates = torch.topk(
            logits[:, -1, :],
            k=beam_width,
            dim=-1,
        )
        line_encoded_candidates = line_encoded_candidates.tolist()
        current_input_token_ids: List[List[int]] = []
        for b_idx in range(bsize):
            # Convert all the candidates to tokens
            candidate_tokens: List[str] = [
                tokenizer.decode(item).lower().strip()
                for item in line_encoded_candidates[b_idx]
            ]
            # Find the first candidate which is not a stopword
            predicted_token_id = None
            for cand_idx, candidate_token in enumerate(candidate_tokens):
                if candidate_token not in STOPWORDS_FROM_GPT2:
                    predicted_token_id = line_encoded_candidates[b_idx][cand_idx]
                    break
            assert predicted_token_id is not None, "No valid candidate found"
            all_token_ids[b_idx].append(predicted_token_id)
            current_input_token_ids.append([predicted_token_id])

        # Update the input tensor to pass to the next step
        current_input_token_ids = torch.tensor(current_input_token_ids).to(model.device)

    # Convert the decoded sequences to a list of strings
    decoded_sequences = [tokenizer.decode(ids) for ids in all_token_ids]
    return decoded_sequences


@hydra.main(version_base=None, config_path="/root/RETRO/config", config_name="config")
def main(cfg: DictConfig) -> None:
    batch_size = 1
    # Check the arguments
    cfg = check_arguments(cfg)

    # Load the pre-trained model and tokenizer
    device = "cuda"
    use_rellama = True
    if use_rellama:
        model = ReLLamaLightningModule.load_from_checkpoint(cfg.ckpt_path).to(device)
        model = model.model
        tokenizer = ReLlamaTokenizer.from_pretrained(cfg.model.base_name)
    else:
        # Use GPT-small
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load the dataset
    dataset = load_dataset(cfg.eval.dataset_name)["test"]

    # Split the last word from the text
    basic_tokenizer = BasicTokenizer()
    dataset = dataset.map(
        lambda x: split_text_into_context_and_last_word(x["text"], basic_tokenizer),
        batched=False,
    )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["context"], return_tensors="pt", padding=True),
        batched=False,
        batch_size=batch_size,
        remove_columns=["domain", "text"],
    )

    # Evaluate the model with the last word prediction
    all_accuracies: List[int] = []
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(tokenized_dataset)), desc="Evaluating"):
            # Get the predicted completions
            predictions: str = predict(
                token_ids=tokenized_dataset["input_ids"][idx],
                tokenizer=tokenizer,
                model=model,
            )[0]
            input_contexts: List[str] = tokenizer.decode(
                tokenized_dataset["input_ids"][idx][0]
            )
            generated_texts: str = predictions[len(input_contexts) :].strip()
            predicted_words: List[str] = basic_tokenizer.tokenize(generated_texts)
            predicted_word: str = (
                "" if len(predicted_words) == 0 else predicted_words[0]
            )
            # Check if the predicted word is the same as the last word
            if predicted_word == tokenized_dataset["last_word"][idx]:
                accuracy = 1
            else:
                accuracy = 0
            all_accuracies.append(accuracy)

            debug = False
            if debug:
                logger.info(f"Input: {input_contexts}")
                logger.info(f"Predicted: {predicted_word}")
                logger.info(f"Last word: {tokenized_dataset['last_word'][idx]}")
                logger.info(f"Accuracy: {accuracy}")
                logger.info("-" * 100)
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
