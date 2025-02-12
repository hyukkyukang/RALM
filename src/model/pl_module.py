import logging
import math
from typing import *

import lightning as L
import torch
import transformers
from lion_pytorch import Lion
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

from src.evaluation.next_word_prediction import evaluate_next_word_prediction
from src.model.rellama.causal_modeling import ReLlamaForCausalLM
from src.model.rellama.model import ReLlama
from src.tokenization import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("LightningModule")


def get_compile_decorator(
    use_compile: bool = True, fullgraph: bool = False, mode: str = "default"
):
    """Returns torch.compile decorator if GPU is capable and use_compile is True, otherwise returns a no-op decorator"""
    if (
        use_compile
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 7
    ):
        log_if_rank_zero(
            logger, f"Compiling the module with torch compile in {mode} mode..."
        )
        return torch.compile(fullgraph=fullgraph, mode=mode)
    return lambda x: x  # no-op decorator


class LightningModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        total_optimization_steps: int = None,
        tokenizer: Optional[ReLlamaTokenizer] = None,
    ) -> None:
        super().__init__()
        ## Need to set this to False to avoid the automatic optimization
        self.automatic_optimization = False
        self.cfg = cfg
        self.total_optimization_steps = total_optimization_steps
        self.bpb_term = math.log2(math.e) / 4  # 4 is bytes_per_token
        self.tokenizer: Union[ReLlamaTokenizer, AutoTokenizer] = (
            self.initialize_tokenizer() if tokenizer is None else tokenizer
        )
        self.model: transformers.LlamaForCausalLM = self.initialize_language_model()
        # Store all arguments within the model checkpoint.
        self.save_hyperparameters(cfg)
        # For efficient training
        self.compiled_step = get_compile_decorator(
            use_compile=cfg.training.get(
                "use_torch_compile", cfg.get("use_torch_compile", False)
            ),
            fullgraph=False,
        )(self._compiled_step)
        # For evaluation
        self.test_step_outputs: List[float] = []
        # Add cumulative tokens counter as int64 tensor
        self.cumulative_tokens = torch.tensor(0, dtype=torch.int64, requires_grad=False)

    def initialize_tokenizer(self) -> ReLlamaTokenizer:
        if self.cfg.model.name == "rellama":
            tokenizer = ReLlamaTokenizer.from_pretrained(self.cfg.model.base_name)
        elif self.cfg.model.name == "gpt":
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.base_name)
        else:
            raise ValueError(f"Tokenizer name {self.cfg.model.name} not supported")
        return tokenizer

    def initialize_language_model(self) -> transformers.LlamaForCausalLM:
        # Initialize the model
        if self.cfg.model.name == "rellama":
            model = ReLlama(self.cfg, self.tokenizer)
            causal_model = ReLlamaForCausalLM(base_model=model)
        elif self.cfg.model.name == "gpt":
            causal_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.base_name
            )
        else:
            raise ValueError(f"Model name {self.cfg.model.name} not supported")

        # Compile the model if the config is set to True and the GPU has the capability to compile the model
        if self.cfg.training.get(
            "use_torch_compile", self.cfg.get("use_torch_compile", False)
        ):
            if torch.cuda.get_device_capability()[0] >= 7:
                log_if_rank_zero(
                    logger,
                    "Compiling the model with torch compile...",
                )
                causal_model = torch.compile(
                    causal_model, dynamic=True, mode="max-autotune"
                )
            else:
                log_if_rank_zero(
                    logger,
                    "Torch compile is not supported on this GPU. Use_torch_compile is set to True, but the GPU does not support torch compile.",
                )

        # Enable gradient checkpointing if specified in config
        if self.cfg.training.get("gradient_checkpointing", False):
            log_if_rank_zero(
                logger,
                "Enabling gradient checkpointing...",
            )
            causal_model.gradient_checkpointing_enable()

        return causal_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        # Use inference_mode when not training
        if not self.training:
            with torch.inference_mode():
                return self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        if batch_idx == 0:
            # Hack to change the device of the cumulative tokens to the device of the batch
            self.cumulative_tokens = self.cumulative_tokens.to(self.device)

        self.compiled_step(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
            batch["avg_char_in_token"],
        )

        # Add the number of valid tokens to the cumulative tokens
        self.cumulative_tokens += batch["num_valid_tokens"]

        # Perform selective logging (i.e., only at the logging steps) of cumulative tokens
        if batch_idx % self.trainer.log_every_n_steps == 0:
            # First gather and sum tokens across processes
            gathered_tokens = self.all_gather(self.cumulative_tokens)
            # Log the total tokens across all processes
            self.logger.log_metrics(
                {"cumulative_num_tokens": torch.sum(gathered_tokens)},
                step=batch_idx,
            )

        if (batch_idx + 1) % self.cfg.training.gradient_accumulation_steps == 0:
            optimizer = self.optimizers()
            # Clip gradients
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.cfg.optimizer.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

            # Update weights
            optimizer.step()
            # Update scheduler
            self.lr_schedulers().step()
            # Zero the gradients
            optimizer.zero_grad()

        return None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        bsize = len(batch["input_ids"])
        if self.cfg.testing.name == "last_word_prediction":
            for b_idx in range(bsize):
                correct = evaluate_next_word_prediction(
                    token_ids=batch["input_ids"][b_idx],
                    last_word=batch["last_word"][b_idx],
                    tokenizer=self.tokenizer,
                    model=self.model,
                )
                self.test_step_outputs.append(correct)
        else:
            raise ValueError(f"Test step {self.cfg.testing.name} not implemented")
        return None

    def on_test_epoch_end(self) -> None:
        # Accumulate the accuracy over all the test steps and multi-processes
        gathered_accuracies: List[torch.Tensor] = self.all_gather(
            self.test_step_outputs
        )
        # Count number of items gathered
        total_items = sum(len(item) for item in gathered_accuracies)
        total_sum = (
            torch.stack([item.sum() for item in gathered_accuracies]).sum().item()
        )
        # Calculate the accuracy
        avg_accuracy = total_sum / total_items
        log_if_rank_zero(logger, f"Accuracy: {avg_accuracy} (Total: {total_items})")
        return None

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        # Choose optimizer based on config
        log_if_rank_zero(logger, f"Using optimizer: {self.cfg.optimizer.name}")
        if self.cfg.optimizer.name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.cfg.lr_scheduler.max_learning_rate,
                weight_decay=self.cfg.optimizer.weight_decay,
                betas=(self.cfg.optimizer.beta1, self.cfg.optimizer.beta2),
            )
        elif self.cfg.optimizer.name == "adafactor":
            optimizer = Adafactor(
                self.parameters(),
                lr=self.cfg.lr_scheduler.max_learning_rate,
                weight_decay=self.cfg.optimizer.weight_decay,
                beta1=self.cfg.optimizer.beta1,
                scale_parameter=False,  # To use a manual (external) learning rate schedule
                relative_step=False,  # We want to use our custom LR schedule
                warmup_init=False,  # We'll handle warmup with our scheduler
            )
        elif self.cfg.optimizer.name == "lion":
            optimizer = Lion(
                self.parameters(),
                lr=self.cfg.lr_scheduler.max_learning_rate,
                weight_decay=self.cfg.optimizer.weight_decay,
                betas=(self.cfg.optimizer.beta1, self.cfg.optimizer.beta2),
                use_triton=True,  # set this to True to use cuda kernel w/ Triton lang (Tillet et al)
            )
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer.name} not supported")

        # Extract values
        warmup_iters = self.cfg.lr_scheduler.warmup_steps
        total_iters = self.total_optimization_steps
        min_lr = self.cfg.lr_scheduler.min_learning_rate
        max_lr = self.cfg.lr_scheduler.max_learning_rate

        # Define a lambda function that wraps the JIT-compiled function
        log_if_rank_zero(
            logger,
            f"Using learning rate scheduler: {self.cfg.lr_scheduler.name} with warmup steps: {warmup_iters} and total steps: {total_iters} (max lr: {max_lr}, min lr: {min_lr})",
        )
        if self.cfg.lr_scheduler.name == "cosine_decay":
            lr_scheduler_fn = lambda it: lr_lambda_cosine_decay(
                it, warmup_iters, total_iters, max_lr, min_lr
            )
        elif self.cfg.lr_scheduler.name == "linear_decay":
            lr_scheduler_fn = lambda it: lr_lambda_linear_decay(
                it, warmup_iters, total_iters, max_lr, min_lr
            )
        else:
            raise ValueError(
                f"Learning rate scheduler {self.cfg.lr_scheduler.name} not supported"
            )

        # Use LambdaLR with the optimized learning rate function
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_scheduler_fn
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _compiled_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        avg_char_in_token: float,
    ) -> None:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # Get the loss and calculate perplexity and bits per byte (BPB)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        # bpb = loss * self.bpb_term

        # Convert loss (nats) to bits
        loss_in_bits = loss * math.log2(math.e)

        # Compute bits per byte (BPB)
        bpb = loss_in_bits / avg_char_in_token

        # Log regular metrics (these will be averaged between logging steps)
        self.log_dict(
            {
                "loss": loss,
                "perplexity": perplexity,
                "bits_per_byte": bpb,
            },
            batch_size=input_ids.size(0),
        )

        # Average the loss over the gradient accumulation steps
        loss = loss / self.cfg.training.gradient_accumulation_steps
        # Backward
        self.manual_backward(loss)
        return None


# Define a TorchScript-compatible function for the learning rate schedule
@torch.jit.script
def lr_lambda_cosine_decay(
    it: int, warmup_iters: int, total_iters: int, max_lr: float, min_lr: float
) -> float:
    """
    Computes learning rate scaling factor for warmup and cosine decay.

    Args:
        it (int): Current optimizer step.
        warmup_iters (int): Number of warmup steps.
        total_iters (int): Total optimizer steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        return float(it) / float(warmup_iters)  # Linear warmup
    elif it >= total_iters:
        return min_lr / max_lr  # Hold at min LR
    else:
        decay_ratio = float(it - warmup_iters) / float(total_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay
        # Ensure we don't go below min_lr
        return max(min_lr / max_lr, (min_lr + coeff * (max_lr - min_lr)) / max_lr)


@torch.jit.script
def lr_lambda_linear_decay(
    it: int, warmup_iters: int, total_iters: int, max_lr: float, min_lr: float = 0.0
) -> float:
    """
    Computes learning rate scaling factor for warmup and linear decay to zero.

    Args:
        it (int): Current optimizer step.
        warmup_iters (int): Number of warmup steps.
        total_iters (int): Total optimizer steps.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        return float(it) / float(warmup_iters)  # Linear warmup
    elif it >= total_iters:
        return min_lr / max_lr
    else:
        # Linear decay to min_lr, normalized by max_lr for use with LambdaLR
        return (
            min_lr
            + (max_lr - min_lr) * (total_iters - it) / (total_iters - warmup_iters)
        ) / max_lr
