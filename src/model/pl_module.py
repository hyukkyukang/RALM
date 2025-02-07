import logging
import math
from typing import *

import lightning as L
import torch
import transformers
from omegaconf import DictConfig

from src.model.rellama.causal_modeling import ReLlamaForCausalLM
from src.model.rellama.model import ReLlama
from src.model.utils import get_llama_config, initialize_weights
from src.tokenizer import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("ReLLamaLightningModule")


class ReLLamaLightningModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        total_steps: int = None,
        tokenizer: Optional[ReLlamaTokenizer] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.total_training_steps = total_steps
        self.bpb_term = math.log2(math.e) / 4  # 4 is bytes_per_token
        self.model: transformers.LlamaForCausalLM = self.initialize_language_model(
            cfg=cfg, tokenizer=tokenizer
        )
        self.save_hyperparameters(cfg)

    def initialize_language_model(
        self, cfg: DictConfig, tokenizer: Optional[ReLlamaTokenizer] = None
    ) -> transformers.LlamaForCausalLM:
        # Get the tokenizer
        if tokenizer is None:
            tokenizer = ReLlamaTokenizer.from_pretrained(cfg.model.base_name)

        llama_config: transformers.LlamaConfig = get_llama_config(cfg, tokenizer)
        llama_config.attn_implementation = "flash_attention_2"

        # Initialize the model
        if cfg.model.name == "llama":
            model = transformers.LlamaModel(config=llama_config)
            causal_model = transformers.LlamaForCausalLM(config=llama_config)
        elif cfg.model.name == "rellama":
            model = ReLlama(llama_config)
            causal_model = ReLlamaForCausalLM(config=llama_config, model=model)
        else:
            raise ValueError(f"Model name {cfg.model.name} not supported")

        # Initialize model weights if not resuming from checkpoint
        if cfg.training.resume_ckpt_path is None:
            log_if_rank_zero(
                logger,
                "Applying xavier uniform initialization to model weights for pretraining from scratch",
            )
            causal_model.apply(initialize_weights)

        # Compile the model if the config is set to True and the GPU has the capability to compile the model
        if cfg.training.use_torch_compile:
            if torch.cuda.get_device_capability()[0] >= 7:
                causal_model = torch.compile(causal_model, dynamic=True)
            else:
                log_if_rank_zero(
                    logger,
                    "Torch compile is not supported on this GPU. Use_torch_compile is set to True, but the GPU does not support torch compile.",
                )

        # Enable gradient checkpointing if specified in config
        if cfg.training.get("gradient_checkpointing", False):
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
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        perplexity = torch.exp(loss)

        # Calculate bits per byte (BPB)
        bpb = loss * self.bpb_term

        # Add metrics logging
        self.log_dict(
            {"loss": loss, "perplexity": perplexity, "bits_per_byte": bpb},
            batch_size=batch["input_ids"].size(0),
        )
        return loss

    # Modify configure_optimizers to use the JIT-compiled function

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.max_learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2),
        )

        # Extract values
        warmup_iters = self.cfg.training.warmup_steps
        total_iters = self.total_training_steps
        min_lr = self.cfg.training.min_learning_rate
        max_lr = self.cfg.training.max_learning_rate

        # Define a lambda function that wraps the JIT-compiled function
        lr_scheduler_fn = lambda it: lr_lambda(
            it, warmup_iters, total_iters, min_lr, max_lr
        )

        # Use LambdaLR with the optimized learning rate function
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_scheduler_fn
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# Define a TorchScript-compatible function for the learning rate schedule
@torch.jit.script
def lr_lambda(
    it: int, warmup_iters: int, total_iters: int, min_lr: float, max_lr: float
) -> float:
    """
    Computes learning rate scaling factor for warmup and cosine decay.

    Args:
        it (int): Current training step.
        warmup_iters (int): Number of warmup steps.
        total_iters (int): Total training steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        return float(it) / float(warmup_iters)  # Linear warmup
    elif it >= total_iters:  # Fix the condition
        return min_lr / max_lr  # Hold at min LR
    else:
        decay_ratio = float(it - warmup_iters) / float(total_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay
        # return coeff * (max_lr - min_lr) / max_lr + min_lr / max_lr  # Adjusted formula
        return (min_lr + coeff * (max_lr - min_lr)) / max_lr
