import logging
import math
from typing import *

import lightning as L
import torch
import transformers
from lion_pytorch import Lion
from omegaconf import DictConfig
from transformers.optimization import Adafactor

from src.model.rellama.causal_modeling import ReLlamaForCausalLM
from src.model.rellama.model import ReLlama
from src.model.utils import get_llama_config, initialize_weights
from src.tokenizer import ReLlamaTokenizer
from src.utils import log_if_rank_zero

logger = logging.getLogger("ReLLamaLightningModule")

def get_compile_decorator(use_compile: bool = True, fullgraph: bool = False, mode: str = "default"):
    """Returns torch.compile decorator if GPU is capable and use_compile is True, otherwise returns a no-op decorator"""
    if use_compile and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        log_if_rank_zero(logger, f"Compiling the module with torch compile in {mode} mode...")
        return torch.compile(fullgraph=fullgraph, mode=mode)
    return lambda x: x  # no-op decorator

class ReLLamaLightningModule(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        total_steps: int = None,
        tokenizer: Optional[ReLlamaTokenizer] = None,
    ) -> None:
        super().__init__()
        ## Need to set this to False to avoid the automatic optimization
        self.automatic_optimization = False
        self.cfg = cfg
        self.total_training_steps = total_steps
        self.bpb_term = math.log2(math.e) / 4  # 4 is bytes_per_token
        self.model: transformers.LlamaForCausalLM = self.initialize_language_model(
            cfg=cfg, tokenizer=tokenizer
        )
        # Store all arguments within the model checkpoint.
        self.save_hyperparameters(cfg)
        self.compiled_step = get_compile_decorator(cfg.training.use_torch_compile, fullgraph=False)(self._compiled_step)

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
                log_if_rank_zero(
                    logger,
                    "Compiling the model with torch compile...",
                )
                causal_model = torch.compile(causal_model, dynamic=True, mode="max-autotune")
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
        
        self.compiled_step(batch["input_ids"], batch["attention_mask"], batch["labels"])

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

    # Modify configure_optimizers to use the JIT-compiled function

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
        total_iters = self.total_training_steps
        min_lr = self.cfg.lr_scheduler.min_learning_rate
        max_lr = self.cfg.lr_scheduler.max_learning_rate

        # Define a lambda function that wraps the JIT-compiled function
        log_if_rank_zero(
            logger,
            f"Using learning rate scheduler: {self.cfg.lr_scheduler.name}",
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

    def _compiled_step(self, input_ids, attention_mask, labels):
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # Get the loss and calculate perplexity and bits per byte (BPB)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        bpb = loss * self.bpb_term

        # Add metrics logging
        self.log_dict(
            {"loss": loss, "perplexity": perplexity, "bits_per_byte": bpb},
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
        it (int): Current training step.
        warmup_iters (int): Number of warmup steps.
        total_iters (int): Total training steps.
        max_lr (float): Maximum learning rate.

    Returns:
        float: The learning rate multiplier.
    """
    if it < warmup_iters:
        return float(it) / float(warmup_iters)  # Linear warmup
    elif it >= total_iters:
        return min_lr
    else:
        # Linear decay to min_lr
        return min_lr + (max_lr - min_lr) * (total_iters - it) / total_iters

