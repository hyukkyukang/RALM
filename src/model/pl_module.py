import logging
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
        total_steps: int,
        tokenizer: Optional[ReLlamaTokenizer] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.total_training_steps = total_steps
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

        return causal_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

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
        # Add metrics logging
        self.log_dict(
            {"loss": loss, "perplexity": perplexity},
            batch_size=batch["input_ids"].size(0),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.max_learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2),
        )

        # Create a chain of schedulers: linear warmup followed by cosine decay to min_lr
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: epoch / self.cfg.training.warmup_steps,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.total_training_steps - self.cfg.training.warmup_steps,
            eta_min=self.cfg.training.min_learning_rate,  # Set your minimum learning rate here
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.cfg.training.warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
