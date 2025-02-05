import logging
from typing import *

import lightning as L
import torch
from omegaconf import DictConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.model.utils import initialize_weights, modify_architecture

logger = logging.getLogger("RETROLightningModule")

class RETROLightningModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, total_steps: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.total_training_steps = total_steps
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.base_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)
        self._init_weights()

    def _init_weights(self) -> None:
        # Modify the architecture of the model
        modify_architecture(model=self.model, cfg=self.cfg.model.architecture)
        
        # Initialize model weights if not resuming from checkpoint
        if self.cfg.training.resume_ckpt_path is None:
            logger.info("Applying xavier uniform initialization to model weights for pretraining from scratch")
            self.model.apply(initialize_weights)
            
        # Check the number of tokens in the vocabulary
        # If the vocabulary size is not multiple of 64, add padding tokens (to increase throughput)
        # https://x.com/karpathy/status/1621578354024677377?lang=en for more details.
        vocab_size = self.model.lm_head.out_features
        if vocab_size % 64 != 0:
            padding_size = 64 - vocab_size % 64
            logger.info(f"Padding vocabulary size to multiple of 64: {vocab_size} -> {vocab_size + padding_size}")
            self.model.resize_token_embeddings(vocab_size + padding_size)
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Any:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
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
            {
                "train_loss": loss,
                "train_perplexity": perplexity
            },
            batch_size=batch["input_ids"].size(0),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        logger.info(f"Train loss: {loss}, perplexity: {perplexity}")
        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        perplexity = torch.exp(loss)
        # Perform logging
        self.log_dict(
            {
                "val_loss": loss,
                "val_perplexity": perplexity
            },
            batch_size=batch["input_ids"].size(0),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return None
    
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.training.warmup_steps, num_training_steps=self.total_training_steps
        )
        return [optimizer], [scheduler]