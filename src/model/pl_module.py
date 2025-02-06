import logging
from typing import *

import lightning as L
import torch
from omegaconf import DictConfig
from transformers import get_cosine_schedule_with_warmup

from src.model.utils import initialize_weights
from src.tokenizer import RETROTokenizer
import transformers

logger = logging.getLogger("RETROLightningModule")


class RETROLightningModule(L.LightningModule):
    def __init__(self, cfg: DictConfig, total_steps: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.total_training_steps = total_steps
        self.model: transformers.LlamaForCausalLM = self.initialize_language_model(
            cfg=cfg
        )
        self.save_hyperparameters(cfg)

    def initialize_language_model(
        self, cfg: DictConfig
    ) -> transformers.LlamaForCausalLM:
        # Get the tokenizer
        tokenizer = RETROTokenizer.from_pretrained(cfg.model.base_name)

        # Get the config from the pre-trained models
        llama_config = transformers.LlamaConfig.from_pretrained(cfg.model.base_name)

        # Modify the configs of the model
        llama_config.vocab_size = len(tokenizer)
        llama_config.pad_token_id = tokenizer.pad_token_id
        llama_config.bos_token_id = tokenizer.bos_token_id
        llama_config.eos_token_id = tokenizer.eos_token_id
        llama_config.num_attention_heads = cfg.model.architecture.num_attention_heads
        llama_config.num_key_value_heads = cfg.model.architecture.num_key_value_heads
        llama_config.hidden_size = cfg.model.architecture.hidden_size
        assert (
            llama_config.hidden_size % llama_config.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"
        llama_config.head_dim = (
            llama_config.hidden_size // llama_config.num_attention_heads
        )
        llama_config.intermediate_size = cfg.model.architecture.intermediate_size
        llama_config.num_hidden_layers = cfg.model.architecture.layers
        llama_config.max_position_embeddings = cfg.model.max_length
        llama_config.torch_dtype = torch.float32
        llama_config.rope_scaling = None
        # Initialize the model
        model = transformers.LlamaForCausalLM(config=llama_config)

        # Initialize model weights if not resuming from checkpoint
        if cfg.training.resume_ckpt_path is None:
            logger.info(
                "Applying xavier uniform initialization to model weights for pretraining from scratch"
            )
            model.apply(initialize_weights)

        # Compile the model if the config is set to True and the GPU has the capability to compile the model
        if cfg.training.use_torch_compile:
            if torch.cuda.get_device_capability()[0] >= 7:
                model = torch.compile(model, dynamic=True)
            else:
                logger.info(
                    "Torch compile is not supported on this GPU. Use_torch_compile is set to True, but the GPU does not support torch compile."
                )

        return model

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

    # def validation_step(
    #     self,
    #     batch: Dict[str, torch.Tensor],
    #     batch_idx: int
    # ) -> None:
    #     outputs = self(
    #         input_ids=batch["input_ids"],
    #         attention_mask=batch["attention_mask"],
    #         labels=batch["labels"],
    #     )
    #     loss = outputs.loss
    #     perplexity = torch.exp(loss)
    #     # Perform logging
    #     self.log_dict(
    #         {
    #             "val_loss": loss,
    #             "val_perplexity": perplexity
    #         },
    #         batch_size=batch["input_ids"].size(0),
    #         on_step=False,
    #         on_epoch=True,
    #         sync_dist=True,
    #     )
    #     return None

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.max_learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

        # Create a chain of schedulers: linear warmup followed by cosine decay to min_lr
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,  # Changed from 0.0 to small positive value
            end_factor=1.0,  # End at max_lr
            total_iters=self.cfg.training.warmup_steps,
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
