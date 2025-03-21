import json
import logging
import math
import os
from typing import *

import lightning as L
import torch
import transformers
from lion_pytorch import Lion
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

from src.evaluation.LM.last_word_prediction import evaluate_last_word_prediction
from src.evaluation.LM.next_token_prediction import (
    compute_perplexity_and_bpb,
    evaluate_next_token_prediction,
)
from src.evaluation.NLU.text_to_text_prediction import evaluate_text_to_text_prediction
from src.model.llama.causal_modeling import LlamaForCausalLM
from src.model.llama.model import Llama
from src.model.rellama.causal_modeling import ReLlamaForCausalLM
from src.model.rellama.model import ReLlama
from src.model.utils import (
    add_to_tensor_dict_safely,
    get_compile_decorator,
    lr_lambda_cosine_decay,
    lr_lambda_linear_decay,
    update_batch_step_in_checkpoint_to_consider_gradient_accumulation,
    update_position_in_checkpoint_for_consistency,
)
from src.tokenization import ReLlamaTokenizer
from src.tokenization.registry import TOKENIZER_REGISTRY
from src.utils import is_torch_compile_possible, log_if_rank_zero

logger = logging.getLogger("LightningModule")


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
        self.test_step_outputs = TensorDict({})
        self.register_buffer("cumulative_tokens", torch.tensor(0, dtype=torch.int64))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.cfg.training.detailed_logging:
            os.makedirs(self.log_dir, exist_ok=True)

    @property
    def log_dir(self) -> str:
        return os.path.join(
            self.cfg.root_dir_path,
            self.cfg.log_dir,
            self.cfg.tag,
        )

    @property
    def log_path(self) -> str:
        return os.path.join(
            self.log_dir, f"intense_logging_rank_{self.trainer.local_rank}.jsonl"
        )

    @property
    def uncompiled_model(self) -> transformers.LlamaForCausalLM:
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def initialize_tokenizer(self) -> ReLlamaTokenizer:
        tokenizer = TOKENIZER_REGISTRY[self.cfg.model.name].from_pretrained(
            self.cfg.model.base_name
        )
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
        elif self.cfg.model.name == "llama":
            model = Llama(self.cfg, self.tokenizer)
            causal_model = LlamaForCausalLM(base_model=model)
        else:
            raise ValueError(f"Model name {self.cfg.model.name} not supported")

        # Compile the model if the config is set to True and the GPU has the capability to compile the model
        if self.cfg.get("use_torch_compile", False):
            if is_torch_compile_possible():
                log_if_rank_zero(
                    logger,
                    "Compiling the model with torch compile...",
                )
                mode = None if self.cfg.model.name == "rellama" else "max-autotune"
                causal_model = torch.compile(causal_model, dynamic=True, mode=mode)
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
        attention_mask: Optional[torch.Tensor] = None,
        retrieved_input_ids: Optional[torch.Tensor] = None,
        num_retrieval_blocks: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Any:
        # Use inference_mode when not training
        if not self.training:
            with torch.inference_mode():
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=use_cache,
                    retrieved_input_ids=retrieved_input_ids,
                    num_retrieval_blocks=num_retrieval_blocks,
                    **kwargs,
                )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            retrieved_input_ids=retrieved_input_ids,
            num_retrieval_blocks=num_retrieval_blocks,
            **kwargs,
        )

    def on_train_batch_start(self, batch: Dict[str, Any], batch_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx)
        # Set the position of the sampler to the batch index
        # This is to update the sampler state to save in the checkpoint
        self.trainer.train_dataloader.sampler.set_position_by_batch_step(
            batch_idx, self.cfg.training.per_device_batch_size
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        if self.cfg.training.detailed_logging:
            # Log the batch_idx and the data index
            with open(self.log_path, "a") as f:
                data_to_dump = {
                    "epoch": self.trainer.current_epoch,
                    "batch_idx": batch_idx,
                    "data_idx": batch["data_idx"],
                    "input_ids_len": [len(item) for item in batch["input_ids"]],
                    "attention_mask_sum": [
                        sum(item).item() for item in batch["attention_mask"]
                    ],
                }
                if (
                    "retrieved_input_ids" in batch
                    and batch["retrieved_input_ids"] is not None
                ):
                    data_to_dump["retrieved_input_ids"] = [
                        len(item) for item in batch["retrieved_input_ids"]
                    ]
                if (
                    "num_retrieval_blocks" in batch
                    and batch["num_retrieval_blocks"] is not None
                ):
                    data_to_dump["num_retrieval_blocks"] = batch["num_retrieval_blocks"]
                f.write(json.dumps(data_to_dump) + "\n")

        self.compiled_step(
            batch["input_ids"],
            batch["attention_mask"],
            batch["retrieved_input_ids"],
            batch["num_retrieval_blocks"],
            batch["labels"],
            batch["avg_char_per_token"],
        )

        # Add the number of valid tokens to the cumulative tokens
        if self.cumulative_tokens.device != self.device:
            # Hack to change the device of the cumulative tokens to the device of the batch
            self.cumulative_tokens = self.cumulative_tokens.to(self.device)
        self.cumulative_tokens += batch["total_valid_tokens_cnt"]

        # Perform selective logging (i.e., only at the logging steps) of cumulative tokens
        if batch_idx % self.trainer.log_every_n_steps == 0:
            # Compute a global step that counts how many times `training_step` has actually been called so far.
            # This is NOT the same as the number of optimizer steps (which is fewer if gradient accumulation > 1).
            global_training_step = (
                self.trainer.current_epoch * self.trainer.num_training_batches
                + batch_idx
            )

            # First gather and sum tokens across processes
            gathered_tokens = self.all_gather(self.cumulative_tokens)

            # Log the total tokens across all processes with that step
            self.logger.log_metrics(
                {
                    "cumulative_num_tokens": torch.sum(gathered_tokens),
                    "global_step": self.global_step,
                },
                step=global_training_step,
            )

        if (batch_idx + 1) % self.cfg.training.gradient_accumulation_steps == 0:
            # Log the calling optimizer
            if self.cfg.training.detailed_logging:
                with open(self.log_path, "a") as f:
                    data_to_dump = {"calling_optimizer": True}
                    f.write(json.dumps(data_to_dump) + "\n")
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
            # Log the optimizer called
            if self.cfg.training.detailed_logging:
                with open(self.log_path, "a") as f:
                    data_to_dump = {"optimzer_called": True}
                    f.write(json.dumps(data_to_dump) + "\n")

        return None

    @torch.compiler.disable()
    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        b_size = len(batch["input_ids"])

        # Identify the validation dataset
        val_dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset.name

        # Lets perform evaluation
        log_dic = {}
        if val_dataset_name == "lambada":
            # Last word prediction
            accuracy: float = evaluate_last_word_prediction(
                model_name=self.cfg.model.name,
                model=self.uncompiled_model,
                tokenizer=self.tokenizer,
                batch_token_ids=batch["input_ids"],
                target_last_words=batch["last_word"],
                retrieved_input_ids=batch["retrieved_input_ids"],
                num_retrieval_blocks=batch["num_retrieval_blocks"],
            )
            log_dic = {"LWP_lambada_acc": accuracy}
        elif val_dataset_name in ["wikitext", "curation"]:
            # Next token prediction
            loss_sum, valid_tokens_cnt = evaluate_next_token_prediction(
                token_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                model=self.uncompiled_model,
                retrieved_input_ids=batch["retrieved_input_ids"],
                num_retrieval_blocks=batch["num_retrieval_blocks"],
            )
            # Compute perplexity and bpb
            perplexity, bpb = compute_perplexity_and_bpb(
                loss_sum, valid_tokens_cnt, batch["total_chars_cnt"]
            )
            log_dic = {
                f"NTP_{val_dataset_name}_perplexity": perplexity,
                f"NTP_{val_dataset_name}_bpb": bpb,
            }
        else:
            raise ValueError(f"Validation step {dataloader_idx} not implemented")

        # Log the results
        if log_dic:
            self.log_dict(
                log_dic,
                batch_size=b_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        return None

    @torch.compiler.disable()
    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        bsize = len(batch["input_ids"])

        # Identify the test dataset
        task_name: str = self.trainer.test_dataloaders[dataloader_idx].dataset.task_name
        test_dataset_name: str = self.trainer.test_dataloaders[
            dataloader_idx
        ].dataset.name

        # Perform evaluation
        if task_name == "last_word_prediction":
            accuracy: float = evaluate_last_word_prediction(
                model_name=self.cfg.model.name,
                model=self.uncompiled_model,
                tokenizer=self.tokenizer,
                batch_token_ids=batch["input_ids"],
                target_last_words=batch["last_word"],
                pad_start_positions=batch["pad_start_positions"],
                retrieved_input_ids=batch["retrieved_input_ids"],
                num_retrieval_blocks=batch["num_retrieval_blocks"],
            )
            add_to_tensor_dict_safely(
                self.test_step_outputs, "LWP_lambada_acc_sum", accuracy * bsize
            )
            add_to_tensor_dict_safely(
                self.test_step_outputs, "LWP_lambada_acc_cnt", bsize
            )
        elif task_name == "next_token_prediction":
            loss_sum, valid_tokens_cnt = self._handle_batch_for_next_token_prediction(
                batch
            )
            add_to_tensor_dict_safely(
                self.test_step_outputs, f"NTP_{test_dataset_name}_loss_sum", loss_sum
            )
            add_to_tensor_dict_safely(
                self.test_step_outputs,
                f"NTP_{test_dataset_name}_valid_tokens_cnt",
                valid_tokens_cnt,
            )
            add_to_tensor_dict_safely(
                self.test_step_outputs,
                f"NTP_{test_dataset_name}_total_chars_cnt",
                batch["total_chars_cnt"],
            )
        elif task_name == "natural_language_understanding":
            # Natural language inference
            accuracy = self._handle_batch_for_natural_language_understanding(batch)
            add_to_tensor_dict_safely(
                self.test_step_outputs,
                f"NLU_{test_dataset_name}_acc_sum",
                accuracy * bsize,
            )
            add_to_tensor_dict_safely(
                self.test_step_outputs, f"NLU_{test_dataset_name}_acc_cnt", bsize
            )
        else:
            raise ValueError(f"Test step {task_name} not implemented")
        return None

    def on_test_epoch_end(self) -> None:
        # Gather and sum over all processes
        gathered_step_outpus: TensorDict = self.all_gather(self.test_step_outputs)
        for key in gathered_step_outpus.keys():
            gathered_step_outpus[key] = gathered_step_outpus[key].sum()
        # Compute the average metrics for each task
        for task_name in self.cfg.testing.task_names:
            if task_name == "last_word_prediction":
                for dataset_name in self.cfg.task[task_name].dataset_names:
                    # Calculate the accuracy
                    avg_accuracy = (
                        gathered_step_outpus[f"LWP_{dataset_name}_acc_sum"]
                        / gathered_step_outpus[f"LWP_{dataset_name}_acc_cnt"]
                    )
                    log_if_rank_zero(
                        logger,
                        f"LWP_{dataset_name} Accuracy: {avg_accuracy} (Total: {gathered_step_outpus[f'LWP_{dataset_name}_acc_cnt']})",
                    )
            elif task_name == "next_token_prediction":
                for dataset_name in self.cfg.task[task_name].dataset_names:
                    # Calculate the perplexity and bpb
                    perplexity, bpb = compute_perplexity_and_bpb(
                        total_loss_sum=gathered_step_outpus[
                            f"NTP_{dataset_name}_loss_sum"
                        ],
                        total_valid_tokens_cnt=gathered_step_outpus[
                            f"NTP_{dataset_name}_valid_tokens_cnt"
                        ],
                        total_chars_cnt=gathered_step_outpus[
                            f"NTP_{dataset_name}_total_chars_cnt"
                        ],
                    )
                    log_if_rank_zero(
                        logger,
                        f"NTP_{dataset_name} Perplexity: {perplexity} (Total tokens: {gathered_step_outpus[f'NTP_{dataset_name}_valid_tokens_cnt']})",
                    )
                    log_if_rank_zero(
                        logger,
                        f"NTP_{dataset_name} Bits per byte: {bpb} (Total characters: {gathered_step_outpus[f'NTP_{dataset_name}_total_chars_cnt']})",
                    )
            elif task_name == "natural_language_understanding":
                for dataset_name in self.cfg.task[task_name].dataset_names:
                    # Calculate the accuracy
                    avg_accuracy = (
                        gathered_step_outpus[f"NLU_{dataset_name}_acc_sum"]
                        / gathered_step_outpus[f"NLU_{dataset_name}_acc_cnt"]
                    )
                    log_if_rank_zero(
                        logger,
                        f"NLU_{dataset_name} Accuracy: {avg_accuracy} (Total: {gathered_step_outpus[f'NLU_{dataset_name}_acc_cnt']})",
                    )
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
        warmup_iters: int = self.cfg.lr_scheduler.warmup_steps
        intermediate_iters: int = self.cfg.lr_scheduler.intermediate_steps
        total_iters: int = self.total_optimization_steps
        max_lr: float = self.cfg.lr_scheduler.max_learning_rate
        min_lr: float = self.cfg.lr_scheduler.min_learning_rate
        intermediate_lr: float = self.cfg.lr_scheduler.intermediate_learning_rate

        if intermediate_iters is None:
            # Check if the configurations are valid
            assert intermediate_lr > min_lr and intermediate_lr < max_lr, (
                f"The intermediate learning rate ({intermediate_lr}) must be greater than "
                f"the minimum learning rate ({min_lr}) and less than the maximum learning rate ({max_lr})"
            )
            assert (
                intermediate_iters > warmup_iters
            ), f"The intermediate steps ({intermediate_iters}) must be greater than the warmup steps ({warmup_iters})"
            assert (
                intermediate_iters < total_iters
            ), f"The intermediate steps ({intermediate_iters}) must be less than the total steps ({total_iters})"

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
                it,
                warmup_iters,
                intermediate_iters,
                total_iters,
                max_lr,
                intermediate_lr,
                min_lr,
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
        retrieved_input_ids: Optional[torch.Tensor] = None,
        num_retrieval_blocks: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        avg_char_per_token: float = 0.0,
    ) -> None:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            retrieved_input_ids=retrieved_input_ids,
            num_retrieval_blocks=num_retrieval_blocks,
            labels=labels,
            use_cache=False,
        )
        # Get the loss and calculate perplexity and bits per byte (BPB)
        loss = outputs.loss
        perplexity = torch.exp(loss)

        # Convert loss (nats) to bits
        loss_in_bits = loss * math.log2(math.e)

        # Compute bits per byte (BPB)
        bpb = loss_in_bits / avg_char_per_token

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

        # Log the loss
        if self.cfg.training.detailed_logging:
            with open(self.log_path, "a") as f:
                data_to_dump = {
                    "loss": loss.item(),
                }
                f.write(json.dumps(data_to_dump) + "\n")

        # Backward
        self.manual_backward(loss)

        # Log the backward
        if self.cfg.training.detailed_logging:
            with open(self.log_path, "a") as f:
                data_to_dump = {"backward": True}
                f.write(json.dumps(data_to_dump) + "\n")

        return None

    def _handle_batch_for_next_token_prediction(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[float, int]:
        """Next token prediction can be evaluated for the whole batch at once."""
        loss_sum, valid_tokens_cnt = evaluate_next_token_prediction(
            token_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            model=self.uncompiled_model,
            retrieved_input_ids=batch["retrieved_input_ids"],
            num_retrieval_blocks=batch["num_retrieval_blocks"],
        )
        return loss_sum, valid_tokens_cnt

    def _handle_batch_for_natural_language_understanding(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """Natural language understanding should be evaluated for each instance in the batch."""
        bsize = len(batch["input_ids"])
        is_correct_list: List[bool] = []
        # last_word_prediction expects no padding
        for b_idx in range(bsize):
            batch_token_ids = batch["input_ids"][b_idx].unsqueeze(0)
            target_texts: List[str] = batch["target"][b_idx : b_idx + 1]
            text_choices: List[str] = batch["choices"]
            retrieved_chunk_ids = (
                None
                if batch["retrieved_chunk_ids"] is None
                else batch["retrieved_chunk_ids"][b_idx].unsqueeze(0)
            )
            is_correct_list.extend(
                evaluate_text_to_text_prediction(
                    batch_token_ids=batch_token_ids,
                    target_texts=target_texts,
                    text_choices=text_choices,
                    tokenizer=self.tokenizer,
                    model=self.uncompiled_model,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                )
            )
        return sum(is_correct_list) / bsize

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Update the batch_step so that we can resume from the correct position
        # even if gradient accumulation is used
        checkpoint = update_batch_step_in_checkpoint_to_consider_gradient_accumulation(
            checkpoint, self.cfg.training.gradient_accumulation_steps
        )

        # Modify the position of the sampler with the batches that stepped
        # This is a hack to make sure the sampler is at the correct position
        # Not sure why the two numbers are different...
        checkpoint = update_position_in_checkpoint_for_consistency(
            checkpoint, self.cfg.training.per_device_batch_size
        )

        # Apply the checkpoint
        return super().on_load_checkpoint(checkpoint)
