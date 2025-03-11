from typing import *

import hkkang_utils.time as time_utils
import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import GenerationMixin
from transformers.models.llama.modeling_llama import (KwargsForCausalLM,
                                                      LlamaModel,
                                                      LlamaPreTrainedModel)

from src.model.rellama.model import ReLlama
from src.model.utils import initialize_weights


class ReLlamaCausalLMOutputWithPast(CausalLMOutputWithPast):
    def __init__(
        self,
        *args,
        retrieval_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retrieval_key_values = retrieval_key_values


class ReLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, base_model: Union[ReLlama]):
        # Get llama config
        super().__init__(base_model.config)
        self.model = base_model
        self.vocab_size = base_model.config.vocab_size
        self.lm_head = torch.nn.Linear(
            base_model.config.hidden_size, base_model.config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()
        self._custom_init()

    def _custom_init(self):
        self.model.apply(initialize_weights)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        retrieval_key_values: Optional[
            List[Tuple[torch.FloatTensor, torch.FloatTensor]]
        ] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        retrieved_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        num_retrieval_blocks: Optional[List[int]] = None,
        **kwargs: KwargsForCausalLM,
    ) -> Union[Tuple, ReLlamaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        bsize = input_ids.shape[0]
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Encode the retrieved data with no grad
        has_retrieval_data = (
            retrieved_input_ids is not None and retrieved_input_ids.numel() > 0
        )
        # Create retrieval key values if there is retrieval data and key values are not provided
        if has_retrieval_data and retrieval_key_values is None:
            with time_utils.Timer("EncodeRetrieval").measure(True):
                with torch.no_grad():
                    max_retrieval_block_num = max(num_retrieval_blocks)
                    block_num, chunk_num, chunk_len = retrieved_input_ids.shape
                # We consider block_num * chunk_num as the batch size when we encode the retrieved data
                retrieved_input_ids = retrieved_input_ids.view(
                    block_num * chunk_num, chunk_len
                )
                retrieved_data_embeds = self.model(
                    input_ids=retrieved_input_ids,
                    use_cache=True,
                    retrieval_block_num=0,
                    is_retrieval=True,
                )
                retrieval_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for layer_idx, (key_vecs, value_vecs) in enumerate(
                    retrieved_data_embeds.past_key_values
                ):
                    _, nhead, chunk_len, head_dim = key_vecs.shape
                    key_vecs = (
                        key_vecs.view(block_num, chunk_num, nhead, chunk_len, head_dim)
                        .permute(2, 0, 1, 3, 4)
                        .reshape(nhead, block_num, chunk_num * chunk_len, head_dim)
                    )
                    value_vecs = (
                        value_vecs.view(
                            block_num, chunk_num, nhead, chunk_len, head_dim
                        )
                        .permute(2, 0, 1, 3, 4)
                        .reshape(nhead, block_num, chunk_num * chunk_len, head_dim)
                    )
                    # Splite by the retrieval block num
                    key_tmp = []
                    value_tmp = []
                    for bidx in range(bsize):
                        retrieval_block_num = num_retrieval_blocks[bidx]
                        block_start_idx = sum(num_retrieval_blocks[:bidx])
                        block_end_idx = block_start_idx + retrieval_block_num
                        # Reshape the key vectors
                        tmp = key_vecs[:, block_start_idx:block_end_idx, :]
                        tmp = tmp.reshape(
                            nhead, retrieval_block_num * chunk_num * chunk_len, head_dim
                        )
                        key_tmp.append(tmp)
                        # Reshape the value vectors
                        tmp = value_vecs[:, block_start_idx:block_end_idx, :]
                        tmp = tmp.reshape(
                            nhead, retrieval_block_num * chunk_num * chunk_len, head_dim
                        )

                        value_tmp.append(tmp)
                        # Add padding to the key and value vectors if the retrieval block num is not the max
                        if retrieval_block_num < max_retrieval_block_num:
                            diff = max_retrieval_block_num - retrieval_block_num
                            key_tmp[-1] = torch.cat(
                                [
                                    key_tmp[-1],
                                    torch.zeros(
                                        (nhead, diff * chunk_num * chunk_len, head_dim),
                                        device=key_tmp[-1].device,
                                    ),
                                ],
                                dim=1,
                            )
                            value_tmp[-1] = torch.cat(
                                [
                                    value_tmp[-1],
                                    torch.zeros(
                                        nhead,
                                        diff * chunk_num * chunk_len,
                                        head_dim,
                                        device=value_tmp[-1].device,
                                    ),
                                ],
                                dim=1,
                            )
                    key_vecs = torch.stack(key_tmp, dim=0)
                    value_vecs = torch.stack(value_tmp, dim=0)
                    retrieval_key_values.append((key_vecs, value_vecs))

        with time_utils.Timer("EncodeMain").measure(True):
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                retrieval_key_values=retrieval_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                is_retrieval=False,
                **kwargs,
            )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ReLlamaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            retrieval_key_values=retrieval_key_values if use_cache else None,
        )
