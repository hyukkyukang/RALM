from typing import *

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import GenerationMixin
from transformers.models.llama.modeling_llama import (
    KwargsForCausalLM,
    LlamaPreTrainedModel,
)

from src.model.rellama.model import ReLlama
from src.model.utils import initialize_weights
from src.objective import disentangle_loss_causalLM


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

    def __init__(self, base_model: ReLlama):
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
        pad_start_positions: Optional[List[int]] = None,
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
            if self.config.oracle_retrieval:
                # Use the input ids to encode and use as the retrieved_input_ids
                # Devide the input ids into chunks. Ex: (2, 86) -> [(2, 64), (2, 22)]
                # Then encode each chunk and use as the retrieval key values
                devided_input_ids: List[torch.Tensor] = list(
                    torch.split(input_ids, 64, dim=1)
                )

                # Pad the last chunk to the same length as the first chunk
                last_chunk = torch.ones_like(devided_input_ids[0])

                # Fill with padding token ids
                last_chunk[:] = self.config.pad_token_id
                last_chunk[:, : devided_input_ids[-1].shape[1]] = devided_input_ids[-1]
                devided_input_ids[-1] = last_chunk

                # Select the input chunks to be used as retrieval data
                selected_input_chunks: List[torch.Tensor] = []
                for b_idx, select_num in enumerate(num_retrieval_blocks):
                    for i in range(select_num):
                        # Skip the first chunk by adding 1 to the index
                        selected_input_chunks.append(devided_input_ids[i + 1][b_idx])
                retrieved_input_ids_by_input_ids = torch.stack(
                    selected_input_chunks, dim=0
                ).unsqueeze(1)

                # Replace the retrieved input ids with the selected input chunks
                assert (
                    retrieved_input_ids_by_input_ids.shape == retrieved_input_ids.shape
                ), f"retrieved_input_ids_by_input_ids.shape: {retrieved_input_ids_by_input_ids.shape}, retrieved_input_ids.shape: {retrieved_input_ids.shape}"
                retrieved_input_ids = retrieved_input_ids_by_input_ids

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
            retrieval_key_values = unpack_and_pad_retrieval_key_value_states(
                retrieved_data_embeds.past_key_values,
                num_retrieval_blocks,
                block_num,
                chunk_num,
                max_retrieval_block_num,
            )
        # Decode with retrieval key values states
        outputs_w_D = self.model(
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
            pad_start_positions=pad_start_positions,
            is_retrieval=False,
            **kwargs,
        )

        hidden_states_w_D = outputs_w_D[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits_w_D = self.lm_head(hidden_states_w_D[:, -num_logits_to_keep:, :])

        # Compute the loss if labels are provided
        loss = None
        if labels is not None:
            if self.config.use_disentangle_loss and self.training:
                # Compute the logits without retrieval
                outputs_wo_D = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                    pad_start_positions=pad_start_positions,
                    is_retrieval=False,
                    **kwargs,
                )

                # Final hidden states without retrieval
                hidden_states_wo_D = outputs_wo_D[0]
                logits_wo_D = self.lm_head(
                    hidden_states_wo_D[:, -num_logits_to_keep:, :]
                )

                # Compute the disentangle loss
                loss = disentangle_loss_causalLM(
                    logits_with_D=logits_w_D,
                    logits_wo_D=logits_wo_D,
                    labels=labels,
                    alpha=self.config.disentangle_alpha,
                    margin=self.config.disentangle_margin,
                )
            else:
                # We don't use disentangle loss, just compute the loss with the logits
                loss = self.loss_function(
                    logits=logits_w_D,
                    labels=labels,
                    vocab_size=self.config.vocab_size,
                    **kwargs,
                )

        if not return_dict:
            output = (logits_w_D,) + outputs_w_D[1:]
            return (loss,) + output if loss is not None else output

        return ReLlamaCausalLMOutputWithPast(
            loss=loss,
            logits=logits_w_D,
            past_key_values=outputs_w_D.past_key_values,
            hidden_states=outputs_w_D.hidden_states,
            attentions=outputs_w_D.attentions,
            retrieval_key_values=retrieval_key_values if use_cache else None,
        )


def unpack_and_pad_retrieval_key_value_states(
    key_value_states: List[Tuple[torch.Tensor, torch.Tensor]],
    num_retrieval_blocks: List[int],
    block_num: int,
    chunk_num: int,
    max_retrieval_block_num: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Unpack and pad the retrieval key and value states into batch-wise retrieval key and value states.
    """
    import torch.nn.functional as F

    # Original shape is (block_num * chunk_num, nhead, chunk_len, head_dim)
    _, nhead, chunk_len, head_dim = key_value_states[0][0].shape
    chunk_size = chunk_num * chunk_len
    target_length = max_retrieval_block_num * chunk_size
    retrieval_key_values = []

    for key_vecs, value_vecs in key_value_states:
        # Reshape: (block_num, chunk_num, nhead, chunk_len, head_dim)
        # Permute to (nhead, block_num, chunk_num, chunk_len, head_dim)
        # Finally reshape to (nhead, block_num, chunk_size, head_dim)
        key_vecs = (
            key_vecs.view(block_num, chunk_num, nhead, chunk_len, head_dim)
            .permute(2, 0, 1, 3, 4)
            .reshape(nhead, block_num, chunk_size, head_dim)
        )
        value_vecs = (
            value_vecs.view(block_num, chunk_num, nhead, chunk_len, head_dim)
            .permute(2, 0, 1, 3, 4)
            .reshape(nhead, block_num, chunk_size, head_dim)
        )

        # Split along the block dimension using num_retrieval_blocks.
        key_splits = torch.split(key_vecs, num_retrieval_blocks, dim=1)
        value_splits = torch.split(value_vecs, num_retrieval_blocks, dim=1)

        padded_keys = []
        padded_values = []
        # Process each batch element
        for ks, vs, nblocks in zip(key_splits, value_splits, num_retrieval_blocks):
            # ks and vs shape: (nhead, nblocks, chunk_size, head_dim)
            # Reshape to (nhead, nblocks * chunk_size, head_dim)
            ks = ks.reshape(nhead, nblocks * chunk_size, head_dim)
            vs = vs.reshape(nhead, nblocks * chunk_size, head_dim)
            pad_len = target_length - ks.shape[1]
            if pad_len > 0:
                ks = F.pad(ks, (0, 0, 0, pad_len))
                vs = F.pad(vs, (0, 0, 0, pad_len))
            padded_keys.append(ks)
            padded_values.append(vs)

        # Stack along batch dimension: (bsize, nhead, target_length, head_dim)
        key_out = torch.stack(padded_keys, dim=0)
        value_out = torch.stack(padded_values, dim=0)
        retrieval_key_values.append((key_out, value_out))

    return retrieval_key_values
