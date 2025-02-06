import torch
import transformers
from omegaconf import DictConfig


def get_llama_config(
    cfg: DictConfig, tokenizer: transformers.PreTrainedTokenizer
) -> transformers.LlamaConfig:
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
    llama_config.head_dim = llama_config.hidden_size // llama_config.num_attention_heads
    llama_config.intermediate_size = cfg.model.architecture.intermediate_size
    llama_config.num_hidden_layers = cfg.model.architecture.layers
    llama_config.max_position_embeddings = cfg.model.max_length
    llama_config.torch_dtype = torch.float32

    return llama_config


# Custom weight initialization function
def initialize_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
        # Initialize weights for Linear and Conv1d layers
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        # Initialize weights for Embedding layers
        torch.nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)
