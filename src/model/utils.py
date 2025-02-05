import torch.nn as nn
from omegaconf import DictConfig
from transformers import LlamaForCausalLM, LlamaConfig


# Custom weight initialization function
def initialize_weights(module: nn.Module) -> None:
    """
    Initializes weights for neural network modules using standard initialization schemes.
    
    - Linear and Embedding layers: Xavier uniform initialization for weights, zeros for biases
    - LayerNorm layers: Ones for weights, zeros for biases
    
    Args:
        module (nn.Module): PyTorch module whose weights need to be initialized
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
        
        
# Modify the architecture of the model
def modify_architecture(model: nn.Module, cfg: DictConfig) -> None:
    """
    Modifies the architecture of a Llama model according to the provided configuration.
    Adjusts the number of layers, attention heads, and dimensions while ensuring compatibility
    with Grouped-Query Attention (GQA). Only works with LlamaForCausalLM models.

    Args:
        model (nn.Module): A LlamaForCausalLM model instance to be modified
        cfg (DictConfig): Configuration containing model architecture parameters:
            - layers: number of transformer layers
            - heads: number of attention heads
            - query_groups: number of GQA query groups
            - intermediate_dim: dimension of MLP intermediate layer
            - embedding_dim: dimension of embeddings and attention
    """
    # Assert that the model is a Llama model
    assert isinstance(model, LlamaForCausalLM), "Model must be a LlamaForCausalLM instance"
    assert isinstance(model.config, LlamaConfig), "Model config must be a LlamaConfig instance"
    
    # Get the architecture parameters
    layer_num = cfg.layers
    head_num = cfg.heads
    query_group_num = cfg.query_groups
    intermediate_dim = cfg.intermediate_dim
    embedding_dim = cfg.embedding_dim
    
    # Ensure embedding_dim is divisible by head_num and query_group_num
    assert embedding_dim % head_num == 0, f"embedding_dim ({embedding_dim}) must be divisible by head_num ({head_num})"
    assert head_num % query_group_num == 0, f"head_num ({head_num}) must be divisible by query_group_num ({query_group_num})"
    
    # Calculate dimensions for GQA
    head_dim = embedding_dim // head_num
    kv_heads = query_group_num  # number of key/value heads
    num_key_value_groups = head_num // query_group_num  # number of query heads per key/value head
    
    # Verify the model has the expected number of layers
    assert hasattr(model, 'model') and hasattr(model.model, 'layers'), "Model structure does not match Llama architecture"
    assert len(model.model.layers) == layer_num, f"Model has {len(model.model.layers)} layers, but config specifies {layer_num}"
    
    # Modify the architecture of all layers in the model
    for i in range(layer_num):
        # Update reference to match Llama architecture
        layer = model.model.layers[i]
        layer.self_attn.num_heads = head_num
        layer.self_attn.num_key_value_heads = kv_heads
        layer.self_attn.head_dim = head_dim
        
        # For GQA, update the linear projections
        q_dim = head_num * head_dim
        kv_dim = kv_heads * head_dim * 2
        layer.self_attn.q_proj = nn.Linear(embedding_dim, q_dim)
        layer.self_attn.k_proj = nn.Linear(embedding_dim, kv_dim // 2)
        layer.self_attn.v_proj = nn.Linear(embedding_dim, kv_dim // 2)
        layer.self_attn.o_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Update MLP layers
        layer.mlp.gate_proj = nn.Linear(embedding_dim, intermediate_dim)
        layer.mlp.up_proj = nn.Linear(embedding_dim, intermediate_dim)
        layer.mlp.down_proj = nn.Linear(intermediate_dim, embedding_dim)
    return None