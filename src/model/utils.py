import torch


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
