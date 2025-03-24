import torch
import torch.nn.functional as F

def disentangle_loss_causalLM(
    logits_with_D: torch.Tensor,
    logits_wo_D: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    margin: float = 1.0,
    ignore_index: int = -100,
):
    """
    Compute a disentanglement loss for causal language modeling.
    
    Args:
        logits_with_D: Tensor of shape (batch_size, seq_len, vocab_size)
        logits_wo_D:   Tensor of shape (batch_size, seq_len, vocab_size)
        labels:        Tensor of shape (batch_size, seq_len)
        alpha:         Weight for the margin penalty term.
        margin:        The margin threshold for the penalty.
        ignore_index:  Token index to ignore during loss computation.
    
    Returns:
        A scalar tensor representing the average loss over all tokens.
    """
    # Ensure logits are in float precision and on the same device as labels.
    device = logits_with_D.device
    logits_with_D = logits_with_D.float()
    logits_wo_D   = logits_wo_D.float()
    labels = labels.to(device)
    
    # Align predictions and targets:
    # Use logits[:, :-1, :] to predict tokens at positions 1...L from inputs at 0...L-1.
    logits_with_D = logits_with_D[:, :-1, :]
    logits_wo_D   = logits_wo_D[:, :-1, :]
    target = labels[:, 1:].contiguous()

    # Flatten tensors for per-token loss computation.
    vocab_size = logits_with_D.size(-1)
    logits_with_D = logits_with_D.reshape(-1, vocab_size)
    logits_wo_D   = logits_wo_D.reshape(-1, vocab_size)
    target = target.reshape(-1)

    # Compute per-token cross-entropy losses.
    ce_with_D = F.cross_entropy(
        logits_with_D, target, ignore_index=ignore_index, reduction='none'
    )
    ce_wo_D = F.cross_entropy(
        logits_wo_D, target, ignore_index=ignore_index, reduction='none'
    )

    # Compute the margin penalty:
    # We desire ce_wo_D - ce_with_D >= margin; if not, we incur a penalty.
    penalty = F.relu(margin - (ce_wo_D - ce_with_D))
    
    # Combine the losses and average over all tokens.
    loss = (ce_with_D + ce_wo_D + alpha * penalty).mean()
    return loss