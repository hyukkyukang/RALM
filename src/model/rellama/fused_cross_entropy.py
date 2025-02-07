from flash_attn.losses.cross_entropy import CrossEntropyLoss


def fixed_cross_entropy(
    source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)(
        source, target
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(
        shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss
