import os

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler)


@torch.compile
def compute_loss_and_backward(model, batch, scaler, gradient_accumulation_steps) -> None:
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()
    return None


def train_gpt2():
    use_flash_attn = True
    use_mixed_precision = True
    use_torch_compile = True
    # Configuration
    # model_name = "gpt2" # GPT-2 Small (~124M parameters)
    # model_name = "gpt2-medium"  # GPT-2 Medium (~355M parameters)
    # model_name = "gpt2-large"  # GPT-2 Large (~774M parameters)
    model_name = "gpt2-xl"  # GPT-2 XL (~1.5B parameters)
    batch_size = 2
    learning_rate = 5e-5
    num_epochs = 1
    max_seq_length = 1024
    gradient_accumulation_steps = 10  # Adjust if GPU memory is limited
    world_size = 4  # Number of GPUs

    # Initialize the process group for DDP
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if use_flash_attn:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2"
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if use_torch_compile:
        model = torch.compile(model)

    # Enable mixed-precision training if the flag is set
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)

    # Add gradient clipping value
    max_grad_norm = 1.0

    # Create dummy dataset for debugging
    dummy_texts = [
        "This is a sample text for debugging GPT-2 training." * 66,
        "Another example of dummy text to test the training pipeline." * 99,
        "Debugging with small datasets can save time and resources." * 99,
        "GPT-2 models are powerful tools for text generation." * 99,
        "This text is part of the dummy dataset for GPT-2 debugging." * 70,
    ] * 10000
    dataset = Dataset.from_dict({"text": dummy_texts})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask"])

    # Print the number of tokens for the dummy text (only on rank 0)
    if local_rank == 0 and len(dummy_texts) < 500:
        for i, text in enumerate(dummy_texts):
            print(f"Text {i}: {text[:100]}")
            print(f"Num tokens: {len(tokenizer(text)['input_ids'])}")
        all_num_tokens = sum(
            min(len(tokenizer(text)["input_ids"]), max_seq_length)
            for text in dummy_texts
        )
        print(f"All num tokens: {all_num_tokens}")

    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(
        tokenized_datasets, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    train_dataloader = DataLoader(
        tokenized_datasets, sampler=train_sampler, batch_size=batch_size
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * num_epochs // world_size
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Shuffle data at the beginning of each epoch
        print(f"Epoch {epoch + 1}/{num_epochs} (Rank {local_rank})")

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1} (Rank {local_rank})",
            disable=local_rank != 0,
        )

        for step, batch in progress_bar:
            batch = {key: val.to(device) for key, val in batch.items()}

            if use_mixed_precision:
                if use_torch_compile:
                    compute_loss_and_backward(model, batch, scaler, gradient_accumulation_steps)
                else:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**batch, labels=batch["input_ids"])
                        loss = outputs.loss
                        loss = loss / gradient_accumulation_steps

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
            else:
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps

                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

            # if local_rank == 0:
            #     progress_bar.set_postfix({"Loss": loss.item()})

    print("Training completed on rank", local_rank)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    train_gpt2()
