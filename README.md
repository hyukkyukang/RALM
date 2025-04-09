# Retrieval-augmented Language Model


## Install dependencies
Install torch and legacy-cgi (for Python3.13)
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
pip install legacy-cgi
```

Install flash-attention
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

Install other dependencies
```bash
pip install -r requirements.txt
```

## Datasets
### Download the Lambada dataset
1. Download the Lambada dataset used in GPT-2 paper from [here](https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl).
2. Move the file to `data/lambada_gpt2_author.jsonl`.
```bash
wget -O data/lambada_gpt2_author.jsonl https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl
```

## Retrieval

```bash
export TORCH_COMPILE_DISABLE=1
export PYTORCH_DISABLE_GPU_KERNELS=1
```

### Conduct retrieval for the target dataset
```bash
python scripts/retrieval/retrieve.py \
    +target_dataset=lambada
```

### Combine retrieved chunks for the target dataset
```bash
python scripts/retrieval/combine_retrieved_chunks.py \
    +target_dataset=lambada
```

## Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/pretrain.py \
    _global.tag=lion_1e-4 \
    optimizer=lion \
    lr_scheduler=linear_decay \
    lr_scheduler.max_learning_rate=1e-4 \
    training.max_epochs=2 \
    training.gradient_accumulation_steps=9 \
    training.per_device_batch_size=38
```

## Evaluation
```bash
python scripts/evaluate.py \
    testing.task_names="[last_word_prediction, next_token_prediction]" \
    model=llama \
    +ckpt_path=/root/RETRO/runs/retro/lion.ckpt \
    testing.per_device_batch_size=1 \
    testing.num_workers=1
```