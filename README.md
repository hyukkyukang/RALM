# retro

## Install dependencies
Install torch
```bash
pip install --pre torch==2.6.0.dev20250104+cu124 --index-url https://download.pytorch.org/whl/nightly/cu124
```

Install flash-attention
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp313-cp313-linux_x86_64.whl 
pip install ./flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

Install legacy-cgi (for Python3.13)
```bash
pip install legacy-cgi
```

Install other dependencies
```bash
pip install -r requirements.txt
```

## Datasets
### Download the dataset
1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ).

2. Remove the blacklisted URLs.
```bash
python datasets/blacklist_urls.py \
            --url_dir_path {datasets/urls/} \
            --output_file_path {clean_url.txt} \
            --print_blacklist
```
3. Download the content from the clean URLs with [openwebtext's utilities](https://github.com/eukaryote31/openwebtext/blob/master/download.py).
```bash
python scripts/dataset/scrape_from_urls.py \
            --url_file=/root/RETRO/clean_urls.txt \
            --output_dir=/root/RETRO/data/texts/ \
            --config_dir_path=/root/RETRO/config/
```
4. Merge the contents into one loose json file with 1 json per newline of the format {'text': text, 'url': unique_url}. It is important for the url to be unique.
```bash
python scripts/dataset/merge_scraped_contents.py \
            --input_dir_path=/root/RETRO/data/texts/ \
            --output_file_path=/root/RETRO/data/texts/merged.jsonl
```

### Preprocess the dataset for pre-training
1. Cleanup the dataset (e.g., removing documents with less than 128 tokens or possible duplicates).
```bash
python scripts/dataset/cleanup_dataset.py \
            --input_file_path=/root/RETRO/data/texts/merged.jsonl \
            --output_file_path=/root/RETRO/data/texts/preprocessed.jsonl \
            --min_tokens=128
```
2. Shuffle the dataset.
```bash
shuf {cleaned_deduplicated_data_file_path} -o {output_file_path}
```

