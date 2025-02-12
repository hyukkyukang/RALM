# retro

## Install dependencies
Install torch and legacy-cgi (for Python3.13)
```bash
pip install torch legacy-cgi
```

Install other dependencies
```bash
pip install -r requirements.txt
```

## Datasets
### Download the Lambada dataset
1. Download the Lambada dataset used in GPT-2 paper from [here](https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl).
2. Move the file to `data/lambada_gpt2_author.jsonl`.

### Download the OpenWebText dataset
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

