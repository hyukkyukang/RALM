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

Install legacy-cgi
```bash
pip install legacy-cgi # This is for Python3.13 
```

Install other dependencies
```bash
pip install -r requirements.txt
```

## Datasets

### Blacklist URLs
1. Download the deduplicated URLs from [jcpeterson](https://mega.nz/#F!EZZD0YwJ!9_PlEQzdMVLaNdKv_ICNVQ!cc4RgQQZ).

2. Remove the blacklisted URLs.
```bash
python datasets/blacklist_urls.py --url_dir_path {datasets/urls/} --output_file_path {clean_url.txt} --print_blacklist
```
3. Download the content from the clean URLs with [openwebtext's utilities](https://github.com/eukaryote31/openwebtext/blob/master/download.py).

4. Merge the contents into one loose json file with 1 json per newline of the format {'text': text, 'url': unique_url}. It is important for the url to be unique.
