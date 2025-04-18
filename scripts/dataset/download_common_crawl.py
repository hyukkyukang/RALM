import os
import requests
import gzip
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup

COMMON_CRAWL_URL = "https://data.commoncrawl.org/"
DOWNLOAD_DIR = "/root/RETRO/data/common_crawl"
TEXT_DIR = "extracted_text"
INDEX_NAME = "CC-MAIN-2024-51"  # Replace with the latest Common Crawl index


def get_warc_paths():
    """Retrieve the list of WARC files for the latest Common Crawl segment."""
    warc_list_url = f"{COMMON_CRAWL_URL}crawl-data/{INDEX_NAME}/warc.paths.gz"
    response = requests.get(warc_list_url, stream=True)

    if response.status_code != 200:
        raise Exception(f"Failed to retrieve WARC paths: {response.status_code}")

    warc_paths = []
    with gzip.GzipFile(fileobj=response.raw) as gz:
        warc_paths = [line.decode("utf-8").strip() for line in gz]

    return warc_paths


def download_and_extract_warc(warc_path):
    """Download a WARC file, extract text content, and remove the WARC file."""
    warc_filename = os.path.basename(warc_path)
    warc_filepath = os.path.join(DOWNLOAD_DIR, warc_filename)
    text_filepath = os.path.join(TEXT_DIR, warc_filename.replace(".warc.gz", ".txt"))

    if os.path.exists(text_filepath):
        print(f"Text file {text_filepath} already exists. Skipping...")
        return

    warc_url = f"{COMMON_CRAWL_URL}{warc_path}"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    print(f"Downloading: {warc_url}")
    response = requests.get(warc_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(warc_filepath, "wb") as f, tqdm(
            desc=warc_filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Downloaded {warc_filepath}")
    else:
        print(f"Failed to download {warc_url}")
        return

    # Extract text content from the WARC file
    print(f"Extracting text from {warc_filepath}")
    with gzip.open(warc_filepath, "rb") as stream, open(
        text_filepath, "w", encoding="utf-8"
    ) as out_file:
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                url = record.rec_headers.get("WARC-Target-URI")
                raw_html = (
                    record.content_stream().read().decode("utf-8", errors="ignore")
                )
                soup = BeautifulSoup(raw_html, "html.parser")
                text_content = soup.get_text()
                out_file.write(f"URL: {url}\n{text_content}\n\n")

    print(f"Extracted text saved to {text_filepath}")

    # Remove the WARC file to save space
    os.remove(warc_filepath)
    print(f"Removed {warc_filepath}")


def main():
    print("Fetching list of WARC files...")
    warc_paths = get_warc_paths()

    print(f"Found {len(warc_paths)} WARC files to process.")

    # Process first 5 WARC files for testing (remove the limit to process all)
    for warc_path in warc_paths[:8]:
        download_and_extract_warc(warc_path)


if __name__ == "__main__":
    main()
