import argparse
import logging
import os
from hashlib import sha256
from typing import *

import hkkang_utils.concurrent as concurrent
import hkkang_utils.pg as pg_utils
import tldextract
import tqdm
import yaml

from src.dataset.scraper import Scraper

logger = logging.getLogger("Scraper")

TOTAL_NUM_URLS = 21269932


def advice_on_process_num(args_process_num: int) -> None:
    n_procs = os.cpu_count()
    if args_process_num > n_procs:
        logger.warning(
            f"Warning: {args_process_num} processes are requested, but the server has only {n_procs} processes available."
        )
    elif args_process_num < n_procs:
        logger.warning(
            f"Warning: {args_process_num} processes are requested, but the server has {n_procs} processes available."
        )
    return None


def get_target_indices_for_process(
    server_num: int, server_idx: int, process_num_per_server: int, process_idx: int
) -> Tuple[int, int]:
    if server_idx >= server_num:
        raise ValueError(
            f"server_idx ({server_idx}) must be less than server_num ({server_num})"
        )
    if process_idx >= process_num_per_server:
        raise ValueError(
            f"process_idx ({process_idx}) must be less than process_num_per_server ({process_num_per_server})"
        )

    total_process_num = server_num * process_num_per_server
    data_size_per_process = (
        TOTAL_NUM_URLS + total_process_num - 1
    ) // total_process_num  # Ceiling division

    # Calculate global process index
    global_process_idx = (server_idx * process_num_per_server) + process_idx

    start_idx = global_process_idx * data_size_per_process
    end_idx = min(start_idx + data_size_per_process, TOTAL_NUM_URLS)

    return start_idx, end_idx


def check_total_lines_in_urls_file(
    url_file_path: str, total_num_urls: int
) -> None:  # First check total number of lines
    logger.info(f"Checking total number of lines in {url_file_path}...")
    with open(url_file_path, "r") as f:
        total_lines = sum(1 for _ in tqdm.tqdm(f, desc="Counting lines in urls file"))
    if total_lines != total_num_urls:
        raise ValueError(
            f"Number of lines in {url_file_path} ({total_lines}) does not match TOTAL_NUM_URLS ({total_num_urls})"
        )
    logger.info(f"Total number of lines in {url_file_path} is {total_lines}.")
    return None


def get_target_urls(url_file_path: str, start_idx: int, end_idx: int) -> List[str]:
    urls: List[str] = []
    with open(url_file_path, "r") as f:
        # Read lines until we reach end_idx
        for idx, line in enumerate(f):
            if idx >= end_idx:
                break
            if idx >= start_idx:
                urls.append(line.strip())
    return urls


def hash_url(url: str) -> str:
    return sha256(url.encode()).hexdigest()


def get_output_file_name(url: str, server_idx: int, process_idx: int) -> str:
    return f"{server_idx}_{process_idx}_{hash_url(url)}.txt"


def fid_in_db(postgres_connector: pg_utils.PostgresConnector, fid: int) -> bool:
    fetch_results: List[Dict[str, Any]] = postgres_connector.execute_and_fetchall(
        f"SELECT fid FROM metadata WHERE fid = {fid}"
    )
    return len(fetch_results) > 0


def get_db_connector(config_dir_path: str) -> pg_utils.PostgresConnector:
    # Read in yaml config file
    with open(os.path.join(config_dir_path, "db.yml"), "r") as f:
        db_config = yaml.safe_load(f)
    # Connect to postgres
    return pg_utils.PostgresConnector(
        user_id=db_config["user_id"],
        passwd=db_config["passwd"],
        host=db_config["host"],
        port=db_config["port"],
        db_id=db_config["db_id"],
    )


def scrape_urls(
    process_idx: int,
    url_file_path: str,
    output_dir_path: str,
    config_dir_path: str,
    start_idx: int,
    end_idx: int,
    server_idx: int,
    timeout: int,
) -> None:
    # Connect to postgres
    postgres_connector = get_db_connector(config_dir_path)

    # Initialize scraper
    scraper = Scraper(timeout=timeout)

    # Get the target urls for this process
    target_urls = get_target_urls(url_file_path, start_idx, end_idx)

    disable_tqdm = process_idx != 0
    for i, url in enumerate(
        tqdm.tqdm(target_urls, desc="Scraping URLs", disable=disable_tqdm)
    ):
        # Get the fid
        fid = start_idx + i

        # Skip if fid already exists in the db
        if fid_in_db(postgres_connector=postgres_connector, fid=fid):
            continue

        # Strip the url
        url = url.strip()

        # Combine domain and suffix into a full domain name
        ext = tldextract.extract(url)
        domain_name = f"{ext.domain}.{ext.suffix}"

        # Scrape the url
        scraped_text, metadata = scraper(url, ext_url=ext)
        metadata["domain"] = domain_name

        # Write to file
        if scraped_text:
            # Get the output file path
            output_file_name = get_output_file_name(
                url=url, server_idx=server_idx, process_idx=process_idx
            )
            output_file_path = os.path.join(output_dir_path, output_file_name)
            # Write to file
            with open(output_file_path, "w") as f:
                f.write(scraped_text)

        # Write to postgres
        postgres_connector.execute(
            f"insert into metadata (fid, url, domain, word_count, elapsed, success) values ({fid}, '{url}', '{domain_name}', {metadata['word_count']}, {metadata['elapsed']}, {metadata['success']})",
        )

    # Close postgres connection
    postgres_connector.close()

    return None


def main(
    url_file_path: str,
    output_dir_path: str,
    config_dir_path: str,
    server_num: int,
    server_idx: int,
    process_num_per_server: int,
    timeout: int,
    **kwargs,
) -> None:
    # Get the number of processes in this server
    advice_on_process_num(process_num_per_server)

    # Prepare directories
    # Make output directory if not exists
    output_dir_path = os.path.join(output_dir_path, "data")
    os.makedirs(output_dir_path, exist_ok=True)

    # Check total number of lines in the urls file
    # check_total_lines_in_urls_file(url_file_path, total_num_urls=TOTAL_NUM_URLS)

    # Initialize multi processor
    multiprocessor = concurrent.MultiProcessor(num_workers=process_num_per_server)

    # Run processes concurrently
    for process_idx in range(process_num_per_server):
        # Get the target indices for this process
        start_idx, end_idx = get_target_indices_for_process(
            server_num=server_num,
            server_idx=server_idx,
            process_num_per_server=process_num_per_server,
            process_idx=process_idx,
        )
        # Run the process
        multiprocessor.run(
            scrape_urls,
            process_idx=process_idx,
            url_file_path=url_file_path,
            output_dir_path=output_dir_path,
            config_dir_path=config_dir_path,
            start_idx=start_idx,
            end_idx=end_idx,
            server_idx=server_idx,
            timeout=timeout,
        )

    # Wait for all processes to finish (this is optional: accessing result will call join method anyway)
    multiprocessor.join()

    return None


def clear_db(config_dir_path: str) -> None:
    logger.info("Clearing database...")
    postgres_connector = get_db_connector(config_dir_path)
    postgres_connector.execute("DELETE FROM metadata")
    postgres_connector.close()
    logger.info("Database cleared.")
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url_file_path", type=str, required=True)
    parser.add_argument("--output_dir_path", type=str, required=True)
    parser.add_argument("--config_dir_path", type=str, required=True)
    parser.add_argument("--server_num", type=int, default=1)
    parser.add_argument("--server_idx", type=int, default=0)
    # parser.add_argument("--process_num_per_server", type=int, default=1)
    parser.add_argument("--process_num_per_server", type=int, default=480)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--clear_db", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_arguments()
    if args.clear_db:
        clear_db(config_dir_path=args.config_dir_path)
    else:
        main(**vars(args))
