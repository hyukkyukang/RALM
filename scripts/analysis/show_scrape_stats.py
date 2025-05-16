from typing import *

import hkkang_utils.file as file_utils
import hkkang_utils.pg as pg_utils

from scripts.dataset.scrape_from_urls import get_db_config


def main(config_dir_path: str, data_dir_path: str) -> None:
    db_config: Dict[str, Any] = get_db_config(config_dir_path)
    postgres_connector = pg_utils.PostgresConnector(
        user_id=db_config["user_id"],
        passwd=db_config["passwd"],
        host=db_config["host"],
        port=db_config["port"],
        db_id=db_config["db_id"],
    )

    total_count = postgres_connector.execute_and_fetchall(
        "SELECT count(*) from metadata"
    )[0][0]
    success_count = postgres_connector.execute_and_fetchall(
        "SELECT count(*) from metadata WHERE success = TRUE"
    )[0][0]
    failed_count = postgres_connector.execute_and_fetchall(
        "SELECT count(*) from metadata WHERE success = FALSE"
    )[0][0]

    print(f"Total count: {total_count}")
    print(f"Success count: {success_count}")
    print(f"Failed count: {failed_count}")

    assert total_count == (
        success_count + failed_count
    ), "Total count does not match success and failed count"

    # Show the disk memory usage of the text data directory (in MB)
    dir_logical_size = file_utils.get_directory_size(data_dir_path)
    dir_physical_size = file_utils.get_disk_usage(data_dir_path)
    print(
        f"Disk memory usage of {data_dir_path}: {file_utils.bytes_to_readable(dir_logical_size)}"
    )
    print(
        f"Disk disk usage of {data_dir_path}: {file_utils.bytes_to_readable(dir_physical_size)}"
    )
    return None


if __name__ == "__main__":
    config_dir_path = "/home/user/RALM/config/"
    data_dir_path = "/home/user/RALM/data/texts/data/"
    main(config_dir_path, data_dir_path)
