import os
from typing import Dict, List

import hkkang_utils.file as file_utils
import tqdm
from datasets import DatasetDict, Dataset
from huggingface_hub import HfApi

DATASET_REPO = "Hyukkyu/superglue2"
SUPERGLUE_DATA_DIR = os.path.join("/home/user/RALM", "superglue_data")

DATASET_NAME_MAPPING = {
    "BoolQ": "boolq",
    "CB": "cb",
    "COPA": "copa",
    "MultiRC": "multirc",
    "ReCoRD": "record",
    "RTE": "rte",
    "WiC": "wic",
    "WSC": "wsc",
}


def get_dummy_value(expected_type):
    """Return a dummy value that matches the expected primitive type."""
    if expected_type == bool:
        return False
    elif expected_type == int:
        return -1
    elif expected_type == float:
        return -1.0
    elif expected_type == str:
        return ""
    else:
        return -1


def infer_schema(value):
    """
    Infer a schema from a value.
    For dictionaries, return a dict with "__type__": "dict" and a nested schema.
    For lists of dicts, return a dict with "__type__": "list" and a nested schema.
    Otherwise, return the Python type.
    """
    if isinstance(value, dict):
        return {"__type__": "dict", "schema": compute_recursive_schema([value])}
    elif isinstance(value, list):
        if value and isinstance(value[0], dict):
            return {"__type__": "list", "schema": compute_recursive_schema(value)}
        else:
            return list  # for lists of primitives
    else:
        return type(value)


def compute_recursive_schema(dict_list: List[Dict]) -> dict:
    """
    Compute a recursive schema for a list of dictionaries.
    The returned schema maps keys to either:
      - A Python type (for primitives), or
      - A dict with a "__type__" key indicating "dict" or "list" along with a nested "schema".
    """
    schema = {}
    for d in dict_list:
        for key, value in d.items():
            s = infer_schema(value)
            if key not in schema:
                schema[key] = s
            else:
                schema[key] = merge_schema(schema[key], s)
    return schema


def merge_schema(s1, s2):
    """
    Merge two schema definitions.
    If both are dicts with a "__type__" key and the same type, merge their inner schemas.
    If both are primitive types, resolve conflicts (e.g. int vs bool).
    Otherwise, default to s1.
    """
    if isinstance(s1, dict) and isinstance(s2, dict):
        if s1.get("__type__") == s2.get("__type__") == "dict":
            merged_inner = merge_dict_schema(s1["schema"], s2["schema"])
            return {"__type__": "dict", "schema": merged_inner}
        if s1.get("__type__") == s2.get("__type__") == "list":
            merged_inner = merge_dict_schema(s1["schema"], s2["schema"])
            return {"__type__": "list", "schema": merged_inner}
    if isinstance(s1, type) and isinstance(s2, type):
        if s1 == s2:
            return s1
        if (s1 == int and s2 == bool) or (s1 == bool and s2 == int):
            return bool
        return s1
    return s1


def merge_dict_schema(dict1: dict, dict2: dict) -> dict:
    merged = {}
    keys = set(dict1.keys()) | set(dict2.keys())
    for key in keys:
        if key in dict1 and key in dict2:
            merged[key] = merge_schema(dict1[key], dict2[key])
        elif key in dict1:
            merged[key] = dict1[key]
        else:
            merged[key] = dict2[key]
    return merged


def fill_dummy_values(item: dict, schema: dict) -> None:
    """
    Recursively traverse the item dictionary using the provided schema and
    add a dummy value (of the correct type) for any missing key.
    """
    for key, expected in schema.items():
        if key not in item:
            # If the expected type is a nested dict or list, initialize accordingly.
            if isinstance(expected, dict):
                if expected.get("__type__") == "dict":
                    item[key] = {}
                    fill_dummy_values(item[key], expected["schema"])
                elif expected.get("__type__") == "list":
                    # For lists, initialize with an empty list.
                    # (If needed, you could insert a dummy element, but that may not be desired.)
                    item[key] = []
            elif isinstance(expected, type):
                item[key] = get_dummy_value(expected)
        else:
            # Key exists: check if we need to fill nested structures.
            if isinstance(expected, dict):
                if expected.get("__type__") == "dict":
                    if isinstance(item[key], dict):
                        fill_dummy_values(item[key], expected["schema"])
                    else:
                        item[key] = {}
                        fill_dummy_values(item[key], expected["schema"])
                elif expected.get("__type__") == "list":
                    if not isinstance(item[key], list):
                        item[key] = []
                    else:
                        for elem in item[key]:
                            if isinstance(elem, dict):
                                fill_dummy_values(elem, expected["schema"])


def add_dummy_labels_recursive(
    test_list: List[Dict], reference_schema: dict
) -> List[Dict]:
    """
    For each test sample, recursively fill in missing keys based on the reference schema.
    """
    for item in test_list:
        fill_dummy_values(item, reference_schema)
    return test_list


def main():
    # Instantiate HfApi to check repository status.
    api = HfApi()
    try:
        existing_files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    except Exception as e:
        print(
            "Dataset repository does not exist yet or is not accessible. A new repository will be created."
        )
        existing_files = []

    # Load the SuperGLUE dataset folders.
    superglue_folders: List[str] = os.listdir(SUPERGLUE_DATA_DIR)
    for folder in superglue_folders:
        assert (
            folder in DATASET_NAME_MAPPING
        ), f"{folder} is not in the DATASET_NAME_MAPPING values"

    for dataset_name in tqdm.tqdm(superglue_folders):
        print(f"Processing {dataset_name}...")

        train_list = file_utils.read_jsonl_file(
            os.path.join(SUPERGLUE_DATA_DIR, dataset_name, "train.jsonl")
        )
        validation_list = file_utils.read_jsonl_file(
            os.path.join(SUPERGLUE_DATA_DIR, dataset_name, "val.jsonl")
        )
        test_list = file_utils.read_jsonl_file(
            os.path.join(SUPERGLUE_DATA_DIR, dataset_name, "test.jsonl")
        )

        # Compute the reference schema from train and validation.
        schema_train = compute_recursive_schema(train_list)
        schema_val = compute_recursive_schema(validation_list)
        reference_schema = merge_dict_schema(schema_train, schema_val)

        # Recursively add dummy values to test samples.
        test_list = add_dummy_labels_recursive(test_list, reference_schema)

        train_dataset = Dataset.from_list(train_list)
        validation_dataset = Dataset.from_list(validation_list)
        test_dataset = Dataset.from_list(test_list)

        dataset = DatasetDict(
            {
                "train": train_dataset,
                "validation": validation_dataset,
                "test": test_dataset,
            }
        )

        # Force all splits to have the same features as the training split.
        common_features = train_dataset.features
        dataset["validation"] = dataset["validation"].cast(common_features)
        dataset["test"] = dataset["test"].cast(common_features)

        config_name = DATASET_NAME_MAPPING[dataset_name]
        config_file = f"{config_name}/dataset_info.json"
        if config_file in existing_files:
            print(f"{config_name} already uploaded, skipping upload.")
            continue

        dataset.push_to_hub(DATASET_REPO, config_name=config_name)
        print(f"Uploaded {dataset_name} successfully!")

    print("All SuperGLUE datasets processed successfully!")


if __name__ == "__main__":
    main()
