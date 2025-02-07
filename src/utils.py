import logging
from typing import *

import torch.distributed as dist
from omegaconf import DictConfig, open_dict
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def log_if_rank_zero(logger: logging.Logger, message: str, level: str = "info") -> None:
    """Helper function to log only on rank 0 process."""
    getattr(logger, level)(message)


def overwrite_config(src: DictConfig, dst: DictConfig) -> DictConfig:
    """Recursively overwrite values in the source config with values from the destination config.

    Args:
        src (DictConfig): Source configuration to be modified
        dst (DictConfig): Destination configuration containing new values

    Returns:
        DictConfig: Modified source configuration with overwritten values

    Note:
        Only overwrites keys that already exist in the source config.
        For nested DictConfigs, recursively updates the nested values.
    """
    with open_dict(src):
        for key, value in dst.items():
            if key in src:
                if isinstance(value, DictConfig):
                    src[key] = overwrite_config(src[key], value)
                else:
                    src[key] = value
    return src


def add_config(cfg: DictConfig, key: str, value: Any) -> DictConfig:
    """Add a new key-value pair to a DictConfig.

    Args:
        cfg (DictConfig): Configuration to modify
        key (str): Key to add
        value (Any): Value to associate with the key

    Returns:
        DictConfig: Modified configuration with the new key-value pair
    """
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def add_global_configs(
    cfg: DictConfig, global_dic: DictConfig = None, exclude_keys: List[str] = None
) -> DictConfig:
    """Recursively add global configuration values to all nested configurations.

    Args:
        cfg (DictConfig): Configuration to modify
        global_dic (DictConfig, optional): Global configuration values to add.
            If None, uses cfg._global
        exclude_keys (List[str], optional): Keys to exclude from global config application

    Returns:
        DictConfig: Modified configuration with global values added to all nested configs

    Raises:
        AssertionError: If global_dic is None and cfg has no _global attribute
    """
    if global_dic is None:
        assert hasattr(cfg, "_global"), "Global configs are not found in the config"
        global_dic = cfg._global

    with open_dict(cfg):
        for sub_cfg_name, sub_cfg in cfg.items():
            if sub_cfg_name != "global" and isinstance(sub_cfg, DictConfig):
                if exclude_keys and sub_cfg_name in exclude_keys:
                    continue
                # Append key to the sub_cfg
                for key, value in global_dic.items():
                    if key not in sub_cfg:
                        sub_cfg[key] = value
                # Recursively add global configs
                add_global_configs(sub_cfg, global_dic=global_dic)
    return cfg


def remove_key_with_none_value(dic: Dict) -> Dict:
    """Remove all key-value pairs where the value is None.

    Args:
        dic (Dict): Dictionary to filter

    Returns:
        Dict: New dictionary with None values removed
    """
    return {key: value for key, value in dic.items() if value is not None}
