import logging
import os
from typing import *

import lightning as L
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoModelForCausalLM


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    # If LOCAL_RANK is set, use it; otherwise, assume a single-process scenario.
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    return True


@rank_zero_only
def log_if_rank_zero(logger: logging.Logger, message: str, level: str = "info") -> None:
    """Helper function to log only on rank 0 process.
    If not distributed, log the message as well."""
    getattr(logger, level)(message)


def overwrite_config(src: DictConfig, dst: DictConfig) -> DictConfig:
    """Recursively overwrite values in the destination config with values from the source config.

    Args:
        src (DictConfig): Source configuration containing values to copy
        dst (DictConfig): Destination configuration to be modified

    Returns:
        DictConfig: Modified destination configuration with overwritten values

    Note:
        Copies all keys from source to destination config. For nested DictConfigs,
        recursively updates the nested values.
    """
    with open_dict(dst):
        for key in src:
            if isinstance(src[key], DictConfig):
                if key not in dst or not isinstance(dst[key], DictConfig):
                    dst[key] = src[key]
                else:
                    dst[key] = overwrite_config(src[key], dst[key])
            else:
                dst[key] = src[key]
    return dst


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


def check_argument(
    dic: Dict,
    name: str,
    arg_type: Type,
    choices: List[Any] = None,
    is_requried: bool = False,
    help: str = None,
) -> bool:
    with open_dict(dic):
        # Check if the argument is required
        if is_requried and name not in dic:
            raise ValueError(f"{name} is required!.({help})")
        # Check argument type and choices
        if name in dic:
            if not isinstance(dic[name], arg_type):
                raise ValueError(f"{name} should be {arg_type}. ({help})")
            if choices is not None and dic[name] not in choices:
                raise ValueError(f"{name} should be in {choices}. ({help})")
        # Set default value for boolean args
        if name not in dic and arg_type == bool:
            dic[name] = False
    return True

def is_model_compiled(model: Union[L.LightningModule, AutoModelForCausalLM]) -> bool:
    """Check if the model is compiled with torch.compile."""
    if isinstance(model, L.LightningModule):
        if (
            "use_torch_compile" in model.cfg
            and model.cfg.use_torch_compile
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 7
        ):
            assert isinstance(
                model.model, torch._dynamo.eval_frame.OptimizedModule
            ), f"Model is not an OptimizedModule?: {type(model.model)}"
            return True
    else:
        return isinstance(model, torch._dynamo.eval_frame.OptimizedModule)