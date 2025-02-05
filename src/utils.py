from typing import *

from omegaconf import DictConfig, open_dict


def handle_old_ckpt(cfg, key) -> Any:
    if key not in cfg:
        return None
    return cfg[key]


def overwrite_config(src: DictConfig, dst: DictConfig) -> DictConfig:
    """Overwrite the src config with the dst config."""
    with open_dict(src):
        for key, value in dst.items():
            if key in src:
                if isinstance(value, DictConfig):
                    src[key] = overwrite_config(src[key], value)
                else:
                    src[key] = value
    return src


def add_config(cfg: DictConfig, key: str, value: Any) -> DictConfig:
    with open_dict(cfg):
        cfg[key] = value
    return cfg


def add_global_configs(
    cfg: DictConfig, global_dic: DictConfig = None, exclude_keys: List[str] = None
) -> DictConfig:
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
    return {key: value for key, value in dic.items() if value is not None}