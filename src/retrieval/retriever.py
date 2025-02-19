from typing import *

from omegaconf import DictConfig

class Retriever:
    def __init__(self, cfg: DictConfig, global_cfg: DictConfig):
        self.cfg = cfg
        self.global_cfg = global_cfg
