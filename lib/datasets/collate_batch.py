from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
from lib.config import cfg

_collators = {}

def make_collator(cfg, is_train):
    collator = cfg.train.collator if is_train else cfg.test.collator #* cfg.test.collator默认是'default'
    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate
