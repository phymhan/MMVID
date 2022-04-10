import os
import random

import numpy as np

import torch
import torch.nn as nn
from torchvision.io import write_video
from torchvision import utils


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True)
        return x / maxes


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_image(ximg, path):
    n_sample = ximg.shape[0]
    utils.save_image(ximg,
                     path,
                     nrow=int(n_sample**0.5),
                     normalize=True,
                     range=(-1, 1))


def save_video(xseq, path):
    video = xseq.data.cpu().clamp(-1, 1)
    video = ((video + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
    write_video(path, video, fps=15)


"""
Copied from VQGAN main.py
"""
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
