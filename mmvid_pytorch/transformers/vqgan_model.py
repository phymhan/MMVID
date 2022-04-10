import io
import sys
import os, sys
import requests
import PIL
import warnings
import os
import hashlib
import urllib
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from mmvid_pytorch import distributed_utils

import importlib
from mmvid_pytorch.transformers.mingpt import GPT


# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

VQGAN_TRANSFORMER_DRIN_PATH = 'pretrained/2020-11-20T12-54-32_drin_transformer/checkpoints/last.ckpt'
VQGAN_TRANSFORMER_DRIN_CONFIG_PATH = 'pretrained/2020-11-20T12-54-32_drin_transformer/configs/2020-11-20T12-54-32-project.yaml'

VQGAN_TRANSFORMER_COCO_PATH = 'pretrained/2021-01-20T16-04-20_coco_transformer/checkpoints/last.ckpt'
VQGAN_TRANSFORMER_COCO_CONFIG_PATH = 'pretrained/2021-01-20T16-04-20_coco_transformer/configs/2021-02-08T17-18-53-project.yaml'

# helpers methods


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


def unmap_pixels(x, eps=0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


def download(url, filename=None, root=CACHE_PATH):
    if (not distributed_utils.is_distributed
            or distributed_utils.backend.is_local_root_worker()):
        os.makedirs(root, exist_ok=True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if (distributed_utils.is_distributed
            and not distributed_utils.backend.is_local_root_worker()
            and not os.path.isfile(download_target)):
        # If the file doesn't exist yet, wait until it's downloaded by the root worker.
        distributed_utils.backend.local_barrier()

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp,
                                                     "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")),
                  ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    if (distributed_utils.is_distributed
            and distributed_utils.backend.is_local_root_worker()):
        distributed_utils.backend.local_barrier()
    return download_target


class VQGanTransformer(nn.Module):
    def __init__(self,
                 which_model='vqgan_drin',
                 seq_len=0,
                 causal=True,
                 mask_type='causal',
                 mask_kwargs={}):
        super().__init__()

        self.causal = causal

        if which_model == 'vqgan_drin':
            model_filename = VQGAN_TRANSFORMER_DRIN_PATH
            config_filename = VQGAN_TRANSFORMER_DRIN_CONFIG_PATH
        elif which_model == 'vqgan_coco':
            model_filename = VQGAN_TRANSFORMER_COCO_PATH
            config_filename = VQGAN_TRANSFORMER_COCO_CONFIG_PATH
        else:
            raise NotImplementedError

        config = OmegaConf.load(config_filename)
        transformer_config = config['model']['params']['transformer_config'][
            'params']
        transformer_config['input_is_embd'] = True
        transformer_config['causal'] = causal
        transformer_config['mask_type'] = mask_type
        transformer_config['mask_kwargs'] = mask_kwargs
        if seq_len > 0:
            transformer_config['block_size'] = seq_len
        model = GPT(**transformer_config)
        state = torch.load(model_filename, map_location='cpu')['state_dict']
        state_dict = {}
        for k in state.keys():
            if k.startswith('transformer.'
                            ) and 'attn.mask' not in k and 'pos_emb' not in k:
                state_dict[k.replace('transformer.', '')] = state[k]
        model.load_state_dict(state_dict, strict=False)

        self.model = model

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embedding.weight)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)[0]


class minGPTTransformer(nn.Module):
    def __init__(self,
                 which_model='mingpt',
                 seq_len=0,
                 causal=True,
                 mask_type='causal',
                 mask_kwargs={},
                 n_head=8,
                 n_layers=24,
                 n_emb=512):
        super().__init__()

        self.causal = causal
        model = GPT(vocab_size=1,
                    block_size=seq_len,
                    n_layer=n_layers,
                    n_head=n_head,
                    n_embd=n_emb,
                    causal=causal,
                    input_is_embd=True,
                    mask_type=mask_type,
                    mask_kwargs=mask_kwargs)
        self.model = model

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if (not distributed_utils.is_distributed
                or not distributed_utils.using_backend(
                    distributed_utils.DeepSpeedBackend)):
            return

        deepspeed = distributed_utils.backend.backend_module
        deepspeed.zero.register_external_parameter(
            self, self.model.quantize.embedding.weight)

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)[0]