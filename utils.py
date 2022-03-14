import os
import sys
import ast
import math
import torch
import shutil
import random
import numpy as np
from torchvision.io import write_video
from torchvision import utils
from torch.nn import functional as F
import torch.nn as nn
import argparse
from pathlib import Path
from copy import deepcopy
# import matplotlib.pyplot as plt
from datetime import datetime
import pdb
st = pdb.set_trace


class Config:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # conflict with DDP


# borrowed from: https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_args(parser, args):
    args = deepcopy(args)
    if hasattr(args, 'parser'):
        delattr(args, 'parser')
    message = f"Name: {getattr(args, 'name', 'NA')} Time: {datetime.now()}\n"
    message += '--------------- Arguments ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = None if parser is None else parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    # print(message)  # suppress messages to std out

    # save to the disk
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    file_name = log_dir / 'args.txt'
    with open(file_name, 'a+') as f:
        f.write(message)
        f.write('\n\n')

    # save command to disk
    file_name = log_dir / 'cmd.txt'
    with open(file_name, 'a+') as f:
        f.write(f'Time: {datetime.now()}\n')
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if getattr(args, 'deepspeed', False) else 'python3 ')
        f.write(' '.join(sys.argv))
        f.write('\n\n')

    # backup train code
    shutil.copyfile(sys.argv[0], log_dir / f'{os.path.basename(sys.argv[0])}.txt')


def print_models(models, args):
    if not isinstance(models, (list, tuple)):
        models = [models]
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    file_name = log_dir / 'models.txt'
    with open(file_name, 'w') as f:
        f.write(f"Name: {getattr(args, 'name', 'NA')} Time: {datetime.now()}\n{'-'*50}\n")
        for model in models:
            f.write(str(model))
            f.write("\n\n")


def save_image(ximg, path):
    n_sample = ximg.shape[0]
    utils.save_image(ximg, path, nrow=int(n_sample ** 0.5), normalize=True, range=(-1, 1))


def save_video(xseq, path):
    video = xseq.data.cpu().clamp(-1, 1)
    video = ((video+1.)/2.*255).type(torch.uint8).permute(0, 2, 3, 1)
    write_video(path, video, fps=15)


"""
copied from VQGAN main
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


"""
copied from MoCoGAN-HD
"""
class GANLoss(nn.Module):
    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = torch.tensor(real_tensor,
                                                   requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.tensor(fake_tensor,
                                                   requires_grad=False)
            target_tensor = self.fake_label_var

        if input.is_cuda:
            target_tensor = target_tensor.cuda(input.get_device())
        return target_tensor

    def __call__(self, input, target_is_real):
        loss = 0
        for input_i in input:
            pred = input_i
            target_tensor = self.get_target_tensor(pred, target_is_real)
            loss += self.loss(pred, target_tensor)
        return loss


class Relativistic_Average_LSGAN(GANLoss):
    '''
        Relativistic average LSGAN
    '''
    def __call__(self, input_1, input_2, target_is_real):
        loss = 0
        for input_i, _input_i in zip(input_1, input_2):
            pred = input_i
            _pred = _input_i
            target_tensor = self.get_target_tensor(pred, target_is_real)
            loss += self.loss(pred - torch.mean(_pred), target_tensor)
        return loss
