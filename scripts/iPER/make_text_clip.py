import argparse
from pathlib import Path
import os
import shutil
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(curr_path).parent))
# from dalle_pytorch.loader import TextVideoDataset
from tqdm import tqdm
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader

import pdb
st = pdb.set_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str, default='Data/iPER/clips_100_256')
    parser.add_argument('--dst_root', type=str, default='Data/iPER/text_100_v1')
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    for video in tqdm(os.listdir(args.src_root)):
        xxx, yyy, zzz = video.split('#')[0].split('_')
        pose = "'A' pose" if zzz == '1' else "random pose"
        desc = f"Person {xxx} dressed in {yyy} is performing {pose}."
        with open(os.path.join(args.dst_root, video+'.txt'), 'w') as f:
            f.write(desc)
