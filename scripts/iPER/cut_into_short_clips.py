import argparse
from pathlib import Path
import os
import shutil
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(curr_path).parent))
# from mmvid_pytorch.loader import TextVideoDataset
from tqdm import tqdm
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from natsort import natsorted

import pdb
st = pdb.set_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src1_dir', type=str, default='Data/iPER/frames_256')
    parser.add_argument('--dst1_dir', type=str, default='Data/iPER/clips_100_256')
    parser.add_argument('--src2_dir', type=str, default='Data/iPER/text_v0')
    parser.add_argument('--dst2_dir', type=str, default='Data/iPER/text_100_v0')
    args = parser.parse_args()

    os.makedirs(args.dst1_dir, exist_ok=True)
    os.makedirs(args.dst2_dir, exist_ok=True)

    n = 100

    for video in tqdm(os.listdir(args.src1_dir)):
        frames = natsorted(os.listdir(os.path.join(args.src1_dir, video)))
        for i in range(0, len(frames), n):
            video_id = f"{video}#{i}"
            os.makedirs(os.path.join(args.dst1_dir, video_id), exist_ok=True)
            for j in range(min(n, len(frames)-i)):
                shutil.copyfile(
                    os.path.join(args.src1_dir, video, frames[i+j]),
                    os.path.join(args.dst1_dir, video_id, frames[i+j]),
                )
            shutil.copyfile(
                os.path.join(args.src2_dir, video+'.txt'),
                os.path.join(args.dst2_dir, video_id+'.txt'),
            )
