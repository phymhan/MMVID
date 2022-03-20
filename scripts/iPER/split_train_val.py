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
import random
from natsort import natsorted

import pdb
st = pdb.set_trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_num', type=int, default=10)
    args = parser.parse_args()

    video_names = os.listdir('Data/iPER/frames_256')
    pid_list = natsorted(list(set([s.split('_')[0] for s in video_names])))

    xxxyyy_dict = {}
    for s in video_names:
        xxx = s.split('_')[0]
        yyy = s.split('_')[1]
        if xxx in xxxyyy_dict:
            if yyy not in xxxyyy_dict[xxx]:
                xxxyyy_dict[xxx].append(yyy)
        else:
            xxxyyy_dict[xxx] = [yyy]

    random.seed(42)
    random.shuffle(pid_list)

    pid_val = set(pid_list[:args.val_num])
    pid_train = set(pid_list[args.val_num:])

    xxxyyyzzz_val = []
    for p in pid_val:
        yyy = random.choice(xxxyyy_dict[p])
        xxxyyyzzz_val.append(f"{p}_{yyy}_1")

    vid_list = os.listdir('Data/iPER/clips_100_256')
    vid_val = [s for s in vid_list if s.split('#')[0] in xxxyyyzzz_val]
    vid_train = list(set(vid_list)-set(vid_val))

    with open('iper_train.txt', 'w') as f:
        f.write('\n'.join(vid_train))
    
    with open('iper_val.txt', 'w') as f:
        f.write('\n'.join(vid_val))
