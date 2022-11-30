import os
from pathlib import Path
import shutil
from tqdm import tqdm

src = '/research/cbim/medical/lh599/data/vox-celeba-alex-train/'
tar = '/research/cbim/medical/lh599/data/vox-celeba-alex-train/demo'

names = os.listdir(os.path.join(src, 'train'))[:10]

for name in tqdm(names):
    # video
    shutil.copytree(os.path.join(src, 'train', name), os.path.join(tar, 'video', name))
    shutil.copytree(os.path.join(src, 'draw_v2/style1', name), os.path.join(tar, 'draw/style1', name))
    shutil.copytree(os.path.join(src, 'mask_v2', name), os.path.join(tar, 'mask', name))
    shutil.copyfile(os.path.join(src, 'text_v2', name+'.txt'), os.path.join(tar, 'text', name+'.txt'))
    shutil.copyfile(os.path.join(src, 'label', name+'.txt'), os.path.join(tar, 'label', name+'.txt'))
