import requests
import os
import sys
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
import shutil
from pathlib import Path

GIF_DIR = 'Data/iPER/iPER_256_video_release'
FRAME_DIR = 'Data/iPER/frames_256'

def extract_one_frame(i, video):
    name = Path(video).stem
    video_path = f"{GIF_DIR}/{video}"
    saveFolder = f"{FRAME_DIR}/{name}"
    fps = 30
    try:
        os.makedirs(saveFolder, exist_ok=True)
        cmd = 'ffmpeg -hide_banner -loglevel error -i %s ' % (video_path)
        cmd += ' -threads 40 -qscale:v 1 '  #-qscale:v 1
        cmd += ' -vf "fps={}" '.format(fps)
        cmd += '%s/%%04d.png' % (saveFolder)
        os.system(cmd)
    except:
        print(f"opening image {name} failed.")
    return None


if __name__ == '__main__':
    
    os.makedirs(FRAME_DIR, exist_ok=True)
    files = os.listdir(GIF_DIR)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap_async(extract_one_frame, [(i, name) for i, name in enumerate(files)]).get()

    print('done!')
