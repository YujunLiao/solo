import os

import cv2
import numpy as np
from PIL import Image
# model = 'lyj_0324_e24'
fps = 5
model = 'solov2_r101_dcn_fpn_8gpu_3x/'

result_dir = f'./result/0530/'
for video_name in os.listdir(f'{result_dir}{model}'):
    if 'fps' in video_name:
        continue
    path = f'{result_dir}{model}{video_name}/'
    names = os.listdir(path)
    names.sort()
    frameSize = Image.open(f'{path}/{names[0]}').size
    n = len(names)
    video_dir = f'{result_dir}{model}fps_{fps}/'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    video_path = f'{video_dir}/{video_name}.mp4'
    print('video path', video_path)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)

    for img_name in names:
        # img = Image.open(f'{path}/{img_name}')
        img = cv2.imread(f'{path}/{img_name}')
        out.write(img)
    out.release()