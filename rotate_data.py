import os

import PIL
from PIL import Image
# n = 4:顺时针转90度
# n = 3:顺时针转180度
# n = 2:顺时针转270度
degree = {
    4: 90, #
    2: 270,
    3: 180
}
n = 3
print(n)
data_dir = 'data/'
for video_name in ['Yotube-vos-3rd']:
    save_video_name = f'{data_dir}{video_name}_rotate{degree[n]}/'
    if not os.path.exists(save_video_name):
        os.makedirs(save_video_name)
    for img_name in os.listdir(f'{data_dir}{video_name}'):
        img = Image.open(f'{data_dir}{video_name}/{img_name}')
        img = img.transpose(n)
        img.save(f'{save_video_name}{img_name}')
        # print(f'{save_video_name}{img_name}')
