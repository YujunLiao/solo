import os
from shutil import copyfile
from tqdm import tqdm


split_size = 0.1

train_data_dir = '/home/lyj/data/coco2017/train2017/'
train_data_split_dir = f'/home/lyj/data/coco2017_1of{int(1 / split_size)}/train2017/'
print(f'{train_data_dir} to {train_data_split_dir}')

if not os.path.exists(train_data_split_dir):
    os.makedirs(train_data_split_dir)

img_names = list(os.listdir(train_data_dir))
img_names.sort()


for i in tqdm(range(int(split_size*len(img_names))), postfix='Copying'):
    name = img_names[i]
    copyfile(f'{train_data_dir}{name}', f'{train_data_split_dir}{name}')






