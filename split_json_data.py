import os
from shutil import copyfile
from tqdm import tqdm
import json

split_size = 0.05

json_path = '/home/lyj/data/coco2017/annotations/instances_train2017.json'
split_json_save_path = f'/home/lyj/data/coco2017_{split_size}/annotations/instances_train2017.json'
print(f'{json_path} to {split_json_save_path}')

if not os.path.exists(os.path.dirname(split_json_save_path)):
    os.makedirs(os.path.dirname(split_json_save_path))

json_file = json.load(open(json_path, 'r'))
n = int(len(json_file['images'])*split_size)
split_json_file = {
    'info': json_file['info'],
    'licenses': json_file['licenses'],
    'categories': json_file['categories'],
    'images': json_file['images'][:n],
    'annotations': []
}

img_ids = {img['id']:img['id']
           for img in split_json_file['images']}

for i in tqdm(range(len(json_file['annotations'])), postfix='Processing'):
    anno = json_file['annotations'][i]
    if anno['image_id'] in img_ids:
        split_json_file['annotations'].append(anno)

json.dump(split_json_file, open(split_json_save_path, 'w'))











