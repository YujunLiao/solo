import json
#导入json头文件

import os,sys
json_path = '/home/lyj/project/SOLO-master/data/coco/annotations/instances_val2017.json'
#json原文件

json_path_new ='/home/lyj/project/SOLO-master/data/coco/annotations/short.json'
#修改json文件后保存的路径

dict={}
#用来存储数据

def get_json_data(json_path):
#获取json里面数据

    with open(json_path,'rb') as f:
    #定义为只读模型，并定义名称为f

        params = json.load(f)
        #加载json文件中的内容给params

        # params['batch_size'] = 16
        #修改内容

        # print("params",params)
        #打印

        dict = params
        #将修改后的内容保存在dict中

    f.close()
    #关闭json读模式

    return dict
    #返回dict字典内容
def write_json_data(dict, new_path):
#写入json文件

    with open(new_path,'w') as r:
    #定义为写模式，名称定义为r

        json.dump(dict,r)
        #将dict写入名称为r的文件中

    r.close()
    #关闭json写模式

the_revised_dict = get_json_data(json_path)

# the_revised_dict["categories"] = [
#    {'supercategory': 'person', 'id': 1, 'name': 'person'},
#    {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
#    {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
#    {'supercategory': 'cartoon', 'id': 88, 'name': 'cartoon'},
# ]
images = {}
new_images = []
for img in the_revised_dict['images']:
    images[img['id']] = img
# new_annos = the_revised_dict['annotations'][:500]
the_revised_dict['annotations'] = the_revised_dict['annotations'][:150]
for anno in the_revised_dict['annotations']:
    img = images[anno['image_id']]
    if img not in new_images:
        new_images.append(img)
the_revised_dict['images'] = new_images


write_json_data(the_revised_dict, json_path_new)
the_revised_dict_new = get_json_data(json_path_new)
print()





# json_path = '/home/lyj/project/SOLO-master/data/coco/annotations/instances_train2017.json'
# json_path_test = '/home/lyj/project/SOLO-master/data/coco/annotations/instances_train2017.json'
# the_revised_dict = get_json_data(json_path)
# # for k in the_revised_dict
#
#
# write_json_data(the_revised_dict, json_path_test)