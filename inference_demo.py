import os
import time
import socket

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv

# map
# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'

# config_file = '../configs/solo/solo_r50_fpn_8gpu_1x.py'
# checkpoint_file = '../checkpoints/SOLO_R50_1x.pth'
#
# config_file = '../configs/solo/solo_r50_fpn_8gpu_3x.py'
# checkpoint_file = '../checkpoints/SOLO_R50_3x.pth'


## AP
#
# config_file = './configs/solo/solo_r101_fpn_8gpu_3x.py'
# checkpoint_file = './checkpoints/SOLO_R101_3x.pth'


# config_file = '../configs/solo/decoupled_solo_r101_fpn_8gpu_3x.py'
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R101_3x.pth'

# config_file = './configs/solov2/solov2_r101_fpn_8gpu_3x.py'
# checkpoint_file = './checkpoints/SOLOv2_R101_3x.pth'

# config_file = './configs/solov2/solov2_r101_dcn_fpn_8gpu_3x.py'
# checkpoint_file = './checkpoints/SOLOv2_R101_DCN_3x.pth'

# config_file = './configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
# checkpoint_file = './checkpoints/SOLOv2_X101_DCN_3x.pth'

## speed

# config_file = '../configs/solo/decoupled_solo_light_dcn_r50_fpn_8gpu_3x.py'
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_LIGHT_DCN_R50_3x.pth'

# config_file = './configs/solov2/solov2_light_512_dcn_r50_fpn_8gpu_3x.py'
# checkpoint_file = './checkpoints/SOLOv2_LIGHT_512_DCN_R50_3x.pth'

config_file = 'configs/solov2/solov2_light_448_r18_fpn_8gpu_3x.py'
checkpoint_file = './work_dir/0602/ps-X10DRG/solov2_light_448_r18_fpn_8gpu_3x/epoch_36.pth'


print(config_file)



# build the model from a config file and a checkpoint file
cuda_n = 0
print('gpu:', cuda_n)
os.environ['CUDA_VISIBLE_DEVICES'] = f'{cuda_n}'
model = init_detector(config_file, checkpoint_file, device=f'cuda')
#
# # test a single image
#
#
# for video_name in ['1', '2', '3']:
score_thr = 0.25
# for video_name in ['coco_72']:
# for video_name in ['Yotube-vos-3rd']:
# for video_name in ['transformed']:

save_dir = f'result/{socket.gethostname()}0530/'


# for video_name in ['cityscape_100', 'GTA5_99']:
for video_name in ['coco_72']:
# for video_name in ['Yotube-vos-3rd_rotate180']:

    data_dir = f'data/{video_name}/'
    out_img_dir = f"{save_dir}{config_file.split('/')[-1].split('.')[0]}/{video_name}_score_thr_{score_thr}/"
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    print('save', save_dir, os.path.abspath(save_dir), out_img_dir)
    n = len(os.listdir(data_dir))
    start = time.time()
    # for i in range(1, 141):
    for img in os.listdir(data_dir):
        # img = f'{i}.jpg'

        result = inference_detector(model, f'{data_dir}{img}')
        show_result_ins(f'{data_dir}{img}', result, model.CLASSES, score_thr=score_thr, out_file=f"./{out_img_dir}{img}")
        # print('save', os.path.abspath(f"../{out_img_dir}{img}"))
    end = time.time()


    # print()
    # for img in os.listdir(directory):
    #     # print(f'{directory}{img}')
    #     # result = inference_detector(model, f'{directory}{img}')
    #     # show_result_ins(f'{directory}{img}', result, model.CLASSES, score_thr=0.25, out_file=f"../data/out/{img}")
    #     break

    print('fps:', n/(end - start), 'n:', n)



