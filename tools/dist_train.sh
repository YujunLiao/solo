#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
#PORT=${PORT:-29527}
#PORT=${PORT:-29528}
#PORT=${PORT:-29529}
#PORT=${PORT:-29530}
#PORT=${PORT:-29501}
PORT=${PORT:-29502}

# 表示','替换为' '空格
#arr=(${CONFIG//,/ })
#work_dir=${arr[2]}
#
#work_dir=(${work_dir//./ })
#work_dir=${arr[1]}



#--work_dir work_dir/0602/$(hostname)/
# --resume_from
# SOLOv2_R50_1x
# ./tools/dist_train.sh configs/solo/solo_r50_fpn_8gpu_1x.py  2 --work_dir work_dir/0602/$(hostname)/solo_r50_fpn_8gpu_1x
# ./tools/dist_train.sh configs/solov2/decoupled_solo_r50_fpn_8gpu_1x.py  2 --work_dir work_dir/0602/$(hostname)/decoupled_solo_r50_fpn_8gpu_1x >> pc2_out.txt



CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG   --launcher pytorch ${@:3}
