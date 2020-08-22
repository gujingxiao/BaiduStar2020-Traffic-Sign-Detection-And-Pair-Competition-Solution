#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
PYTHON=${PYTHON:-"python"}

$PYTHON -u train_pair.py --model=SE_ResNeXt50_vd_32x4d --pretrained_model=output_pair/SE_ResNeXt50_vd_32x4d/1500_160 \
  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train --train_batch_size=72 --lr_steps=2100,3600,4200 \
  --image_shape=3,160,160 --total_iter_num=4500 --save_iter_step=300 --lr=0.005