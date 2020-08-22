#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1

PYTHON=${PYTHON:-"python"}

$PYTHON -u tools/train.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal.yml \
 -r output/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/40000