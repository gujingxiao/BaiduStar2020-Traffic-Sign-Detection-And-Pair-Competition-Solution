#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
PYTHON=${PYTHON:-"python"}

#---------------------------------------------------------- test ----------------------------------------------------------------

#$PYTHON -u eval_pair.py --model=ResNet50 --pretrained_model=output_pair/ResNet50/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test --image_shape=3,160,160 \
#  --detect_path=../output_results/test/ensemble_test_050_filter --output_path=../output_results/test/ensemble_test_match --thres=0.65

#$PYTHON -u eval_pair.py --model=ResNeXt101_vd_64x4d --pretrained_model=output_pair/ResNeXt101_vd_64x4d/final.pdparams \
# --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test --image_shape=3,160,160 \
# --detect_path=../output_results/test/ensemble_test_050_filter --output_path=../output_results/test/ensemble_test_match --thres=0.65

#$PYTHON -u eval_pair.py --model=SE_ResNeXt50_vd_32x4d --pretrained_model=output_pair/SE_ResNeXt50_vd_32x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test --image_shape=3,160,160 \
#  --detect_path=../output_results/test/ensemble_test_050_filter --output_path=../output_results/test/ensemble_test_match --thres=0.65

#$PYTHON -u eval_pair.py --model=ResNeXt152_vd_32x4d --pretrained_model=output_pair/ResNeXt152_vd_32x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test --image_shape=3,160,160 \
#  --detect_path=../output_results/test/ensemble_test_050_filter --output_path=../output_results/test/ensemble_test_match --thres=0.65

#$PYTHON -u eval_pair.py --model=ResNeXt152_vd_64x4d --pretrained_model=output_pair/ResNeXt152_vd_64x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test --image_shape=3,160,160 \
#  --detect_path=../output_results/test/ensemble_test_050_filter --output_path=../output_results/test/ensemble_test_match --thres=0.65

#---------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------- val ----------------------------------------------------------------

#$PYTHON -u eval_pair.py --model=ResNet50 --pretrained_model=output_pair/ResNet50/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train --image_shape=3,160,160 \
#  --detect_path=../output_results/val/ensemble_val_050_filter --output_path=../output_results/val/ensemble_val_050_match --thres=0.9

#$PYTHON -u eval_pair.py --model=ResNeXt101_vd_64x4d --pretrained_model=output_pair/ResNeXt101_vd_64x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train --image_shape=3,160,160 \
#  --detect_path=../output_results/val/ensemble_val_050_filter --output_path=../output_results/val/ensemble_val_050_match --thres=0.9

#$PYTHON -u eval_pair.py --model=SE_ResNeXt50_vd_32x4d --pretrained_model=output_pair/SE_ResNeXt50_vd_32x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train --image_shape=3,160,160 \
#  --detect_path=../output_results/val/ensemble_val_050_filter --output_path=../output_results/val/ensemble_val_050_match --thres=0.9

#$PYTHON -u eval_pair.py --model=ResNeXt152_vd_32x4d --pretrained_model=output_pair/ResNeXt152_vd_32x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train --image_shape=3,160,160 \
#  --detect_path=../output_results/val/ensemble_val_050_filter --output_path=../output_results/val/ensemble_val_050_match --thres=0.9

#$PYTHON -u eval_pair.py --model=ResNeXt152_vd_64x4d --pretrained_model=output_pair/ResNeXt152_vd_64x4d/final.pdparams \
#  --data_path=/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train --image_shape=3,160,160 \
#  --detect_path=../output_results/val/ensemble_val_050_filter --output_path=../output_results/val/ensemble_val_050_match --thres=0.9

#---------------------------------------------------------------------------------------------------------------------------------





