#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

PYTHON=${PYTHON:-"python"}

#--------------------------------------------------------------------- eval --------------------------------------------------------------------
#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/30000


#-------------------------------------------------------------------- test ----------------------------------------------------------------------
#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal/cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_cbr50_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas/cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas_test.yml \
# -o weights=output/cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas/model_final save_prediction_only=true -p ../output_results/test/cascade_senet154_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_r101_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn/cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn_test.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn/model_final save_prediction_only=true -p ../output_results/test/cascade_x101_c19_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_r200_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_cbr200_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_r152_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_r2n200_c3_test

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal_test.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/test/cascade_r2n101_c3_test

#--------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------- val ---------------------------------------------------------------------
#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal/cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cbr50_vd_fpn_dcnv2_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_cbr50_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas/cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas_val.yml \
# -o weights=output/cascade_rcnn_dcnv2_se154_vd_fpn_gn_cas/model_final -p ../output_results/val/cascade_se154_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r101_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_r101_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn/cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_x101_vd_64x4d_fpn/model_final save_prediction_only=true -p ../output_results/val/cascade_x101_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r200_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_r200_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cbr200_vd_fpn_dcnv2_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_cbr200_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_r152_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_r152_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_res2net200_vd_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_r2n200_c3_val

#$PYTHON -u tools/eval.py -c configs/star2020/cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal/cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal_val.yml \
# -o weights=output/cascade_rcnn_cls_aware_dcn_res2net101_vd_fpn_nonlocal/model_final save_prediction_only=true -p ../output_results/val/cascade_r2n101_c3_val

#------------------------------------------------------------------------------------------------------------------------------------------------
