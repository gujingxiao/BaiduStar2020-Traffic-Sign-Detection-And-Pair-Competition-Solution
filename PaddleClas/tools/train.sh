export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD:$PYTHONPATH
python -m paddle.distributed.launch --selected_gpus="0,1" tools/train.py -c ./configs/Res2Net/Res2Net101_vd_26w_4s.yaml