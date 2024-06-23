#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='2mqvwyxz'
export NNI_SYS_DIR='/home/huanting/nni-experiments/2mqvwyxz/trials/XYJvW'
export NNI_TRIAL_JOB_ID='XYJvW'
export NNI_OUTPUT_DIR='/home/huanting/nni-experiments/2mqvwyxz/trials/XYJvW'
export NNI_TRIAL_SEQ_ID='5'
export NNI_CODE_DIR='/home/huanting/model/compy-learn-master'
export CUDA_VISIBLE_DEVICES='-1'
cd $NNI_CODE_DIR
eval 'python /home/huanting/model/compy-learn-master/devmap_exploration.py' 1>/home/huanting/nni-experiments/2mqvwyxz/trials/XYJvW/stdout 2>/home/huanting/nni-experiments/2mqvwyxz/trials/XYJvW/stderr
echo $? `date +%s%3N` >'/home/huanting/nni-experiments/2mqvwyxz/trials/XYJvW/.nni/state'