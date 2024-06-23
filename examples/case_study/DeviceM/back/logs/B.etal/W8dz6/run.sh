#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='cedisl1k'
export NNI_SYS_DIR='/home/huanting/nni-experiments/cedisl1k/trials/W8dz6'
export NNI_TRIAL_JOB_ID='W8dz6'
export NNI_OUTPUT_DIR='/home/huanting/nni-experiments/cedisl1k/trials/W8dz6'
export NNI_TRIAL_SEQ_ID='19'
export NNI_CODE_DIR='/home/huanting/model/compy-learn-master'
export CUDA_VISIBLE_DEVICES='-1'
cd $NNI_CODE_DIR
eval 'python /home/huanting/model/compy-learn-master/devmap_exploration.py' 1>/home/huanting/nni-experiments/cedisl1k/trials/W8dz6/stdout 2>/home/huanting/nni-experiments/cedisl1k/trials/W8dz6/stderr
echo $? `date +%s%3N` >'/home/huanting/nni-experiments/cedisl1k/trials/W8dz6/.nni/state'