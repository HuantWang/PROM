#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='bn28mpu6'
export NNI_SYS_DIR='/home/huanting/nni-experiments/bn28mpu6/trials/Tnm79'
export NNI_TRIAL_JOB_ID='Tnm79'
export NNI_OUTPUT_DIR='/home/huanting/nni-experiments/bn28mpu6/trials/Tnm79'
export NNI_TRIAL_SEQ_ID='7'
export NNI_CODE_DIR='/home/huanting/model/compy-learn-master'
export CUDA_VISIBLE_DEVICES='-1'
cd $NNI_CODE_DIR
eval 'python /home/huanting/model/compy-learn-master/devmap_exploration.py' 1>/home/huanting/nni-experiments/bn28mpu6/trials/Tnm79/stdout 2>/home/huanting/nni-experiments/bn28mpu6/trials/Tnm79/stderr
echo $? `date +%s%3N` >'/home/huanting/nni-experiments/bn28mpu6/trials/Tnm79/.nni/state'