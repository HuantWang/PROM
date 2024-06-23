#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='ovhm7lga'
export NNI_SYS_DIR='/home/huanting/nni-experiments/ovhm7lga/trials/pqW2t'
export NNI_TRIAL_JOB_ID='pqW2t'
export NNI_OUTPUT_DIR='/home/huanting/nni-experiments/ovhm7lga/trials/pqW2t'
export NNI_TRIAL_SEQ_ID='157'
export NNI_CODE_DIR='/home/huanting/model/compy-learn-master'
export CUDA_VISIBLE_DEVICES='-1'
cd $NNI_CODE_DIR
eval 'python /home/huanting/model/compy-learn-master/devmap_exploration.py' 1>/home/huanting/nni-experiments/ovhm7lga/trials/pqW2t/stdout 2>/home/huanting/nni-experiments/ovhm7lga/trials/pqW2t/stderr
echo $? `date +%s%3N` >'/home/huanting/nni-experiments/ovhm7lga/trials/pqW2t/.nni/state'