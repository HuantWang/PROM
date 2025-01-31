#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

python ae_cd_thread.py
python ae_cd_loop.py
python ae_cd_vul.py --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../benchmark/Bug/train.jsonl     --eval_data_file=../../benchmark/Bug/valid.jsonl     --test_data_file=../../benchmark/Bug/test.jsonl --evaluate_during_training


conda activate dev
echo "Environment for 'C3' activated."
cd ../case_study/DeviceM/
python ae_dev_cov_dev.py
cd ../../tutorial/

conda activate tvm
python ae_cd_tlp.py