import sys
sys.path.append('../case_study/BugD/')

from ae_VD_codebert import ae_vul_codebert
from ae_VD_linevul import ae_vul_linevul
from ae_VD_vulde import ae_vul_vulde
# ae_vul_codebert()
# ae_vul_linevul()
ae_vul_vulde()
"""
--output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../benchmark/Bug/train.jsonl     --eval_data_file=../../benchmark/Bug/valid.jsonl     --test_data_file=../../benchmark/Bug/test.jsonl --evaluate_during_training
"""

