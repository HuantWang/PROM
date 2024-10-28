import sys
sys.path.append('../case_study/BugD/')

from ae_VD_codebert import ae_vul_codebert
from ae_VD_linevul import ae_vul_linevul
from ae_VD_vulde import ae_vul_vulde

print("\nCase 4:\n")
print("\n--- The Evaluation on CodeBERT ---\n")
ae_vul_codebert()

print("\n--- The Evaluation on Linevul ---\n")
ae_vul_linevul()

print("\n--- CThe Evaluation on VUlDE ---\n")
ae_vul_vulde()
"""
--output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../benchmark/Bug/train.jsonl     --eval_data_file=../../benchmark/Bug/valid.jsonl     --test_data_file=../../benchmark/Bug/test.jsonl --evaluate_during_training
"""

