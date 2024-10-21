import subprocess
# setting the environments
subprocess.run(["bash", "env.sh", "C5"])
import sys
sys.path.append('/home/huanting/PROM/examples/case_study/tlp/scripts/')
sys.path.append('/home/huanting/PROM/examples/case_study/tlp/python')
sys.path.append('/home/huanting/PROM/thirdpackage')
sys.path.append('/home/huanting/PROM')
sys.path.append('/home/huanting/PROM/src')

import src.prom.prom_util as util
from sklearn.neural_network import MLPRegressor
from prom.regression import MapieQuantileRegressor, MapieRegressor
from prom.metrics import regression_coverage_score


from ae_train_tlp import ae_eval_model,AttentionModule
from ae_deploy_tlp import ae_deploy_model





# ae_eval_model("all")
ae_deploy_model("all")

"""
--output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../benchmark/Bug/train.jsonl     --eval_data_file=../../benchmark/Bug/valid.jsonl     --test_data_file=../../benchmark/Bug/test.jsonl --evaluate_during_training
"""

