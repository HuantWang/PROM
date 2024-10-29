
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
from ae_train_tlp import AttentionModule

from ae_cd_tlp_script import ae_cd_model



ae_cd_model("all")

"""
--output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../benchmark/Bug/train.jsonl     --eval_data_file=../../benchmark/Bug/valid.jsonl     --test_data_file=../../benchmark/Bug/test.jsonl --evaluate_during_training
"""

