import pickle
import numpy as np
import torch
import argparse
# from tlp_train import *
from mtl_tlp_train import MTLTLPAttentionModule







# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cuda", type=str, default='cpu')
#     parser.add_argument("--start_idx", type=int, default=0)
#     parser.add_argument("--test_dataset_name", type=str,
#     default='/home/huanting/model/cost_model/new/tlp/scripts/data_model/resnet50_test.pkl')
#     parser.add_argument("--load_name", type=str,
#     default='/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/resnet50.pkl')
#     parser.add_argument("--platform", type=str, default='llvm')  # or cuda
#     args = parser.parse_args()
#     # print(args)
#     model_paths=['/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/bert.pkl',
#                  '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/bert_tiny.pkl',
#                  '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/large.pkl',
#                  '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/medium.pkl',
#                  '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/resnet50.pkl']
#
#     device = 'cpu'
#
#     with open(args.test_dataset_name, 'rb') as f:
#         test_datasets = pickle.load(f)
#
#     for i in model_paths:
#         args.load_name = i
#         print(i)
#         eval_model(args.load_name)
