import pickle
import numpy as np
import torch
import argparse
from tlp_train import *
from mtl_tlp_train import MTLTLPAttentionModule


top_ks = [1, 5, 10, 20]


def pred_a_dataset(datas, task_pred_dict, model):

    datas_new = []
    for data_idx, data in enumerate([datas]):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        datas_new.extend(line_vecs)

    if isinstance(model, BertModule):
        test_loader = BertSegmentDataLoader(datas_new, 128, False)
    elif isinstance(model, GPTModule):
        test_loader = GPTSegmentDataLoader(datas_new, 128, False)
    else:
        test_loader = SegmentDataLoader(datas_new, 128, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_all = []
    labels_all = []

    for batch_datas_steps, batch_labels in test_loader:
        batch_datas_steps = batch_datas_steps.to("cpu")
        preds = model(batch_datas_steps)
        if isinstance(preds, list) and len(preds) > 1:
            preds = preds[0]
        preds_all.append(preds.detach().cpu())
        labels_all.append(batch_labels.detach().cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    task_pred_dict[workloadkey] = (preds_all.detach().cpu().numpy(
    ), test_loader.min_latency.min().numpy(), labels_all.numpy())


def eval_model(model_file='',test_datasets=''):

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    model.eval()
    task_pred_dict = {}

    pred_a_dataset_dict = {}
    for data_idx, data in enumerate(test_datasets):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

    folder_path = '/home/huanting/PROM/benchmark/TensorT/network_info'  # 替换成你的文件夹路径
    # 获取文件夹中所有文件名
    file_names = os.listdir(folder_path)
    # 筛选出满足条件的文件名
    files = [os.path.join(folder_path, file_name) for file_name in file_names if file_name.endswith(',llvm).task.pkl')]
    top_1_total = []
    top_5_total = []
    top_10_total = []
    top_20_total = []
    best_latency_total_list = []
    best_latency_total = 0
    top1_total = 0
    top5_total = 0
    top10_total = 0
    top20_total = 0

    for file in files:
        tasks, task_weights = pickle.load(open(file, "rb"))
        latencies = [0] * len(top_ks)
        best_latency = 0
        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                # print('error task.workload_key not in pred_a_dataset_dict')
                continue

            pred_a_dataset(
                pred_a_dataset_dict[task.workload_key], task_pred_dict, model)
            preds, min_latency, labels = task_pred_dict[task.workload_key]
            real_values = labels[np.argsort(-preds)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight #预测
            best_latency += min_latency * weight
        try:
            top_1_total.append(best_latency/latencies[0])
            # print(f"top 1 score: {best_latency / latencies[0]}")
        except:
            top_1_total.append(0)
            # print(f"top 1 score: {0}")

        try:
            top_5_total.append(best_latency / latencies[1])
            # print(f"top 5 score: {best_latency / latencies[1]}")
        except:
            top_5_total.append(0)
            # print(f"top 5 score: {0}")


        best_latency_total_list.append(best_latency)
        best_latency_total += best_latency
        top1_total += latencies[0]
        top5_total += latencies[1]
        top10_total += latencies[2]
        top20_total += latencies[3]

    if top1_total == 0:
        print(f"average top 1 score is {0}")
        top_1_total.append(0)
    else:
        print(f"average top 1 score is {best_latency_total / top1_total}")
        top_1_total.append(best_latency_total / top1_total)

    if top5_total == 0:
        print(f"average top 5 score is {0}")
        top_5_total.append(0)
    else:
        print(f"average top 5 score is {best_latency_total / top5_total}")
        top_5_total.append(best_latency_total / top5_total)

    if top10_total == 0:
        print(f"average top 10 score is {0}")
        top_10_total.append(0)
    else:
        print(f"average top 10 score is {best_latency_total / top10_total}")
        top_10_total.append(best_latency_total / top10_total)

    if top20_total == 0:
        print(f"average top 20 score is {0}")
        top_20_total.append(0)
    else:
        print(f"average top 20 score is {best_latency_total / top20_total}")
        top_20_total.append(best_latency_total / top20_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default='cpu')
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--test_dataset_name", type=str,
    default='/home/huanting/model/cost_model/new/tlp/scripts/data_model/resnet50_test.pkl')
    parser.add_argument("--load_name", type=str,
    default='/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/resnet50.pkl')
    parser.add_argument("--platform", type=str, default='llvm')  # or cuda
    args = parser.parse_args()
    # print(args)
    model_paths=['/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/bert.pkl',
                 '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/bert_tiny.pkl',
                 '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/large.pkl',
                 '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/medium.pkl',
                 '/home/huanting/model/cost_model/new/tlp/scripts/tlp_i7/resnet50.pkl']

    device = 'cpu'

    with open(args.test_dataset_name, 'rb') as f:
        test_datasets = pickle.load(f)

    for i in model_paths:
        args.load_name = i
        print(i)
        eval_model(args.load_name)
