import os
import pickle
import torch
import time
import numpy as np
import random
import math
from torch import nn
from torch import optim
import argparse
import sys
import nni

# sys.path.append('/home/huanting/PROM/src')


sys.path.append('/home/huanting/PROM/examples/case_study/tlp/python')
sys.path.append('/home/huanting/PROM/thirdpackage')
sys.path.append('/home/huanting/PROM')
sys.path.append('/home/huanting/PROM/src')

import src.prom.prom_util as util
from sklearn.neural_network import MLPRegressor
from prom.regression import MapieQuantileRegressor, MapieRegressor
from prom.metrics import regression_coverage_score


def pred_a_dataset(datas, task_pred_dict, model):
    """
    Generates predictions for a given dataset using the specified model.

    Args:
        datas: Data to predict.
        task_pred_dict: Dictionary to store task predictions.
        model: Trained model for making predictions.
    """
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


def eval_model(model_file='', test_datasets=''):
    """
    Evaluates the performance of a model on a test dataset.

    Args:
        model_file (str): Path to the saved model file.
        test_datasets: Test dataset for evaluation.

    Returns:
        float: Average performance score.
    """
    top_ks = [1, 5, 10, 20]
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
                latencies[i] += np.min(real_latency[:top_k]) * weight  # 预测
            best_latency += min_latency * weight
        try:
            top_1_total.append(best_latency / latencies[0])
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

    return best_latency_total / top5_total


def get_cosine_schedule_with_warmup(
        optimizer: optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class Tlp_prom(util.ModelDefinition):
    """
    Model definition for Tlp_prom, managing data partitioning and feature extraction.
    """
    def __init__(self, model=None, dataset=None, calibration_data=None, args=None):
        """
        Initializes the model, dataset, and calibration data.
        """
        # self.model =
        self.calibration_data = None
        self.dataset = None

    def data_partitioning(self, train_dataset, test_dataset, calibration_ratio=0.2, args=None):
        """
        Partitions data into training and test datasets.

        Args:
            train_dataset: Training dataset path.
            test_dataset: Test dataset path.
            calibration_ratio: Ratio for calibration data.
            args: Additional arguments.
        """
        # prepare dataset
        if os.path.exists(args.save_folder) is False:
            print('create folder', args.save_folder)
            os.makedirs(args.save_folder, exist_ok=True)

        with open(train_dataset, 'rb') as f:
            datasets_global = pickle.load(f)

        train_data = load_datas(datasets_global)
        # print('create dataloader done.')
        del datasets_global

        # random.seed(args.seed)
        with open(test_dataset, 'rb') as f:
            # with open(r'/home/huanting/PROM/examples/case_study/tlp/scripts/data_model/bert_tiny_test.pkl', 'rb') as f:
            test_datasets = pickle.load(f)

        random.shuffle(test_datasets)
        length = len(test_datasets)
        # print("length", length)
        test_dataset = test_datasets[:1]

        return train_data, test_dataset

    def predict(self, X, significant_level=0.1):
        """
        Predicts outcomes with a given confidence level.

        Args:
            X: Input data.
            significant_level: Confidence level for prediction.

        Returns:
            Tuple containing predictions and probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        pred = self.model.predict(self, sequences='')
        probability = self.model.predict_proba(self, sequences='')
        return pred, probability

    def feature_extraction(self, srcs):
        pass


class AttentionModule(nn.Module):
    """
    Neural network module implementing an attention mechanism.
    """
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size
        self.res_block_cnt = args.res_block_cnt

        in_dim = self.fea_size
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim_1, args.attention_head)

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.l_list = []
        for i in range(self.res_block_cnt):
            self.l_list.append(nn.Sequential(
                nn.Linear(hidden_dim_1, hidden_dim_1),
                nn.ReLU()
            ))
        self.l_list = nn.Sequential(*self.l_list)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):

        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.attention(encoder_output, encoder_output, encoder_output)[0] + encoder_output

        for l in self.l_list:
            output = l(output) + output

        output = self.decoder(output).sum(0)

        return output.squeeze()

    def fit(self, batch_datas_steps):
        return 1

    def predict(self, batch_datas_steps):
        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.attention(encoder_output, encoder_output, encoder_output)[0] + encoder_output

        for l in self.l_list:
            output = l(output) + output

        output = self.decoder(output).sum(0)
        return output.squeeze()


class TransformerEncoderLayerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size

        in_dim = self.fea_size
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim_1, dim_feedforward=256, nhead=args.attention_head
        )

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):
        batch_datas_steps = batch_datas_steps[:,
                            :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)
        encoder_output = encoder_output.transpose(0, 1)
        output = self.transformer_encoder_layer(encoder_output)
        output = self.l0(output) + output
        output = self.l1(output) + output
        output = self.decoder(output).sum(0)
        return output.squeeze()


class TransformerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size

        in_dim = self.fea_size
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim_1 = hidden_dim[-1]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim_1, dim_feedforward=256, nhead=args.attention_head
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_1, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):
        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]
        encoder_output = self.encoder(batch_datas_steps)

        encoder_output = encoder_output.transpose(0, 1)
        output = self.transformer(encoder_output)

        output = self.decoder(output).sum(0)

        return output.squeeze()


class LSTMModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fea_size = args.fea_size
        self.step_size = args.step_size

        lstm_linar_in_dim = self.fea_size
        lstm_linar_hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        hidden_dim = lstm_linar_hidden_dim[-1]

        self.lstm_linar_encoder = nn.Sequential(
            nn.Linear(lstm_linar_in_dim, lstm_linar_hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[0], lstm_linar_hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[1], lstm_linar_hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(lstm_linar_hidden_dim[2], lstm_linar_hidden_dim[3]),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            lstm_linar_hidden_dim[-1], lstm_linar_hidden_dim[-1])

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, out_dim[0]),
            nn.ReLU(),
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
            nn.ReLU(),
            nn.Linear(out_dim[2], out_dim[3]),
        )

    def forward(self, batch_datas_steps):
        batch_datas_steps = batch_datas_steps[:, :self.step_size, :self.fea_size]

        batch_datas_steps = batch_datas_steps.transpose(0, 1)
        lstm_output = self.lstm_linar_encoder(batch_datas_steps)
        _, (h, c) = self.lstm(lstm_output)
        lstm_output = h[0]

        output = lstm_output

        output = self.l0(output) + output
        output = self.l1(output) + output

        output = self.decoder(output)

        return output.squeeze()


class GPTModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_sup_model = args.self_sup_model

        from minGPT.gpt_model import GPUModel
        self.gpt = GPUModel(self.self_sup_model).model
        out_dim = [23, 256, 1]

        self.decoder = nn.Sequential(
            nn.Linear(out_dim[0], out_dim[1]),
            nn.ReLU(),
            nn.Linear(out_dim[1], out_dim[2]),
        )

    def forward(self, batch_datas_steps):
        output = self.gpt(batch_datas_steps)[0].mean(1)
        output = self.decoder(output)
        return output.squeeze()


class BertModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_sup_model = args.self_sup_model

        from bert.bert_model import BertModel
        self.bert = BertModel(self.self_sup_model).model

        self.decode = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, batch_datas_steps):
        output = self.bert(batch_datas_steps).logits[:, 0, :]
        output = self.decode(output)
        return output.squeeze()


class LambdaRankLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1.):
        device = self.device
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :,
                     None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(
            ((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (
                y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(
            sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


class SegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.iter_order = self.pointer = None

        datas_steps = []
        labels = []
        min_latency = []
        for data_idx, data in enumerate(dataset):
            data = data[:3]
            datas_step, label, min_lat = data
            datas_steps.append(datas_step)
            labels.append(label)
            min_latency.append(min_lat)

        self.datas_steps = torch.FloatTensor(datas_steps)
        self.labels = torch.FloatTensor(labels)
        self.min_latency = torch.FloatTensor(min_latency)

        self.number = len(self.datas_steps)

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):

        batch_datas_steps = self.datas_steps[indices]
        batch_datas_steps = nn.utils.rnn.pad_sequence(
            batch_datas_steps, batch_first=True)
        batch_labels = self.labels[indices]

        return (batch_datas_steps, batch_labels)

    def __len__(self):
        return self.number


class GPTSegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.iter_order = self.pointer = None

        datas_steps = []
        labels = []
        min_latency = []
        for data_idx, data in enumerate(dataset):
            datas_step, label, min_lat = data
            datas_step_new = []
            for step in datas_step:
                step_new = step.copy()
                step_new.insert(0, 1 - sum(step_new[:11]))
                datas_step_new.append(step_new)

            datas_steps.append(datas_step_new)
            labels.append(label)
            min_latency.append(min_lat)

        self.datas_steps = torch.FloatTensor(datas_steps)
        self.labels = torch.FloatTensor(labels)
        self.min_latency = torch.FloatTensor(min_latency)

        self.number = len(self.datas_steps)

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        batch_datas_steps = self.datas_steps[indices]
        batch_datas_steps = nn.utils.rnn.pad_sequence(
            batch_datas_steps, batch_first=True)
        batch_labels = self.labels[indices]
        return (batch_datas_steps[:, :-1, :], batch_labels)

    def __len__(self):
        return self.number


class BertSegmentDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.iter_order = self.pointer = None

        datas_steps = []
        labels = []
        min_latency = []
        for data_idx, data in enumerate(dataset):
            datas_step, label, min_lat = data
            datas_steps.append(datas_step)
            labels.append(label)
            min_latency.append(min_lat)

        self.datas_steps = torch.LongTensor(datas_steps)
        self.labels = torch.FloatTensor(labels)
        self.min_latency = torch.FloatTensor(min_latency)

        self.number = len(self.datas_steps)

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):

        batch_datas_steps = self.datas_steps[indices]
        batch_labels = self.labels[indices]
        return batch_datas_steps, batch_labels

    def __len__(self):
        return self.number


def load_datas(datasets_global):
    """
    Loads and partitions global dataset into training and validation sets.

    Args:
        datasets_global: Complete dataset.

    Returns:
        Tuple of training and validation data loaders.
    """
    datasets = np.array(datasets_global, dtype=object)
    if args.data_cnt > 0:
        train_len = int(args.data_cnt * 1000 * 0.9)
        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:args.data_cnt * 1000]
        # train_indices, val_indices = perm[:1], perm[train_len:args.data_cnt * 1000]
    else:
        train_len = int(len(datasets) * 0.9)
        train_len = int(500 * 0.8)
        # print("valid length",len(datasets))
        perm = np.random.permutation(len(datasets))
        train_indices, val_indices = perm[:train_len], perm[train_len:500]
        # train_indices, val_indices = perm, perm
    train_datas, val_datas = datasets[train_indices], datasets[val_indices]

    n_gpu = int(8)
    if args.attention_class == 'gpt':
        train_dataloader = GPTSegmentDataLoader(train_datas, args.train_size_per_gpu * n_gpu, True)
        val_dataloader = GPTSegmentDataLoader(val_datas, args.train_size_per_gpu * n_gpu, False)
    elif args.attention_class == 'bert':
        train_dataloader = BertSegmentDataLoader(train_datas, args.train_size_per_gpu * n_gpu, True)
        val_dataloader = BertSegmentDataLoader(val_datas, args.train_size_per_gpu * n_gpu, False)
    else:
        train_dataloader = SegmentDataLoader(
            train_datas, args.train_size_per_gpu * n_gpu, True)
        val_dataloader = SegmentDataLoader(
            val_datas, args.val_size_per_gpu * n_gpu, False)

    return train_dataloader, val_dataloader


def validate(model, valid_loader, loss_func, device):
    """
    Validates the model on the validation dataset.

    Args:
        model: Trained model.
        valid_loader: Validation data loader.
        loss_func: Loss function for validation.
        device: Device for model and data.

    Returns:
        Sum of validation losses.
    """
    model.eval()
    valid_losses = []

    for batch_datas_steps, batch_labels in valid_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        batch_labels = batch_labels.to(device)

        preds = model(batch_datas_steps)
        valid_losses.append(loss_func(preds, batch_labels).item())

    return np.sum(valid_losses)


def train(train_loader, val_dataloader, device, test_tlp):
    """
    Trains the model and performs validation periodically.

    Args:
        train_loader: Training data loader.
        val_dataloader: Validation data loader.
        device: Device for model and data.
        test_tlp: Test data for evaluation.

    Returns:
        Tuple of model save file name and best performance score.
    """
    performance = 0
    best_performance = 0
    if args.attention_class == 'default':
        args.hidden_dim = [64, 128, 256, 256]
        args.out_dim = [256, 128, 64, 1]
        net = AttentionModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'transformer':
        print('TransformerModule')
        net = TransformerModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'attention_encoder_layer':
        print('TransformerEncoderLayerModule')
        net = TransformerEncoderLayerModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'lstm':
        print('LSTMModule')
        net = LSTMModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'gpt':
        print('GPTModule')
        net = GPTModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'bert':
        print('BertModule')
        net = BertModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'attention_512':
        args.hidden_dim = [64, 128, 256, 512]
        args.out_dim = [256, 128, 64, 1]
        print('Attention512Module')
        net = AttentionModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'attention_768':
        args.hidden_dim = [64, 256, 512, 768]
        args.out_dim = [512, 256, 128, 1]
        print('Attention768Module')
        net = AttentionModule().to(device)
        net = net.to(torch.device("cpu"))
    elif args.attention_class == 'attention_1024':
        args.hidden_dim = [64, 256, 512, 1024]
        args.out_dim = [512, 256, 128, 1]
        print('Attention1024Module')
        net = AttentionModule().to(device)
        net = net.to(torch.device("cpu"))

    if args.rank_mse == 'rank':
        loss_func = LambdaRankLoss(device)
    else:
        loss_func = nn.MSELoss()

    n_epoch = args.n_epoch
    if args.optimizer == 'default':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=1)
    elif args.optimizer == 'decrease_per_17_0.8':
        print('optimizer', 'decrease_per_17_0.8')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.8)
    elif args.optimizer == 'decrease_per_17_0.5':
        print('optimizer', 'decrease_per_17_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.5)
    elif args.optimizer == 'decrease_per_12_0.5':
        print('optimizer', 'decrease_per_12_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 4, gamma=0.5)
    elif args.optimizer == 'decrease_per_10_0.5':
        print('optimizer', 'decrease_per_10_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 5, gamma=0.5)
    elif args.optimizer == 'decrease_per_17_0.5_no_decay':
        print('optimizer', 'decrease_per_17_0.5')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.5)

    train_loss = None
    print('start train...')
    # print(len(train_loader), len(val_dataloader))
    best_valid_loss = 1e6
    for epoch in range(n_epoch):
        tic = time.time()

        net.train()
        train_loss = 0
        for batch, (batch_datas_steps, batch_labels) in enumerate(train_loader):
            batch_datas_steps = batch_datas_steps.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = loss_func(net(batch_datas_steps), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        lr_scheduler.step()

        train_time = time.time() - tic

        if epoch % 10 == 0 or epoch == n_epoch - 1:
            valid_loss = validate(net, val_dataloader,
                                  loss_func, device=device)
            loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (
                train_loss, valid_loss)
            print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                epoch, batch, loss_msg, len(train_loader) / train_time,))
            model_save_file_name = '%s/tlp_model_%d.pkl' % (args.save_folder, args.seed)

        if best_valid_loss > valid_loss:
            print("save the current model...")
            best_valid_loss = valid_loss
            with open(model_save_file_name, 'wb') as f:
                pickle.dump(net.cpu(), f)
            print("evaluating...")
            performance = eval_model(model_file=model_save_file_name, test_datasets=test_tlp)

            if best_performance < performance:
                best_performance = performance
                print("The best top-1 performance is", best_performance)
                model_save_file_name = '%s/tlp_model_%d_best.pkl' % (args.save_folder, args.seed)
                with open(model_save_file_name, 'wb') as f:
                    pickle.dump(net.cpu(), f)

    net = net.to(device)
    return model_save_file_name, best_performance


def il(test_loader, device, pre_trained_model, aug_data, args):
    """
    Implements incremental learning using pre-trained model on augmented data.

    Args:
        test_loader: Test data loader.
        device: Device for model and data.
        pre_trained_model: Path to pre-trained model.
        aug_data: Augmented data for incremental learning.
        args: Additional arguments.

    Returns:
        Path to the updated model after fine-tuning.
    """

    if pre_trained_model:
        with open(pre_trained_model, 'rb') as f:
            net = pickle.load(f)
            print("Loaded pre-trained model for fine-tuning.")
    else:
        raise ValueError("No pre-trained model path provided")

    datas_new = []
    for data_idx, data in enumerate(test_loader):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        datas_new.extend(line_vecs)

    selected_il = np.random.choice(aug_data, len(aug_data) // 20, replace=False)

    il_dataset = [datas_new[i] for i in selected_il]
    test_dataset = [datas_new[i] for i in range(len(datas_new)) if i not in selected_il]

    il_dataset_loader = SegmentDataLoader(il_dataset, 128, False)
    test_dataset_loader = SegmentDataLoader(test_dataset, 128, False)


    if args.rank_mse == 'rank':
        loss_func = LambdaRankLoss(device)  # 假设 LambdaRankLoss 是已经定义好的类
    else:
        loss_func = nn.MSELoss()


    optimizer = optim.Adam(net.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_epoch // 3, gamma=0.8)


    print('Start fine-tuning...')
    for epoch in range(args.n_epoch):
        net.train()
        train_loss = 0
        tic = time.time()
        for batch_idx, (inputs, targets) in enumerate(il_dataset_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()

        lr_scheduler.step()


        if epoch % 5 == 0 or epoch == args.n_epoch - 1:
            valid_loss = validate(net, test_dataset_loader, loss_func, device)
            print(f"Epoch: {epoch} Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f}")


    model_save_file_name = f'{args.save_folder}/tlp_model_{args.seed}.pkl'
    with open(model_save_file_name, 'wb') as f:
        pickle.dump(net.cpu(), f)
    net.to(device)

    return model_save_file_name


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def conformal_prediction(datas, task_pred_dict, model, mapie, task_drift_dict, task_after_dict,cluster_num):
    """
    Performs conformal prediction for anomaly detection.

    Args:
        datas: Input data.
        task_pred_dict: Dictionary to store task predictions.
        model: Trained model.
        mapie: Conformal prediction model.
        task_drift_dict: Dictionary for detected drift data.
        task_after_dict: Dictionary for post-drift data.
        cluster_num: Number of clusters for prediction.

    Returns:
        Indices of detected drifted data, F1 scores, precision, and recall.
    """
    datas_new = []
    for data_idx, data in enumerate([datas]):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        datas_new.extend(line_vecs)

    if isinstance(model, BertModule):
        test_loader = BertSegmentDataLoader(datas_new, 512, False)
    elif isinstance(model, GPTModule):
        test_loader = GPTSegmentDataLoader(datas_new, 512, False)
    else:
        test_loader = SegmentDataLoader(datas_new, 512, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_all = []
    labels_all = []
    data_test = []

    for batch_datas_steps, batch_labels in test_loader:
        batch_datas_steps = batch_datas_steps.to('cpu')
        data_test.append(batch_datas_steps)
        preds = model(batch_datas_steps)
        if isinstance(preds, list) and len(preds) > 1:
            preds = preds[0]
        preds_all.append(preds.detach().cpu())
        labels_all.append(batch_labels.detach().cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    # 计算绝对差异百分比
    # a=test_loader.min_latency.min()
    percentage_difference = np.abs((test_loader.min_latency.min() - labels_all) / labels_all) * 100
    # 找到差异大于 20% 的索引
    indices_real = np.where(percentage_difference > 20)[0]
    print("The size of indices is :", indices_real.size)
    """cp"""
    data_test = torch.cat(data_test, dim=0)
    y_test = labels_all
    alphas = np.arange(0.1, 1, 0.1)
    credibility_result, confidence_result = mapie.predict(data_test, alpha=alphas)
    # results = [regression_coverage_score(y_test, y_pis[:, 0, i], y_pis[:, 1, i]) for i, _ in enumerate(alphas)]
    # coverage = [result[0] for result in results]
    # TF = [result[1] for result in results]

    task_pred_dict[workloadkey] = (preds_all.detach().cpu().numpy(
    ), test_loader.min_latency.min().numpy(), labels_all.numpy())

    all_indices = np.arange(credibility_result.shape[0])

    # 创建一个字典来存储每个 alpha 的结果
    indices_detec_dict = {}
    indices_not_detec_dict = {}
    best_f1 = 0
    best_indices_detec = []
    f1_score_all=[]
    precision_all=[]
    recall_all=[]
    for index, pvalue in enumerate(alphas):
        # if index==num_set:
        #     if all(pvalue) or not any(pvalue):
        #         print("all ture/false, break")

        # indices_p = np.where(pvalue == False)[0]
        indices_cre = np.where(credibility_result[:, index] == True)[0]
        indices_con = np.where(confidence_result[:, index] == True)[0]

        indices_detec = np.union1d(indices_cre, indices_con)  # np.intersect1d 交
        indices_detec_dict[pvalue] = indices_detec
        # 获取所有索引中不在 indices_detec 中的索引
        indices_not_detec = np.setdiff1d(all_indices, indices_detec)
        indices_not_detec_dict[pvalue] = indices_not_detec

        true_positives = np.sum(np.isin(indices_real, indices_detec))
        # 计算假正例（False Positives）
        false_positives = len(indices_detec) - true_positives

        # 计算真负例（True Negatives）
        true_negatives = np.sum(indices_real != indices_detec)

        # 计算假负例（False Negatives）
        false_negatives = len(indices_real) - true_positives

        # 计算准确率（Accuracy）
        accuracy = (true_positives + true_negatives) / len(indices_real)

        # 计算精度（Precision）
        precision = true_positives / (true_positives + false_positives)

        # 计算召回率（Recall）
        recall = true_positives / (true_positives + false_negatives)

        # 计算 F1 分数
        f1_score = 2 * (precision * recall) / (precision + recall)
        if best_f1 < f1_score:
            best_f1 = f1_score
            best_indices_detec = indices_detec
        print("The alpha is:", pvalue)
        # print(f"Detection f1_score is: {f1_score:.2%}, "
        #       f"accuracy is: {accuracy:.2%}, "
        #       f"precision is: {precision:.2%}, recall is: {recall:.2%}")
        f1_score_all.append(f1_score)
        precision_all.append(precision)
        recall_all.append(recall)

    task_after_dict[workloadkey] = (preds_all[indices_detec].detach().cpu().numpy(),
                                    test_loader.min_latency[indices_detec].min().numpy(),
                                    labels_all[indices_detec].numpy()
                                    )
    try:
        task_drift_dict[workloadkey] = (preds_all[indices_not_detec].detach().cpu().numpy(),
                                        test_loader.min_latency[indices_not_detec].min().numpy(),
                                        labels_all[indices_not_detec].numpy()
                                        )
    except:
        task_drift_dict[workloadkey] = (preds_all[indices_not_detec].detach().cpu().numpy(),
                                        test_loader.min_latency[indices_detec].min().numpy(),
                                        labels_all[indices_not_detec].numpy()
                                        )

    return best_indices_detec, f1_score_all, precision_all, recall_all


def cp(train_loader, val_dataloader, test_datasets, underlying_path, file_pattern,cluster_num=2):
    """
    Runs conformal prediction and anomaly detection on test data.

    Args:
        train_loader: Training data loader.
        val_dataloader: Validation data loader.
        test_datasets: Test datasets.
        underlying_path: Path to pre-trained model.
        file_pattern: Pattern for file matching.
        cluster_num: Number of clusters for conformal prediction.

    Returns:
        Indices of detected anomalies, F1 scores, precision, and recall.
    """
    model_save_file_name = underlying_path
    with open(model_save_file_name, 'rb') as f:
        net = pickle.load(f)
    """calibration data"""
    data_valid = []
    y_valid = []
    for batch_datas_steps, batch_labels in val_dataloader:
        # a=batch_datas_steps.shape
        # b=batch_labels.shape
        data_valid.append(batch_datas_steps)
        y_valid.append(batch_labels)
    data_valid = torch.cat(data_valid, dim=0)
    y_valid = torch.cat(y_valid, dim=0)
    """CP"""
    import numpy as np

    print("Train the anamoly detection model...")
    # alphas = np.arange(0.1, 1, 0.1)
    # for batch_datas_steps, batch_labels in val_dataloader:
    est_mlp = net
    mapie = MapieRegressor(est_mlp, cv="prefit")
    mapie.fit(data_valid, y_valid,cluster_num=cluster_num)

    # Evaluate prediction and coverage level on testing set
    """test data"""
    net.eval()
    pred_a_dataset_dict = {}
    print("Use the anamoly detection model to detect drifting data...")
    for data_idx, data in enumerate(test_datasets):
        file, file_idx, workloadkey_idx, workloadkey, workload_args, flop_ct, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data


    folder_path = '/home/huanting/PROM/benchmark/TensorT/network_info'  #

    import glob
    files = glob.glob(f"{folder_path}/{file_pattern}")
    # files = [os.path.join(folder_path, file_name) for file_name in file_names if file_name.endswith(',llvm).task.pkl')]

    top_ks = [1, 5]
    # for num_set in range(len(np.arange(0.1, 1, 0.1))):
    task_pred_dict = {}
    task_drift_dict = {}
    task_after_dict = {}
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
    # try:
    print("_________________________________________________")
    # print("value = ", num_set)
    # indices_detec_all=[]
    for file in files:
        tasks, task_weights = pickle.load(open(file, "rb"))
        latencies = [0] * len(top_ks)
        best_latency = 0
        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                # print('error task.workload_key not in pred_a_dataset_dict')
                continue
            indices_detec,f1,precision,recall = conformal_prediction(
                pred_a_dataset_dict[task.workload_key], task_pred_dict,
                net, mapie, task_drift_dict, task_after_dict,cluster_num)
            # f1_maxindex = f1.index(max(f1))
            # f1_max= f1[f1_maxindex]
            # pre_max = precision[f1_maxindex]
            # rec_max = recall[f1_maxindex]
            # indices_detec.append(indices_detec)
    return indices_detec,f1,precision,recall
    # p-value
    #         preds, min_latency, labels = task_pred_dict[task.workload_key]
    #         real_values = labels[np.argsort(-preds)]
    #         real_latency = min_latency / np.maximum(real_values, 1e-5)
    #         for i, top_k in enumerate(top_ks):
    #             latencies[i] += np.min(real_latency[:top_k]) * weight
    #         best_latency += min_latency * weight
    #
    #     best_latency_total += best_latency
    #     top1_total += latencies[0]
    #     top5_total += latencies[1]
    #
    # if top1_total == 0:
    #     print(f"average top 1 score is {0}")
    #     top_1_total.append(0)
    # else:
    #     print(f"average top 1 score is {best_latency_total / top1_total}")
    #     top_1_total.append(best_latency_total / top1_total)
    #
    # if top5_total == 0:
    #     print(f"average top 5 score is {0}")
    #     top_5_total.append(0)
    # else:
    #     print(f"average top 5 score is {best_latency_total / top5_total}")
    #     top_5_total.append(best_latency_total / top5_total)




def init_args():
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "seed": 1730,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, default='models/il/tlp_i7_tiny')
    parser.add_argument('--mode', choices=['train', 'deploy'], help="Mode to run: train or deploy")
    parser.add_argument("--under_train_dataset", type=str,
                        default='./data_model/bert_base_train_and_val.pkl')
    parser.add_argument("--under_test_dataset", type=str,
                        default='./data_model/bert_base_test.pkl')
    parser.add_argument("--under_model", type=str,
                        default='./models/tlp_i7_base/tlp_model_2705.pkl')
    parser.add_argument("--test_dataset", type=str,
                        default='./data_model/bert_large_test.pkl')
    parser.add_argument("--path", type=str, default="((bert_large*.task.pkl")
    parser.add_argument("--cuda", type=str, default='cpu')
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--rank_mse", type=str, default='rank')
    parser.add_argument("--optimizer", type=str, default='default')
    parser.add_argument("--attention_head", type=int, default=8)
    parser.add_argument("--attention_class", type=str, default='default')
    parser.add_argument("--step_size", type=int, default=64)
    parser.add_argument("--fea_size", type=int, default=22)
    parser.add_argument("--res_block_cnt", type=int, default=2)
    parser.add_argument("--self_sup_model", type=str, default='')
    parser.add_argument("--data_cnt", type=int, default=-1)  # data_cnt * 1000
    parser.add_argument("--seed", type=int, default=params["seed"])
    parser.add_argument("--train_size_per_gpu", type=int, default=64)
    parser.add_argument("--val_size_per_gpu", type=int, default=64)
    parser.add_argument("--n_epoch", type=int, default=50)
    args = parser.parse_args()
    args.seed = int(args.seed)
    print("seed: ", args.seed)
    set_seed(args.seed)
    return args


def train_model(args):
    # init args
    tlp_prom = Tlp_prom()
    # split data to train and test
    print("Load data and split data to train and test...")
    # set test length
    train_data, test_data = tlp_prom.data_partitioning \
        (train_dataset=args.under_train_dataset, test_dataset=args.under_test_dataset, args=args)
    under_model_name, performance = train(*train_data, device="cpu", test_tlp=test_data)
    # origin_testdata = test_data
    # under_model_name='/home/huanting/PROM/examples/case_study/tlp/scripts/tlp_i7/bert.pkl'
    # performance = eval_model(model_file=under_model_name, test_datasets=test_data)
    print("Load data and evaluate the data on new benchmark...")
    nni.report_final_result(performance)


def deploy_model(args):
    # init args
    tlp_prom = Tlp_prom()
    # split data to train and test
    print("Load data and split data to train and test...")
    train_data, test_data = tlp_prom.data_partitioning \
        (train_dataset=args.under_train_dataset, test_dataset=args.test_dataset, args=args)

    underlying_model_name = args.under_model
    # print("Evaluate the data on new benchmark...")
    # deploy_perm = eval_model(model_file=underlying_model_name, test_datasets=test_data)

    """cp"""
    print("Conformal prediction...")
    cluster_f1 = []
    cluster_pre = []
    cluster_rec = []
    cluster_all = []
    for cluster_num in tqdm(range(2,70), desc="Inner loop", leave=False):
        indices_detec,f1,pre,rec = cp(*train_data, test_datasets=test_data,
                           underlying_path=underlying_model_name, file_pattern=args.path,cluster_num=cluster_num)
        cluster_f1.append(f1)
        cluster_pre.append(pre)
        cluster_rec.append(rec)
        cluster_all.append(cluster_num)
    return cluster_f1,cluster_pre,cluster_rec,cluster_all
    """incremental learning"""
    # print("Incremental learning...")
    #
    # il_model = il(test_data, device="cpu", pre_trained_model=underlying_model_name,
    #               aug_data=indices_detec, args=args)
    # print("Evaluate the data on new benchmark...")
    # il_perm = eval_model(model_file=il_model, test_datasets=test_data)
    # improve_perm = il_perm - deploy_perm
    # print("improve_perm: ", improve_perm)
    # nni.report_final_result(improve_perm)


if __name__ == "__main__":
    print("initial parameters...")
    from tqdm import tqdm
    args = init_args()

    for i in tqdm(range(1), desc="Outer loop"):
        cluster_f1,cluster_pre,cluster_rec,cluster_all=deploy_model(args)

    data = {
        'cluster_f1': cluster_f1,
        'cluster_pre': cluster_pre,
        'cluster_rec': cluster_rec,
        'cluster_all': cluster_all
    }

    with open('cluster_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    # print("save to 'cluster_data.pkl'")

    # train_model(args)
    # nnictl create --config /home/huanting/PROM/examples/case_study/tlp/scripts/config.yaml --port 8088

    """
    --mode deploy --save_folder models/il/tlp_i7_tiny --under_model ./models/train/tlp_i7_base/tlp_model_533_best.pkl --test_data ./data_model/bert_tiny_test.pkl --path ((bert_tiny*.task.pkl
    """
