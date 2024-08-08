import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import nni
from torch import nn
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from typing import Callable, Sequence, Tuple
from compy.models.model import Model
# from simple_uq.util.mlp import MLP
import torch.nn.functional as F
# import uncertainty_toolbox as uct

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sys
sys.path.append('./case_study/DeviceM')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
import src.prom.prom_util as prom_utils
from src.prom.prom_util import Prom_utils
from prom.prom_classification import MapieClassifier
from prom.metrics import (classification_coverage_score,
                           classification_mean_width_score)

import random
class Dev_gnn(prom_utils.ModelDefinition):
    def __init__(self,model=None,dataset=None,calibration_data=None,args=None):
        # self.model = DeepTune()
        self.calibration_data = None
        self.dataset = None

    def data_partitioning(self, dataset, test_ratio=0.5,random_seed=0):
        # dict_add={"npb-3.3": {"subdir": ""}}
        keys = list(dataset.keys())
        random.seed(random_seed)
        random.shuffle(keys)  # 随机打乱键的顺序
        n_test = int(len(keys) * test_ratio)  # 计算测试集大小
        test_keys = keys[:n_test]  # 获取测试集的键
        train_keys = keys[n_test:] # 获取训练集的键
        test_dict = {key: dataset[key] for key in test_keys}  # 构建测试集字典
        train_dict = {key: dataset[key] for key in train_keys}  # 构建训练集字典
        # train_dict.update(dict_add)
        return train_dict, test_dict

    def predict(self, X, significant_level=0.1):
        if self.model is None:
            raise ValueError("Model is not initialized.")

        pred=self.model.predict(self, sequences='')
        probability=self.model.predict_proba(self, sequences='')
        return pred, probability

    def feature_extraction(self, srcs):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        code_tokens = [tokenizer.tokenize(src) for src in srcs]
        seqs = [tokenizer.convert_tokens_to_ids(src) for src in code_tokens]
        # seqs = [tokenizer.tokenize(src) for src in tokens_ids]
        # pad_val = atomizer.vocab_size
        pad_val = len(seqs)
        encoded = np.array(pad_sequences(seqs, maxlen=1024, value=pad_val))
        return np.vstack([np.expand_dims(x, axis=0) for x in encoded])

class Net(torch.nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        annotation_size = config["hidden_size_orig"]
        hidden_size = config["gnn_h_size"]
        n_steps = config["num_timesteps"]
        num_cls = 2

        self.reduce = nn.Linear(annotation_size, hidden_size)
        self.conv = GatedGraphConv(hidden_size, n_steps)
        self.agg = GlobalAttention(nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 2))
        self.uq = GlobalAttention(nn.Linear(hidden_size, 1), nn.Linear(hidden_size, 1))
        self.lin = nn.Linear(hidden_size, num_cls)
        # self.mean_head = MLP(
        #     input_dim=1,
        #     output_dim=1,
        #     hidden_sizes=[2, 2],
        #     hidden_activation=  F.relu,
        # )
        # self.logvar_head = MLP(
        #     input_dim=1,
        #     output_dim=1,
        #     hidden_sizes=[2, 2],
        #     hidden_activation=F.relu,
        # )

    # def get_mean_and_standard_deviation(
    #         self, graph,
    # ):
    #     """Get the mean and standard deviation prediction.
    #
    #     Args:
    #         x_data: The data in numpy ndarray form.
    #         device: The device to use. Should be the same as the device
    #             the model is currently on.
    #
    #     Returns:
    #         Mean and standard deviation as ndarrays
    #     """
    #
    #     with torch.no_grad():
    #         data, edge_index, batch = graph.x, graph.edge_index, graph.batch
    #         latent = self.get_data(graph)
    #         # mean, logvar = self.forward(graph)
    #         mean = self.mean_head(latent)
    #         logvar = self.logvar_head(latent)
    #     mean = mean.numpy()
    #     std = (logvar / 2).exp().numpy()
    #     return mean, std, latent

    # def get_data(
    #     self, graph,
    # ):
    #     x, edge_index, batch = graph.x, graph.edge_index, graph.batch
    #     x = self.reduce(x)
    #     x = self.conv(x, edge_index)
    #     x = self.agg(x, batch)
    #     return x

    def forward(
        self, graph,
    ):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch

        x = self.reduce(x)
        x = self.conv(x, edge_index)
        x = self.agg(x, batch)
        x = torch.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)

        return x

    def fit(
            self
    ):
        return 1

    def predict_proba(
        self, graph,
    ):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.reduce(x)
        x = self.conv(x, edge_index)
        x = self.agg(x, batch)
        x = torch.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)
        return x.detach().numpy()

    def predict(
        self, graph,
    ):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.reduce(x)
        x = self.conv(x, edge_index)
        x = self.agg(x, batch)

        x = torch.softmax(x, dim=1)
        pred = x.max(dim=1)[1]
        return pred.numpy().tolist()


class GnnPytorchGeomModel(Model):
    def __init__(self, config_input=None, num_types=None,mode='train',model_path=None,random_seed=None):
        if not config_input:
            config = {
                "num_timesteps": 4,
                "hidden_size_orig": num_types,
                "gnn_h_size": 32,
                "gnn_m_size": 2,
                "learning_rate": 0.001,
                "batch_size": 8,
                "num_epochs": 5,
            }
            # nni
            tuner_params = nni.get_next_parameter()  # 这会获得一组搜索空间中的参数
            try:
                config.update(tuner_params)
            except:
                pass
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if mode == "test":
            self.model = torch.load(model_path)
        else:
            self.model = Net(config)
        self.model = self.model.to(self.device)

    def __process_data(self, data):
        return [
            {
                "nodes": data["x"]["code_rep"].get_node_list(),
                "edges": data["x"]["code_rep"].get_edge_list(),
                "aux_in": data["x"]["aux_in"],
                "label": data["y"],
                "cpu_time": data["cpu_time"],
                "gpu_time": data["gpu_time"],
            }
            for data in data
        ]

    def __build_pg_graphs(self, batch_graphs):
        pg_graphs = []

        for batch_graph in batch_graphs:
            # Graph
            # - nodes
            one_hot = np.zeros(
                (len(batch_graph["nodes"]), self.config["hidden_size_orig"])
            )
            one_hot[np.arange(len(batch_graph["nodes"])), batch_graph["nodes"]] = 1
            x = torch.tensor(one_hot, dtype=torch.float)

            # -edges
            edge_index, edge_features = [], []
            for edge in batch_graph["edges"]:
                edge_index.append([edge[0], edge[2]])
                edge_features.append([edge[1]])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_features = torch.tensor(edge_features, dtype=torch.long)

            graph = Data(
                x=x,
                edge_index=edge_index.t().contiguous(),
                edge_features=edge_features,
                y=batch_graph["label"],
                cpu_time=batch_graph["cpu_time"],
                gpu_time=batch_graph["gpu_time"],
            )
            pg_graphs.append(graph)

        return pg_graphs

    def _train_init(self, data_train, data_valid):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        return self.__process_data(data_train), self.__process_data(data_valid)

    def _test_data_init(self, data_test):
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        return self.__process_data(data_test)

    def _train_with_batch(self, batch):
        loss_sum = 0
        correct_sum = 0

        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)

        baseline_speedup = []
        oracle_percent=[]
        ##########
        for data in loader:
            data = data.to(self.device)

            self.model.train()
            self.opt.zero_grad()

            pred = self.model(data)
            loss = F.nll_loss(pred, data.y)
            loss.backward()
            self.opt.step()

            loss_sum += loss
            correct_sum += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

            #########################
            cpu_time = data.cpu_time.view(-1)
            gpu_time = data.gpu_time.view(-1)

            pred_label = pred.max(dim=1)[1]
            origin_label = data.y.view(-1)
            # for label calculate

            # run time for now
            #1=gpu 0=cpu
            for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
                    zip(pred_label, origin_label, cpu_time, gpu_time):
                    # baseline:gpu
                    baseline_speedup.append(gpu_time_tem / cpu_time_tem)
                    #oracle:
                    if pre_label_tem==1:
                        oracle_percent.append(min(gpu_time_tem.item(),cpu_time_tem.item())/gpu_time_tem.item())
                    else:
                        oracle_percent.append(min(gpu_time_tem.item(), cpu_time_tem.item()) / cpu_time_tem.item())

            baseline_speedup = np.mean(baseline_speedup)

        train_accuracy = correct_sum / len(loader.dataset)
        train_loss = loss_sum / len(loader.dataset)

        return train_loss, train_accuracy,baseline_speedup,oracle_percent


    def _test_init(self):
        self.model.eval()

    def _model_save(self,name='best_model.pkl'):
        torch.save(self.model, name)

    def _predict_with_batch(self, batch):
        correct = 0

        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)

        baseline_speedup = []
        oracle_percent = []
        for data in loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)

            cpu_time = data.cpu_time.view(-1)
            gpu_time = data.gpu_time.view(-1)

            pred_label=pred.max(dim=1)[1]
            origin_label=data.y.view(-1)
            # for label calculate

            # run time for now
            for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
                    zip(pred_label, origin_label, cpu_time, gpu_time):
                baseline_speedup.append(gpu_time_tem / cpu_time_tem)
                # oracle:
                if pre_label_tem == 1:
                    oracle_percent.append(min(gpu_time_tem.item(), cpu_time_tem.item()) / gpu_time_tem.item())
                else:
                    oracle_percent.append(min(gpu_time_tem.item(), cpu_time_tem.item()) / cpu_time_tem.item())
            baseline_speedup = np.mean(baseline_speedup)

            correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        valid_accuracy = correct / len(loader.dataset)

        return valid_accuracy, pred, baseline_speedup,oracle_percent

    def _predict_uq_batch(self, train_batches, valid_batches,test_batches,random_seed):
        # print("start conformal prediction")
        F1_all=[]
        Pre_all=[]
        Rec_all=[]
        Acc_all=[]
        clf = self.model

        method_params = {
            # "naive": ("naive", False),
            # "score": ("score", False),
            # "cumulated_score": ("cumulated_score", True),
            # "random_cumulated_score": ("cumulated_score", "randomized"),
            # "top_k": ("top_k", False),
        "lac": ("score", True),
        "top_k": ("top_k", True),
        "aps": ("cumulated_score", True),
        # "raps": ("raps", True)
        }
        # y_preds, y_pss, y_pred_merged,p_value = {}, {}, {},{}
        #
        # import math
        # def find_alpha_range(n):
        #     alpha_min = max(1 / n, 0)
        #     alpha_max = min(1 - 1 / n, 1)
        #     return math.ceil(alpha_min * 100) / 100, math.floor(alpha_max * 100) / 100
        #
        # alpha_min, alpha_max = find_alpha_range(len(valid_batches))
        # alphas = np.arange(alpha_min, alpha_max, 0.05)
        # # alphas = [alpha_min]
        graphs_valid = self.__build_pg_graphs(valid_batches)
        loader_valid = DataLoader(graphs_valid, batch_size=999999)
        for X_cal in loader_valid:
            X_cal = X_cal.to(self.device)
        y_cal=X_cal.y.view(-1)

        graphs_test = self.__build_pg_graphs(test_batches)
        loader_test = DataLoader(graphs_test, batch_size=999999)
        for X_test in loader_test:
            X_test = X_test.to(self.device)
        y_test = X_test.y.view(-1)

        with torch.no_grad():
            pred = self.model(X_test)
        all_pre = pred.max(dim=1)[1]

        #
        # for name, (method, include_last_label) in method_params.items():
        #     mapie = MapieClassifier(
        #         estimator=clf, method=method, cv="prefit", random_state=random_seed,classes=y_test
        #     )
        #     mapie.fit(X_cal, y_cal)
        #     mapie.prom_test_ncm(X_test)
        #     y_preds[name], y_pss[name], p_value[name] = \
        #         mapie.predict(X_cal,X_test, alpha=alphas, include_last_label=include_last_label)
        #     # """ mobi """
        #
        #     # accuracies[name] = accuracy_score(y_test, y_preds[name])
        #
        #     # probability
        #     # valid_probability = clf.predict_proba(X_cal)
        #     # p-value : y_pvalue
        #     # label : y_preds_valid
        #     # y_preds_valid[name], y_pss_valid[name], y_pvalue[name] = mapie.predict(
        #     #     X_cal, alpha=alphas, include_last_label=include_last_label
        #     # )
        #
        #     # # 提取valid_probability的第一列和第二列
        #     # valid_prob_col1 = valid_probability[:, 0]
        #     # valid_prob_col2 = valid_probability[:, 1]
        #     #
        #     # # 提取y_pvalue[name]的第一列和第二列
        #     # y_pvalue_col1 = y_pvalue[name][:, :, 0][:, 0]
        #     # y_pvalue_col2 = y_pvalue[name][:, :, 0][:, 1]
        #     #
        #     # # 使用argsort()函数获取排序索引
        #     # rank_indices = np.argsort(y_pvalue_col1)
        #     # rank_indices_2 = np.argsort(y_pvalue_col2)
        #     # # 计算排名的比例
        #     # y_pvalue_col1_rank = (rank_indices + 1) / len(y_pvalue_col1)
        #     # y_pvalue_col2_rank = (rank_indices_2 + 1) / len(y_pvalue_col2)
        #     # # 将两个列拼接在一起，创建两个新的(4, 2)数组
        #     # vec_label0 = np.column_stack((valid_prob_col1, y_pvalue_col1_rank))
        #     # vec_label1 = np.column_stack((valid_prob_col2, y_pvalue_col2_rank))
        #     # y_preds_tensor = torch.tensor(y_preds_valid[name])
        #     # # 创建一个数组，其中预测错误的地方标记为1，预测正确的地方标记为0
        #     # labels = torch.where(y_cal != y_preds_tensor, torch.tensor([1]), torch.tensor([0]))
        #     #
        #     # from sklearn import svm
        #     # clf0 = svm.SVC(probability=True)
        #     # clf0.fit(vec_label0, labels)
        #     #
        #     # clf1 = svm.SVC(probability=True)
        #     # clf1.fit(vec_label1, labels)
        #     # print("_____________________________________________")
        #         #
        #     ###################
        #     # test_probability = clf.predict_proba(X_test)
        #     # y_preds[name], y_pss[name], y_pvalue[name] = mapie.predict(
        #     #     X_test, alpha=alphas, include_last_label=include_last_label
        #     # )
        #     # # 提取valid_probability的第一列和第二列
        #     # valid_prob_col1 = test_probability[:, 0]
        #     # valid_prob_col2 = test_probability[:, 1]
        #     #
        #     # # 提取y_pvalue[name]的第一列和第二列
        #     # y_pvalue_col1 = y_testp[:, :, 0][:, 0]
        #     # y_pvalue_col2 = y_testp[:, :, 0][:, 1]
        #     # # 使用argsort()函数获取排序索引
        #     # rank_indices = np.argsort(y_pvalue_col1)
        #     # rank_indices_2 = np.argsort(y_pvalue_col2)
        #     # # 计算排名的比例
        #     # y_pvalue_col1_rank = (rank_indices + 1) / len(y_pvalue_col1)
        #     # y_pvalue_col2_rank = (rank_indices_2 + 1) / len(y_pvalue_col2)
        #     # # 将两个列拼接在一起，创建两个新的(4, 2)数组
        #     # vec_label0 = np.column_stack((valid_prob_col1, y_pvalue_col1_rank))
        #     # vec_label1 = np.column_stack((valid_prob_col2, y_pvalue_col2_rank))
        #     #
        #     # y_pred_0  = clf0.predict(vec_label0)
        #     # y_pred_1  = clf1.predict(vec_label1)
        #     # y_pred_merged[name] = (y_pred_0 | y_pred_1)
        #
        # def count_null_set(y: np.ndarray) -> int:
        #     count = 0
        #     for pred in y[:, :]:
        #         if np.sum(pred) == 0:
        #             count += 1
        #     return count

        ############
        Prom_thread = Prom_utils(clf, method_params, task="thread")

        y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
            cal_x=X_cal, cal_y=y_cal, test_x=X_test, test_y=y_test, significance_level="auto")
        #################
        # MAPIE
        Prom_thread.evaluate_mapie \
            (y_preds=y_preds,
             y_pss=y_pss,
             p_value=p_value,
             all_pre=all_pre,
             y=y_test,
             significance_level=0.05)

        Prom_thread.evaluate_rise \
            (y_preds=y_preds,
             y_pss=y_pss,
             p_value=p_value,
             all_pre=all_pre,
             y=y_test,
             significance_level=0.05)

        index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all, _, _ \
            = Prom_thread.evaluate_conformal_prediction \
            (y_preds=y_preds,
             y_pss=y_pss,
             p_value=p_value,
             all_pre=all_pre,
             y=y_test,
             significance_level=0.05)
        ##############
        # nulls, coverages, accuracies, confidence_sizes,credibility_sizes = {}, {}, {}, {},{}
        # credibility_score={}
        # for name, (method, include_last_label) in method_params.items():
        #     results_array = np.zeros((len(p_value[name]), len(alphas)), dtype=bool)
        #     #step1, use p-value to select
        #     for i, alpha in enumerate(alphas):
        #         results_array[:, i] = np.array([value > (1 - alpha) for value in p_value[name]])
        #     credibility_score[name] = results_array
        #
        # for name, (method, include_last_label) in method_params.items():
        #     # accuracies[name] = accuracy_score(y_test, y_preds[name])
        #     # nulls[name] = [self.count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(self.alphas)]
        #     coverages[name] = [classification_coverage_score(y_test, y_pss[name][:, :, i]) for i, _ in
        #                        enumerate(alphas)]
        #     confidence_sizes[name] = [y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)]
        #     credibility_sizes[name] = [credibility_score[name][i, :].sum().mean() for i, _ in enumerate(alphas)]
        #
        # confidence_result = {key: min(range(len(lst)), key=lambda i: abs(lst[i] - 1)) for key, lst in
        #                      confidence_sizes.items()}
        # credibility_result = {key: min(range(len(lst)), key=lambda i: abs(lst[i] - 1)) for key, lst in
        #                       credibility_sizes.items()}
        # # Extract the y_ps at the closest index
        # confidence_result_ps = {method: y_pss[method][:, :, confidence_result[method]] for method in y_pss}
        # credibility_result_ps = {method: credibility_score[method][:, credibility_result[method]] for method in
        #                          credibility_score}
        # # Determine indices for each method
        # index_all_tem, index_all_right_tem = {}, {}
        # for (method, y_ps1), (_, y_ps2) in zip(credibility_result_ps.items(), confidence_result_ps.items()):
        #     for index, confidence_single in enumerate(y_ps2):
        #         confidence_true = sum(confidence_single)
        #         credibility_true = y_ps1[index]
        #         if method not in index_all_tem:
        #             index_all_tem[method] = []
        #             index_all_right_tem[method] = []
        #         if confidence_true == 1 and credibility_true == 1:
        #         # elif confidence_true == 1:
        #             index_all_right_tem[method].append(index)
        #         else:
        #             index_all_tem[method].append(index)
        # index_all = list(set([idx for indices in index_all_tem.values() for idx in indices]))
        # index_list = list(index_all_tem.values())
        # index_list.append(index_all)
        #
        # index_all_right = list(set(range(len(y_test))) - set(index_all))
        # index_list_right = list(index_all_right_tem.values())
        # index_list_right.append(index_all_right)

        # result = {}  # 用于存储结果的字典
        # for key, lst in sizes.items():  # 遍历字典的键值对
        #     closest_index = min(
        #         range(len(lst)), key=lambda i: abs(lst[i] - 1)
        #     )  # 找到最接近1的数字的索引
        #     # closest_index = min(
        #     #     range(len(lst)), key=lambda i: abs(lst[i] - 1) if lst[i] != 1 else float('inf')
        #     # )  # 找到最接近1且不等于1的数字的索引
        #     result[key] = closest_index  # 将结果存入字典
        # # y_ps_90中提出来那个最接近1的位置
        # result_ps = {}
        # for method, y_ps in y_pss.items():
        #     result_ps[method] = y_ps[:, :, result[method]]
        #
        # index_all_tem = {}
        # index_all_right_tem = {}
        # for method, y_ps in y_pred_merged.items():
        #     for index, i in enumerate(y_ps):
        #         num_true = i
        #         if method not in index_all_tem:
        #             index_all_tem[method] = []
        #             index_all_right_tem[method] = []
        #         if num_true != 0:
        #             index_all_tem[method].append(index)
        #         elif num_true == 0:
        #             index_all_right_tem[method].append(index)
        # index_all = []
        # index_list = []
        # # 遍历字典中的每个键值对
        # for key, value in index_all_tem.items():
        #     # 使用集合对列表中的元素进行去重，并转换为列表
        #     list_length = len(value)
        #     # print(f"Length of {key}: {list_length}")
        #     # 将去重后的列表添加到新列表中
        #     index_all.extend(value)
        #     index_list.append(value)
        # index_all = list(set(index_all))
        # # print(f"Length of index_all: {len(index_all)}")
        # index_list.append(index_all)
        #
        # index_all_right = []
        # index_list_right = []
        # # 遍历字典中的每个键值对
        # for key, value in index_all_right_tem.items():
        #     # 使用集合对列表中的元素进行去重，并转换为列表
        #     list_length = len(value)
        #     # print(f"Length of {key}: {list_length}")
        #     # 将去重后的列表添加到新列表中
        #     index_all_right.extend(value)
        #     index_list_right.append(value)
        #
        #
        # index_all_right = list(set(list(range(len(y_test)))) - set(index_all))
        # # print(f"Length of index_all: {len(index_all_right)}")
        # index_list_right.append(index_all_right)
        """ compute metircs"""
        # with torch.no_grad():
        #     pred = self.model(X_test)
        # all_pre = pred.max(dim=1)[1]
        # # 所有错误的np.where(all_pre != y_test)[0]
        # different_indices = np.where(all_pre != y_test)[0]
        # different_indices_right = np.where(all_pre == y_test)[0]
        # # 找到的错误的： index_list
        # # 找的真的错的：num_common_elements
        # acc_best = 0
        # F1_best = 0
        # pre_best = 0
        # rec_best = 0
        # method_name_best = " NONE"
        # method_name = {
        #     "lac": ("score", True),
        #     "top_k": ("top_k", True),
        #     "aps": ("cumulated_score", True),
        #     # "raps": ("raps", True)
        #     "mixture": ("mixture", False)
        # }
        # for index, (single_list, single_list_right) in enumerate(zip(index_list, index_list_right)):
        #     common_elements = np.intersect1d(single_list, different_indices)
        #     num_common_elements = len(common_elements)
        #     common_elements_right = np.intersect1d(single_list_right, different_indices_right)
        #     num_common_elements_right = len(common_elements_right)
        #     try:
        #         accuracy = (num_common_elements + num_common_elements_right) / len(all_pre)
        #     except:
        #         accuracy = 0
        #     try:
        #         precision = num_common_elements / len(single_list)
        #     except:
        #         precision = 0
        #     try:
        #         recall = num_common_elements / len(different_indices)
        #     except:
        #         recall = 0
        #     try:
        #         F1 = 2 * precision * recall / (precision + recall)
        #     except:
        #         F1 = 0
        #     print(
        #         f"{list(method_name.keys())[index]} find accuracy: {accuracy * 100:.2f}%, "
        #         f"precision: {precision * 100:.2f}%, "
        #         f"recall: {recall * 100:.2f}%, "
        #         f"F1: {F1 * 100:.2f}%"
        #     )

            # if F1 > F1_best:
            #     method_name_best = list(method_name.keys())[index]
            #     acc_best = accuracy
            #     F1_best = F1
            #     pre_best = precision
            #     rec_best = recall
        # print(
        #     f"{method_name_best} is the best approach"
        #     f"best accuracy: {accuracy * 100:.2f}%, "
        #     f"best precision: {pre_best * 100:.2f}%, "
        #     f"best recall: {rec_best * 100:.2f}%, "
        #     f"best F1: {F1_best * 100:.2f}%"
        # )
        # Acc_all.append(acc_best)
        # F1_all.append(F1_best)
        # Pre_all.append(pre_best)
        # Rec_all.append(rec_best)
        # nni.report_final_result(F1_all)
        nni.report_intermediate_result(F1_all)
        """"IL"""
        selected_count = max(int(len(y_test) * 0.05), 1)
        np.random.seed(random_seed)
        # try:
        #     random_element = np.random.choice(common_elements, selected_count, replace=False)
        # except:
        random_element = np.random.choice(range(len(y_test)), selected_count)

        sample = [test_batches[index] for index in random_element]

        train_batches = np.concatenate((train_batches, sample))
        test_batches = [item for item in test_batches if item not in sample]



        return train_batches,test_batches

    def _predict_test_batch(self, batch):
        correct = 0
        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)
        baseline_speedup = []
        GPUbaseline_speedup = 0.0
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = self.model(data)

            cpu_time = data.cpu_time.view(-1)
            gpu_time = data.gpu_time.view(-1)
            pred_label=pred.max(dim=1)[1]
            origin_label=data.y.view(-1)

            # run time for now

            for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
                    zip(pred_label, origin_label, cpu_time, gpu_time):
                if pre_label_tem == 0:
                    baseline_speedup.append(gpu_time_tem / cpu_time_tem)
                if pre_label_tem == 1:
                    baseline_speedup.append(gpu_time_tem / cpu_time_tem)
            baseline_speedup = np.mean(baseline_speedup)

            correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        valid_accuracy = correct / len(loader.dataset)
        return valid_accuracy, pred, baseline_speedup

    def _predict_val_batch(self, batch):
        correct = 0
        graphs = self.__build_pg_graphs(batch)
        loader = DataLoader(graphs, batch_size=999999)
        baseline_speedup = []

        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = self.model(data)

            cpu_time = data.cpu_time.view(-1)
            gpu_time = data.gpu_time.view(-1)

            pred_label=pred.max(dim=1)[1]
            origin_label=data.y.view(-1)


            # run time for now
            for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
                    zip(pred_label, origin_label, cpu_time, gpu_time):
                if pre_label_tem == 0:
                    baseline_speedup.append(gpu_time_tem / cpu_time_tem)
                if pre_label_tem == 1:
                    baseline_speedup.append(gpu_time_tem / cpu_time_tem)
            baseline_speedup = np.mean(baseline_speedup)

            correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
        valid_accuracy = correct / len(loader.dataset)



        return valid_accuracy, pred, baseline_speedup
