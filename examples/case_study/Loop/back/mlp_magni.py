import pandas as pd
from transformers import AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import torch
import pickle
import sys
import nni
import argparse
from mapie.classification import MapieClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import random
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from sklearn.model_selection import train_test_split
import os.path as fs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from progressbar import ProgressBar
# during grid search, not all parameters will converge. Ignore these warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
import re
filterwarnings("ignore", category=ConvergenceWarning)

class ThreadCoarseningModel(object):
    """
    A model for predicting OpenCL thread coarsening factors.

    Attributes
    ----------
    __name__ : str
        Model name
    __basename__ : str
        Shortened name, used for files
    """

    __name__ = None
    __basename__ = None

    def init(self, seed: int) -> None:
        """
        Initialize the model.

        Do whatever is required to setup a new thread coarsening model here.
        This method is called prior to training and predicting.
        This method may be omitted if no initial setup is required.

        Parameters
        ----------
        seed : int
            The seed value used to reproducible results. May be 'None',
            indicating that no seed is to be used.
        """
        pass

    def save(self, outpath: str) -> None:
        """
        Save model state.

        This must capture all of the relevant state of the model. It is up
        to implementing classes to determine how best to save the model.

        Parameters
        ----------
        outpath : str
            The path to save the model state to.
        """
        raise NotImplementedError

    def restore(self, inpath: str) -> None:
        """
        Load a trained model from file.

        This is called in place of init() if a saved model file exists. It
        must restore all of the required model state.

        Parameters
        ----------
        inpath : str
            The path to load the model from. This is the same path as
            was passed to save() to create the file.
        """
        raise NotImplementedError

    def train(
        self, cascading_features: np.array, cascading_y: np.array, verbose: bool = False
    ) -> None:
        """
        Train a model.

        Parameters
        ----------
        cascading_features : np.array
            An array of feature vectors of shape (n,7,7). Used for the cascading
            model, there are 7 vectors of 7 features for each benchmark, one for
            each coarsening factor.

        cascading_y : np.array
            An array of classification labels of shape(n,7). Used for the cascading
            model.

        sequences : np.array
            An array of encoded source code sequences of shape (n,seq_length).

        y_1hot : np.array
            An array of optimal coarsening factors of shape (n,6), in 1-hot encoding.

        verbose: bool, optional
            Whether to print verbose status messages during training.
        """
        raise NotImplementedError

    def predict(self, cascading_features: np.array, sequences: np.array) -> np.array:
        """
        Make predictions for programs.

        Parameters
        ----------
        cascading_features : np.array
            An array of feature vectors of shape (n,7,7). Used for the cascading
            model, there are 7 vectors of 7 features for each benchmark, one for
            each coarsening factor.

        sequences : np.array
            An array of encoded source code sequences of shape (n,seq_length).

        Returns
        -------
        np.array
            Predicted 'y' values (optimal thread coarsening factors) with shape (n,1).
        """
        raise NotImplementedError

    def predict_proba(self, sequences: np.array):
        raise NotImplementedError

    def predict_model(self, sequences: np.array):
        raise NotImplementedError


class Magni(ThreadCoarseningModel):
    __name__ = "Magni et al."
    __basename__ = "magni"

    def init(self, seed: int = None):
        # the neural network
        nn = MLPClassifier(random_state=seed, shuffle=True)

        # cross-validation over the training set. We train on 16 programs,
        # so with k=16 and no shuffling of the data, we're performing
        # nested leave-one-out cross-validation
        inner_cv = KFold(n_splits=2, shuffle=False)

        # hyper-parameter combinations to try
        params = {
            "max_iter": [200, 500, 1000, 2000],
            "hidden_layer_sizes": [
                (32,),
                (32, 32),
                (32, 32, 32),
                (64,),
                (64, 64),
                (64, 64, 64),
                (128,),
                (128, 128),
                (128, 128, 128),
                (256,),
                (256, 256),
                (256, 256, 256),
            ],
        }
        params = {
            "max_iter": [200],
            "hidden_layer_sizes": [
                (128,),
            ],
        }

        self.model = GridSearchCV(nn, cv=inner_cv, param_grid=params, n_jobs=1)

    def save(self, outpath):
        with open(outpath, "wb") as outfile:
            pickle.dump(self.model, outfile)

    def restore(self, inpath):
        with open(inpath, "rb") as infile:
            self.model = pickle.load(infile)

    def train(
        self, cascading_features: np.array, cascading_y: np.array, verbose: bool = False
    ) -> None:
        self.model.fit(cascading_features, cascading_y)

    def predict_proba(
        self, cascading_features: np.array, sequences: np.array
    ) -> np.ndarray:
        """
        Returns the predicted probabilities of the images in X.

        Paramters:
        X: np.ndarray of shape (n_sample, width, height, n_channels)
            Images to predict.

        Returns:
        np.ndarray of shape (n_samples, n_labels)
        """
        for i in range(len(cascading_features)):
            # predict whether to coarsen, using the program features of
            # the current coarsening level:
            a = cascading_features[i]
            should_coarsen = self.model.predict_proba([cascading_features[i]])
            # if not should_coarsen:
            #     break
        return should_coarsen

    def predict(self, sequences: np.array) -> np.array:
        # directly predict optimal thread coarsening factor from source sequences:


        p = self.model.predict(sequences)
        preds = torch.softmax(torch.tensor(p, dtype=torch.float32), dim=1)
        p = preds.detach().numpy()
        indices = [np.argmax(x) for x in p]
        return indices

    def predict_proba(self, sequences: np.array):
        """
        Returns the predicted probabilities of the images in X.

        Paramters:
        X: np.ndarray of shape (n_sample, width, height, n_channels)
            Images to predict.

        Returns:
        np.ndarray of shape (n_samples, n_labels)
        """
        # p = np.array(self.model.predict(sequences, batch_size=64, verbose=0))
        p = self.model.predict(sequences)
        preds = torch.softmax(torch.tensor(p, dtype=torch.float32), dim=1)
        return preds.detach().numpy()

    def predict_model(self, sequences: np.array):
        # directly predict optimal thread coarsening factor from source sequences:
        p = self.model.predict(sequences)
        preds = torch.softmax(torch.tensor(p, dtype=torch.float32), dim=1)
        p = preds.detach().numpy()
        indices = [np.argmax(x) for x in p]
        indice = np.array(indices)
        return indice

    # def predict_ori(self, cascading_features: np.array, sequences: np.array) -> np.array:
    #     # we only support leave-one-out cross-validation (implementation detail):
    #     assert(len(sequences) == 1)
    #
    #     # The binary cascading model:
    #     #
    #     # iteratively apply thread coarsening, using a new feature vector
    #     # every time coarsening is applied
    #     for i in range(len(cascading_features)):
    #         # predict whether to coarsen, using the program features of
    #         # the current coarsening level:
    #         should_coarsen = self.model.predict([cascading_features[i]])[0]
    #         if not should_coarsen:
    #             break
    #     p = cfs[i]
    #     return [cfs[i]]

import os


def get_c_code_from_file(file_name):
    with open(file_name, "r") as file:
        c_code = file.read()
    return c_code


def extract_file_names_from_paths(file_paths):
    file_names = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_names.append(file_name)
    return file_names



import clang.cindex


def remove_comments(text):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        if match.group(2) is not None:
            return ""
        else:
            return match.group(1)

    return regex.sub(_replacer, text)


def find_for_loops(node, file_content, depth=0):
    loops = []
    if node.kind == clang.cindex.CursorKind.FOR_STMT:
        for child in node.get_children():
            if child.kind == clang.cindex.CursorKind.COMPOUND_STMT:
                start_offset = child.extent.start.offset
                end_offset = child.extent.end.offset
                loops.append(file_content[start_offset:end_offset])
    # Recurse for children of this node
    for child in node.get_children():
        loops.extend(find_for_loops(child, file_content, depth + 1))
    return loops


def extract_loops_from_files(file_list):
    index = clang.cindex.Index.create()
    all_loops = []
    for file_path in file_list:
        tu = index.parse(file_path)
        with open(file_path, "r") as file:
            content = file.read()

        loops = find_for_loops(tu.cursor, content)
        if len(loops) > 0:
            # 使用列表推导和嵌套循环来合并成一个平铺的列表
            loops = loops[0]
            loops = remove_comments(loops)
            loops = loops.replace("\n", "")

        all_loops.append(loops)
    return all_loops


def load_data(path="/home/huanting/model/loop/deeptune/data/bruteforce_runtimes.pkl"):
    import pickle

    with open(path, "rb") as f:
        dict_all = pickle.load(f)
    file_name = []
    label = []
    data = dict_all.get("opt_factors")
    for key in data:
        value = data[key]
        file_name.append(key)
        label.append(value)
        # print("Key:", key, "Value:", value)
    # 获取 A 中所有不同的元组（不重复）
    unique_tuples = list(set(label))
    # 创建一个字典，将每个元组映射为唯一的整数编码
    tuple_to_int = {tuple_: i for i, tuple_ in enumerate(unique_tuples)}
    # 初始化一个全零矩阵，用于存储 one-hot 编码后的结果
    one_hot_matrix = np.zeros((len(label), len(unique_tuples))).astype(int)
    # 为每个元组做 one-hot 编码
    for i, tuple_ in enumerate(label):
        index = tuple_to_int[tuple_]
        one_hot_matrix[i, index] = 1
    # 将矩阵转换为列表
    label = one_hot_matrix.tolist()
    # 使用示例
    file_name_cut = extract_file_names_from_paths(file_name)
    folder_path = "/home/huanting/model/loop/deeptune/data"  # 替换为您实际的文件夹路径
    # for key, value in c_codes.items():
    file_path = []
    for i in file_name_cut:
        file_path_single = os.path.join(folder_path, i)
        file_path.append(file_path_single)
    #time
    time = dict_all.get("all_program_runtimes")
    time_all=[]
    for key in time:
        value = time[key]
        time_all.append(value)
    data_dict = {}  # 创建一个空字典

    all_loops = extract_loops_from_files(file_path)
    filtered_data = [(a, b, c, d) for a, b, c,d in zip(all_loops, label, time_all,file_path) if a and b and c and d]
    all_loops, label, time_all,file_path = zip(*filtered_data)
    # data_dict = {"feature": all_loops, "label": label}
    for i, file_name in enumerate(file_path):
        data_dict[file_name] = {
            'time': time_all[i],
            'feature': all_loops[i],
            'label': label[i]
        }
    with open("data_dict.pkl", "wb") as f:
        pickle.dump(data_dict, f)

    # print("a")


def evaluate(model, args):
    progressbar = [0, ProgressBar(max_value=4)]

    X_seq = None  # defer sequence encoding (it's expensive)
    F1_all = []
    Pre_all = []
    Rec_all = []
    Acc_all = []
    speed_up_all = []
    improved_spp_all = []
    # load data
    X_seq, y_1hot,time = get_feature(path="data_dict.pkl")
    numbers = np.arange(1, len(y_1hot))
    # for j, (train_index, test_index) in enumerate(kf.split(y)):
    seed = int(args.seed)
    train_index, temp_set = train_test_split(numbers, train_size=0.6, random_state=seed)
    valid_index, test_index = train_test_split(
        temp_set, train_size=0.5, random_state=seed
    )
    train_index=train_index
    valid_index=valid_index
    test_index = test_index
    model_name = model.__name__
    model_basename = model.__basename__
    model_path = f"./models/{model_basename}-{1}.model"
    predictions_path = (
        f"./predictions/{model_basename}-{1}.result"
    )
    # cache the model
    model.init(seed=seed)
    model.train(
        cascading_features=X_seq[train_index],
        cascading_y=y_1hot[train_index],
        verbose=True,
    )


    # cache the model
    try:
        os.mkdir(fs.dirname(model_path))
    except:
        pass
    model.save(model_path)

    # make prediction
    all_pre = model.predict(sequences=X_seq[test_index])
    """speed up"""
    p_speedup_all = []
    non_speedup_all = []
    # oracle prediction
    for i, (o, p) in enumerate(zip(y_1hot[test_index], all_pre)):
        # get runtime without thread coarsening
        time_single=[]
        a=time[test_index]
        for row in time[test_index][i]:
            for num in row:
                time_single.append(num)
        non_runtime=time_single[0]
        # get runtime of prediction
        p_runtime = time_single[p]
        # get runtime of oracle coarsening factor
        o_runtime = min(time_single)
        # speedup and % oracle
        non_speedup = non_runtime / p_runtime
        speedup_origin = non_runtime / o_runtime
        speedup_prediction = non_runtime / p_runtime
        percent = speedup_prediction / speedup_origin
        non_speedup_all.append(percent)
    origin_speedup = sum(non_speedup_all) / len(non_speedup_all)
    print("origin_speedup is", origin_speedup)

    #########plot
    # plt.boxplot(non_speedup_all)
    import seaborn as sns
    import pandas as pd
    data_df = pd.DataFrame({'Data': non_speedup_all})

    sns.violinplot(data=data_df, y='Data')
    seed_save = str(int(seed))
    import matplotlib.pyplot as plt
    plt.title('violin Plot Example ' + seed_save)
    plt.ylabel('Values')
    plt.savefig('/home/huanting/model/loop/deeptune/figures/' + 'box_plot_' +
                str(origin_speedup) + '_' + str(seed_save) + '.png')
    data_df.to_pickle('/home/huanting/model/loop/deeptune/figures/data/' +
                      str(origin_speedup) + '_' + str(seed_save) + '_data.pkl')
    # plt.show()
    ##########
    nni.report_final_result(origin_speedup)
    return origin_speedup
    # """SpeedUp"""
    #
    #
    # # p_model = model.predict_model(cascading_features=X_cc[test_index[0]], sequences=X_seq[test_index])[0]
    # # pred = model.predict_proba(cascading_features=X_cc[test_index[0]], sequences=X_seq[test_index])[0]
    # acc = accuracy_score(np.argmax(y_1hot[test_index], axis=1), all_pre)
    # print(" acc is", acc)
    # """ conformal prediction"""
    # print("start conformal prediction")
    # clf = model
    # method_params = {
    #     "naive": ("naive", False),
    #     "score": ("score", False),
    #     "cumulated_score": ("cumulated_score", True),
    #     "random_cumulated_score": ("cumulated_score", "randomized"),
    #     "top_k": ("top_k", False),
    # }
    # y_preds, y_pss = {}, {}
    #
    # def find_alpha_range(n):
    #     alpha_min = max(1 / n, 0)
    #     alpha_max = min(1 - 1 / n, 1)
    #     return alpha_min, alpha_max
    #
    # alpha_min, alpha_max = find_alpha_range(len(y_1hot[valid_index]))
    # alphas = np.arange(alpha_min, alpha_max, 0.1)
    # # alphas=[0.1]
    #
    # X_cal = X_seq[valid_index]
    # one_hot_matrix = y_1hot[valid_index]
    # y_cal = np.argmax(one_hot_matrix, axis=1)
    # # y_tr = np.argmax(y_1hot[train_index], axis=1)
    # y_test = np.argmax(y_1hot[test_index], axis=1)
    # X_test = X_seq[test_index]
    # for name, (method, include_last_label) in method_params.items():
    #     mapie = MapieClassifier(
    #         estimator=clf, method=method, cv="prefit", random_state=42
    #     )
    #     mapie.fit(X_cal, y_cal)
    #     y_preds[name], y_pss[name] = mapie.predict(
    #         X_test, alpha=alphas, include_last_label=include_last_label
    #     )
    #
    # def count_null_set(y: np.ndarray) -> int:
    #     count = 0
    #     for pred in y[:, :]:
    #         if np.sum(pred) == 0:
    #             count += 1
    #     return count
    #
    # nulls, coverages, accuracies, sizes = {}, {}, {}, {}
    #
    #
    # for name, (method, include_last_label) in method_params.items():
    #     accuracies[name] = accuracy_score(y_test, y_preds[name])
    #     nulls[name] = [
    #         count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(alphas)
    #     ]
    #     coverages[name] = [
    #         classification_coverage_score(y_test, y_pss[name][:, :, i])
    #         for i, _ in enumerate(alphas)
    #     ]
    #     sizes[name] = [
    #         y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
    #     ]
    # # sizes里每个method最接近1的
    # result = {}  # 用于存储结果的字典
    # for key, lst in sizes.items():  # 遍历字典的键值对
    #     closest_index = min(
    #         range(len(lst)), key=lambda i: abs(lst[i] - 1)
    #     )  # 找到最接近1的数字的索引
    #     result[key] = closest_index  # 将结果存入字典
    # # y_ps_90中提出来那个最接近1的位置
    # result_ps = {}
    # for method, y_ps in y_pss.items():
    #     result_ps[method] = y_ps[:, :, result[method]]
    #
    # index_all_tem = {}
    # index_all_right_tem = {}
    # for method, y_ps in result_ps.items():
    #     for index, i in enumerate(y_ps):
    #         num_true = sum(i)
    #         if method not in index_all_tem:
    #             index_all_tem[method] = []
    #             index_all_right_tem[method] = []
    #         if num_true != 1:
    #             index_all_tem[method].append(index)
    #         elif num_true == 1:
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
    # y=np.argmax(y_1hot, axis=1)
    # index_all_right = list(set(list(range(len(y[test_index])))) - set(index_all))
    # # print(f"Length of index_all: {len(index_all_right)}")
    # index_list_right.append(index_all_right)
    # """ compute metircs"""
    # # 所有错误的
    # different_indices = np.where(all_pre != y[test_index])[0]
    # different_indices_right = np.where(all_pre == y[test_index])[0]
    # # 找到的错误的： index_list
    # # 找的真的错的：num_common_elements
    # acc_best = 0
    # F1_best = 0
    # pre_best = 0
    # rec_best = 0
    # method_name_best = " NONE"
    # method_name = {
    #     "naive": ("naive", False),
    #     "score": ("score", False),
    #     "cumulated_score": ("cumulated_score", True),
    #     "random_cumulated_score": ("cumulated_score", "randomized"),
    #     "top_k": ("top_k", False),
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
    #
    #     if F1 > F1_best:
    #         method_name_best = list(method_name.keys())[index]
    #         acc_best = accuracy
    #         F1_best = F1
    #         pre_best = precision
    #         rec_best = recall
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
    # """""speed up"""
    # origin_accuracy = accuracy_score(y_test, y_preds[name])
    #
    # #
    # selected_count = max(int(len(y_test) * 0.05), 1)
    # np.random.seed(seed)
    # try:
    #     random_element = np.random.choice(common_elements, selected_count, replace=False)
    # except:
    #     random_element = np.random.choice(range(len(test_index)), selected_count)
    # sample = [test_index[index] for index in random_element]
    #
    # train_index = np.concatenate((train_index, sample))
    # test_index = [item for item in test_index if item not in sample]
    #
    # # create a new model and train it
    # model.init(seed=seed)
    # model.train(
    #     cascading_features=X_seq[train_index],
    #     cascading_y=y_1hot[train_index],
    #     verbose=True,
    # )
    #
    #
    # # make prediction
    # all_pre = model.predict(sequences=X_seq[test_index])
    # """speed up"""
    # p_speedup_all = []
    # non_speedup_all = []
    # # oracle prediction
    # for i, (o, p) in enumerate(zip(y_1hot[test_index], all_pre)):
    #     # get runtime without thread coarsening
    #     time_single = []
    #     a = time[test_index]
    #     for row in time[test_index][i]:
    #         for num in row:
    #             time_single.append(num)
    #     non_runtime = time_single[0]
    #     # get runtime of prediction
    #     p_runtime = time_single[p]
    #     # get runtime of oracle coarsening factor
    #     o_runtime = time_single[np.argmax(o)]
    #     # speedup and % oracle
    #     non_speedup = non_runtime / p_runtime
    #     # p_oracle = o_runtime / p_runtime
    #     non_speedup_all.append(non_speedup)
    # il_speedup = sum(non_speedup_all) / len(non_speedup_all)
    #
    # improved_sp = il_speedup - origin_speedup
    # # print("origin_speedup",origin_speedup,"il_speedup : ", il_speedup,"improved_speedup : ", improved_sp)
    # """ compute metircs"""
    #
    # progressbar[0] += 1  # update progress bar
    # progressbar[1].update(progressbar[0])
    #
    # mean_acc = sum(Acc_all) / len(Acc_all)
    # mean_f1 = sum(F1_all) / len(F1_all)
    # mean_pre = sum(Pre_all) / len(Pre_all)
    # mean_rec = sum(Rec_all) / len(Rec_all)
    # # mean_improved = sum(improve_accuracy_all) / len(improve_accuracy_all)
    # # mean_il = sum(IL_accuracy_all) / len(improve_accuracy_all)
    # # mean_speed_up = sum(speed_up_all) / len(speed_up_all)
    # # meanimproved_speed_up = sum(improved_spp_all) / len(improved_spp_all)
    # print(
    #     f"mean accuracy: {mean_acc * 100:.2f}%, "
    #     f"mean precision: {mean_pre * 100:.2f}%, "
    #     f"mean recall: {mean_rec * 100:.2f}%, "
    #     f"mean F1: {mean_f1 * 100:.2f}%, "
    # )
    # print(
    #     f"origin speed up: {origin_speedup}, "
    #     f"Imroved speed up: {il_speedup}, "
    #     f"Imroved mean speed up: {improved_sp}, "
    # )
    # nni.report_final_result(improved_sp)
    # return improved_sp

def get_feature(path="data_dict.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # feature = list(data["feature"])
    # label = np.array(list(data["label"]))

    time = []
    feature = []
    label = []

    for file_info in data.values():
        time.append(file_info['time'])
        feature.append(file_info['feature'])
        label.append(file_info['label'])

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    code_tokens = [tokenizer.tokenize(src) for src in feature]

    seqs = [tokenizer.convert_tokens_to_ids(src) for src in code_tokens]
    pad_val = len(seqs)
    encoded = np.array(pad_sequences(seqs, maxlen=100, value=pad_val))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded]), np.array(label), np.array(time)



def main():
    # load_data()
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "epoch": 1,
            "batch_size": 32,
            "seed": 123456,
        }
    print("________________")
    print(params)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=params["seed"],
        help="random seed for initialization",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=params["epoch"],
        help="random seed for initialization",
    )
    parser.add_argument(
        "--batch_size",
        default=params["batch_size"],
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    args = parser.parse_args()


    magni = evaluate(Magni(), args)
    # deeptune.groupby('Platform')['Platform', 'Speedup', 'Oracle'].mean()


main()
