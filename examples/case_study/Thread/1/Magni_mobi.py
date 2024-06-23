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

from mapie.metrics import (classification_coverage_score,
                           classification_mean_width_score)
from sklearn.model_selection import train_test_split
import os.path as fs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
# during grid search, not all parameters will converge. Ignore these warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
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

    def train(self, cascading_features: np.array, cascading_y: np.array,
               verbose: bool=False) -> None:
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
    def predict_proba(self,  sequences: np.array):

        raise NotImplementedError

    def predict_model(self,  sequences: np.array) :
        raise NotImplementedError
import tensorflow as tf
class Magni(ThreadCoarseningModel):
    __name__ = "Magni et al."
    __basename__ = "magni"

    def init(self, seed=1):
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
            ]
        }
        params = {
            "max_iter": [200],
            "hidden_layer_sizes": [
                (128,),
            ]
        }

        self.model = GridSearchCV(nn, cv=inner_cv, param_grid=params, n_jobs=1)

    def save(self, outpath):
        with open(outpath, 'wb') as outfile:
            pickle.dump(self.model, outfile)

    def restore(self, inpath):
        with open(inpath, 'rb') as infile:
            self.model = pickle.load(infile)

    def train(self, cascading_features: np.array, cascading_y: np.array,
              verbose: bool=False) -> None:
        self.model.fit(cascading_features, cascading_y)

    def predict_proba(self,cascading_features: np.array, sequences: np.array) -> np.ndarray:
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
            a=cascading_features[i]
            should_coarsen = self.model.predict_proba([cascading_features[i]])
            # if not should_coarsen:
            #     break
        return should_coarsen

    def predict(self,  sequences: np.array) -> np.array:
        # directly predict optimal thread coarsening factor from source sequences:
        cfs = [1, 2, 4, 8, 16, 32]
        p = self.model.predict(sequences)
        preds=torch.softmax(torch.tensor(p, dtype=torch.float32),dim=1)
        p=preds.detach().numpy()
        indices = [np.argmax(x) for x in p]
        return [cfs[x] for x in indices]

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
        preds=torch.softmax(torch.tensor(p, dtype=torch.float32),dim=1)
        return preds.detach().numpy()

    def predict_model(self, sequences: np.array):
        # directly predict optimal thread coarsening factor from source sequences:
        p = self.model.predict(sequences)
        preds = torch.softmax(torch.tensor(p, dtype=torch.float32),dim=1)
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
import random
def evaluate(model,args):
    # report progress:
    from progressbar import ProgressBar
    progressbar = [0, ProgressBar(max_value=4)]
    pd.set_option('display.max_rows', 5)
    df = pd.read_csv("data/case-study-b/pact-2014-runtimes.csv")
    oracles = pd.read_csv("data/case-study-b/pact-2014-oracles.csv")
    cfs = [1, 2, 4, 8, 16, 32]  # thread coarsening factors
    data = []

    X_seq = None  # defer sequence encoding (it's expensive)
    Acc_all = []
    F1_all = []
    Pre_all = []
    Rec_all = []
    for i, platform in enumerate(["Kepler","Fermi" ,"Cypress", "Tahiti" ]):
        platform_name = platform2str(platform)

        # load data
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        # y_1hot = get_onehot(oracles, platform)
        X_cc, y_cc = get_magni_features(df, oracles, platform)

        # LOOCV
        # kf = KFold(n_splits=len(y), shuffle=False)
        # seed = int(args.seed)
        seed = int(args.seed)
        numbers = np.arange(1, len(y))

        device_tasks = {
            "AMD SDK": [
                "binarySearch", "convolution", "dwtHaar1D", "fastWalsh",
                "floydWarshall", "mt", "mtLocal", "nbody", "reduce", "sobel"
            ],
            "Nvidia SDK": [
                "blackscholes", "mvCoal", "mvUncoal"
            ],
            "Parboil": [
                "mriQ", "sgemm", "spmv", "stencil"
            ]
        }
        # 将设备的键转换为一个无序的列表
        random.seed(seed)
        devices_list = list(device_tasks.keys())
        # 随机打乱列表
        random.shuffle(devices_list)
        # 创建一个新的无序字典
        shuffled_device_tasks = {device: device_tasks[device] for device in devices_list}
        device_indices = {}
        for device, tasks in shuffled_device_tasks.items():
            device_indices[device] = []
            for i in tasks:
                device_index = oracles["kernel"][oracles["kernel"] == i].index[0]
                if device_index:
                    device_indices[device].append(device_index)

        lists = []
        # 遍历字典，将每个键的值单独添加到列表中
        for values in device_indices.values():
            lists.append(values)
        train_index = lists[0]
        valid_index = lists[1]
        test_index = lists[2]


        # train_index, temp_set = train_test_split(numbers, train_size=0.4, random_state=seed)
        # valid_index, test_index = train_test_split(temp_set, train_size=0.6, random_state=seed)
        # for j, (train_index, test_index) in enumerate(kf.split(y)):
        kernel = sorted(set(df["kernel"]))[test_index[0]]

        model_name = model.__name__
        model_basename = model.__basename__

        model_path = f"models/magni/{platform}-{seed}.model"
        predictions_path = f"data/case-study-b/predictions/{model_basename}-{platform}-{1}.result"


        def align_lists(lists):

            max_length = max(len(lst) for lst in lists)  # 获取最长的array长度
            aligned_lists = []
            for lst in lists:
                a = len(lst)
                aligned_array = lst.tolist() + [0] * (max_length - len(lst))  # 补0对齐每个array
                aligned_lists.append(np.array(aligned_array))
            return aligned_lists
        flattened_array_method2 = [array.flatten() for array in X_cc]
        aligned_lists = align_lists(flattened_array_method2)
        X_feature=np.array(aligned_lists)
        y_1hot = get_onehot(oracles, platform)
        # b=y_1hot[train_index]
        model.init(seed=seed)
        model.train(cascading_features=X_feature[train_index],
                    cascading_y=y_1hot[train_index],
                    verbose=True)
        # model.train(cascading_features=np.concatenate(X_cc[train_index]),
        #             cascading_y=y_1hot[train_index],
        #             verbose=True)

        # cache the model
        try:
            os.mkdir(fs.dirname(model_path))
        except:
            pass
        model.save(model_path)


        # make prediction
        all_pre = []
        for i in range(len(X_feature[test_index])):
            p = model.predict(sequences=X_feature[test_index])[i]
            p = min(p, 2 ** (len(X_feature[test_index[0]]) - 1))
            all_pre.append(p)

        acc=accuracy_score(np.argmax(y_1hot[test_index], axis=1), all_pre)
        print(platform," acc is",acc)
        """ conformal prediction"""
        clf = model
        method_params = {
            "naive": ("naive", False),
            "score": ("score", False),
            "cumulated_score": ("cumulated_score", True),
            "random_cumulated_score": ("cumulated_score", "randomized"),
            "top_k": ("top_k", False)
        }
        y_preds, y_pss = {}, {}

        def find_alpha_range(n):
            alpha_min = max(1 / n, 0)
            alpha_max = min(1 - 1 / n, 1)
            return alpha_min, alpha_max

        alpha_min, alpha_max = find_alpha_range(len(y_1hot[valid_index]))
        alphas = np.arange(alpha_min, alpha_max, 0.1)
        # alphas=[0.1]

        X_cal = X_feature[valid_index]
        one_hot_matrix = y_1hot[valid_index]
        y_cal = np.argmax(one_hot_matrix, axis=1)
        y_tr = np.argmax(y_1hot[train_index], axis=1)
        y_test = np.argmax(y_1hot[test_index], axis=1)
        X_test = X_feature[test_index]

        for name, (method, include_last_label) in method_params.items():
            mapie = MapieClassifier(estimator=clf, method=method, cv="prefit", random_state=seed)
            mapie.fit(X_cal, y_cal)
            y_preds[name], y_pss[name] = mapie.predict(X_test, alpha=alphas, include_last_label=include_last_label)
        def count_null_set(y: np.ndarray) -> int:
            """
            Count the number of empty prediction sets.

            Parameters
            ----------
            y: np.ndarray of shape (n_sample, )

            Returns
            -------
            int
            """
            count = 0
            for pred in y[:, :]:
                if np.sum(pred) == 0:
                    count += 1
            return count

        nulls, coverages, accuracies, sizes = {}, {}, {}, {}
        for name, (method, include_last_label) in method_params.items():
            accuracies[name] = accuracy_score(y_test, y_preds[name])
            nulls[name] = [
                count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(alphas)
            ]
            coverages[name] = [
                classification_coverage_score(
                    y_test, y_pss[name][:, :, i]
                ) for i, _ in enumerate(alphas)
            ]
            sizes[name] = [
                y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
            ]

        # sizes里每个method最接近1的
        result = {}  # 用于存储结果的字典
        for key, lst in sizes.items():  # 遍历字典的键值对
            closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - 1))  # 找到最接近1的数字的索引
            result[key] = closest_index  # 将结果存入字典
        # y_ps_90中提出来那个最接近1的位置
        result_ps = {}
        for method, y_ps in y_pss.items():
            result_ps[method] = y_ps[:, :, result[method]]

        index_all_tem = {}
        index_all_right_tem = {}
        for method, y_ps in result_ps.items():
            for index, i in enumerate(y_ps):
                num_true = sum(i)
                if method not in index_all_tem:
                    index_all_tem[method] = []
                    index_all_right_tem[method] = []
                if num_true != 1:
                    index_all_tem[method].append(index)
                elif num_true == 1:
                    index_all_right_tem[method].append(index)
        index_all = []
        index_list = []
        # 遍历字典中的每个键值对
        for key, value in index_all_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all.extend(value)
            index_list.append(value)
        index_all = list(set(index_all))
        # print(f"Length of index_all: {len(index_all)}")
        index_list.append(index_all)

        index_all_right = []
        index_list_right = []
        # 遍历字典中的每个键值对
        for key, value in index_all_right_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all_right.extend(value)
            index_list_right.append(value)
        index_all_right = list(set(list(range(len(y[test_index])))) - set(index_all))
        # print(f"Length of index_all: {len(index_all_right)}")
        index_list_right.append(index_all_right)
        """ compute metircs"""
        # o = y[test_index]
        # correct = p == o
        # 所有错误的
        different_indices = np.where(all_pre != y[test_index])[0]
        different_indices_right = np.where(all_pre == y[test_index])[0]
        # 找到的错误的： index_list
        # 找的真的错的：num_common_elements
        acc_best = 0
        F1_best = 0
        pre_best = 0
        rec_best = 0
        method_name_best = " NONE"
        method_name = {
            "naive": ("naive", False),
            "score": ("score", False),
            "cumulated_score": ("cumulated_score", True),
            "random_cumulated_score": ("cumulated_score", "randomized"),
            "top_k": ("top_k", False),
            "mixture": ("mixture", False)
        }
        for index, (single_list, single_list_right) in enumerate(zip(index_list, index_list_right)):
            common_elements = np.intersect1d(single_list, different_indices)
            num_common_elements = len(common_elements)
            common_elements_right = np.intersect1d(single_list_right, different_indices_right)
            num_common_elements_right = len(common_elements_right)
            try:
                accuracy = (num_common_elements + num_common_elements_right) / len(all_pre)
            except:
                accuracy = 0
            try:
                precision = num_common_elements / len(single_list)
            except:
                precision = 0
            try:
                recall = num_common_elements / len(different_indices)
            except:
                recall = 0
            try:
                F1 = 2 * precision * recall / (precision + recall)
            except:
                F1 = 0
            print(
                f"{list(method_name.keys())[index]} find accuracy: {accuracy * 100:.2f}%, "
                f"precision: {precision * 100:.2f}%, "
                f"recall: {recall * 100:.2f}%, "
                f"F1: {F1 * 100:.2f}%"
            )

            if F1 > F1_best:
                method_name_best = list(method_name.keys())[index]
                acc_best = accuracy
                F1_best = F1
                pre_best = precision
                rec_best = recall
        print(
            f"{method_name_best} is the best approach"
            f"best accuracy: {accuracy * 100:.2f}%, "
            f"best precision: {pre_best * 100:.2f}%, "
            f"best recall: {rec_best * 100:.2f}%, "
            f"best F1: {F1_best * 100:.2f}%"
        )
        Acc_all.append(acc_best)
        F1_all.append(F1_best)
        Pre_all.append(pre_best)
        Rec_all.append(rec_best)
        # precision=找到的真的错误的/找到的错误的
        # recall=找到的真的错误的/所有错误的
        """ compute metircs"""

        progressbar[0] += 1  # update progress bar
        progressbar[1].update(progressbar[0])

    mean_acc = sum(Acc_all) / len(Acc_all)
    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    print(
        f"4 device mean accuracy: {mean_acc * 100:.2f}%, "
        f"mean precision: {mean_pre * 100:.2f}%, "
        f"mean recall: {mean_rec * 100:.2f}%, "
        f"mean F1: {mean_f1 * 100:.2f}%"
    )
    nni.report_final_result(mean_f1)
    return mean_f1


def IL(model, args):
    # report progress:
    from progressbar import ProgressBar
    progressbar = [0, ProgressBar(max_value=4)]
    pd.set_option('display.max_rows', 5)
    df = pd.read_csv("data/case-study-b/pact-2014-runtimes.csv")
    oracles = pd.read_csv("data/case-study-b/pact-2014-oracles.csv")
    cfs = [1, 2, 4, 8, 16, 32]  # thread coarsening factors
    data = []

    X_seq = None  # defer sequence encoding (it's expensive)
    Acc_all = []
    F1_all = []
    Pre_all = []
    Rec_all = []
    improve_accuracy_all = []
    IL_accuracy_all = []
    speed_up_all=[]
    improved_spp_all=[]
    # ["Kepler", "Fermi", "Cypress", "Tahiti"]
    for i, platform in enumerate(["Kepler", "Fermi", "Cypress", "Tahiti"]):
        platform_name = platform2str(platform)

        # load data
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        # y_1hot = get_onehot(oracles, platform)
        X_cc, y_cc = get_magni_features(df, oracles, platform)

        # LOOCV
        # kf = KFold(n_splits=len(y), shuffle=False)
        # seed = int(args.seed)
        seed = int(args.seed)
        numbers = np.arange(1, len(y))

        device_tasks = {
            "AMD SDK": [
                "binarySearch", "convolution", "dwtHaar1D", "fastWalsh",
                "floydWarshall", "mt", "mtLocal", "nbody", "reduce", "sobel"
            ],
            "Nvidia SDK": [
                "blackscholes", "mvCoal", "mvUncoal"
            ],
            "Parboil": [
                "mriQ", "sgemm", "spmv", "stencil"
            ]
        }
        # 将设备的键转换为一个无序的列表
        random.seed(seed)
        devices_list = list(device_tasks.keys())
        # 随机打乱列表
        random.shuffle(devices_list)
        # 创建一个新的无序字典
        shuffled_device_tasks = {device: device_tasks[device] for device in devices_list}
        device_indices = {}
        for device, tasks in shuffled_device_tasks.items():
            device_indices[device] = []
            for i in tasks:

                device_index = oracles["kernel"][oracles["kernel"] == i].index[0]

                if device_index:
                    device_indices[device].append(device_index)
        print("platform is: ",platform,"device list is ",devices_list)
        lists = []
        # 遍历字典，将每个键的值单独添加到列表中
        for values in device_indices.values():
            lists.append(values)
        train_index = lists[0]
        valid_index = lists[1]
        test_index = lists[2]

        # train_index, temp_set = train_test_split(numbers, train_size=0.4, random_state=seed)
        # valid_index, test_index = train_test_split(temp_set, train_size=0.6, random_state=seed)
        # for j, (train_index, test_index) in enumerate(kf.split(y)):

        model_name = model.__name__
        model_basename = model.__basename__

        model_path = f"models/magni/{platform}-{seed}.model"
        predictions_path = f"data/case-study-b/predictions/{model_basename}-{platform}-{1}.result"

        def align_lists(lists):

            max_length = max(len(lst) for lst in lists)  # 获取最长的array长度
            aligned_lists = []
            for lst in lists:
                a = len(lst)
                aligned_array = lst.tolist() + [0] * (max_length - len(lst))  # 补0对齐每个array
                aligned_lists.append(np.array(aligned_array))
            return aligned_lists

        flattened_array_method2 = [array.flatten() for array in X_cc]
        aligned_lists = align_lists(flattened_array_method2)
        X_feature = np.array(aligned_lists)
        y_1hot = get_onehot(oracles, platform)
        # b=y_1hot[train_index]
        model.init(seed=seed)
        model.train(cascading_features=X_feature[train_index],
                    cascading_y=y_1hot[train_index],
                    verbose=True)

        all_pre = []
        for i in range(len(X_feature[train_index])):
            p = model.predict(sequences=X_feature[train_index])[i]
            p = min(p, 2 ** (len(X_feature[train_index[0]]) - 1))
            all_pre.append(p)

        acc = accuracy_score(np.argmax(y_1hot[train_index], axis=1), all_pre)
        print(platform, "Train acc is", acc)
        # model.train(cascading_features=np.concatenate(X_cc[train_index]),
        #             cascading_y=y_1hot[train_index],
        #             verbose=True)

        # cache the model
        try:
            os.mkdir(fs.dirname(model_path))
        except:
            pass
        model.save(model_path)

        # make prediction
        all_pre = []
        for i in range(len(X_feature[test_index])):
            p = model.predict(sequences=X_feature[test_index])[i]
            p = min(p, 2 ** (len(X_feature[test_index[0]]) - 1))
            all_pre.append(p)
        """speed up"""
        s_oracle_all = []
        p_speedup_all = []
        p_oracle_all = []
        # oracle prediction
        for i,(o,p) in enumerate(zip(y[test_index],all_pre)):
            # o = y[test_index]
            # correct = p == o
            # get runtime without thread coarsening
            kernel=oracles["kernel"][test_index[i]]
            row = df[(df["kernel"] == kernel) & (df["cf"] == 1)]
            assert (len(row) == 1)  # sanity check
            nocf_runtime = float(row["runtime_" + platform])

            # get runtime of prediction
            row = df[(df["kernel"] == kernel) & (df["cf"] == p)]
            if (len(row) != 1):
                row = df[(df["kernel"] == kernel) & (df["cf"] == 1)]# sanity check
            p_runtime = float(row["runtime_" + platform])

            # get runtime of oracle coarsening factor
            o_runtime = oracle_runtimes[test_index[i]]

            # speedup and % oracle
            s_oracle = nocf_runtime / o_runtime
            p_speedup = nocf_runtime / p_runtime
            p_oracle = o_runtime / p_runtime
            s_oracle_all.append(s_oracle)
            p_speedup_all.append(p_speedup)
            p_oracle_all.append(p_oracle)
        origin_speedup=sum(p_speedup_all) / len(p_speedup_all)

        """"""

        acc = accuracy_score(np.argmax(y_1hot[test_index], axis=1), all_pre)

        print("Test acc is", acc)
        """ conformal prediction"""
        clf = model
        method_params = {
            "naive": ("naive", False),
            # "score": ("score", False),
            # "cumulated_score": ("cumulated_score", True),
            # "random_cumulated_score": ("cumulated_score", "randomized"),
            # "top_k": ("top_k", False)
        }
        y_preds, y_pss = {}, {}

        def find_alpha_range(n):
            alpha_min = max(1 / n, 0)
            alpha_max = min(1 - 1 / n, 1)
            return alpha_min, alpha_max

        alpha_min, alpha_max = find_alpha_range(len(y_1hot[valid_index]))
        alphas = np.arange(alpha_min, alpha_max, 0.1)
        alphas=[alpha_min]

        X_cal = X_feature[valid_index]
        one_hot_matrix = y_1hot[valid_index]
        y_cal = np.argmax(one_hot_matrix, axis=1)
        y_tr = np.argmax(y_1hot[train_index], axis=1)
        y_test = np.argmax(y_1hot[test_index], axis=1)
        X_test = X_feature[test_index]

        for name, (method, include_last_label) in method_params.items():
            mapie = MapieClassifier(estimator=clf, method=method, cv="prefit", random_state=42)
            mapie.fit(X_cal, y_cal)
            y_preds[name], y_pss[name] = mapie.predict(X_test, alpha=alphas, include_last_label=include_last_label)

        def count_null_set(y: np.ndarray) -> int:
            """
            Count the number of empty prediction sets.

            Parameters
            ----------
            y: np.ndarray of shape (n_sample, )

            Returns
            -------
            int
            """
            count = 0
            for pred in y[:, :]:
                if np.sum(pred) == 0:
                    count += 1
            return count

        nulls, coverages, accuracies, sizes = {}, {}, {}, {}
        for name, (method, include_last_label) in method_params.items():
            accuracies[name] = accuracy_score(y_test, y_preds[name])
            nulls[name] = [
                count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(alphas)
            ]
            coverages[name] = [
                classification_coverage_score(
                    y_test, y_pss[name][:, :, i]
                ) for i, _ in enumerate(alphas)
            ]
            sizes[name] = [
                y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
            ]

        # sizes里每个method最接近1的
        result = {}  # 用于存储结果的字典
        for key, lst in sizes.items():  # 遍历字典的键值对
            closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - 1))  # 找到最接近1的数字的索引
            result[key] = closest_index  # 将结果存入字典
        # y_ps_90中提出来那个最接近1的位置
        result_ps = {}
        for method, y_ps in y_pss.items():
            result_ps[method] = y_ps[:, :, result[method]]

        index_all_tem = {}
        index_all_right_tem = {}
        for method, y_ps in result_ps.items():
            for index, i in enumerate(y_ps):
                num_true = sum(i)
                if method not in index_all_tem:
                    index_all_tem[method] = []
                    index_all_right_tem[method] = []
                if num_true != 1:
                    index_all_tem[method].append(index)
                elif num_true == 1:
                    index_all_right_tem[method].append(index)
        index_all = []
        index_list = []
        # 遍历字典中的每个键值对
        for key, value in index_all_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all.extend(value)
            index_list.append(value)
        index_all = list(set(index_all))
        # print(f"Length of index_all: {len(index_all)}")
        index_list.append(index_all)

        index_all_right = []
        index_list_right = []
        # 遍历字典中的每个键值对
        for key, value in index_all_right_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all_right.extend(value)
            index_list_right.append(value)
        index_all_right = list(set(list(range(len(y[test_index])))) - set(index_all))
        # print(f"Length of index_all: {len(index_all_right)}")
        index_list_right.append(index_all_right)
        """ compute metircs"""
        # o = y[test_index]
        # correct = p == o
        # 所有错误的
        different_indices = np.where(all_pre != y[test_index])[0]
        different_indices_right = np.where(all_pre == y[test_index])[0]
        # 找到的错误的： index_list
        # 找的真的错的：num_common_elements
        acc_best = 0
        F1_best = 0
        pre_best = 0
        rec_best = 0
        method_name_best = " NONE"
        method_name = {
            "naive": ("naive", False),
            # "score": ("score", False),
            # "cumulated_score": ("cumulated_score", True),
            # "random_cumulated_score": ("cumulated_score", "randomized"),
            # "top_k": ("top_k", False),
            "mixture": ("mixture", False)
        }
        for index, (single_list, single_list_right) in enumerate(zip(index_list, index_list_right)):
            common_elements = np.intersect1d(single_list, different_indices)
            num_common_elements = len(common_elements)
            common_elements_right = np.intersect1d(single_list_right, different_indices_right)
            num_common_elements_right = len(common_elements_right)
            try:
                accuracy = (num_common_elements + num_common_elements_right) / len(all_pre)
            except:
                accuracy = 0
            try:
                precision = num_common_elements / len(single_list)
            except:
                precision = 0
            try:
                recall = num_common_elements / len(different_indices)
            except:
                recall = 0
            try:
                F1 = 2 * precision * recall / (precision + recall)
            except:
                F1 = 0
            print(
                f"{list(method_name.keys())[index]} find accuracy: {accuracy * 100:.2f}%, "
                f"precision: {precision * 100:.2f}%, "
                f"recall: {recall * 100:.2f}%, "
                f"F1: {F1 * 100:.2f}%"
            )

            if F1 > F1_best:
                method_name_best = list(method_name.keys())[index]
                acc_best = accuracy
                F1_best = F1
                pre_best = precision
                rec_best = recall
        print(
            f"{method_name_best} is the best approach"
            f"best accuracy: {accuracy * 100:.2f}%, "
            f"best precision: {pre_best * 100:.2f}%, "
            f"best recall: {rec_best * 100:.2f}%, "
            f"best F1: {F1_best * 100:.2f}%"
        )
        Acc_all.append(acc_best)
        F1_all.append(F1_best)
        Pre_all.append(pre_best)
        Rec_all.append(rec_best)
        # precision=找到的真的错误的/找到的错误的
        # recall=找到的真的错误的/所有错误的
        origin_accuracy=accuracy_score(y_test, y_preds[name])
        np.random.seed(seed)
        #
        selected_count = max(int(len(y_test) * 0.05), 1)
        try:
            random_element = random.sample(list(common_elements),selected_count)
        except:
            random_element = random.sample(range(len(test_index)), selected_count)
        sample = [test_index[index] for index in random_element]

        train_index=train_index+sample
        test_index = [item for item in test_index if item not in sample]

        # create a new model and train it
        model.init(seed=seed)
        model.train(cascading_features=X_feature[train_index],
                    cascading_y=y_1hot[train_index],
                    verbose=True)

        all_pre = []
        for i in range(len(X_feature[test_index])):
            p = model.predict(sequences=X_feature[test_index])[i]
            p = min(p, 2 ** (len(X_feature[test_index[0]]) - 1))
            all_pre.append(p)
        s_oracle_all = []
        p_speedup_all = []
        p_oracle_all = []
        # oracle prediction
        for i,(o,p) in enumerate(zip(y[test_index],all_pre)):
            # o = y[test_index]
            # correct = p == o
            # get runtime without thread coarsening
            kernel=oracles["kernel"][test_index[i]]
            row = df[(df["kernel"] == kernel) & (df["cf"] == 1)]
            assert (len(row) == 1)  # sanity check
            nocf_runtime = float(row["runtime_" + platform])

            # get runtime of prediction
            row = df[(df["kernel"] == kernel) & (df["cf"] == p)]
            if (len(row) != 1):
                row = df[(df["kernel"] == kernel) & (df["cf"] == 1)]# sanity check
            p_runtime = float(row["runtime_" + platform])

            # get runtime of oracle coarsening factor
            o_runtime = oracle_runtimes[test_index[i]]

            # speedup and % oracle
            s_oracle = nocf_runtime / o_runtime
            p_speedup = nocf_runtime / p_runtime
            p_oracle = o_runtime / p_runtime
            s_oracle_all.append(s_oracle)
            p_speedup_all.append(p_speedup)
            p_oracle_all.append(p_oracle)

        retrained_speedup = sum(p_speedup_all) / len(p_speedup_all)
        inproved_speedup=retrained_speedup-origin_speedup
        print("origin speed up is ",origin_speedup, " retrained speed up is ", retrained_speedup, "improved speed up is ",inproved_speedup)
        speed_up_all.append(retrained_speedup)
        improved_spp_all.append(inproved_speedup)
        IL_accuracy = accuracy_score(np.argmax(y_1hot[test_index], axis=1), all_pre)
        improve_accuracy=IL_accuracy-origin_accuracy
        print(
            f"improved accuracy: {improve_accuracy * 100:.2f}%, "
            f"Increment accuracy: {IL_accuracy * 100:.2f}%, ")
        improve_accuracy_all.append(improve_accuracy)
        IL_accuracy_all.append(IL_accuracy)

        progressbar[0] += 1  # update progress bar
        progressbar[1].update(progressbar[0])

    mean_acc = sum(Acc_all) / len(Acc_all)
    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    mean_improved = sum(improve_accuracy_all) / len(improve_accuracy_all)
    mean_il = sum(IL_accuracy_all) / len(improve_accuracy_all)
    mean_speed_up = sum(speed_up_all) / len(speed_up_all)
    meanimproved_speed_up = sum(improved_spp_all) / len(improved_spp_all)
    print(
        f"4 device mean accuracy: {mean_acc * 100:.2f}%, "
        f"mean precision: {mean_pre * 100:.2f}%, "
        f"mean recall: {mean_rec * 100:.2f}%, "
        f"mean F1: {mean_f1 * 100:.2f}%, "

    )
    print(
        f"Imroved Increment Learning acc: {mean_improved * 100:.2f}%, "
        f"Increment Learning acc: {mean_il * 100:.2f}%, "
        f"Imroved speed up: {mean_speed_up}, "
        f"Imroved mean speed up: {meanimproved_speed_up}, "
    )
    nni.report_final_result(mean_f1)

    return meanimproved_speed_up

def get_onehot(df, platform):
    cfs = [1, 2, 4, 8, 16, 32]
    hot = np.zeros((len(df), len(cfs)), dtype=np.int32)
    for i, cf in enumerate(df[f"cf_{platform}"]):
        hot[i][cfs.index(cf)] = 1

    return hot


def load_predict(model, args, model_pretrained=''):
    pd.set_option('display.max_rows', 5)
    df = pd.read_csv("data/case-study-b/pact-2014-runtimes.csv")
    oracles = pd.read_csv("data/case-study-b/pact-2014-oracles.csv")
    # thread coarsening factors
    # report progress:
    from progressbar import ProgressBar
    progressbar = [0, ProgressBar(max_value=4)]

    data = []

    X_seq = None  # defer sequence encoding (it's expensive)
    Acc_all = []
    F1_all = []
    Pre_all = []
    Rec_all = []

    for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
        platform_name = platform2str(platform)
        # load data
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        y_1hot = get_onehot(oracles, platform)
        X_cc, y_cc = get_magni_features(df, oracles, platform)

        # LOOCV
        kf = KFold(n_splits=len(y), shuffle=False)
        numbers = np.arange(1, len(y))
        # for j, (train_index, test_index) in enumerate(kf.split(y)):

        model_name = model.__name__
        model_basename = model.__basename__

        seed = args.seed

        model_path = f"models/{platform}-{seed}.model"
        predictions_path = f"data/case-study-b/predictions/{model_basename}-{platform}-{1}.result"

        model_pretrainpath = model_pretrained + '/' + model_path

        train_index, temp_set = train_test_split(numbers, train_size=0.4, random_state=seed)
        valid_index, test_index = train_test_split(temp_set, train_size=0.6, random_state=seed)

        model.restore(model_pretrainpath)

        if X_seq is None:
            X_seq = encode_srcs(df["src"].values)

        # make prediction
        all_pre = []
        for i in range(len(X_seq[test_index])):
            p = model.predict(sequences=X_seq[test_index])[i]
            p = min(p, 2 ** (len(X_cc[test_index[0]]) - 1))
            all_pre.append(p)

        """ conformal prediction"""
        clf = model
        method_params = {
            "naive": ("naive", False),
            "score": ("score", False),
            "cumulated_score": ("cumulated_score", True),
            "random_cumulated_score": ("cumulated_score", "randomized"),
            "top_k": ("top_k", False)
        }
        y_preds, y_pss = {}, {}

        def find_alpha_range(n):
            alpha_min = max(1 / n, 0)
            alpha_max = min(1 - 1 / n, 1)
            return alpha_min, alpha_max

        alpha_min, alpha_max = find_alpha_range(len(y_1hot[valid_index]))
        alphas = np.arange(alpha_min, alpha_max, 0.1)

        X_cal = X_seq[valid_index]
        one_hot_matrix = y_1hot[valid_index]
        y_cal = np.argmax(one_hot_matrix, axis=1)
        y_test = np.argmax(y_1hot[test_index], axis=1)

        X_test = X_seq[test_index]
        for name, (method, include_last_label) in method_params.items():
            mapie = MapieClassifier(estimator=clf, method=method, cv="prefit", random_state=seed)
            mapie.fit(X_cal, y_cal)
            y_preds[name], y_pss[name] = mapie.predict(X_test, alpha=alphas, include_last_label=include_last_label)

        def count_null_set(y: np.ndarray) -> int:
            """
            Count the number of empty prediction sets.

            Parameters
            ----------
            y: np.ndarray of shape (n_sample, )

            Returns
            -------
            int
            """
            count = 0
            for pred in y[:, :]:
                if np.sum(pred) == 0:
                    count += 1
            return count

        nulls, coverages, accuracies, sizes = {}, {}, {}, {}
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        for name, (method, include_last_label) in method_params.items():
            accuracies[name] = accuracy_score(y_test, y_preds[name])
            nulls[name] = [
                count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(alphas)
            ]
            coverages[name] = [
                classification_coverage_score(
                    y_test, y_pss[name][:, :, i]
                ) for i, _ in enumerate(alphas)
            ]
            sizes[name] = [
                y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
            ]
        # sizes里每个method最接近1的
        result = {}  # 用于存储结果的字典
        for key, lst in sizes.items():  # 遍历字典的键值对
            closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - 1))  # 找到最接近1的数字的索引
            result[key] = closest_index  # 将结果存入字典
        # y_ps_90中提出来那个最接近1的位置
        result_ps = {}
        for method, y_ps in y_pss.items():
            result_ps[method] = y_ps[:, :, result[method]]

        index_all_tem = {}
        index_all_right_tem = {}
        for method, y_ps in result_ps.items():
            for index, i in enumerate(y_ps):
                num_true = sum(i)
                if method not in index_all_tem:
                    index_all_tem[method] = []
                    index_all_right_tem[method] = []
                if num_true != 1:
                    index_all_tem[method].append(index)
                elif num_true == 1:
                    index_all_right_tem[method].append(index)

        index_all = []
        index_list = []
        # 遍历字典中的每个键值对
        for key, value in index_all_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all.extend(value)
            index_list.append(value)
        index_all = list(set(index_all))
        # print(f"Length of index_all: {len(index_all)}")
        index_list.append(index_all)

        index_all_right = []
        index_list_right = []
        # 遍历字典中的每个键值对
        for key, value in index_all_right_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all_right.extend(value)
            index_list_right.append(value)
        index_all_right = list(set(list(range(len(y[test_index])))) - set(index_all))
        # print(f"Length of index_all: {len(index_all_right)}")
        index_list_right.append(index_all_right)
        """ compute metircs"""
        # o = y[test_index]
        # correct = p == o
        # 所有错误的
        different_indices = np.where(all_pre != y[test_index])[0]
        different_indices_right = np.where(all_pre == y[test_index])[0]
        # 找到的错误的： index_list
        # 找的真的错的：num_common_elements
        acc_best = 0
        F1_best = 0
        pre_best = 0
        rec_best = 0
        method_name_best = " NONE"
        method_name = {
            "naive": ("naive", False),
            "score": ("score", False),
            "cumulated_score": ("cumulated_score", True),
            "random_cumulated_score": ("cumulated_score", "randomized"),
            "top_k": ("top_k", False),
            "mixture": ("mixture", False)
        }
        for index, (single_list, single_list_right) in enumerate(zip(index_list, index_list_right)):
            common_elements = np.intersect1d(single_list, different_indices)
            num_common_elements = len(common_elements)
            common_elements_right = np.intersect1d(single_list_right, different_indices_right)
            num_common_elements_right = len(common_elements_right)
            try:
                accuracy = (num_common_elements + num_common_elements_right) / len(all_pre)
            except:
                accuracy = 0
            try:
                precision = num_common_elements / len(single_list)
            except:
                precision = 0
            try:
                recall = num_common_elements / len(different_indices)
            except:
                recall = 0
            try:
                F1 = 2 * precision * recall / (precision + recall)
            except:
                F1 = 0
            print(
                f"{list(method_name.keys())[index]} find accuracy: {accuracy * 100:.2f}%, "
                f"precision: {precision * 100:.2f}%, "
                f"recall: {recall * 100:.2f}%, "
                f"F1: {F1 * 100:.2f}%"
            )

            if F1 > F1_best:
                method_name_best = list(method_name.keys())[index]
                acc_best = accuracy
                F1_best = F1
                pre_best = precision
                rec_best = recall
        print(
            f"{method_name_best} is the best approach"
            f"best accuracy: {accuracy * 100:.2f}%, "
            f"best precision: {pre_best * 100:.2f}%, "
            f"best recall: {rec_best * 100:.2f}%, "
            f"best F1: {F1_best * 100:.2f}%"
        )
        Acc_all.append(acc_best)
        F1_all.append(F1_best)
        Pre_all.append(pre_best)
        Rec_all.append(rec_best)
        # precision=找到的真的错误的/找到的错误的
        # recall=找到的真的错误的/所有错误的
        """ compute metircs"""

        progressbar[0] += 1  # update progress bar
        progressbar[1].update(progressbar[0])

    mean_acc = sum(Acc_all) / len(Acc_all)
    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    print(
        f"4 device mean accuracy: {mean_acc * 100:.2f}%, "
        f"mean precision: {mean_pre * 100:.2f}%, "
        f"mean recall: {mean_rec * 100:.2f}%, "
        f"mean F1: {mean_f1 * 100:.2f}%, "

    )
    nni.report_final_result(mean_f1)

    return mean_f1

    # return pd.DataFrame(data, columns=[
    #     "Model", "Platform", "Kernel", "Oracle-CF", "Predicted-CF", "Speedup", "Oracle"])

def get_onehot(df, platform):
    cfs = [1, 2, 4, 8, 16, 32]
    hot = np.zeros((len(df), len(cfs)), dtype=np.int32)
    for i, cf in enumerate(df[f"cf_{platform}"]):
        hot[i][cfs.index(cf)] = 1

    return hot

def platform2str(platform):
    if platform == "Fermi":
        return "NVIDIA GTX 480"
    elif platform == "Kepler":
        return "NVIDIA Tesla K20c"
    elif platform == "Cypress":
        return "AMD Radeon HD 5900"
    elif platform == "Tahiti":
        return "AMD Tahiti 7970"
    else:
        raise LookupError

def get_magni_features(df, oracles, platform):
    """
    Assemble cascading data.
    """
    X_cc, y_cc, = [], []
    for kernel in sorted(set(df["kernel"])):
        _df = df[df["kernel"] == kernel]

        oracle_cf = int(oracles[oracles["kernel"] == kernel][f"cf_{platform}"].values[0])

        feature_vectors = np.asarray([
            _df['PCA1'].values,
            _df['PCA2'].values,
            _df['PCA3'].values,
            _df['PCA4'].values,
            _df['PCA5'].values,
            _df['PCA6'].values,
            _df['PCA7'].values,
        ]).T

        X_cc.append(feature_vectors)
        y = []
        cfs__ = []

        def modify_array(arr):
            found_zero = False

            for i in range(len(arr)):
                if not found_zero and arr[i] == 0:
                    arr[i] = 1
                    found_zero = True
                else:
                    arr[i] = 0

            return arr

        cfs = [1, 2, 4, 8, 16, 32]
        for i, cf in enumerate(cfs[:len(feature_vectors)]):
            y_ = 1 if cf < oracle_cf else 0
            y.append(y_)
        modified_array = modify_array(y)
        y_cc.append(modified_array)

        assert len(feature_vectors) == len(y)

    assert len(X_cc) == len(y_cc) == 17

    return np.asarray(X_cc), np.asarray(y_cc)

def encode_srcs(srcs):
    """ encode and pad source code for learning """


    # seqs = [atomizer.atomize(src) for src in srcs]
    # seqs = [tokenizer.tokenize(src) for src in srcs]
    code_tokens=[tokenizer.tokenize(src) for src in srcs]
    seqs = [tokenizer.convert_tokens_to_ids(src) for src in code_tokens]
    # seqs = [tokenizer.tokenize(src) for src in tokens_ids]
    # pad_val = atomizer.vocab_size
    pad_val = len(seqs)
    encoded = np.array(pad_sequences(seqs, maxlen=1024, value=pad_val))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded])

def load_predict(model,args,model_pretrained=''):

    # report progress:
    from progressbar import ProgressBar
    progressbar = [0, ProgressBar(max_value=4)]
    pd.set_option('display.max_rows', 5)
    df = pd.read_csv("data/case-study-b/pact-2014-runtimes.csv")
    oracles = pd.read_csv("data/case-study-b/pact-2014-oracles.csv")
    cfs = [1, 2, 4, 8, 16, 32]  # thread coarsening factors
    data = []

    X_seq = None  # defer sequence encoding (it's expensive)
    Acc_all = []
    F1_all = []
    Pre_all = []
    Rec_all = []
    for i, platform in enumerate(["Kepler","Fermi" ,"Cypress", "Tahiti" ]):
        platform_name = platform2str(platform)

        # load data
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        # y_1hot = get_onehot(oracles, platform)
        X_cc, y_cc = get_magni_features(df, oracles, platform)

        # LOOCV
        # kf = KFold(n_splits=len(y), shuffle=False)
        # seed = int(args.seed)
        seed = int(args.seed)
        numbers = np.arange(1, len(y))
        model_path = f"models/magni/{platform}-{seed}.model"
        model_pretrainpath = model_pretrained + '/' + model_path

        train_index, temp_set = train_test_split(numbers, train_size=0.4, random_state=seed)
        valid_index, test_index = train_test_split(temp_set, train_size=0.6, random_state=seed)

        model.restore(model_pretrainpath)

        # for j, (train_index, test_index) in enumerate(kf.split(y)):
        kernel = sorted(set(df["kernel"]))[test_index[0]]

        model_name = model.__name__
        model_basename = model.__basename__

        predictions_path = f"data/case-study-b/predictions/{model_basename}-{platform}-{1}.result"

        # if 0:
        #     # load result from cache
        #     with open(predictions_path, 'rb') as infile:
        #         p = pickle.load(infile)
        # else:
            # if fs.exists(model_path):
                # load a trained model from cache
                # model.restore(model_path)
            # else:
                # encode source codes
        # if X_seq is None:
        #     X_seq = encode_srcs(df["src"].values)

        # create a new model and train it
        # a=X_cc[train_index]

        def align_lists(lists):

            max_length = max(len(lst) for lst in lists)  # 获取最长的array长度
            aligned_lists = []
            for lst in lists:
                a = len(lst)
                aligned_array = lst.tolist() + [0] * (max_length - len(lst))  # 补0对齐每个array
                aligned_lists.append(np.array(aligned_array))
            return aligned_lists
        flattened_array_method2 = [array.flatten() for array in X_cc]
        aligned_lists = align_lists(flattened_array_method2)
        X_feature=np.array(aligned_lists)
        y_1hot = get_onehot(oracles, platform)




        # make prediction
        all_pre = []
        for i in range(len(X_feature[test_index])):
            p = model.predict(sequences=X_feature[test_index])[i]
            p = min(p, 2 ** (len(X_feature[test_index[0]]) - 1))
            all_pre.append(p)

        acc=accuracy_score(np.argmax(y_1hot[test_index], axis=1), all_pre)
        print()
        print(platform," acc is",acc)
        print()
        """ conformal prediction"""
        clf = model
        method_params = {
            "naive": ("naive", False),
            "score": ("score", False),
            "cumulated_score": ("cumulated_score", True),
            "random_cumulated_score": ("cumulated_score", "randomized"),
            "top_k": ("top_k", False)
        }
        y_preds, y_pss = {}, {}

        def find_alpha_range(n):
            alpha_min = max(1 / n, 0)
            alpha_max = min(1 - 1 / n, 1)
            return alpha_min, alpha_max

        alpha_min, alpha_max = find_alpha_range(len(y_1hot[valid_index]))
        alphas = np.arange(alpha_min, alpha_max, 0.1)
        # alphas=[0.1]

        X_cal = X_feature[valid_index]
        one_hot_matrix = y_1hot[valid_index]
        y_cal = np.argmax(one_hot_matrix, axis=1)
        y_tr = np.argmax(y_1hot[train_index], axis=1)
        y_test = np.argmax(y_1hot[test_index], axis=1)
        X_test = X_feature[test_index]

        for name, (method, include_last_label) in method_params.items():
            mapie = MapieClassifier(estimator=clf, method=method, cv="prefit", random_state=seed)
            mapie.fit(X_cal, y_cal)
            y_preds[name], y_pss[name] = mapie.predict(X_test, alpha=alphas, include_last_label=include_last_label)
        def count_null_set(y: np.ndarray) -> int:
            """
            Count the number of empty prediction sets.

            Parameters
            ----------
            y: np.ndarray of shape (n_sample, )

            Returns
            -------
            int
            """
            count = 0
            for pred in y[:, :]:
                if np.sum(pred) == 0:
                    count += 1
            return count

        nulls, coverages, accuracies, sizes = {}, {}, {}, {}
        for name, (method, include_last_label) in method_params.items():
            accuracies[name] = accuracy_score(y_test, y_preds[name])
            nulls[name] = [
                count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(alphas)
            ]
            coverages[name] = [
                classification_coverage_score(
                    y_test, y_pss[name][:, :, i]
                ) for i, _ in enumerate(alphas)
            ]
            sizes[name] = [
                y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
            ]

        # sizes里每个method最接近1的
        result = {}  # 用于存储结果的字典
        for key, lst in sizes.items():  # 遍历字典的键值对
            closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - 1))  # 找到最接近1的数字的索引
            result[key] = closest_index  # 将结果存入字典
        # y_ps_90中提出来那个最接近1的位置
        result_ps = {}
        for method, y_ps in y_pss.items():
            result_ps[method] = y_ps[:, :, result[method]]

        index_all_tem = {}
        index_all_right_tem = {}
        for method, y_ps in result_ps.items():
            for index, i in enumerate(y_ps):
                num_true = sum(i)
                if method not in index_all_tem:
                    index_all_tem[method] = []
                    index_all_right_tem[method] = []
                if num_true != 1:
                    index_all_tem[method].append(index)
                elif num_true == 1:
                    index_all_right_tem[method].append(index)
        index_all = []
        index_list = []
        # 遍历字典中的每个键值对
        for key, value in index_all_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all.extend(value)
            index_list.append(value)
        index_all = list(set(index_all))
        # print(f"Length of index_all: {len(index_all)}")
        index_list.append(index_all)

        index_all_right = []
        index_list_right = []
        # 遍历字典中的每个键值对
        for key, value in index_all_right_tem.items():
            # 使用集合对列表中的元素进行去重，并转换为列表
            list_length = len(value)
            # print(f"Length of {key}: {list_length}")
            # 将去重后的列表添加到新列表中
            index_all_right.extend(value)
            index_list_right.append(value)
        index_all_right = list(set(list(range(len(y[test_index])))) - set(index_all))
        # print(f"Length of index_all: {len(index_all_right)}")
        index_list_right.append(index_all_right)
        """ compute metircs"""
        # o = y[test_index]
        # correct = p == o
        # 所有错误的
        different_indices = np.where(all_pre != y[test_index])[0]
        different_indices_right = np.where(all_pre == y[test_index])[0]
        # 找到的错误的： index_list
        # 找的真的错的：num_common_elements
        acc_best = 0
        F1_best = 0
        pre_best = 0
        rec_best = 0
        method_name_best = " NONE"
        method_name = {
            "naive": ("naive", False),
            "score": ("score", False),
            "cumulated_score": ("cumulated_score", True),
            "random_cumulated_score": ("cumulated_score", "randomized"),
            "top_k": ("top_k", False),
            "mixture": ("mixture", False)
        }
        for index, (single_list, single_list_right) in enumerate(zip(index_list, index_list_right)):
            common_elements = np.intersect1d(single_list, different_indices)
            num_common_elements = len(common_elements)
            common_elements_right = np.intersect1d(single_list_right, different_indices_right)
            num_common_elements_right = len(common_elements_right)
            try:
                accuracy = (num_common_elements + num_common_elements_right) / len(all_pre)
            except:
                accuracy = 0
            try:
                precision = num_common_elements / len(single_list)
            except:
                precision = 0
            try:
                recall = num_common_elements / len(different_indices)
            except:
                recall = 0
            try:
                F1 = 2 * precision * recall / (precision + recall)
            except:
                F1 = 0
            print(
                f"{list(method_name.keys())[index]} find accuracy: {accuracy * 100:.2f}%, "
                f"precision: {precision * 100:.2f}%, "
                f"recall: {recall * 100:.2f}%, "
                f"F1: {F1 * 100:.2f}%"
            )

            if F1 > F1_best:
                method_name_best = list(method_name.keys())[index]
                acc_best = accuracy
                F1_best = F1
                pre_best = precision
                rec_best = recall
        print(
            f"{method_name_best} is the best approach"
            f"best accuracy: {accuracy * 100:.2f}%, "
            f"best precision: {pre_best * 100:.2f}%, "
            f"best recall: {rec_best * 100:.2f}%, "
            f"best F1: {F1_best * 100:.2f}%"
        )
        Acc_all.append(acc_best)
        F1_all.append(F1_best)
        Pre_all.append(pre_best)
        Rec_all.append(rec_best)
        # precision=找到的真的错误的/找到的错误的
        # recall=找到的真的错误的/所有错误的
        """ compute metircs"""

        progressbar[0] += 1  # update progress bar
        progressbar[1].update(progressbar[0])

    mean_acc = sum(Acc_all) / len(Acc_all)
    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    print(
        f"4 device mean accuracy: {mean_acc * 100:.2f}%, "
        f"mean precision: {mean_pre * 100:.2f}%, "
        f"mean recall: {mean_rec * 100:.2f}%, "
        f"mean F1: {mean_f1 * 100:.2f}%"
    )
    nni.report_final_result(mean_f1)
    return mean_f1

def main():
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "seed": int(123456),
        }
    print("________________")
    print(params)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    args = parser.parse_args()

    print("Evaluating Magni et al. ...", file=sys.stderr)
    magni = IL(Magni(),args)
    # load_predict(Magni(),args,model_pretrained='/home/huanting/model/Thread/paper-end2end-dl')

main()
# magni.groupby('Platform')['Platform', 'Speedup', 'Oracle'].mean()