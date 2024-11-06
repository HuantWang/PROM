import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import logging

logging.set_verbosity_error()
from transformers import AutoTokenizer
from keras_preprocessing.sequence import pad_sequences
import nni
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import re
import pickle
import sys
import os.path as fs
import torch
from keras.layers import Input, Dropout, Embedding, LSTM, Dense
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model, Sequential, load_model
# during grid search, not all parameters will converge. Ignore these warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from sklearn.model_selection import train_test_split
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from progressbar import ProgressBar
import matplotlib.pyplot as plt

filterwarnings("ignore", category=ConvergenceWarning)
import tensorflow as tf
import os

sys.path.append('./case_study/Loop')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
sys.path.append('/home/huanting/PROM')
import src.prom.prom_util as util
import clang.cindex
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold


class LoopT(util.ModelDefinition):
    def __init__(self, model=None, dataset=None, calibration_data=None, args=None):
        self.model = Magni()
        self.calibration_data = None
        self.dataset = None

    def data_partitioning(self, dataset=r'data_dict.pkl', calibration_ratio=0.2, args=None):
        X_seq, y_1hot, time = get_feature(path=dataset)

        numbers = np.arange(1, len(y_1hot))
        # for j, (train_index, test_index) in enumerate(kf.split(y)):
        seed = int(args.seed)
        np.random.seed(seed)
        # test 500 samples

        train_index, temp_set = train_test_split(numbers, train_size=0.6, random_state=seed)
        valid_index, test_index = train_test_split(
            temp_set, train_size=0.5, random_state=seed
        )

        return X_seq, y_1hot, time, train_index, valid_index, test_index

    def predict(self, X, significant_level=0.1):
        if self.model is None:
            raise ValueError("Model is not initialized.")

        pred = self.model.predict(self, sequences='')
        probability = self.model.predict_proba(self, sequences='')
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


class Magni():
    __name__ = "Magni et al."
    __basename__ = "magni"

    def init(self, args: int = None):
        # the neural network
        nn = MLPClassifier(random_state=args.seed, shuffle=True)

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


def make_prediction(model=None, X_feature=None, y_1hot=None, time=None, test_index=None):
    # make prediction
    all_pre = []
    for i in range(len(X_feature[test_index])):
        p = model.predict(sequences=X_feature[test_index])[i]
        p = min(p, 2 ** (len(X_feature[test_index[0]]) - 1))
        all_pre.append(p)
    """speed up"""
    p_speedup_all = []
    non_speedup_all = []
    data_distri = []
    # oracle prediction
    for i, (o, p) in enumerate(zip(y_1hot[test_index], all_pre)):
        # get runtime without thread coarsening
        time_single = []
        a = time[test_index]
        for row in time[test_index][i]:
            for num in row:
                time_single.append(num)
        non_runtime = time_single[0]
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
        data_distri.append(percent)
    origin_speedup = sum(non_speedup_all) / len(non_speedup_all)
    # print("origin_speedup is", origin_speedup)
    return origin_speedup, all_pre, data_distri


def make_prediction_il(model_il=None, X_feature=None, y_1hot=None, time=None,
                       test_index=None, origin_speedup=None):
    # make prediction
    all_pre = []
    for i in range(len(X_feature[test_index])):
        p = model_il.predict(sequences=X_feature[test_index])[i]
        p = min(p, 2 ** (len(X_feature[test_index[0]]) - 1))
        all_pre.append(p)
    """speed up"""
    p_speedup_all = []
    non_speedup_all = []
    # oracle prediction
    for i, (o, p) in enumerate(zip(y_1hot[test_index], all_pre)):
        # get runtime without thread coarsening
        time_single = []
        a = time[test_index]
        for row in time[test_index][i]:
            for num in row:
                time_single.append(num)
        non_runtime = time_single[0]
        # get runtime of prediction
        p_runtime = time_single[p]
        # get runtime of oracle coarsening factor
        o_runtime = min(time_single)
        # speedup and % oracle
        # non_speedup = non_runtime / p_runtime
        speedup_origin = non_runtime / o_runtime
        speedup_prediction = non_runtime / p_runtime
        percent = speedup_prediction / speedup_origin
        # non_speedup_all.append(non_speedup)
        non_speedup_all.append(percent)
    retrained_speedup = sum(non_speedup_all) / len(non_speedup_all)

    inproved_speedup = retrained_speedup - origin_speedup
    # print("origin_speedup",origin_speedup,"il_speedup : ", il_speedup,"improved_speedup : ", improved_sp)
    print("The retrained speed up is ", retrained_speedup,
          "the improved speed up is ", inproved_speedup)
    return retrained_speedup, inproved_speedup


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
    # time
    time = dict_all.get("all_program_runtimes")
    time_all = []
    for key in time:
        value = time[key]
        time_all.append(value)
    data_dict = {}  # 创建一个空字典

    all_loops = extract_loops_from_files(file_path)
    filtered_data = [(a, b, c, d) for a, b, c, d in zip(all_loops, label, time_all, file_path) if a and b and c and d]
    all_loops, label, time_all, file_path = zip(*filtered_data)
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

# main()
