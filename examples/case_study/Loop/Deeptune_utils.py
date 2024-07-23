import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import logging
logging.set_verbosity_error()
from transformers import AutoTokenizer
from keras.utils import pad_sequences
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
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
import src.prom_util as util
import clang.cindex

class LoopT(util.ModelDefinition):
    def __init__(self,model=None,dataset=None,calibration_data=None,args=None):
        self.model = DeepTune()
        self.calibration_data = None
        self.dataset = None

    def data_partitioning(self, dataset=r'data_dict.pkl', calibration_ratio=0.2,args=None):
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

        return X_seq, y_1hot,time,train_index, valid_index, test_index


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


class DeepTune():
    __name__ = "DeepTune"
    __basename__ = "deeptune"

    def init(self, args):
        self.seed = args.seed
        self.epoch = args.epoch
        self.batch = args.batch_size

        # Vocabulary has a padding character
        # vocab_size = atomizer.vocab_size + 1
        np.random.seed(int(self.seed))
        tf.random.set_seed(self.seed)
        vocab_size = 99999
        # Language model. Takes as inputs source code sequences.
        seq_inputs = Input(shape=(100,), dtype="int32")
        x = Embedding(
            input_dim=vocab_size, input_length=100, output_dim=64, name="embedding"
        )(seq_inputs)
        x = LSTM(64, return_sequences=True, implementation=1, name="lstm_1")(x)
        x = LSTM(64, implementation=1, name="lstm_2")(x)

        # Heuristic model. Takes as inputs the language model,
        #   outputs 1-of-6 thread coarsening factor
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        outputs = Dense(35, activation="sigmoid")(x)

        self.model = Model(inputs=seq_inputs, outputs=outputs)
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def save(self, outpath: str):
        self.model.save(outpath)

    def fit(self, seed):
        return

    def restore(self, inpath: str):
        self.model = load_model(inpath)

    def train(
        self, sequences: np.array, y_1hot: np.array, verbose: bool = False
    ) -> None:
        self.model.fit(
            sequences,
            y_1hot,
            epochs=self.epoch,
            batch_size=self.batch,
            verbose=verbose,
            shuffle=True,
        )

    def predict(self, sequences: np.array) -> np.array:
        # directly predict optimal thread coarsening factor from source sequences:
        p = np.array(self.model.predict(sequences, verbose=0))
        preds = torch.softmax(torch.tensor(p), dim=1)
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
        p = np.array(self.model.predict(sequences, verbose=0))
        preds = torch.softmax(torch.tensor(p), dim=1)
        return preds.detach().numpy()

    def predict_model(self, sequences: np.array):
        # directly predict optimal thread coarsening factor from source sequences:
        p = np.array(self.model.predict(sequences, verbose=0))
        preds = torch.softmax(torch.tensor(p), dim=1)
        p = preds.detach().numpy()
        indices = [np.argmax(x) for x in p]
        indice = np.array(indices)
        return indice

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

def deeptune_make_prediction(model=None, X_seq=None, y_1hot=None, time=None, test_index=None):
    # make prediction
    all_pre = []
    all_pre = model.predict(sequences=X_seq[test_index])
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
        non_speedup = non_runtime / p_runtime
        speedup_origin = non_runtime / o_runtime
        speedup_prediction = non_runtime / p_runtime
        percent = speedup_prediction / speedup_origin
        non_speedup_all.append(percent)
    origin_speedup = sum(non_speedup_all) / len(non_speedup_all)
    # print("origin_speedup is", origin_speedup)
    return origin_speedup, all_pre

def deeptune_make_prediction_il(model_il=None, X_seq=None, y_1hot=None, time=None,
                       test_index=None,  origin_speedup=None):
    # make prediction
    all_pre = model_il.predict(sequences=X_seq[test_index])
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
    return retrained_speedup,inproved_speedup

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
    # load data
    X_seq, y_1hot,time = get_feature(path="data_dict.pkl")
    numbers = np.arange(1, len(y_1hot))
    # for j, (train_index, test_index) in enumerate(kf.split(y)):
    seed = int(args.seed)
    np.random.seed(seed)
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
    model.init(args)
    model.train(
        sequences=X_seq[train_index], verbose=False, y_1hot=y_1hot[train_index]
    )

    # cache the model
    try:
        os.mkdir(fs.dirname(model_path))
    except:
        pass
    model.save(model_path)

    # make prediction
    all_pre = []
    all_pre = model.predict(sequences=X_seq[test_index])
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
        non_speedup = non_runtime / p_runtime
        speedup_origin=non_runtime/o_runtime
        speedup_prediction=non_runtime/p_runtime
        percent= speedup_prediction/speedup_origin
        non_speedup_all.append(percent)
    origin_speedup = sum(non_speedup_all) / len(non_speedup_all)
    print ( "origin_speedup is", origin_speedup)

    #########plot
    # plt.boxplot(non_speedup_all)
    import seaborn as sns
    import pandas as pd
    data_df = pd.DataFrame({'Data': non_speedup_all})

    sns.violinplot(data=data_df, y='Data')
    seed_save=str(int(seed))
    plt.title('violin Plot Example '+seed_save)
    plt.ylabel('Values')
    plt.savefig('/home/huanting/model/loop/deeptune/figures/'+'box_plot_'+
                str(origin_speedup)+'_'+str(seed_save)+'.png')
    data_df.to_pickle('/home/huanting/model/loop/deeptune/figures/data/'+
                      str(origin_speedup)+'_'+str(seed_save)+'_data.pkl')
    # plt.show()
    ##########
    nni.report_final_result(origin_speedup)
    return origin_speedup
    """ conformal prediction"""
    print("start conformal prediction")
    clf = model
    method_params = {
        "naive": ("naive", False),
        "score": ("score", False),
        "cumulated_score": ("cumulated_score", True),
        "random_cumulated_score": ("cumulated_score", "randomized"),
        "top_k": ("top_k", False),
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
        mapie = MapieClassifier(
            estimator=clf, method=method, cv="prefit", random_state=seed
        )
        mapie.fit(X_cal, y_cal)
        y_preds[name], y_pss[name] = mapie.predict(
            X_test, alpha=alphas, include_last_label=include_last_label
        )
        # print("a")

    def count_null_set(y: np.ndarray) -> int:
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
            classification_coverage_score(y_test, y_pss[name][:, :, i])
            for i, _ in enumerate(alphas)
        ]
        sizes[name] = [
            y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
        ]
    # sizes里每个method最接近1的
    result = {}  # 用于存储结果的字典
    for key, lst in sizes.items():  # 遍历字典的键值对
        closest_index = min(
            range(len(lst)), key=lambda i: abs(lst[i] - 1)
        )  # 找到最接近1的数字的索引
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
    y = np.argmax(y_1hot, axis=1)
    index_all_right = list(set(list(range(len(y[test_index])))) - set(index_all))
    # print(f"Length of index_all: {len(index_all_right)}")
    index_list_right.append(index_all_right)
    """ compute metircs"""
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
    """""IL"""
    origin_accuracy = accuracy_score(y_test, y_preds[name])

    #
    selected_count = max(int(len(y_test) * 0.05), 1)
    np.random.seed(seed)
    try:
        random_element = np.random.choice(common_elements, selected_count, replace=False)
    except:
        random_element = np.random.choice(range(len(test_index)), selected_count)
    sample = [test_index[index] for index in random_element]

    train_index = np.concatenate((train_index, sample))
    test_index = [item for item in test_index if item not in sample]

    # create a new model and train it
    model.init(args)
    model.train(
        sequences=X_seq[train_index], verbose=False, y_1hot=y_1hot[train_index]  # TODO
    )

    # make prediction
    all_pre = model.predict(sequences=X_seq[test_index])
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
    il_speedup = sum(non_speedup_all) / len(non_speedup_all)

    improved_sp = il_speedup - origin_speedup
    # print("origin_speedup",origin_speedup,"il_speedup : ", il_speedup,"improved_speedup : ", improved_sp)
    """ compute metircs"""
    progressbar[0] += 1  # update progress bar
    progressbar[1].update(progressbar[0])

    mean_acc = sum(Acc_all) / len(Acc_all)
    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    # mean_improved = sum(improve_accuracy_all) / len(improve_accuracy_all)
    # mean_il = sum(IL_accuracy_all) / len(improve_accuracy_all)
    # mean_speed_up = sum(speed_up_all) / len(speed_up_all)
    # meanimproved_speed_up = sum(improved_spp_all) / len(improved_spp_all)
    print(
        f"mean accuracy: {mean_acc * 100:.2f}%, "
        f"mean precision: {mean_pre * 100:.2f}%, "
        f"mean recall: {mean_rec * 100:.2f}%, "
        f"mean F1: {mean_f1 * 100:.2f}%, "
    )
    print(
        f"origin speed up: {origin_speedup}, "
        f"Imroved speed up: {il_speedup}, "
        f"Imroved mean speed up: {improved_sp}, "
    )
    nni.report_final_result(improved_sp)
    return improved_sp

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
            "epoch": 5,
            "batch_size": 128,
            "seed": 3407,
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

    print("Evaluating Deeptune et al. ...", file=sys.stderr)


    deeptune_model = DeepTune()
    deeptune_model.init(args)
    # deeptune_model.model.summary()
    print("Evaluating DeepTune ...", file=sys.stderr)
    deeptune = evaluate(DeepTune(), args)
    # deeptune.groupby('Platform')['Platform', 'Speedup', 'Oracle'].mean()


# main()
