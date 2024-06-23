from transformers import AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
import nni
import argparse
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import re
import pickle
import sys
import os.path as fs
import torch
from keras.layers import Input, Dropout, Embedding, merge, LSTM, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
# during grid search, not all parameters will converge. Ignore these warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from sklearn.model_selection import train_test_split
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score, classification_mean_width_score
from progressbar import ProgressBar

filterwarnings("ignore", category=ConvergenceWarning)



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
        print("Key:", key, "Value:", value)
    # 获取 A 中所有不同的元组（不重复）
    unique_tuples = list(set(label))
    # 创建一个字典，将每个元组映射为唯一的整数编码
    tuple_to_int = {tuple_: i for i, tuple_ in enumerate(unique_tuples)}
    # 初始化一个全零矩阵，用于存储 one-hot 编码后的结果
    one_hot_matrix = np.zeros((len(label), len(unique_tuples)))
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

    all_loops = extract_loops_from_files(file_path)
    all_loops, label = zip(*[(a, b) for a, b in zip(all_loops, label) if a and b])
    data_dict = {"feature": all_loops, "label": label}
    with open("data_dict.pkl", "wb") as f:
        pickle.dump(data_dict, f)

    # print("a")


def evaluate( args):
    progressbar = [0, ProgressBar(max_value=4)]

    X_seq = None  # defer sequence encoding (it's expensive)
    F1_all = []
    Pre_all = []
    Rec_all = []

    # load data
    X_seq, y_1hot = get_feature(path="data_dict.pkl")
    numbers = np.arange(1, len(y_1hot))
    # for j, (train_index, test_index) in enumerate(kf.split(y)):
    seed = int(args.seed)
    train_index, temp_set = train_test_split(numbers, train_size=0.6, random_state=seed)
    valid_index, test_index = train_test_split(
        temp_set, train_size=0.5, random_state=seed
    )
    # make prediction
    all_pre = []
    # all_pre = model.predict(sequences=X_seq[test_index])
    # p_model = model.predict_model(cascading_features=X_cc[test_index[0]], sequences=X_seq[test_index])[0]
    # pred = model.predict_proba(cascading_features=X_cc[test_index[0]], sequences=X_seq[test_index])[0]

    """ conformal prediction"""
    print("start conformal prediction,GaussianNB")
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    # 构建决策树分类器并进行训练
    # clf = DecisionTreeClassifier()
    clf = KNeighborsClassifier(n_neighbors=3).fit(X_seq[train_index], y_1hot[train_index])
    y_pred_train = clf.predict(X_seq[train_index])
    accuracy_train = accuracy_score(y_1hot[train_index], y_pred_train)
    print("accuracy_train", accuracy_train)
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
    # alphas=[0.1]

    X_cal = X_seq[valid_index]
    y_cal = y_1hot[valid_index]
    X_test = X_seq[test_index]
    y_test = y_1hot[test_index]

    all_pre = clf.predict(X_test)
    # y_pred_proba = clf.predict_proba(X_test)
    # y_pred_proba_max = np.max(y_pred_proba, axis=1)


    for name, (method, include_last_label) in method_params.items():
        mapie = MapieClassifier(
            estimator=clf, method=method, cv="prefit", random_state=42
        )
        mapie.fit(X_cal, y_cal)
        y_preds[name], y_pss[name] = mapie.predict(
            X_test, alpha=alphas, include_last_label=include_last_label
        )

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
    for method, y_ps in result_ps.items():
        for index, i in enumerate(y_ps):
            num_true = sum(i)
            if method not in index_all_tem:
                index_all_tem[method] = []  # 将键的值初始化为空列表
            if num_true != 1:
                index_all_tem[method].append(index)
    index_all = []
    index_list = []
    # 遍历字典中的每个键值对
    for key, value in index_all_tem.items():
        # 使用集合对列表中的元素进行去重，并转换为列表
        list_length = len(value)
        print(f"Length of {key}: {list_length}")
        # 将去重后的列表添加到新列表中
        index_all.extend(value)
        index_list.append(value)
    index_all = list(set(index_all))
    print(f"Length of index_all: {len(index_all)}")
    index_list.append(index_all)
    """ compute metircs"""
    # o = y[test_index]
    # correct = p == o
    # 所有错误的
    # y=np.argmax(y_1hot, axis=1)
    different_indices = np.where(all_pre != y_1hot[test_index])[0]
    # 找到的错误的： index_list
    # 找的真的错的：num_common_elements
    F1_best = 0
    pre_best = 0
    rec_best = 0
    method_name = {
        "naive": ("naive", False),
        "score": ("score", False),
        "cumulated_score": ("cumulated_score", True),
        "random_cumulated_score": ("cumulated_score", "randomized"),
        "top_k": ("top_k", False),
        "all": ("all", False),
    }
    for index, single_list in enumerate(index_list):
        common_elements = np.intersect1d(single_list, different_indices)
        num_common_elements = len(common_elements)
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
            list(method_name.keys())[index],
            "find precision为：%.2f%%" % (precision * 100),
        )
        print(list(method_name.keys())[index], "find recall：%.2f%%" % (recall * 100))
        print(list(method_name.keys())[index], "find F1：%.2f%%" % (F1 * 100))
        if F1 > F1_best:
            F1_best = F1
            pre_best = precision
            rec_best = recall
    print("best precision为：%.2f%%" % (pre_best * 100))
    print("best recall：%.2f%%" % (rec_best * 100))
    print("best F1：%.2f%%" % (F1_best * 100))
    F1_all.append(F1_best)
    Pre_all.append(pre_best)
    Rec_all.append(rec_best)
    # precision=找到的真的错误的/找到的错误的
    # recall=找到的真的错误的/所有错误的
    """ compute metircs"""

    progressbar[0] += 1  # update progress bar
    progressbar[1].update(progressbar[0])

    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    print("All best precision为：%.2f%%" % (mean_pre * 100))
    print("All best recall：%.2f%%" % (mean_rec * 100))
    print("All best F1：%.2f%%" % (mean_f1 * 100))
    nni.report_final_result(mean_f1)
    return mean_f1

def get_feature(path="data_dict.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    feature = list(data["feature"])
    label = np.array(list(data["label"]))
    y = np.argmax(label, axis=1)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    code_tokens = [tokenizer.tokenize(src) for src in feature]

    seqs = [tokenizer.convert_tokens_to_ids(src) for src in code_tokens]
    pad_val = len(seqs)
    encoded = np.array(pad_sequences(seqs, maxlen=100, value=pad_val))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded]), y



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

    # print("Evaluating Deeptune et al. ...", file=sys.stderr)


    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    # deeptune_model.model.summary()
    # print("Evaluating DeepTune ...", file=sys.stderr)
    deeptune = evaluate( args)
    # deeptune.groupby('Platform')['Platform', 'Speedup', 'Oracle'].mean()


main()
