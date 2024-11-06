#%%

# Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
# Department of Computer Science and Engineering, IIT Hyderabad
#
# This software is available under the BSD 4-Clause License. Please see LICENSE
# file in the top-level directory for more details.
#
import nni
import xgboost as xgb
import pandas as pd
import numpy as np
import sys, re
from sklearn.model_selection import KFold
import os
from scipy.stats.mstats import gmean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('/cgo/prom/PROM')
sys.path.append('./case_study/DeviceM')
sys.path.append('/cgo/prom/PROM/src')
sys.path.append('/cgo/prom/PROM/thirdpackage')
from src.prom.prom_util import Prom_utils
def readEmd_program(filename):
    lines = [line.strip("\n\t") for line in open(filename)]
    entity = []
    rep = []
    targetLabel = []
    flag = 0
    for line in lines:
        r = line.split("\t")
        targetLabel.append(int(r[0]))
        res = r[1:]
        res_double = [float(val) for val in res]
        rep.append(res_double)
    return rep, targetLabel


static_pred_vals = [58.823529, 56.911765]
static_pred_mean = [57.867647]
static_sp_vals = [1.0, 1.0]
static_sp_mean = [1.0]

llfiles = pd.read_csv("../../../benchmark/DeviceM/all.txt", sep="\s+")
fileNum = llfiles["FileNum"]
filesname = llfiles["ProgramName"]

# device_dict = {"amd": "AMD Tahiti 7970", "nvidia": "NVIDIA GTX 970"}
device_dict = {"amd": "AMD Tahiti 7970"}
#%% md

# Classification Model

#%%

def train(max_depth=4, learning_rate=0.1, n_estimators=200, args=None):

    print("Start training...")
    raw_embeddings, fileIndexNum = readEmd_program(
        "../../../benchmark/DeviceM/output/embeddings/Device_Mapping_FlowAware_llvm17.txt"
    )
    data = []
    rt_label_dict = {"amd": "runtime_cpu", "nvidia": "runtime_gpu"}

    for i, platform in enumerate(device_dict.keys()):
        platform_name = device_dict[platform]

        # Load runtime data
        df = pd.read_csv("../../../benchmark/DeviceM/cgo17-{}.csv".format(platform))
        df["bench_data"] = (
            df.loc[df["dataset"] != "default", "benchmark"]
            + str("_")
            + df.loc[df["dataset"] != "default", "dataset"]
        )

        df.loc[df["dataset"] == "default", "bench_data"] = df.loc[
            df["dataset"] == "default", "benchmark"
        ]
        df["bench_data_path"] = str("./") + df["bench_data"] + str(".ll")

        raw_embeddings_pd = pd.DataFrame(raw_embeddings, columns=range(1, 301))
        efileNum = pd.DataFrame(fileIndexNum)
        embeddings = raw_embeddings_pd
        embeddingsData = pd.concat([efileNum, embeddings], axis=1)
        embeddingsData = embeddingsData.merge(llfiles, left_on=0, right_on="FileNum")

        df = pd.merge(
            embeddingsData, df, left_on="ProgramName", right_on="bench_data_path"
        )
        targetLabel = np.array([1 if x == "GPU" else 0 for x in df["oracle"].values])

        embeddings = df.drop(
            columns=[
                "dataset",
                "comp",
                "rational",
                "mem",
                "localmem",
                "coalesced",
                "atomic",
                "runtime_cpu",
                "runtime_gpu",
                0,
                "src",
                "seq",
                "bench_data",
                "bench_data_path",
                "ProgramName",
                "FileNum",
                "Unnamed: 0",
                "benchmark",
                "oracle",
            ]
        )
        embeddings = (embeddings - embeddings.min()) / (
            embeddings.max() - embeddings.min()
        )
        embeddings = np.array(embeddings)

        from sklearn.model_selection import StratifiedKFold


        # 假设 total_length 是数据集的总长度
        total_length = len(targetLabel)  # 示例长度

        # 计算每个部分的长度
        train_length = int(0.6 * total_length)
        val_length = int(0.2 * total_length)
        test_length = total_length - train_length - val_length  # 防止精度丢失
        # 10-fold cross-validation
        train_indices = int(0.6 * total_length)
        val_indices = int(0.2 * total_length)
        test_indices = total_length - train_length - val_length  # 防止精度丢失

        indices = np.arange(total_length)
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        train = indices[:train_indices]
        val = indices[train_indices:train_indices + val_indices]
        test = indices[train_indices + val_indices:]

        model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_jobs=10,
            random_state=args.seed
        )
        model.fit(embeddings[train], targetLabel[train])
        predictions = model.predict(embeddings[test])

        predictions = [
            "CPU" if prediction == 0 else "GPU" for prediction in predictions
        ]
        test_df = df.iloc[test].reset_index()
        assert test_df.shape[0] == len(predictions)
        test_df = pd.concat(
            [test_df, pd.DataFrame(predictions, columns=["predictions"])], axis=1
        )

        rt_label = rt_label_dict[platform]
        for idx, row in test_df.iterrows():
            oracle = row["oracle"]
            pred = row["predictions"]
            rt_baseline = row[rt_label]
            rt_oracle = (
                row["runtime_cpu"] if oracle == "CPU" else row["runtime_gpu"]
            )
            rt_pred = row["runtime_cpu"] if pred == "CPU" else row["runtime_gpu"]
            data.append(
                {
                    "Model": "IR2vec",
                    "Platform": platform_name,
                    "Oracle Mapping": oracle,
                    "Predicted Mapping": pred,
                    "Correct?": oracle == pred,
                    "Speedup": rt_oracle / rt_pred,
                    "OracleSpeedUp": rt_oracle / rt_pred,
                }
            )
        ir2vec = pd.DataFrame(data, index=range(1, len(data) + 1))

    # print("\nSpeedup Matrix: IR2Vec Vs. others\n")
    ir2vec_sp_vals = ir2vec.groupby(["Platform"])["Speedup"].apply(lambda x: gmean(x)).values
    o_percent_all=ir2vec.groupby(["Platform"])["Speedup"].apply(lambda x: x).values


    ir2vec_sp_mean = ir2vec_sp_vals.mean()

    ###
    plt.boxplot(o_percent_all)
    data_df = pd.DataFrame({'Data': o_percent_all})
    sns.violinplot(data=data_df, y='Data')
    seed_save = str(int(args.seed))
    plt.title('Box Plot Example ' + seed_save)
    plt.ylabel('Values')
    plt.savefig('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_train' +
                str(ir2vec_sp_mean) + '_' + str(seed_save) + '.png')
    data_df.to_pickle('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_train' +
                      str(ir2vec_sp_mean) + '_' + str(seed_save) + '_data.pkl')
    # plt.show()

    sp_df = pd.DataFrame(
        {
            "IR2Vec": list(ir2vec_sp_vals) + [ir2vec_sp_mean],
        },
        index=["AMD Tahiti 7970", "Average"],
    )
    print("The Training performance is:", ir2vec_sp_mean)
    # print(sp_df)
    # nni.report_final_result(ir2vec_sp_mean)

def deploy(max_depth=4, learning_rate=0.1, n_estimators=200, args=None):

    print("Loading dataset and underlying model...")
    raw_embeddings, fileIndexNum = readEmd_program(
        "../../../benchmark/DeviceM/output/embeddings/Device_Mapping_FlowAware_llvm17.txt"
    )
    data = []
    rt_label_dict = {"amd": "runtime_cpu", "nvidia": "runtime_gpu"}
    np.random.seed(args.seed)
    for i, platform in enumerate(device_dict.keys()):
        platform_name = device_dict[platform]

        # Load runtime data
        df = pd.read_csv("../../../benchmark/DeviceM/cgo17-{}.csv".format(platform))
        df["bench_data"] = (
            df.loc[df["dataset"] != "default", "benchmark"]
            + str("_")
            + df.loc[df["dataset"] != "default", "dataset"]
        )

        df.loc[df["dataset"] == "default", "bench_data"] = df.loc[
            df["dataset"] == "default", "benchmark"
        ]
        df["bench_data_path"] = str("./") + df["bench_data"] + str(".ll")

        raw_embeddings_pd = pd.DataFrame(raw_embeddings, columns=range(1, 301))
        efileNum = pd.DataFrame(fileIndexNum)
        embeddings = raw_embeddings_pd
        embeddingsData = pd.concat([efileNum, embeddings], axis=1)
        embeddingsData = embeddingsData.merge(llfiles, left_on=0, right_on="FileNum")

        df = pd.merge(
            embeddingsData, df, left_on="ProgramName", right_on="bench_data_path"
        )
        targetLabel = np.array([1 if x == "GPU" else 0 for x in df["oracle"].values])

        embeddings = df.drop(
            columns=[
                "dataset",
                "comp",
                "rational",
                "mem",
                "localmem",
                "coalesced",
                "atomic",
                "runtime_cpu",
                "runtime_gpu",
                0,
                "src",
                "seq",
                "bench_data",
                "bench_data_path",
                "ProgramName",
                "FileNum",
                "Unnamed: 0",
                "benchmark",
                "oracle",
            ]
        )
        embeddings = (embeddings - embeddings.min()) / (
            embeddings.max() - embeddings.min()
        )
        embeddings = np.array(embeddings)

        from sklearn.model_selection import StratifiedKFold

        ###############
        # 创建一个 defaultdict，用于存储分组后的索引
        from collections import defaultdict
        index_dict = defaultdict(list)

        # 遍历 DataFrame 的每一行，根据第一个'-'之前的部分进行分组
        for idx, value in enumerate(df['bench_data']):
            key = value.split('-')[0]  # 取第一个'-'之前的部分作为键
            index_dict[key].append(idx)  # 将索引添加到相应的分组中

        # 将 defaultdict 转换为普通字典
        index_dict = dict(index_dict)
        # 获取所有的 keys
        keys = list(index_dict.keys())
        # 设置随机种子以确保可重复性
        np.random.seed(args.seed)

        # 打乱 keys 的顺序
        np.random.shuffle(keys)

        # 按照60:20:20的比例分配 keys
        n_total = len(keys)
        n_train = int(0.6 * n_total)
        n_val = int(0.3 * n_total)
        n_test = n_total - n_train - n_val

        train_keys = keys[:n_train]
        val_keys = keys[n_train:n_train + n_val]
        test_keys = keys[n_train + n_val:]

        # 创建用于存储最终的索引的列表
        train = []
        val = []
        test = []

        # 将分配好的 keys 对应的索引列表拼接起来
        for key in train_keys:
            train.extend(index_dict[key])

        for key in val_keys:
            val.extend(index_dict[key])

        for key in test_keys:
            test.extend(index_dict[key])


        model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_jobs=10,
            random_state=args.seed
        )
        model.fit(embeddings[train], targetLabel[train])
        predictions = model.predict(embeddings[test])
        all_pre = predictions
        predictions = [
            "CPU" if prediction == 0 else "GPU" for prediction in predictions
        ]
        test_df = df.iloc[test].reset_index()
        assert test_df.shape[0] == len(predictions)
        test_df = pd.concat(
            [test_df, pd.DataFrame(predictions, columns=["predictions"])], axis=1
        )

        rt_label = rt_label_dict[platform]
        for idx, row in test_df.iterrows():
            oracle = row["oracle"]
            pred = row["predictions"]
            rt_baseline = row[rt_label]
            rt_oracle = (
                row["runtime_cpu"] if oracle == "CPU" else row["runtime_gpu"]
            )
            rt_pred = row["runtime_cpu"] if pred == "CPU" else row["runtime_gpu"]
            data.append(
                {
                    "Model": "IR2vec",
                    "Platform": platform_name,
                    "Oracle Mapping": oracle,
                    "Predicted Mapping": pred,
                    "Correct?": oracle == pred,
                    "Speedup": rt_oracle / rt_pred,
                    "OracleSpeedUp": rt_oracle / rt_pred,
                }
            )
        ir2vec = pd.DataFrame(data, index=range(1, len(data) + 1))

    # print("\nSpeedup Matrix: IR2Vec Vs. others\n")
    ir2vec_sp_vals = ir2vec.groupby(["Platform"])["Speedup"].apply(lambda x: gmean(x)).values
    ir2vec_sp_mean = ir2vec_sp_vals.mean()
    sp_df = pd.DataFrame(
        {
            "IR2Vec": list(ir2vec_sp_vals) + [ir2vec_sp_mean],
        },
        index=["AMD Tahiti 7970", "Average"],
    )
    original_performance = ir2vec_sp_mean
    print("The performance of underlying model during deployment phase is", ir2vec_sp_mean)

    o_percent_all = ir2vec.groupby(["Platform"])["Speedup"].apply(lambda x: x).values
    ###
    plt.boxplot(o_percent_all)
    data_df = pd.DataFrame({'Data': o_percent_all})
    sns.violinplot(data=data_df, y='Data')
    seed_save = str(int(args.seed))
    plt.title('Box Plot Example ' + seed_save)
    plt.ylabel('Values')
    plt.savefig('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_deploy' +
                str(ir2vec_sp_mean) + '_' + str(seed_save) + '.png')
    data_df.to_pickle('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_deploy' +
                      str(ir2vec_sp_mean) + '_' + str(seed_save) + '_data.pkl')
    # plt.show()

    """conformal prediction"""
    # Conformal Prediction
    # the underlying model
    print(f"Start conformal prediction on {platform}...")
    clf = model
    # the prom parameters
    method_params = {
        "lac": ("score", True),
        "top_k": ("top_k", True),
        "aps": ("cumulated_score", True),
        "raps": ("raps", True)
    }
    # the prom object

    calibration_data=embeddings[val]
    cal_y=targetLabel[val]
    test_x = embeddings[test]
    test_y = targetLabel[test]

    Prom_thread = Prom_utils(clf, method_params, task="thread")
    # conformal prediction
    y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
        cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")

    # evaluate conformal prediction
    Prom_thread.evaluate_conformal_cd \
        (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=test_y, significance_level='auto')



    # evaluate conformal prediction
    # index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all, _, _ \
    #     = Prom_thread.evaluate_conformal_prediction \
    #     (y_preds=y_preds,
    #      y_pss=y_pss,
    #      p_value=p_value,
    #      all_pre=all_pre,
    #      y=test_y,
    #      significance_level='auto')



import nni
import argparse
def load_args(mode):
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {} and mode == 'train':
        params = {
            "seed": 6794,
        }
    elif params == {} and mode == 'deploy':
        params = {
            "seed": 8882,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--mode', choices=['train', 'deploy'],  help="Mode to run: train or deploy")
    args = parser.parse_args()

    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args

def ae_dev_i2v():
    print("_________Start training phase________")
    args = load_args("train")
    train(max_depth=10, learning_rate=0.5, n_estimators=70, args=args)

    print("_________Start deployment phase________")
    args = load_args("deploy")
    deploy(max_depth=10, learning_rate=0.5, n_estimators=70, args=args)

# if __name__ == '__main__':
#     args=load_args()
#     raw_embeddings, fileIndexNum = readEmd_program(
#         "../../../benchmark/DeviceM/output/embeddings/Device_Mapping_FlowAware_llvm17.txt"
#     )
    # if args.mode == 'train':
    #     train(max_depth=10, learning_rate=0.5, n_estimators=70,args=args)
    # elif args.mode == 'deploy':
    #     deploy(max_depth=10, learning_rate=0.5, n_estimators=70,args=args)
    # train(max_depth=10, learning_rate=0.5, n_estimators=70, args=args)
    # deploy(max_depth=10, learning_rate=0.5, n_estimators=70, args=args)

    # nnictl create --config /cgo/prom/PROM/examples/case_study/DeviceM/config.yml --port 8088
