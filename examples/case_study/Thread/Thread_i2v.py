#%%

# Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
# Department of Computer Science and Engineering, IIT Hyderabad
#
# This software is available under the BSD 4-Clause License. Please see LICENSE
# file in the top-level directory for more details.
#
import pandas as pd
import numpy as np
import heapq
import sys, re
from sklearn.model_selection import KFold
import os
import xgboost as xgb
from scipy.stats import gmean
sys.path.append('/home/huanting/PROM')
sys.path.append('./case_study/Thread')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
from src.prom.prom_util import Prom_utils

#%%

# Check the data

#%%

assert (
    os.path.exists("../../../benchmark/Thread/kernels_ir")
    and os.path.exists("../../../benchmark/Thread/pact-2014-oracles.csv")
    and os.path.exists("../../../benchmark/Thread/pact-2014-runtimes.csv")
), "Dataset is not present. Please down load"

#%%

assert os.path.exists("../../../benchmark/Thread/output/embeddings"), "Embeddings are not generated"

#%% md

# Read data from input file

#%%

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

#%%

_FLAG_TO_DEVICE_NAME = {
    "Cypress": "AMD Radeon HD 5900",
    "Tahiti": "AMD Tahiti 7970",
    "Fermi": "NVIDIA GTX 480",
    "Kepler": "NVIDIA Tesla K20c",
}

device_list = ["Tahiti"]

oracle_file = os.path.join("../../../benchmark/Thread/pact-2014-oracles.csv")
oracles = pd.read_csv(oracle_file)

runtimes_file = os.path.join("../../../benchmark/Thread/pact-2014-runtimes.csv")
df = pd.read_csv(runtimes_file)

#%% md

# Results from other works

# The accuracies and speedups are taken from the results quoted by NCC in their work for the purpose of comparison. For detailed analysis (discussed later), we run these models and the obtained results are stored as pickle files in ./data/prior_art_results.

#%%

magni_sp_vals = [1.21, 1.01, 0.86, 0.94]
magni_sp_mean = [1.005]
deeptune_sp_vals = [1.10, 1.05, 1.10, 0.99]
deeptune_sp_mean = [1.06]
deeptuneTL_sp_vals = [1.17, 1.23, 1.14, 0.93]
deeptuneTL_sp_mean = [1.1175]
ncc_sp_vals = [1.29, 1.07, 0.97, 1.01]
ncc_sp_mean = [1.086]

#%%

cfs = np.array([1, 2, 4, 8, 16, 32])
kernel_freq = df["kernel"].value_counts().sort_index().reset_index()

#%% md

# Classification Model

#%%
import math




def find_runtime(df, kernel, cf, platform):
    filter1 = df["kernel"] == kernel
    filter2 = df["cf"] == cf
    return df.where(filter1 & filter2)["runtime_" + platform].dropna()

import random
def data_partitioning(platform='', mode='train', calibration_ratio=0.2, args=None):
        pd.set_option('display.max_rows', 5)
        df = pd.read_csv("../../../benchmark/Thread/pact-2014-runtimes.csv")
        oracles = pd.read_csv("../../../benchmark/Thread/pact-2014-oracles.csv")
        data = []

        # load data
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        #
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

        random.seed(args.seed)
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
        #
        lists = []
        # 遍历字典，将每个键的值单独添加到列表中
        for values in device_indices.values():
            lists.append(values)
        random.seed(args.seed)
        if mode == 'train':
            lists = lists[0] + lists[1] + lists[2]
            random.shuffle(lists)
            train_index = lists[:-4]
            valid_index = lists[-4:-2]
            test_index = lists[-4:]
        elif mode == 'test':
            lists_tandc = lists[0] + lists[1]
            random.shuffle(lists_tandc)
            train_index = lists_tandc[:-2]
            valid_index = lists_tandc[-2:]
            test_index = lists[2]
        return train_index, valid_index, test_index

def train(max_depth, learning_rate, n_estimators,args):
    inferencetime = []
    raw_embeddings_pd = pd.DataFrame(raw_embeddings, columns=range(1, 301))
    efileNum = pd.DataFrame(fileIndex)
    embeddings = pd.concat([efileNum, raw_embeddings_pd], axis=1)

    llfiles = pd.read_csv("../../../benchmark/Thread/all.txt", sep="\s+")
    fileNum = llfiles["FileNum"]
    filesname = llfiles["ProgramName"]

    oracles["kernel_path"] = str("./") + oracles["kernel"] + str(".ll")

    df["kernel_path"] = str("./") + df["kernel"] + str(".ll")

    resultant_data = pd.DataFrame()
    for i, platform in enumerate(device_list):
        embeddingsData_tmp = embeddings
        embeddingsData_tmp = embeddingsData_tmp.merge(
            llfiles, left_on=0, right_on="FileNum"
        )
        embeddingsData_tmp = pd.merge(
            embeddingsData_tmp, oracles, left_on="ProgramName", right_on="kernel_path"
        )
        embeddingsData_tmp["cf"] = embeddingsData_tmp["cf_" + platform]
        embeddingsData_tmp["device"] = i + 1
        resultant_data = pd.concat([resultant_data, embeddingsData_tmp])

    resultant_data = pd.get_dummies(resultant_data, columns=["device"])
    resultant_data.reset_index(inplace=True)

    targetLabel = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)

    data = resultant_data
    data = data.drop(
        columns=[
            "index",
            0,
            "FileNum",
            "ProgramName",
            "kernel",
            "cf_Fermi",
            "runtime_Fermi",
            "cf_Kepler",
            "runtime_Kepler",
            "cf_Cypress",
            "runtime_Cypress",
            "cf_Tahiti",
            "runtime_Tahiti",
            "kernel_path",
            "cf",
        ]
    )
    embeddings = (data - data.min()) / (data.max() - data.min())
    embeddings = np.array(embeddings)
    data = []

    train_index, valid_index, test_index = \
        data_partitioning(platform=platform, mode='train', calibration_ratio=0.2, args=args)
    # kf = KFold(n_splits=len(targetLabel), shuffle=False)
    ##############
    actual_labels = np.array([1., 2., 4., 8., 16., 32.])
    expected_labels = np.array([0, 1, 2, 3, 4, 5])
    label_mapping = {old: new for old, new in zip(actual_labels, expected_labels)}
    mapping = {expected: actual for expected, actual in zip(expected_labels, actual_labels)}
    y = targetLabel
    targetLabel = np.array([label_mapping[val] for val in y])
    train_index=targetLabel[train_index]


    ###############
    kernel = sorted(set(df["kernel"]))[test_index[0] % 17]
    gbc = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        n_jobs=10,
        random_state=args.seed
    )

    a=targetLabel[train_index]
    gbc.fit(embeddings[train_index], targetLabel[train_index])


    # test
    prediction = gbc.predict(embeddings[test_index])
    data_distri=[]
    for prediction_single in prediction:
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        prediction_single = min(
            prediction_single, 2 ** (kernel_freq["kernel"][test_index[0] % 17] - 1)
        )
        # 再次转换 targetLabel 回实际标签
        prediction_single = mapping[prediction_single]
        oracle = targetLabel[test_index[0]]

        rt_baseline = float(find_runtime(df, kernel, 1, platform))
        # a=find_runtime(df, kernel, prediction, platform)
        rt_pred = float(find_runtime(df, kernel, prediction_single, platform))
        # a=oracle_runtimes[test_index[0] % 17]
        rt_oracle = float(oracle_runtimes[test_index[0] % 17])
        data_distri.append(rt_oracle / rt_pred)
        data.append(
            {
                "Model": "IR2vec",
                "Platform": _FLAG_TO_DEVICE_NAME[platform],
                "Kernel": kernel,
                "Oracle-CF": oracle,
                "Predicted-CF": prediction,
                "Speedup": rt_baseline / rt_pred,
                "Oracle": rt_oracle / rt_pred,
                "OracleSpeedUp": rt_baseline / rt_oracle,
            }
        )
        ir2vec = pd.DataFrame(
            data,
            columns=[
                "Model",
                "Platform",
                "Kernel",
                "Oracle-CF",
                "Predicted-CF",
                "Speedup",
                "Oracle",
                "OracleSpeedUp",
            ],
        )

    # print("\nSpeedup Matrix: IR2Vec Vs. others\n")
    ir2vec_sp_vals = ir2vec.groupby(["Platform"])["Oracle"].mean().values
    # ir2vec_sp_mean = ir2vec_sp_vals.mean()



    print("training finished")

    model_dir_path = "logs/train/models/i2v/"
    model_path = f"logs/train/models/i2v/{platform}-{args.seed}-{ir2vec_sp_vals[0]}.bin"
    plot_figure_path = 'logs/train/figs/i2v/plot'
    plot_figuredata_path = 'logs/train/figs/i2v/data'
    os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
    gbc.save_model(model_path)

    ###########
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.boxplot(data_distri)
    data_df = pd.DataFrame({'Data': data_distri})
    sns.violinplot(data=data_df, y='Data')
    seed_save = str(int(args.seed))
    plt.title('Box Plot Example ' + seed_save)
    plt.ylabel('Values')
    plt.savefig(plot_figure_path + str(ir2vec_sp_vals[0]) + '_' + str(seed_save) + '.png')
    data_df.to_pickle(plot_figuredata_path + str(ir2vec_sp_vals[0]) + '_' + str(seed_save) + '_data.pkl')
    # plt.show()

    print("The trained model performance is",ir2vec_sp_vals)
    nni.report_final_result(ir2vec_sp_vals[0])




    # sp_df = pd.DataFrame(
    #     {
    #         "Magni et al.": magni_sp_vals + magni_sp_mean,
    #         "DeepTune": deeptune_sp_vals + deeptune_sp_mean,
    #         "DeepTune-TL": deeptuneTL_sp_vals + deeptuneTL_sp_mean,
    #         "NCC": ncc_sp_vals + ncc_sp_mean,
    #         "IR2Vec": list(ir2vec_sp_vals) + [ir2vec_sp_mean],
    #     },
    #     index=[
    #         "AMD Radeon HD 5900",
    #         "AMD Tahiti 7970",
    #         "NVIDIA GTX 480",
    #         "NVIDIA Tesla K20c",
    #         "Average",
    #     ],
    # )
    # print(sp_df)```


def deploy(max_depth, learning_rate, n_estimators,args):
    inferencetime = []
    raw_embeddings_pd = pd.DataFrame(raw_embeddings, columns=range(1, 301))
    efileNum = pd.DataFrame(fileIndex)
    embeddings = pd.concat([efileNum, raw_embeddings_pd], axis=1)

    llfiles = pd.read_csv("../../../benchmark/Thread/all.txt", sep="\s+")
    fileNum = llfiles["FileNum"]
    filesname = llfiles["ProgramName"]

    oracles["kernel_path"] = str("./") + oracles["kernel"] + str(".ll")

    df["kernel_path"] = str("./") + df["kernel"] + str(".ll")

    resultant_data = pd.DataFrame()
    for i, platform in enumerate(device_list):
        embeddingsData_tmp = embeddings
        embeddingsData_tmp = embeddingsData_tmp.merge(
            llfiles, left_on=0, right_on="FileNum"
        )
        embeddingsData_tmp = pd.merge(
            embeddingsData_tmp, oracles, left_on="ProgramName", right_on="kernel_path"
        )
        embeddingsData_tmp["cf"] = embeddingsData_tmp["cf_" + platform]
        embeddingsData_tmp["device"] = i + 1
        resultant_data = pd.concat([resultant_data, embeddingsData_tmp])

    resultant_data = pd.get_dummies(resultant_data, columns=["device"])
    resultant_data.reset_index(inplace=True)

    # targetLabel = np.array(resultant_data["cf"])
    targetLabel = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
    data = resultant_data
    data = data.drop(
        columns=[
            "index",
            0,
            "FileNum",
            "ProgramName",
            "kernel",
            "cf_Fermi",
            "runtime_Fermi",
            "cf_Kepler",
            "runtime_Kepler",
            "cf_Cypress",
            "runtime_Cypress",
            "cf_Tahiti",
            "runtime_Tahiti",
            "kernel_path",
            "cf",
        ]
    )
    embeddings = (data - data.min()) / (data.max() - data.min())
    embeddings = np.array(embeddings)
    data = []

    train_index, valid_index, test_index = \
        data_partitioning(platform=platform, mode='test', calibration_ratio=0.2, args=args)


    # kf = KFold(n_splits=len(targetLabel), shuffle=False)
    ##############
    actual_labels = np.array([1., 2., 4., 8., 16., 32.])
    expected_labels = np.array([0, 1, 2, 3, 4, 5])
    label_mapping = {old: new for old, new in zip(actual_labels, expected_labels)}
    mapping = {expected: actual for expected, actual in zip(expected_labels, actual_labels)}
    y = targetLabel
    targetLabel = np.array([label_mapping[val] for val in y])

    cal_x = embeddings[valid_index]
    calibration_data = cal_x
    cal_y = targetLabel[valid_index]
    test_x = embeddings[test_index]
    test_y = targetLabel[test_index]
    train_y = targetLabel[train_index]
    print("TRain",train_y)
    print("Test",test_y)
    print("Cal",cal_y)
    ###############
    kernel = sorted(set(df["kernel"]))[test_index[0] % 17]
    gbc = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        n_jobs=10,
        random_state=args.seed
    )

    gbc.fit(embeddings[train_index], targetLabel[train_index])

    # test
    prediction = gbc.predict(embeddings[test_index])
    for prediction_single in prediction:
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        prediction_single = min(
            prediction_single, 2 ** (kernel_freq["kernel"][test_index[0] % 17] - 1)
        )
        # 再次转换 targetLabel 回实际标签
        prediction_single = mapping[prediction_single]
        oracle = targetLabel[test_index[0]]

        rt_baseline = float(find_runtime(df, kernel, 1, platform))
        # a=find_runtime(df, kernel, prediction, platform)
        rt_pred = float(find_runtime(df, kernel, prediction_single, platform))
        # a=oracle_runtimes[test_index[0] % 17]
        rt_oracle = float(oracle_runtimes[test_index[0] % 17])
        data.append(
            {
                "Model": "IR2vec",
                "Platform": _FLAG_TO_DEVICE_NAME[platform],
                "Kernel": kernel,
                "Oracle-CF": oracle,
                "Predicted-CF": prediction,
                "Speedup": rt_baseline / rt_pred,
                "Oracle": rt_oracle / rt_pred,
                "OracleSpeedUp": rt_baseline / rt_oracle,
            }
        )
        ir2vec = pd.DataFrame(
            data,
            columns=[
                "Model",
                "Platform",
                "Kernel",
                "Oracle-CF",
                "Predicted-CF",
                "Speedup",
                "Oracle",
                "OracleSpeedUp",
            ],
        )

    # print("\nSpeedup Matrix: IR2Vec Vs. others\n")
    ir2vec_sp_vals_origin = ir2vec.groupby(["Platform"])["Oracle"].mean().values
    # ir2vec_sp_mean = ir2vec_sp_vals.mean()

    print("The trained model performance is:",ir2vec_sp_vals_origin)

    """conformal prediction"""
    # Conformal Prediction
    # the underlying model
    print(f"Start conformal prediction on {platform}...")
    clf = gbc
    # the prom parameters
    method_params = {
        "lac": ("score", True),
        "top_k": ("top_k", True),
        "aps": ("cumulated_score", True),
        "raps": ("raps", True)
    }
    # the prom object

    Prom_thread = Prom_utils(clf, method_params, task="thread")
    # conformal prediction
    y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
        cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")

    # evaluate conformal prediction
    Prom_thread.evaluate_mapie \
        (y_preds=y_preds,
         y_pss=y_pss,
         p_value=p_value,
         all_pre=prediction,
         y=y[test_index],
         significance_level=0.05)

    Prom_thread.evaluate_rise \
        (y_preds=y_preds,
         y_pss=y_pss,
         p_value=p_value,
         all_pre=prediction,
         y=y[test_index],
         significance_level=0.05)

    # evaluate conformal prediction
    index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all, _, _ \
        = Prom_thread.evaluate_conformal_prediction \
        (y_preds=y_preds,
         y_pss=y_pss,
         p_value=p_value,
         all_pre=prediction,
         y=y[test_index],
         significance_level=0.05)

    # Increment learning
    print("Finding the most valuable instances for incremental learning...")
    train_index, test_index = Prom_thread.incremental_learning \
        (args.seed, test_index, train_index)
    # retrain the model
    print(f"Retraining the model on {platform}...")
    gbc.fit(embeddings[train_index], targetLabel[train_index])

    # test
    prediction = gbc.predict(embeddings[test_index])
    data_distri=[]
    for prediction_single in prediction:
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        prediction_single = min(
            prediction_single, 2 ** (kernel_freq["kernel"][test_index[0] % 17] - 1)
        )
        # 再次转换 targetLabel 回实际标签
        prediction_single = mapping[prediction_single]
        oracle = targetLabel[test_index[0]]

        rt_baseline = float(find_runtime(df, kernel, 1, platform))
        # a=find_runtime(df, kernel, prediction, platform)
        rt_pred = float(find_runtime(df, kernel, prediction_single, platform))
        # a=oracle_runtimes[test_index[0] % 17]
        rt_oracle = float(oracle_runtimes[test_index[0] % 17])
        data_distri.append(rt_oracle / rt_pred)
        data.append(
            {
                "Model": "IR2vec",
                "Platform": _FLAG_TO_DEVICE_NAME[platform],
                "Kernel": kernel,
                "Oracle-CF": oracle,
                "Predicted-CF": prediction,
                "Speedup": rt_baseline / rt_pred,
                "Oracle": rt_oracle / rt_pred,
                "OracleSpeedUp": rt_baseline / rt_oracle,
            }
        )
        ir2vec = pd.DataFrame(
            data,
            columns=[
                "Model",
                "Platform",
                "Kernel",
                "Oracle-CF",
                "Predicted-CF",
                "Speedup",
                "Oracle",
                "OracleSpeedUp",
            ],
        )

    ir2vec_sp_vals = ir2vec.groupby(["Platform"])["Oracle"].mean().values
    improved = ir2vec_sp_vals[0] - ir2vec_sp_vals_origin[0]
    print("The improved performance is:",improved)

    model_dir_path = "logs/deploy/models/i2v/"
    model_path = f"logs/deploy/models/i2v/{platform}-{args.seed}-{ir2vec_sp_vals[0]}.bin"
    plot_figure_path = 'logs/deploy/figs/i2v/plot'
    plot_figuredata_path = 'logs/deploy/figs/i2v/data'
    os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
    gbc.save_model(model_path)

    ###########
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.boxplot(data_distri)
    data_df = pd.DataFrame({'Data': data_distri})
    sns.violinplot(data=data_df, y='Data')
    seed_save = str(int(args.seed))
    plt.title('Box Plot Example ' + seed_save)
    plt.ylabel('Values')
    plt.savefig(plot_figure_path + str(ir2vec_sp_vals[0]) + '_' + str(seed_save) + '.png')
    data_df.to_pickle(plot_figuredata_path + str(ir2vec_sp_vals[0]) + '_' + str(seed_save) + '_data.pkl')
    # plt.show()

    print("The trained model performance is", ir2vec_sp_vals)

    nni.report_final_result(improved)



import nni
import argparse

def load_args():
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "seed": 11,
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

if __name__ == '__main__':
    raw_embeddings, fileIndex = readEmd_program(
        "../../../benchmark/Thread/output/embeddings/Thread_Coarsening_FlowAware_llvm17.txt"
    )
    args=load_args()

    if args.mode == 'train':
        train(max_depth=1, learning_rate=0.05, n_estimators=140,args=args)
    elif args.mode == 'deploy':
        deploy(max_depth=1, learning_rate=0.05, n_estimators=140, args=args)
    # train(max_depth=1, learning_rate=0.05, n_estimators=140, args=args)
    # deploy(max_depth=1, learning_rate=0.05, n_estimators=140, args=args)
    #
    # nnictl create --config /home/huanting/PROM/examples/case_study/Thread/config.yaml --port 8088

