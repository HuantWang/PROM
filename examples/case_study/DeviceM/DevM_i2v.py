#%%

# Copyright (c) 2021, S. VenkataKeerthy, Rohit Aggarwal
# Department of Computer Science and Engineering, IIT Hyderabad
#
# This software is available under the BSD 4-Clause License. Please see LICENSE
# file in the top-level directory for more details.
#
import xgboost as xgb
import pandas as pd
import numpy as np
import sys, re
from sklearn.model_selection import KFold
import os
from scipy.stats.mstats import gmean

#%%

# assert (
#     os.path.exists("data/kernels_ir")
#     and os.path.exists("data/cgo17-amd.csv")
#     and os.path.exists("data/cgo17-nvidia.csv")
# ), "Dataset is not present. Please download"

#%%

# assert os.path.exists("output/embeddings"), "Embeddings are not generated"

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

#%% md

# Results from other works

# The accuracies and speedups are taken from the results quoted by NCC in their work for the purpose of comparison. For detailed analysis (discussed later), we run these models and the obtained results are stored as pickle files in ./data/prior_art_results.

#%%

static_pred_vals = [58.823529, 56.911765]
static_pred_mean = [57.867647]
static_sp_vals = [1.0, 1.0]
static_sp_mean = [1.0]
grewe_pred_vals = [73.382353, 72.941176]
grewe_pred_mean = [73.161765]
grewe_sp_vals = [2.905822, 1.264801]
grewe_sp_mean = [2.085312]
deeptune_pred_vals = [83.676471, 80.294118]
deeptune_pred_mean = [81.985294]
deeptune_sp_vals = [3.335612, 1.412222]
deeptune_sp_mean = [2.373917]
ncc_pred_vals = [82.79, 81.76]
ncc_pred_mean = [82.275]
ncc_sp_vals = [3.42, 1.39]
ncc_sp_mean = [2.405]

llfiles = pd.read_csv("../../../benchmark/DeviceM/all.txt", sep="\s+")
fileNum = llfiles["FileNum"]
filesname = llfiles["ProgramName"]

device_dict = {"amd": "AMD Tahiti 7970", "nvidia": "NVIDIA GTX 970"}

#%% md

# Classification Model

#%%

def evaluate(max_depth=4, learning_rate=0.1, n_estimators=200, seed=204):
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

        # 10-fold cross-validation
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for j, (train, test) in enumerate(kf.split(embeddings, targetLabel)):

            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                n_jobs=10,
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
                        "Speedup": rt_baseline / rt_pred,
                        "OracleSpeedUp": rt_baseline / rt_oracle,
                    }
                )
        ir2vec = pd.DataFrame(data, index=range(1, len(data) + 1))

    # print("Accuracy Matrix: IR2Vec Vs. others\n")
    # ir2vec_pred_vals = ir2vec.groupby(["Platform"])["Correct?"].mean().values * 100
    # ir2vec_pred_mean = ir2vec_pred_vals.mean()
    # accuracy_df = pd.DataFrame(
    #     {
    #         "Static Mapping": static_pred_vals + static_pred_mean,
    #         "Grewe et al.": grewe_pred_vals + grewe_pred_mean,
    #         "DeepTune": deeptune_pred_vals + deeptune_pred_mean,
    #         "NCC": ncc_pred_vals + ncc_pred_mean,
    #         "IR2Vec": list(ir2vec_pred_vals) + [ir2vec_pred_mean],
    #     },
    #     index=["AMD Tahiti 7970", "NVIDIA GTX 970", "Average"],
    # )
    # print(accuracy_df)

    print("\nSpeedup Matrix: IR2Vec Vs. others\n")
    ir2vec_sp_vals = ir2vec.groupby(["Platform"])["Speedup"].mean().values
    ir2vec_sp_mean = ir2vec_sp_vals.mean()
    sp_df = pd.DataFrame(
        {
            "Static Mapping": static_sp_vals + static_sp_mean,
            "Grewe et al.": grewe_sp_vals + grewe_sp_mean,
            "DeepTune": deeptune_sp_vals + deeptune_sp_mean,
            "NCC": ncc_sp_vals + ncc_sp_mean,
            "IR2Vec": list(ir2vec_sp_vals) + [ir2vec_sp_mean],
        },
        index=["AMD Tahiti 7970", "NVIDIA GTX 970", "Average"],
    )
    print(sp_df)

    return ir2vec

#%% md

# IR2Vec Symbolic Vs. Others

#%%

raw_embeddings, fileIndexNum = readEmd_program(
    "../../../benchmark/DeviceM/output/embeddings/Device_Mapping_FlowAware_llvm17.txt"
)
ir2vec_sym = evaluate(max_depth=10, learning_rate=0.5, n_estimators=70, seed=104)

# #%% raw
# # Expected Results
# Accuracy Matrix: IR2Vec Vs. others
#
#                  Static Mapping  Grewe et al.   DeepTune     NCC     IR2Vec
# AMD Tahiti 7970       58.823529     73.382353  83.676471  82.790  90.284006
# NVIDIA GTX 970        56.911765     72.941176  80.294118  81.760  87.144993
# Average               57.867647     73.161765  81.985294  82.275  88.714499
#
# Speedup Matrix: IR2Vec Vs. others
#
#                  Static Mapping  Grewe et al.  DeepTune    NCC    IR2Vec
# AMD Tahiti 7970             1.0      2.905822  3.335612  3.420  3.471963
# NVIDIA GTX 970              1.0      1.264801  1.412222  1.390  1.433372
# Average                     1.0      2.085312  2.373917  2.405  2.452667
#%% md

# IR2Vec Flow-Aware Vs. Others

#%%

raw_embeddings, fileIndexNum = readEmd_program(
    "../../../benchmark/DeviceM/output/embeddings/Device_Mapping_FlowAware_llvm17.txt"
)
ir2vec_fa = evaluate(max_depth=10, learning_rate=0.5, n_estimators=70, seed=104)

#%% raw
# Expected Results
# Accuracy Matrix: IR2Vec Vs. others
#
#                  Static Mapping  Grewe et al.   DeepTune     NCC     IR2Vec
# AMD Tahiti 7970       58.823529     73.382353  83.676471  82.790  92.825112
# NVIDIA GTX 970        56.911765     72.941176  80.294118  81.760  89.686099
# Average               57.867647     73.161765  81.985294  82.275  91.255605
#
# Speedup Matrix: IR2Vec Vs. others
#
#                  Static Mapping  Grewe et al.  DeepTune    NCC    IR2Vec
# AMD Tahiti 7970             1.0      2.905822  3.335612  3.420  3.510104
# NVIDIA GTX 970              1.0      1.264801  1.412222  1.390  1.467221
# Average                     1.0      2.085312  2.373917  2.405  2.488663
#
# #%% md
#
# # Other related observations
# For the comparison, we use the results obtained on training the earlier works

#%%

# deeptune_res = pd.read_pickle("data/prior_art_results/deeptune_dm.results")
# grewe_res = pd.read_pickle("data/prior_art_results/grewe_dm.results")
# static_res = pd.read_pickle("data/prior_art_results/static_dm.results")
# ncc_res = pd.read_pickle("data/prior_art_results/ncc_fix_DM.results")

#%% md

## Speedup comparisons

#%%

def calcSpeedup(platform):
    # grewe_geomean = gmean(
    #     grewe_res[grewe_res["Platform"] == platform]["Speedup"].values
    # )
    # deeptune_geomean = gmean(
    #     deeptune_res[deeptune_res["Platform"] == platform]["Speedup"].values
    # )
    # ncc_geomean = gmean(ncc_res[ncc_res["Platform"] == platform]["Speedup"].values)
    ir2vec_sym_geomean = gmean(
        ir2vec_sym[ir2vec_sym["Platform"] == platform]["Speedup"].values
    )
    ir2vec_fa_geomean = gmean(
        ir2vec_fa[ir2vec_fa["Platform"] == platform]["Speedup"].values
    )

    # print(f"Geometric mean of Grewe et al. {grewe_geomean:.2f}x")
    # print(f"Geometric mean of DeepTune {deeptune_geomean:.2f}x")
    # print(f"Geometric mean of Inst2Vec {ncc_geomean:.2f}x")
    # print(f"Geometric mean of IR2Vec Symbolic {ir2vec_sym_geomean:.3f}x")
    # print(f"Geometric mean of IR2Vec Flow-Aware {ir2vec_fa_geomean:.3f}x")

    return (
        # grewe_geomean,
        # deeptune_geomean,
        # ncc_geomean,
        ir2vec_sym_geomean,
        ir2vec_fa_geomean,
    )

#%% md

### On AMD Tahiti 7970

#%%

tah_ir2vSym, tah_ir2vFA = calcSpeedup("AMD Tahiti 7970")

#%% md

### On NVIDIA GTX 970

#%%

gtx_ir2vSym, gtx_ir2vFA = calcSpeedup("NVIDIA GTX 970")

#%% md

### On both the platforms

#%%

# grewe_geomean = gmean(grewe_res["Speedup"].values)
# deeptune_geomean = gmean(deeptune_res["Speedup"].values)
# ncc_geomean = gmean(ncc_res["Speedup"].values)
# ir2vec_sym_geomean = gmean(ir2vec_sym["Speedup"].values)
ir2vec_fa_geomean = gmean(ir2vec_fa["Speedup"].values)

# print(f"Geometric mean of Grewe et al. - {grewe_geomean:.2f}x")
# print(f"Geometric mean of DeepTune - {deeptune_geomean:.2f}x")
# print(f"Geometric mean of Inst2Vec - {ncc_geomean:.2f}x")
# print(f"Geometric mean of IR2Vec Symbolic {ir2vec_sym_geomean:.2f}x")
print(f"Geometric mean of IR2Vec Flow-Aware {ir2vec_fa_geomean:.2f}x")

#%% md

# Percentage of increase in speedup by IR2Vec Flow-Aware encodings over others

#%%

def slowDown(value1, value2):
    return round(np.abs(((value2 - value1) / value2) * 100), 2)

#%%

print("\nAMD Tahiti 7970")
# print(" % Increase in SpeedUp over Grewe et al - ", slowDown(tah_ir2vFA, tah_grewe))
# print(" % Increase in SpeedUp over DeepTune - ", slowDown(tah_ir2vFA, tah_dt))
# print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(tah_ir2vFA, tah_ncc))
print(
    " % Increase in SpeedUp over IR2Vec Symbolic - ",
    slowDown(tah_ir2vFA, tah_ir2vSym),
)

print("\nNVIDIA GTX 970")
# print(" % Increase in SpeedUp over Grewe et al - ", slowDown(gtx_ir2vFA, gtx_grewe))
# print(" % Increase in SpeedUp over DeepTune - ", slowDown(gtx_ir2vFA, gtx_dt))
# print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(gtx_ir2vFA, gtx_ncc))
print(
    " % Increase in SpeedUp over IR2Vec Symbolic - ",
    slowDown(gtx_ir2vFA, gtx_ir2vSym),
)

#%% md

## Accuracy Comparisons

#%%

#%% md

### On AMD Tahiti 7970

#%%


#%% md

### On NVIDIA GTX 970

#%%

#%% md

## Percentage of improvement in accuracy obtained by Flow Aware embeddings when compared to other methods

# Calculated based on the reference values taken from https://github.com/spcl/ncc/blob/master/train_task_devmap.py

#%% md

### On AMD Tahiti 7970

#%%

# AMD Tahiti 7970
tah_grewe = 73.382353
tah_dt = 83.676471
tah_ncc = 82.790
tah_nccimm = 88.09

print("\nAMD Tahiti 7970")
# print(" % Increase in SpeedUp over Grewe et al - ", slowDown(tah_ir2vFA, tah_grewe))
# print(" % Increase in SpeedUp over DeepTune - ", slowDown(tah_ir2vFA, tah_dt))
# print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(tah_ir2vFA, tah_ncc))
print(" % Increase in SpeedUp over Inst2Vec-imm - ", slowDown(tah_ir2vFA, tah_nccimm))
print(
    " % Increase in SpeedUp over IR2Vec Symbolic - ",
    slowDown(tah_ir2vFA, tah_ir2vSym),
)

#%% md

### On NVIDIA GTX 970

#%%

# NVIDIA GTX 970
static = 56.911765
gtx_grewe = 72.941176
gtx_dt = 80.294118
gtx_ncc = 81.760
gtx_nccimm = 86.62


print("\nNVIDIA GTX 970")
# print(" % Increase in SpeedUp over Grewe et al - ", slowDown(gtx_ir2vFA, gtx_grewe))
# print(" % Increase in SpeedUp over DeepTune - ", slowDown(gtx_ir2vFA, gtx_dt))
# print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(gtx_ir2vFA, gtx_ncc))
print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(gtx_ir2vFA, gtx_nccimm))
print(
    " % Increase in SpeedUp over IR2Vec Symbolic - ",
    slowDown(gtx_ir2vFA, gtx_ir2vSym),
)

#%% md

### On both the platforms

#%%

dt = 81.99
ncc = 82.275
nccimm = (88.09 + 86.62) / 2
ir2vSym = ir2vec_sym["Correct?"].mean() * 100
ir2vFA = ir2vec_fa["Correct?"].mean() * 100

# print(" % Increase in SpeedUp over DeepTune - ", slowDown(ir2vFA, dt))
# print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(ir2vFA, ncc))
print("")
print("On both the platforms")
print(" % Increase in SpeedUp over Inst2Vec - ", slowDown(ir2vFA, nccimm))
print(
    " % Increase in SpeedUp over IR2Vec Symbolic - ",
    slowDown(ir2vFA, ir2vSym),
)
