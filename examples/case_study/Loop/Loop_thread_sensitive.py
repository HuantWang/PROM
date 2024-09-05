import sys
import os
import warnings
import pickle
# from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
import random
sys.path.append('./case_study/Loop')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
from Deeptune_utils import DeepTune, LoopT, deeptune_make_prediction, deeptune_make_prediction_il

import numpy as np
import nni
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from prom_util_sensitive import Prom_utils


def load_deeptune_args():
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "epoch": 20,
            "batch_size": 128,
            "seed": 6116,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=params['epoch'],
                        help="random seed for initialization")
    parser.add_argument("--batch_size", default=params['batch_size'], type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--mode', choices=['train', 'deploy'], help="Mode to run: train or deploy")
    args = parser.parse_args()

    args.seed=int(random.randint(0, 10000))
    print("seeds is", args.seed)
    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args


def loop_deploy_de(args):
    # load args

    prom_loop = LoopT(model=DeepTune())
    deeptune_model = DeepTune()

    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []
    deeptune_model.init(args)

    # Load dataset
    # print(f"*****************{platform}***********************")
    print(f"Loading dataset")
    X_seq, y_1hot, time, train_index, valid_index, test_index = \
        prom_loop.data_partitioning(dataset=r'../../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1, args=args)
    train_index = train_index[:1000]
    valid_index = valid_index[:400]
    test_index = test_index[:200]
    # train_index = train_index[:100]
    # valid_index = valid_index[:100]
    # test_index = test_index[:100]
    #  init the model
    # print(f"Loading underlying model...")
    prom_loop.model.init(args)
    seed_value = int(args.seed)
    prom_loop.model.train(
        sequences=X_seq[train_index], verbose=False, y_1hot=y_1hot[train_index]
    )
    # save the model
    # model_dir_path = "logs/train/models/ma/"
    plot_figure_path = 'logs/train/figs/de/plot'
    plot_figuredata_path = 'logs/train/figs/de/data'
    # os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)

    # load the model
    # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
    # make prediction
    origin_speedup, all_pre, data_distri = deeptune_make_prediction \
        (model=deeptune_model, X_seq=X_seq, y_1hot=y_1hot,
         time=time, test_index=test_index)
    # print(f"Loading successful, the speedup is {origin_speedup}")
    # nni.report_final_result(origin)

    # Conformal Prediction
    # the underlying model
    print(f"Start conformal prediction...")
    clf = prom_loop.model
    # the prom parameters
    method_params = {
        "lac": ("score", True),
        "top_k": ("top_k", True),
        "aps": ("cumulated_score", True),
        "raps": ("raps", True)
    }
    # the prom object
    Prom_thread = Prom_utils(clf, method_params, task="loop")
    # conformal prediction

    calibration_data = X_seq[valid_index]
    cal_y = y_1hot[valid_index]
    # cal_y = np.argmax(y_1hot[valid_index], axis=1)

    test_x = X_seq[test_index]
    test_y = y_1hot[test_index]
    y = np.argmax(y_1hot[test_index], axis=1)  # the true label with 0/1/2/3/4
    y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
        cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")

    # evaluate conformal prediction
    # MAPIE
    # Prom_thread.evaluate_mapie \
    #     (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y,
    #      significance_level=0.05)

    # Prom_thread.evaluate_rise \
    #     (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y,
    #      significance_level=0.05)

    index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all, _, _,alpha \
        = Prom_thread.evaluate_conformal_prediction \
        (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y,significance_level='auto')

    return Acc_all, F1_all, Pre_all, Rec_all, alpha,seed_value


def loop_train_de(args):
    # load args

    prom_loop = LoopT(model=DeepTune())
    deeptune_model = DeepTune()

    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []
    deeptune_model.init(args)

    # Load dataset
    # print(f"*****************{platform}***********************")
    print(f"Loading dataset")
    X_seq, y_1hot, time, train_index, valid_index, test_index = \
        prom_loop.data_partitioning(dataset=r'../../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1, args=args)
    train_index = train_index[:1000]
    valid_index = valid_index[:400]
    test_index = test_index[:200]
    # train_index = train_index[:100]
    # valid_index = valid_index[:100]
    # test_index = test_index[:100]
    #  init the model
    print(f"Loading underlying model...")
    prom_loop.model.init(args)
    seed_value = int(args.seed)
    prom_loop.model.train(
        sequences=X_seq[train_index], verbose=False, y_1hot=y_1hot[train_index]
    )
    # save the model
    # model_dir_path = "logs/train/models/ma/"
    plot_figure_path = 'logs/train/figs/de/plot'
    plot_figuredata_path = 'logs/train/figs/de/data'
    # os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)

    # load the model
    # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
    # make prediction
    origin_speedup, all_pre, data_distri = deeptune_make_prediction \
        (model=deeptune_model, X_seq=X_seq, y_1hot=y_1hot,
         time=time, test_index=test_index)
    print(f"Loading successful, the speedup is {origin_speedup}")
    # origin = sum(speed_up_all) / len(speed_up_all)
    # print("final percent:", origin)
    ######
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    plt.boxplot(data_distri)
    data_df = pd.DataFrame({'Data': data_distri})
    sns.violinplot(data=data_df, y='Data')
    seed_save = str(int(seed_value))
    plt.title('Box Plot Example ' + seed_save)
    plt.ylabel('Values')

    plt.savefig(plot_figure_path + str(origin_speedup) + '_' + str(
        seed_save) + '.png')
    data_df.to_pickle(
        plot_figuredata_path + str(origin_speedup) + '_' + str(
            seed_save) + '_data.pkl')
    # plt.show()
    print("training finished")
    nni.report_final_result(origin_speedup)

def sensitive():
    from tqdm import tqdm
    Acc_all = []
    F1_all = []
    Pre_all = []
    Rec_all = []
    alpha_all = []
    seed_all = []

    for i in tqdm(range(100)):
        args = load_deeptune_args()
        Acc, F1, Pre, Rec, alpha,seed = loop_deploy_de(args)
        # print(f"Acc: {Acc}, F1: {F1}, Pre: {Pre}, Rec: {Rec}, alpha: {alpha}")
        Acc_all.append(Acc)
        F1_all.append(F1)
        Pre_all.append(Pre)
        Rec_all.append(Rec)
        seed_all.append(seed)

    # 将这些列表保存到一个pkl文件中
    with open('results.pkl', 'wb') as f:
        pickle.dump(
            {'Acc_all': Acc_all, 'F1_all': F1_all, 'Pre_all': Pre_all, 'Rec_all': Rec_all, 'alpha_all': alpha_all, 'seed_all':seed_all}, f)

def load_pkc():
    import matplotlib.pyplot as plt
    from scipy.stats import gmean
    with open('results.pkl', 'rb') as f:
        data = pickle.load(f)

    # 提取各个列表
    Acc_all = data['Acc_all']
    F1_all = data['F1_all']
    Pre_all = data['Pre_all']
    Rec_all = data['Rec_all']
    alpha_all = data['alpha_all']
    # seed_all = data['seed_all']
    Acc_all=np.array(Acc_all)
    F1_all=np.array(F1_all)
    Pre_all=np.array(Pre_all)
    Rec_all=np.array(Rec_all)
    Acc_mean = gmean(Acc_all, axis=1)
    F1_mean = gmean(F1_all, axis=1)
    Pre_mean = gmean(Pre_all, axis=1)
    Rec_mean = gmean(Rec_all, axis=1)

    Pre_mean_sort = sorted(Pre_mean, reverse=False)
    Rec_mean_sort = sorted(Rec_mean, reverse=True)
    F1_scores = [2 * (pre * rec) / (pre + rec) for pre, rec in zip(Pre_mean_sort, Rec_mean_sort)]

    # 绘制排序后的图表
    plt.plot(Pre_mean_sort, label="Precision", )
    plt.plot(Rec_mean_sort, label="recall",)
    plt.plot(F1_scores, label="F1", )
    plt.xlabel('Alpha')
    plt.ylabel('Precision')
    plt.legend()
    plt.title("Precision vs Alpha (Sorted by Precision)")
    plt.show()
    print("over")


if __name__ == '__main__':

    # sensitive()
    load_pkc()





    # nnictl create --config /home/huanting/PROM/examples/case_study/Loop/config.yaml --port 8088

