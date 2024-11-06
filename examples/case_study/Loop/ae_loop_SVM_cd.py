import sys
import os
import warnings

os.environ['CURL_CA_BUNDLE'] = ''
# from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
import random
sys.path.append('./case_study/Loop')
sys.path.append('/cgo/prom/PROM/')
sys.path.append('/cgo/prom/PROM/src')
sys.path.append('/cgo/prom/PROM/thirdpackage')

from Magni_utils import Magni,LoopT,make_prediction,make_prediction_il


import numpy as np
import nni
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from src.prom.prom_util import Prom_utils

def load_deeptune_args(mode=''):
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {} and mode == 'train':
        params = {
            "seed": 6039,
            "epoch": 20,
            "batch_size": 32,
        }
    elif params == {} and mode == 'deploy':
        params = {
            "seed": 5220,
            "epoch": 20,
            "batch_size": 32,
        }
    elif params == {} and mode == '':
        params = {
            "seed": 7375,
            "epoch": 20,
            "batch_size": 32,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--mode', choices=['train', 'deploy'], help="Mode to run: train or deploy")
    args = parser.parse_args()
    args.seed = int(args.seed)
    # print("seeds is", args.seed)
    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args

def loop_train_svm(args):
    # load args
    prom_loop = LoopT(model=Magni())
    Magni_model = Magni()
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []
    Magni_model.init(args)

    # Load dataset
    # print(f"*****************{platform}***********************")
    print(f"Loading dataset")

    X_seq, y_1hot, time, train_index, valid_index, test_index = \
        prom_loop.data_partitioning(dataset=r'../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1,
                                    args=args)
    train_index = train_index[:1500]
    valid_index = valid_index[:500]
    test_index = test_index[:500]
    #  init the model
    print(f"Training underlying model...")
    prom_loop.model.init(args)
    seed_value = int(args.seed)

    from sklearn.svm import SVC

    # 训练模型
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    y_1hot = np.argmax(y_1hot, axis=1)
    clf = SVC().fit(X_seq[train_index], y_1hot[train_index])
    y_pred_train = clf.predict(X_seq[train_index])
    ###
    data_distri=[]
    non_speedup_all = []
    for i, (o, p) in enumerate(zip(y_1hot[test_index], y_pred_train)):
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

    # save the model
    # model_dir_path = "logs/train/models/ma/"
    plot_figure_path = 'logs/train/figs/sv/plot'
    plot_figuredata_path = 'logs/train/figs/sv/data'
    # os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
    # model_path = f"logs/train/models/ma/magni_{seed_value}.model"
    # prom_loop.model.save(model_path)

    # load the model
    # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
    # make prediction

    print(f"Training successful, the speedup is {origin_speedup}")

    ######
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # plt.boxplot(data_distri)
    # data_df = pd.DataFrame({'Data': data_distri})
    # sns.violinplot(data=data_df, y='Data')
    # seed_save = str(int(seed_value))
    # plt.title('Box Plot Example ' + seed_save)
    # plt.ylabel('Values')
    #
    # plt.savefig(plot_figure_path + str(origin_speedup) + '_' + str(
    #     seed_save) + '.png')
    # data_df.to_pickle(
    #     plot_figuredata_path + str(origin_speedup) + '_' + str(
    #         seed_save) + '_data.pkl')
    # plt.show()
    # print("training finished")
    # nni.report_final_result(origin_speedup)
    return origin_speedup

def loop_deploy_svm(args):
    # load args
    prom_loop = LoopT(model=Magni())
    Magni_model = Magni()
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []
    Magni_model.init(args)

    # Load dataset
    # print(f"*****************{platform}***********************")
    print(f"Loading dataset")
    X_seq, y_1hot, time, train_index, valid_index, test_index = \
        prom_loop.data_partitioning(dataset=r'../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1,
                                    args=args)
    train_index = train_index[:1500]
    valid_index = valid_index[:500]
    test_index = test_index[:500]
    #  init the model
    print(f"Loading underlying model...")
    prom_loop.model.init(args)
    seed_value = int(args.seed)

    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.ensemble import RandomForestClassifier

    # 训练模型
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    y_1hot = np.argmax(y_1hot, axis=1)
    a=set(y_1hot[train_index])
    # print(len(a))
    clf = SVC(probability=True,random_state=args.seed).fit(X_seq[train_index], y_1hot[train_index])
    # clf = RandomForestClassifier().fit(X_seq[train_index], y_1hot[train_index])

    all_pre = clf.predict(X_seq[test_index])
    ###
    data_distri = []
    non_speedup_all = []
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

    # save the model
    # model_dir_path = "logs/train/models/ma/"
    plot_figure_path = 'logs/train/figs/sv/plot'
    plot_figuredata_path = 'logs/train/figs/sv/data'
    # os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
    # model_path = f"logs/train/models/ma/magni_{seed_value}.model"
    # prom_loop.model.save(model_path)

    # load the model
    # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
    # make prediction

    print(f"Loading successful, the speedup during deployment is {origin_speedup}")

    ######
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # plt.boxplot(data_distri)
    # data_df = pd.DataFrame({'Data': data_distri})
    # sns.violinplot(data=data_df, y='Data')
    # seed_save = str(int(seed_value))
    # plt.title('Box Plot Example ' + seed_save)
    # plt.ylabel('Values')

    # plt.savefig(plot_figure_path + str(origin_speedup) + '_' + str(
    #     seed_save) + '.png')
    # data_df.to_pickle(
    #     plot_figuredata_path + str(origin_speedup) + '_' + str(
    #         seed_save) + '_data.pkl')
    # plt.show()
    # print("training finished")

    # Conformal Prediction
    # the underlying model
    print(f"Start conformal prediction...")
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
    y = y_1hot[test_index]
    y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
        cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")

    # evaluate conformal prediction
    # MAPIE
    # Prom_thread.evaluate_mapie \
    #     (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y,
    #      significance_level="auto")
    #
    # Prom_thread.evaluate_rise \
    #     (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y,
    #      significance_level="auto")

    Prom_thread.evaluate_conformal_cd \
        (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y, significance_level='auto')


def ae_loop_svm_script():
    print("_________Start training phase________")
    args = load_deeptune_args("train")
    loop_train_svm(args=args)

    print("_________Start deploy phase________")
    args = load_deeptune_args("deploy")
    loop_deploy_svm(args=args)

# if __name__=='__main__':
#     args = load_deeptune_args()
#     if args.mode == 'train':
#         loop_train_svm(args=args)
#     elif args.mode == 'deploy':
#         loop_deploy_svm(args=args)
    # loop_train_svm(args)
    # loop_deploy_svm(args)
    # nnictl create --config /home/huanting/PROM/examples/case_study/Loop/config.yaml --port 8088
