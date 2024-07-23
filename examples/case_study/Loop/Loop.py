import sys
import os
import warnings

# from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
import random
sys.path.append('./case_study/Loop')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
from Ml_utils import data_partitioning
from Deeptune_utils import DeepTune,LoopT,deeptune_make_prediction,deeptune_make_prediction_il
from Magni_utils import Magni,LoopT,make_prediction,make_prediction_il
from Ml_utils import ml_make_prediction,ml_make_prediction_il

import numpy as np
import nni
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from src.prom_util import Prom_utils

def load_deeptune_args():
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "epoch": 3,
            "batch_size": 8,
            "seed": 123,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=params['epoch'],
                        help="random seed for initialization")
    parser.add_argument("--batch_size", default=params['batch_size'], type=int,
                        help="Batch size per GPU/CPU for training.")
    args = parser.parse_args()
    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args

def loop_main(tasks="deeptune"):
    # load args

    prom_loop = LoopT(model=DeepTune())
    deeptune_model = DeepTune()


    args = load_deeptune_args()
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []
    deeptune_model.init(args)

    # Load dataset
    # print(f"*****************{platform}***********************")
    print(f"Loading dataset")
    X_seq, y_1hot,time,train_index, valid_index, test_index = \
        prom_loop.data_partitioning(dataset=r'../../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1, args=args)
    train_index=train_index[:600]
    valid_index=valid_index[:300]
    test_index=test_index[:100]
    #  init the model
    print(f"Loading underlying model...")
    prom_loop.model.init(args)
    seed_value = int(args.seed)
    prom_loop.model.train(
        sequences=X_seq[train_index], verbose=False, y_1hot=y_1hot[train_index]
    )
    # save the model
    model_path = f"models/loop/{seed_value}.model"
    try:
        os.mkdir(os.save.dirname(model_path))
    except:
        pass
    prom_loop.model.save(model_path)

    # load the model
    # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
    # make prediction
    origin_speedup, all_pre = deeptune_make_prediction\
        (model=deeptune_model, X_seq=X_seq, y_1hot=y_1hot,
         time=time, test_index=test_index)
    print(f"Loading successful, the speedup is {origin_speedup:.2%}")
    # origin = sum(speed_up_all) / len(speed_up_all)
    # print("final percent:", origin)
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
    cal_y= y_1hot[valid_index]
    # cal_y = np.argmax(y_1hot[valid_index], axis=1)

    test_x = X_seq[test_index]
    test_y = y_1hot[test_index]
    y = np.argmax(y_1hot[test_index], axis=1) # the true label with 0/1/2/3/4
    y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
        cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")

    # evaluate conformal prediction
    index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all \
        = Prom_thread.evaluate_conformal_prediction \
        (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y)

    # Increment learning
    print("Finding the most valuable instances for incremental learning...")
    train_index, test_index = Prom_thread.incremental_learning \
        (args.seed, test_index, train_index)
    # retrain the model
    print("Retraining the model...")

    prom_loop.model.train(
            sequences=X_seq[train_index], verbose=0, y_1hot=y_1hot[train_index]
        )
    # test the pretrained model
    retrained_speedup, inproved_speedup = deeptune_make_prediction_il\
        (model_il=deeptune_model, X_seq=X_seq, y_1hot=y_1hot, time=time,
                       test_index=test_index, origin_speedup=origin_speedup)

    print(
        f"origin speed up: {origin_speedup}, "
        f"Imroved speed up: {retrained_speedup}, "
        f"Imroved mean speed up: {inproved_speedup}, "
    )

def loop_main(tasks="mlp"):
    # load args

    prom_loop = LoopT(model=Magni())
    Magni_model = Magni()

    args = load_deeptune_args()
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []
    Magni_model.init(args)

    # Load dataset
    # print(f"*****************{platform}***********************")
    print(f"Loading dataset")
    X_seq, y_1hot, time, train_index, valid_index, test_index = \
        prom_loop.data_partitioning(dataset=r'../../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1,
                                    args=args)
    train_index = train_index[:600]
    valid_index = valid_index[:300]
    test_index = test_index[:100]
    #  init the model
    print(f"Loading underlying model...")
    prom_loop.model.init(args)
    seed_value = int(args.seed)
    prom_loop.model.train(
        cascading_features=X_seq[train_index],
        cascading_y=y_1hot[train_index],
        verbose=True,
    )

    # save the model
    model_path = f"models/loop/magni_{seed_value}.model"
    try:
        os.mkdir(os.save.dirname(model_path))
    except:
        pass
    prom_loop.model.save(model_path)

    # load the model
    # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
    # make prediction
    origin_speedup, all_pre = make_prediction \
        (model=prom_loop.model, X_feature=X_seq, y_1hot=y_1hot,
         time=time, test_index=test_index)
    print(f"Loading successful, the speedup is {origin_speedup:.2%}")
    # origin = sum(speed_up_all) / len(speed_up_all)
    # print("final percent:", origin)
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
    index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all \
        = Prom_thread.evaluate_conformal_prediction \
        (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y)

    # Increment learning
    print("Finding the most valuable instances for incremental learning...")
    train_index, test_index = Prom_thread.incremental_learning \
        (args.seed, test_index, train_index)
    # retrain the model
    print("Retraining the model...")
    prom_loop.model.train(
        cascading_features=X_seq[train_index],
        cascading_y=y_1hot[train_index],
        verbose=0,
    )

    # test the pretrained model
    retrained_speedup, inproved_speedup = make_prediction_il \
        (model_il=prom_loop.model, X_feature=X_seq, y_1hot=y_1hot, time=time,
         test_index=test_index, origin_speedup=origin_speedup)
    print(
        f"origin speed up: {origin_speedup}, "
        f"Imroved speed up: {retrained_speedup}, "
        f"Imroved mean speed up: {inproved_speedup}, "
    )

loop_main(tasks="mlp")
# if __name__=='__main__':
#     # load args
#     args = load_deeptune_args()
#     origin_speedup_all = []
#     speed_up_all = []
#     improved_spp_all = []
#     # Load dataset
#     print(f"Loading dataset")
#     X_seq, y_1hot,time,train_index, valid_index, test_index = \
#         data_partitioning(dataset=r'../../../benchmark/Loop/data_dict.pkl', calibration_ratio=0.1, args=args)
#     train_index=train_index[:1200]
#     valid_index=valid_index[:100]
#     test_index=test_index[:100]
#     #  init the model
#     print(f"Loading underlying model...")
#
#     # save the model
#     y_1hot = np.argmax(y_1hot, axis=1)
#     clf = SVC(kernel='linear',random_state=42,probability=True).fit(X_seq[train_index], y_1hot[train_index])
#     # load the model
#     # prom_loop.model.restore(r'/home/huanting/PROM/examples/case_study/Loop/models/loop/123.model')
#     # make prediction
#     origin_speedup, all_pre = ml_make_prediction \
#         (model=clf, X_feature=X_seq, y_1hot=y_1hot,
#          time=time, test_index=test_index)
#     print(f"Loading successful, the speedup is {origin_speedup:.2%}")
#
#     # origin = sum(speed_up_all) / len(speed_up_all)
#     # print("final percent:", origin)
#     # nni.report_final_result(origin)
#
#     # Conformal Prediction
#     # the underlying model
#     print(f"Start conformal prediction...")
#     # the prom parameters
#     method_params = {
#         "lac": ("score", True),
#         "top_k": ("top_k", True),
#         "aps": ("cumulated_score", True),
#         "raps": ("raps", True)
#     }
#     # the prom object
#     Prom_thread = Prom_utils(clf, method_params, task="loop")
#     # conformal prediction
#     calibration_data = X_seq[valid_index]
#     cal_y = y_1hot[valid_index]
#     test_x = X_seq[test_index]
#     test_y = y_1hot[test_index]
#     y=test_y
#     # y = np.argmax(y_1hot[test_index], axis=1) # the true label with 0/1/2/3/4
#     y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
#         cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")
#
#     # evaluate conformal prediction
#     index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all \
#         = Prom_thread.evaluate_conformal_prediction \
#         (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y)
#
#     # Increment learning
#     print("Finding the most valuable instances for incremental learning...")
#     train_index, test_index = Prom_thread.incremental_learning \
#         (args.seed, test_index, train_index)
#     # retrain the model
#     print("Retraining the model...")
#     clf_rl = SVC(kernel='linear',random_state=42,probability=True).fit(X_seq[train_index], y_1hot[train_index])
#     # test the pretrained model
#     retrained_speedup, inproved_speedup = ml_make_prediction_il\
#         (model_il=clf_rl, X_feature=X_seq, y_1hot=y_1hot, time=time,
#                        test_index=test_index, origin_speedup=origin_speedup)
#     print(
#         f"origin speed up: {origin_speedup}, "
#         f"Imroved speed up: {retrained_speedup}, "
#         f"Imroved mean speed up: {inproved_speedup}, "
#     )