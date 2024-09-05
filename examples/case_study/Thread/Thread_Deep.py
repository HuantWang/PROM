import sys
import os
import warnings
warnings.filterwarnings("ignore")
import random
# 将根目录添加到path中
sys.path.append('/home/huanting/PROM')
sys.path.append('./case_study/Thread')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
import nni
import argparse
from Deeptune_utils import ThreadCoarseningDe,DeepTune,make_predictionDe,make_prediction_ilDe
from Magni_utils import ThreadCoarseningMa,Magni,make_predictionMa,make_prediction_ilMa
from src.prom.prom_util import Prom_utils

def load_deeptune_args():
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "epoch": 10,
            "batch_size": 16,
            "seed": 5072,
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
    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args


def Thread_DeepTune_train(args):
    # load args
    prom_thread=ThreadCoarseningDe(model=DeepTune())
    dataset_path= "../../../benchmark/Thread/pact-2014-runtimes.csv"

    # train the underlying model

    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []

    for i, platform in enumerate(["Tahiti"]):
        # Load dataset
        print(f"*****************{platform}***********************")
        print(f"Loading dataset on {platform}...")
        X_cc, y_cc,train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data,cal_y,train_index, valid_index, test_index,y,X_seq,y_1hot=\
            prom_thread.data_partitioning(dataset_path,platform=platform, calibration_ratio=0.1,args=args)

        #  init the model
        import time
        start_time = time.time()
        print(f"Loading underlying model...")
        prom_thread.model.init(args)
        seed_value = int(args.seed)
        prom_thread.model.train(
                sequences=train_x,
                verbose=True,
                y_1hot=train_y)

        # load the model
        # prom_thread.model.restore(r'/home/huanting/PROM/examples/case_study/Thread/models/de/Cypress-123.model')
        # make prediction
        origin_speedup,all_pre,data_distri=make_predictionDe(speed_up_all=speed_up_all,
                                               platform=platform,model=prom_thread.model,
                        test_x=test_x,test_index=test_index,X_cc=X_cc)
        print(f"Loading successful, the speedup on the {platform} is {origin_speedup:.2%}")
        origin = sum(speed_up_all) / len(speed_up_all)

        endtime=time.time()
        time_cost=endtime-start_time
        print("Training cost time:",time_cost)
        # save the model
        model_dir_path="logs/train/models/de/"
        plot_figure_path = 'logs/train/figs/de/plot'
        plot_figuredata_path = 'logs/train/figs/de/data'
        os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)

        model_path = f"logs/train/models/de/{platform}-{seed_value}-{origin}.model"

        prom_thread.model.save(model_path)
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

        plt.savefig(plot_figure_path + str(origin) + '_' + str(
            seed_save) + '_test.png')
        data_df.to_pickle(
            plot_figuredata_path + str(origin) + '_' + str(
                seed_save) + '_data_test.pkl')
        plt.show()
        print("training finished")
    print("final percent:", origin)
    nni.report_final_result(origin)

def Thread_DeepTune_deploy(args):
    # load args
    prom_thread = ThreadCoarseningDe(model=DeepTune())
    dataset_path = "../../../benchmark/Thread/pact-2014-runtimes.csv"

    # train the underlying model
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []

    for i, platform in enumerate(["Tahiti"]):
        # Load dataset
        print(f"*****************{platform}***********************")
        print(f"Loading dataset on {platform}...")
        X_cc, y_cc, train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data, cal_y, train_index, valid_index, test_index, y, X_seq, y_1hot = \
            prom_thread.data_partitioning(dataset_path, platform=platform, mode='test', calibration_ratio=0.1,
                                          args=args)

        #  init the model
        import time
        start_time = time.time()
        print(f"Loading underlying model...")
        prom_thread.model.init(args)
        seed_value = int(args.seed)
        prom_thread.model.train(
            sequences=train_x,
            verbose=True,
            y_1hot=train_y)

        # load the model
        # prom_thread.model.restore(r'/home/huanting/PROM/examples/case_study/Thread/models/de/Cypress-123.model')
        # make prediction
        origin_speedup, all_pre, data_distri = make_predictionDe(speed_up_all=speed_up_all,
                                                                 platform=platform, model=prom_thread.model,
                                                                 test_x=test_x, test_index=test_index, X_cc=X_cc)
        print(f"Loading successful, the speedup on the {platform} is {origin_speedup:.2%}")
        origin = sum(speed_up_all) / len(speed_up_all)

        endtime = time.time()
        time_cost = endtime - start_time
        print("Training cost time:", time_cost)
        # save the model
        model_dir_path = "logs/deploy/models/de/"
        plot_figure_path = 'logs/deploy/figs/de/plot'
        plot_figuredata_path = 'logs/deploy/figs/de/data'
        os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
        model_path = f"logs/deploy/models/de/{platform}-{seed_value}-{origin}.model"
        prom_thread.model.save(model_path)
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

        plt.savefig(plot_figure_path + str(origin) + '_' + str(
            seed_save) + '.png')
        data_df.to_pickle(
            plot_figuredata_path + str(origin) + '_' + str(
                seed_save) + '_data.pkl')
        # plt.show()
        print("training finished")

        # Conformal Prediction
        # the underlying model
        print(f"Start conformal prediction on {platform}...")
        clf = prom_thread.model
        # the prom parameters
        method_params = {
            "lac": ("score", True),
            "top_k": ("top_k", True),
            "aps": ("cumulated_score", True),
            "raps": ("raps", True)
        }
        # the prom object
        Prom_thread=Prom_utils(clf, method_params,task="thread")
        # conformal prediction
        y_preds, y_pss, p_value=Prom_thread.conformal_prediction(
                             cal_x=calibration_data, cal_y=cal_y, test_x=test_x,test_y=test_y,significance_level="auto")

        # evaluate conformal prediction
        #MAPIE
        Prom_thread.evaluate_mapie \
            (y_preds=y_preds,
             y_pss=y_pss,
             p_value=p_value,
             all_pre=all_pre,
             y=y[test_index],
             significance_level=0.05)

        Prom_thread.evaluate_rise \
            (y_preds=y_preds,
             y_pss=y_pss,
             p_value=p_value,
             all_pre=all_pre,
             y=y[test_index],
             significance_level=0.05)

        index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all, _, _ \
            = Prom_thread.evaluate_conformal_prediction \
            (y_preds=y_preds,
             y_pss=y_pss,
             p_value=p_value,
             all_pre=all_pre,
             y=y[test_index],
             significance_level=0.05)
        # Increment learning
        print("Finding the most valuable instances for incremental learning...")
        train_index, test_index=Prom_thread.incremental_learning\
            (seed_value, test_index, train_index)
        # retrain the model
        print(f"Retraining the model on {platform}...")
        prom_thread.model.train(
                sequences=X_seq[train_index],
                verbose=True,
                y_1hot=y_1hot[train_index])
        # test the pretrained model
        retrained_speedup,inproved_speedup,data_distri=make_prediction_ilDe\
            (speed_up_all=speed_up_all, platform=platform,model=prom_thread.model,
             test_x=X_seq[test_index],test_index=test_index, X_cc=X_cc,
             origin_speedup=origin_speedup,improved_spp_all=improved_spp_all)

        origin_speedup_all.append(origin_speedup)
        speed_up_all.append(retrained_speedup)
        improved_spp_all.append(inproved_speedup)

        plt.boxplot(data_distri)
        data_df = pd.DataFrame({'Data': data_distri})
        sns.violinplot(data=data_df, y='Data')
        seed_save = str(int(seed_value))
        plt.title('Box Plot Example ' + seed_save)
        plt.ylabel('Values')

        plt.savefig(plot_figure_path + str(origin) + '_' + str(
            seed_save) + '_il.png')
        data_df.to_pickle(
            plot_figuredata_path + str(origin) + '_' + str(
                seed_save) + '_data_il.pkl')
        # print("____________________________________")
    mean_acc = sum(Acc_all) / len(Acc_all)
    mean_f1 = sum(F1_all) / len(F1_all)
    mean_pre = sum(Pre_all) / len(Pre_all)
    mean_rec = sum(Rec_all) / len(Rec_all)
    mean_speed_up = sum(speed_up_all) / len(speed_up_all)
    meanimproved_speed_up = sum(improved_spp_all) / len(improved_spp_all)
    print(
        f"The average accuracy is: {mean_acc * 100:.2f}%, "
        f"average precision is: {mean_pre * 100:.2f}%, "
        f"average recall is: {mean_rec * 100:.2f}%, "
        f"average F1 is: {mean_f1 * 100:.2f}%, "

    )
    print(
        f"Imroved speed up is: {mean_speed_up}, "
        f"Imroved averaged speed up is: {meanimproved_speed_up}, "
    )
    print("final improved percent:", meanimproved_speed_up)
    nni.report_final_result(meanimproved_speed_up)

if __name__=='__main__':
    args = load_deeptune_args()
    if args.mode == 'train':
        Thread_DeepTune_train(args=args)
    elif args.mode == 'deploy':
        Thread_DeepTune_deploy(args=args)
    Thread_DeepTune_train(args)
    # Thread_DeepTune_deploy(args)
    # nnictl create --config /home/huanting/PROM/examples/case_study/Thread/config.yaml --port 8088