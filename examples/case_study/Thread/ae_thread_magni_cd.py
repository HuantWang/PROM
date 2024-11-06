import sys
import os
import warnings
warnings.filterwarnings("ignore")
import random
# 将根目录添加到path中
sys.path.append('/cgo/prom/PROM')
sys.path.append('./case_study/Thread')
sys.path.append('/cgo/prom/PROM/src')
sys.path.append('/cgo/prom/PROM/thirdpackage')
import nni
import argparse
from Deeptune_utils import ThreadCoarseningDe,DeepTune,make_predictionDe,make_prediction_ilDe
from Magni_utils import ThreadCoarseningMa,Magni,make_predictionMa,make_prediction_ilMa
from src.prom.prom_util import Prom_utils

def load_magni_args_ae(mode=''):
    # get parameters from tuner
    params = nni.get_next_parameter()

    if params == {} and mode == 'train':
        params = {
            "seed": 6349,
            "epoch": 10,
            "batch_size": 32,
        }
    elif params == {} and mode == 'deploy':
        params = {
            "seed": 2494,
            "epoch": 10,
            "batch_size": 32,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--mode', choices=['train', 'deploy'], help="Mode to run: train or deploy")
    args, unknown = parser.parse_known_args()
    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args


def Thread_train_magni(args):
    # load args
    prom_thread=ThreadCoarseningMa(model=Magni())
    dataset_path= "../../../benchmark/Thread/pact-2014-runtimes.csv"

    # train the underlying model

    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []

    # for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
    import time
    start_time = time.time()
    for i, platform in enumerate(["Tahiti"]):
        # Load dataset

        print(f"Loading dataset...")
        X_cc, y_cc,train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data,cal_y,train_index, valid_index, test_index,y,X_seq,y_1hot=\
            prom_thread.data_partitioning(dataset_path,platform=platform,mode='train', calibration_ratio=0.1,args=args)
        #  init the model
        print(f"Training underlying model...")
        prom_thread.model.init(args)
        seed_value = int(args.seed)
        prom_thread.model.train(
                cascading_features=train_x,
                verbose=True,
                cascading_y=train_y)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"The time about the training model: {training_time} s")


        # load the model
        # prom_thread.model.restore(r'/home/huanting/PROM/examples/case_study/Thread/models/depptune/Cypress-123.model')
        # make prediction
        origin_speedup,all_pre,data_distri=make_predictionMa(speed_up_all=speed_up_all,
                                               platform=platform,model=prom_thread.model,
                        test_x=test_x,test_index=test_index,X_cc=X_cc)
        print(f"Training model successful, the speedup is {origin_speedup:.2%}")
        origin = sum(speed_up_all) / len(speed_up_all)

        # save the model
        model_dir_path="logs/train/models/magni/"
        model_path = f"logs/train/models/magni/{platform}-{seed_value}-{origin}.model"
        plot_figure_path = 'logs/train/figs/magni/plot'
        plot_figuredata_path = 'logs/train/figs/magni/data'
        os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
        try:
            os.mkdir(os.path.dirname(model_path))
        except:
            pass
        # prom_thread.model.save(model_path)
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
        # plt.savefig(plot_figure_path+str(origin)+'_' + str(seed_save) + '.png')
        # data_df.to_pickle(plot_figuredata_path +str(origin)+'_' + str(seed_save) + '_data.pkl')
        # plt.show()
        # print("training finished")
    print(f"Training phase finished. The speedup percentage is: {origin:.2f}")
    # nni.report_final_result(origin)

def Thread_deploy_magni(args):
    # load args
    prom_thread=ThreadCoarseningMa(model=Magni())
    dataset_path= "../../../benchmark/Thread/pact-2014-runtimes.csv"

    # train the underlying model
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []

    # for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
    import time
    start_time = time.time()
    for i, platform in enumerate(["Tahiti"]):
        # Load dataset

        print(f"Loading dataset...")
        X_cc, y_cc,train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data,cal_y,train_index, valid_index, test_index,y,X_seq,y_1hot=\
            prom_thread.data_partitioning(dataset_path,platform=platform,mode='test', calibration_ratio=0.1,args=args)
        #  init the model
        print(f"Loading underlying model...")
        prom_thread.model.init(args)
        seed_value = int(args.seed)
        prom_thread.model.train(
                cascading_features=train_x,
                verbose=True,
                cascading_y=train_y)

        end_time = time.time()
        training_time = end_time - start_time
        # print(f"The time about the training model: {training_time} s")


        # load the model
        # prom_thread.model.restore(r'/home/huanting/PROM/examples/case_study/Thread/models/depptune/Cypress-123.model')
        # make prediction
        origin_speedup,all_pre,data_distri=make_predictionMa(speed_up_all=speed_up_all,
                                               platform=platform,model=prom_thread.model,
                        test_x=test_x,test_index=test_index,X_cc=X_cc)
        print(f"Loading successful, the speedup during deployment is {origin_speedup:.2%}")
        origin = sum(speed_up_all) / len(speed_up_all)

        # save the model
        model_dir_path = f"logs/deploy/models/magni/"
        plot_figure_path = 'logs/deploy/figs/magni/plot'
        plot_figuredata_path = 'logs/deploy/figs/magni/data'
        os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figure_path), exist_ok=True)
        os.makedirs(os.path.dirname(plot_figuredata_path), exist_ok=True)
        model_path = f"logs/deploy/models/magni/{platform}-{seed_value}-{origin}.model"
        try:
            os.mkdir(os.path.dirname(model_path))
        except:
            pass
        # prom_thread.model.save(model_path)
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
        # plt.savefig(plot_figure_path+str(origin)+'_' + str(seed_save) + '.png')
        # data_df.to_pickle(plot_figuredata_path +str(origin)+'_' + str(seed_save) + '_data.pkl')
        # plt.show()
        # print("training finished")

        """conformal prediction"""
        # Conformal Prediction
        # the underlying model
        print(f"Start conformal prediction...")
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
        y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
                             cal_x=calibration_data, cal_y=cal_y, test_x=test_x,
            test_y=test_y,significance_level="auto")

        # evaluate conformal prediction
        # Prom_thread.evaluate_mapie \
        #     (y_preds=y_preds,
        #      y_pss=y_pss,
        #      p_value=p_value,
        #      all_pre=all_pre,
        #      y=y[test_index],
        #      significance_level=0.05)
        #
        # Prom_thread.evaluate_rise \
        #     (y_preds=y_preds,
        #      y_pss=y_pss,
        #      p_value=p_value,
        #      all_pre=all_pre,
        #      y=y[test_index],
        #      significance_level=0.05)

        Prom_thread.evaluate_conformal_cd(y_preds=y_preds,
                                          y_pss=y_pss,
                                          p_value=p_value,
                                          all_pre=all_pre,
                                          y=y[test_index],
                                          significance_level=0.05)

def ae_thread_magni_script():
    print("_________Start training phase________")
    args = load_magni_args_ae("train")
    Thread_train_magni(args=args)

    print("_________Start deploy phase________")
    args = load_magni_args_ae("deploy")
    Thread_deploy_magni( args=args)

# if __name__=='__main__':
#     args = load_magni_args()
#     if args.mode == 'train':
#         Thread_train_magni(args=args)
#     elif args.mode == 'deploy':
#         Thread_deploy_magni( args=args)
    # Thread_deploy_magni( args=args)
    # Thread_train_magni(args)
    # Thread_deploy_magni(args)
    # nnictl create --config /home/huanting/PROM/examples/case_study/Thread/config.yaml --port 8088