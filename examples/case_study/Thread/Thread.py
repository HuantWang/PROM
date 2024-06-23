import sys
import os
import warnings
warnings.filterwarnings("ignore")
import random
sys.path.append('./case_study/Thread')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
import nni
import argparse
# from Deeptune_utils import ThreadCoarsening,DeepTune,make_prediction,make_prediction_il
from Magni_utils import ThreadCoarsening,Magni,make_prediction,make_prediction_il
from src.util import Prom_utils



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

def load_magni_args():
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "seed": 123,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    args = parser.parse_args()
    # train the underlying model
    # deeptune_model = DeepTune()
    # deeptune_model.init(args)
    return args
if __name__=='__main__':
    # load args
    prom_thread=ThreadCoarsening(model=Magni())
    dataset_path= "../../../benchmark/Thread/pact-2014-runtimes.csv"

    # train the underlying model
    args = load_magni_args()
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []

    for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
        # Load dataset
        print(f"*****************{platform}***********************")
        print(f"Loading dataset on {platform}...")
        X_cc, y_cc,train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data,cal_y,train_index, valid_index, test_index,y,X_seq,y_1hot=\
            prom_thread.data_partitioning(dataset_path,platform=platform, calibration_ratio=0.1,args=args)
        #  init the model
        print(f"Training/Loading underlying model...")
        prom_thread.model.init(args)
        seed_value = int(args.seed)
        prom_thread.model.train(
                cascading_features=train_x,
                verbose=True,
                cascading_y=train_y)
        # save the model
        model_path = f"models/magni/{platform}-{seed_value}.model"
        try:
            os.mkdir(os.path.dirname(model_path))
        except:
            pass
        prom_thread.model.save(model_path)

        # load the model
        # prom_thread.model.restore(r'/home/huanting/PROM/examples/case_study/Thread/models/depptune/Cypress-123.model')
        # make prediction
        origin_speedup,all_pre=make_prediction(speed_up_all=speed_up_all,
                                               platform=platform,model=prom_thread.model,
                        test_x=test_x,test_index=test_index,X_cc=X_cc)
        print(f"Loading successful, the speedup on the {platform} is {origin_speedup:.2%}")
        # origin = sum(speed_up_all) / len(speed_up_all)
        # print("final percent:", origin)
        # nni.report_final_result(origin)

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
                             cal_x=calibration_data, cal_y=cal_y, test_x=test_x,
            test_y=test_y,significance_level="auto")

        # evaluate conformal prediction
        index_all_right, index_list_right,Acc_all,F1_all,Pre_all,Rec_all\
            =Prom_thread.evaluate_conformal_prediction\
            (y_preds=y_preds, y_pss=y_pss,p_value=p_value,all_pre=all_pre,y=y[test_index])

        # Increment learning
        print("Finding the most valuable instances for incremental learning...")
        train_index, test_index=Prom_thread.incremental_learning\
            (seed_value, test_index, train_index)
        # retrain the model
        print("Retraining the model on {platform}...")
        prom_thread.model.train(
                cascading_features=X_seq[train_index],
                cascading_y=y_1hot[train_index],verbose=True)
        # test the pretrained model
        retrained_speedup,inproved_speedup=make_prediction_il\
            (speed_up_all=speed_up_all, platform=platform,model=prom_thread.model,
             test_x=X_seq[test_index],test_index=test_index, X_cc=X_cc,
             origin_speedup=origin_speedup,improved_spp_all=improved_spp_all)

        origin_speedup_all.append(origin_speedup)
        speed_up_all.append(retrained_speedup)
        improved_spp_all.append(inproved_speedup)
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



def Thread(method="DeepTune"):
    # load args
    prom_thread=ThreadCoarsening(model=DeepTune())
    dataset_path= "../../../benchmark/Thread/pact-2014-runtimes.csv"

    # train the underlying model
    args = load_args()
    origin_speedup_all = []
    speed_up_all = []
    improved_spp_all = []

    for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
        # Load dataset
        print(f"*****************{platform}***********************")
        print(f"Loading dataset on {platform}...")
        X_cc, y_cc,train_x, valid_x, test_x, train_y, valid_y, test_y, \
        calibration_data,cal_y,train_index, valid_index, test_index,y,X_seq,y_1hot=\
            prom_thread.data_partitioning(dataset_path,platform=platform, calibration_ratio=0.1,args=args)
        #  init the model
        print(f"Loading underlying model...")
        prom_thread.model.init(args)
        seed_value = int(args.seed)
        # prom_thread.model.train(
        #         sequences=train_x,
        #         verbose=True,
        #         y_1hot=train_y)
        # save the model
        # model_path = f"models/depptune/{platform}-{seed_value}.model"
        # try:
        #     os.mkdir(fs.dirname(model_path))
        # except:
        #     pass
        # prom_thread.model.save(model_path)

        # load the model
        prom_thread.model.restore(r'/home/huanting/PROM/examples/case_study/Thread/models/depptune/Cypress-123.model')
        # make prediction
        origin_speedup,all_pre=make_prediction(speed_up_all=speed_up_all,
                                               platform=platform,model=prom_thread.model,
                        test_x=test_x,test_index=test_index,X_cc=X_cc)
        print(f"Loading successful, the speedup on the {platform} is {origin_speedup:.2%}")
        # origin = sum(speed_up_all) / len(speed_up_all)
        # print("final percent:", origin)
        # nni.report_final_result(origin)

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
        index_all_right, index_list_right,Acc_all,F1_all,Pre_all,Rec_all\
            =Prom_thread.evaluate_conformal_prediction\
            (y_preds=y_preds, y_pss=y_pss,p_value=p_value,all_pre=all_pre,y=y[test_index])

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
        retrained_speedup,inproved_speedup=make_prediction_il\
            (speed_up_all=speed_up_all, platform=platform,model=prom_thread.model,
             test_x=X_seq[test_index],test_index=test_index, X_cc=X_cc,
             origin_speedup=origin_speedup,improved_spp_all=improved_spp_all)

        origin_speedup_all.append(origin_speedup)
        speed_up_all.append(retrained_speedup)
        improved_spp_all.append(inproved_speedup)
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