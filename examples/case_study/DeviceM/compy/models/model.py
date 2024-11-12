import pprint
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息
import warnings
warnings.filterwarnings("ignore")
class Model(object):
    def __init__(self, config):
        pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(config)

        self.config = config

    def train(self, data_train, data_valid,data_test, args):
        train_summary = []
        best_speedup= [0]

        data_train, data_valid = self._train_init(data_train, data_valid)
        data_test = self._test_data_init(data_test)

        # print()
        # print("Training model...")
        for epoch in range(self.config["num_epochs"]):
            batch_size = self.config["batch_size"]
            np.random.seed(args.seed)
            np.random.shuffle(data_train)
            train_batches = [
                data_train[i * batch_size : (i + 1) * batch_size]
                for i in range((len(data_train) + batch_size - 1) // batch_size)
            ]
            train_speedup_total=[]
            valid_count=0
            # Train
            # start_time = time.time()

            for batch in train_batches:
                train_loss, train_accuracy,baseline_speedup,oracle_percent_train = self._train_with_batch(batch)
                valid_count += train_accuracy * len(batch)
                train_speedup_total.append(baseline_speedup)
            if len(train_speedup_total)==1:
                train_speed_up_geomean=train_speedup_total[0]
            else:
                train_speed_up_geomean = np.exp(np.mean(np.log(train_speedup_total)))
            train_accuracy = valid_count / len(data_train)
            # print("origin speedup is: ",train_speed_up_geomean)
            # Valid
            self._test_init()
            np.random.seed(args.seed)
            np.random.shuffle(data_valid)
            valid_batches = [
                data_valid[i * batch_size : (i + 1) * batch_size]
                for i in range((len(data_valid) + batch_size - 1) // batch_size)
            ]

            valid_count = 0
            baseline_speedup_total = []
            for batch in valid_batches:
                batch_accuracy, _, baseline_speedup,oracle_percent_valid = self._predict_with_batch(batch)
                valid_count += batch_accuracy * len(batch)
                baseline_speedup_total.append(baseline_speedup)
            if len(baseline_speedup_total) == 1:
                valid_speed_up = baseline_speedup_total[0]
            else:
                valid_speed_up = np.exp(np.mean(np.log(baseline_speedup_total)))
            valid_accuracy = valid_count / len(data_valid)
            # print("origin valid speedup is: ", train_speed_up_geomean)
            #make prediction
            np.random.seed(args.seed)
            np.random.shuffle(data_test)
            self._test_init()
            test_batches = [
                data_test[i * batch_size: (i + 1) * batch_size]
                for i in range((len(data_test) + batch_size - 1) // batch_size)
            ]

            test_count = 0
            pre_speedup_total = []
            o_percent_all=[]
            for batch in test_batches:
                batch_accuracy, _, baseline_speedup,oracle_percent_test = self._predict_with_batch(batch)
                test_count += batch_accuracy * len(batch)
                pre_speedup_total.append(baseline_speedup)
                o_percent_all+=oracle_percent_test

            if len(pre_speedup_total) == 1:
                test_speed_up = pre_speedup_total[0]
            else:
                test_speed_up = np.exp(np.mean(np.log(pre_speedup_total)))

            test_accuracy = test_count / len(data_test)
            percent_mean=sum(o_percent_all)/len(o_percent_all)
            # print(epoch,"  ","train_accuracy ", train_accuracy,
            #       " valid_accuracy ", valid_accuracy,
            #       " test_accuracy ", test_accuracy)
            # print(epoch, "  ", "train percent", train_speed_up_geomean, " valid speedup ", valid_speed_up, " test speedup ",
            #       test_speed_up)
            # print(epoch, "  ", "train percent", train_speed_up_geomean, " valid speedup ", valid_speed_up,
            #       " test speedup ",
            #       test_speed_up)
        # print("The training performance is:",percent_mean)

        # #####

        # plt.boxplot(o_percent_all)
        # data_df = pd.DataFrame({'Data': o_percent_all})
        # sns.violinplot(data=data_df, y='Data')
        # seed_save = str(int(args.seed))
        # plt.title('Box Plot Example ' + seed_save)
        # plt.ylabel('Values')
        # plt.savefig('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_deploy' +
        #             str(percent_mean) + '_' + str(seed_save) + '.png')
        # data_df.to_pickle('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_deploy' +
        #                   str(percent_mean) + '_' + str(seed_save) + '_data.pkl')
        # plt.show()

        # # UQ
        # cp_valid_batches = data_valid
        # cp_test_batches = data_test
        # train_batches,test_batches = self._predict_uq_batch(data_train,cp_valid_batches, cp_test_batches,random_seed)
        #
        # """IL"""
        # for epoch in range(self.config["num_epochs"]):
        #     np.random.seed(random_seed)
        #     np.random.shuffle(data_train)
        #     train_batches = [
        #         data_train[i * batch_size: (i + 1) * batch_size]
        #         for i in range((len(data_train) + batch_size - 1) // batch_size)
        #     ]
        #     train_speedup_total = []
        #     valid_count = 0
        #     # Train
        #     # start_time = time.time()
        #     for batch in train_batches:
        #         train_loss, train_accuracy, baseline_speedup = self._train_with_batch(batch)
        #         valid_count += train_accuracy * len(batch)
        #         train_speedup_total.append(baseline_speedup)
        #
        #     if len(train_speedup_total)==1:
        #         train_speed_up_geomean=train_speedup_total[0]
        #     else:
        #         train_speed_up_geomean = np.exp(np.mean(np.log(train_speedup_total)))
        #
        #     train_accuracy = valid_count / len(data_train)
        #
        #     np.random.seed(random_seed)
        #     np.random.shuffle(data_test)
        #     self._test_init()
        #     test_batches = [
        #         data_test[i * batch_size: (i + 1) * batch_size]
        #         for i in range((len(data_test) + batch_size - 1) // batch_size)
        #     ]
        #
        #     test_count = 0
        #     pre_speedup_total = []
        #     for batch in test_batches:
        #         batch_accuracy, _, baseline_speedup = self._predict_with_batch(batch)
        #         test_count += batch_accuracy * len(batch)
        #         pre_speedup_total.append(baseline_speedup)
        #
        #     if len(pre_speedup_total)==1:
        #         il_speed_up=pre_speedup_total[0]
        #     else:
        #         il_speed_up = np.exp(np.mean(np.log(pre_speedup_total)))
        #     test_accuracy = test_count / len(data_test)
        #     impoved_sp = il_speed_up-test_speed_up
        # print("increment_train",train_speed_up_geomean,"increment speedup ",il_speed_up, "improved speedup",impoved_sp)



        impoved_sp = 0
        model_path = ''
        import os
        if args.method == 'Deeptune':
            model_dir_path = f'/cgo/prom/PROM/examples/case_study/DeviceM/save_model/De/'
            os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
            model_path = model_dir_path+ \
                         f'{args.seed}_{percent_mean}.pkl'
        elif args.method == 'Programl':
            model_dir_path = f'/cgo/prom/PROM/examples/case_study/DeviceM/save_model/Programl/'
            model_path = model_dir_path + \
                            f'{args.seed}_{percent_mean}.pkl'

        self._model_save(model_path)
        print("Suceessfully")

        return impoved_sp,best_speedup,model_path,percent_mean

    def deploy_dev(self, data_train, data_valid,data_test, args):
        train_summary = []
        best_speedup= [0]

        data_train, data_valid = self._train_init(data_train, data_valid)
        data_test = self._test_data_init(data_test)

        # print()
        print("Loading model...")
        for epoch in range(self.config["num_epochs"]):
            batch_size = self.config["batch_size"]
            np.random.seed(args.seed)
            np.random.shuffle(data_train)
            train_batches = [
                data_train[i * batch_size : (i + 1) * batch_size]
                for i in range((len(data_train) + batch_size - 1) // batch_size)
            ]
            train_speedup_total=[]
            valid_count=0
            # Train
            # start_time = time.time()

            for batch in train_batches:
                train_loss, train_accuracy,baseline_speedup,oracle_percent_train = self._train_with_batch(batch)
                valid_count += train_accuracy * len(batch)
                train_speedup_total.append(baseline_speedup)
            if len(train_speedup_total)==1:
                train_speed_up_geomean=train_speedup_total[0]
            else:
                train_speed_up_geomean = np.exp(np.mean(np.log(train_speedup_total)))
            train_accuracy = valid_count / len(data_train)
            # print("origin speedup is: ",train_speed_up_geomean)
            # Valid
            self._test_init()
            np.random.seed(args.seed)
            np.random.shuffle(data_valid)
            valid_batches = [
                data_valid[i * batch_size : (i + 1) * batch_size]
                for i in range((len(data_valid) + batch_size - 1) // batch_size)
            ]

            valid_count = 0
            baseline_speedup_total = []
            for batch in valid_batches:
                batch_accuracy, _, baseline_speedup,oracle_percent_valid = self._predict_with_batch(batch)
                valid_count += batch_accuracy * len(batch)
                baseline_speedup_total.append(baseline_speedup)
            if len(baseline_speedup_total) == 1:
                valid_speed_up = baseline_speedup_total[0]
            else:
                valid_speed_up = np.exp(np.mean(np.log(baseline_speedup_total)))
            valid_accuracy = valid_count / len(data_valid)
            # print("origin valid speedup is: ", train_speed_up_geomean)
            #make prediction
            np.random.seed(args.seed)
            np.random.shuffle(data_test)
            self._test_init()
            test_batches = [
                data_test[i * batch_size: (i + 1) * batch_size]
                for i in range((len(data_test) + batch_size - 1) // batch_size)
            ]

            test_count = 0
            pre_speedup_total = []
            o_percent_all=[]
            for batch in test_batches:
                batch_accuracy, _, baseline_speedup,oracle_percent_test = self._predict_with_batch(batch)
                test_count += batch_accuracy * len(batch)
                pre_speedup_total.append(baseline_speedup)
                o_percent_all+=oracle_percent_test

            if len(pre_speedup_total) == 1:
                test_speed_up = pre_speedup_total[0]
            else:
                test_speed_up = np.exp(np.mean(np.log(pre_speedup_total)))

            test_accuracy = test_count / len(data_test)
            percent_mean=sum(o_percent_all)/len(o_percent_all)
            # print(epoch,"  ","train_accuracy ", train_accuracy,
            #       " valid_accuracy ", valid_accuracy,
            #       " test_accuracy ", test_accuracy)
            # print(epoch, "  ", "train percent", train_speed_up_geomean, " valid speedup ", valid_speed_up, " test speedup ",
            #       test_speed_up)
            # print(epoch, "  ", "train percent", train_speed_up_geomean, " valid speedup ", valid_speed_up,
            #       " test speedup ",
            #       test_speed_up)
        print("The underlying model performance during deployment is:",percent_mean)

        # #####

        # plt.boxplot(o_percent_all)
        # data_df = pd.DataFrame({'Data': o_percent_all})
        # sns.violinplot(data=data_df, y='Data')
        # seed_save = str(int(args.seed))
        # plt.title('Box Plot Example ' + seed_save)
        # plt.ylabel('Values')
        # plt.savefig('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_deploy' +
        #             str(percent_mean) + '_' + str(seed_save) + '.png')
        # data_df.to_pickle('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_deploy' +
        #                   str(percent_mean) + '_' + str(seed_save) + '_data.pkl')
        # plt.show()

        # # UQ
        # cp_valid_batches = data_valid
        # cp_test_batches = data_test
        # train_batches,test_batches = self._predict_uq_batch(data_train,cp_valid_batches, cp_test_batches,random_seed)
        #
        # """IL"""
        # for epoch in range(self.config["num_epochs"]):
        #     np.random.seed(random_seed)
        #     np.random.shuffle(data_train)
        #     train_batches = [
        #         data_train[i * batch_size: (i + 1) * batch_size]
        #         for i in range((len(data_train) + batch_size - 1) // batch_size)
        #     ]
        #     train_speedup_total = []
        #     valid_count = 0
        #     # Train
        #     # start_time = time.time()
        #     for batch in train_batches:
        #         train_loss, train_accuracy, baseline_speedup = self._train_with_batch(batch)
        #         valid_count += train_accuracy * len(batch)
        #         train_speedup_total.append(baseline_speedup)
        #
        #     if len(train_speedup_total)==1:
        #         train_speed_up_geomean=train_speedup_total[0]
        #     else:
        #         train_speed_up_geomean = np.exp(np.mean(np.log(train_speedup_total)))
        #
        #     train_accuracy = valid_count / len(data_train)
        #
        #     np.random.seed(random_seed)
        #     np.random.shuffle(data_test)
        #     self._test_init()
        #     test_batches = [
        #         data_test[i * batch_size: (i + 1) * batch_size]
        #         for i in range((len(data_test) + batch_size - 1) // batch_size)
        #     ]
        #
        #     test_count = 0
        #     pre_speedup_total = []
        #     for batch in test_batches:
        #         batch_accuracy, _, baseline_speedup = self._predict_with_batch(batch)
        #         test_count += batch_accuracy * len(batch)
        #         pre_speedup_total.append(baseline_speedup)
        #
        #     if len(pre_speedup_total)==1:
        #         il_speed_up=pre_speedup_total[0]
        #     else:
        #         il_speed_up = np.exp(np.mean(np.log(pre_speedup_total)))
        #     test_accuracy = test_count / len(data_test)
        #     impoved_sp = il_speed_up-test_speed_up
        # print("increment_train",train_speed_up_geomean,"increment speedup ",il_speed_up, "improved speedup",impoved_sp)



        impoved_sp = 0
        model_path = ''
        import os
        if args.method == 'Deeptune':
            model_dir_path = f'/cgo/prom/PROM/examples/case_study/DeviceM/save_model/De/'
            os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
            model_path = model_dir_path+ \
                         f'{args.seed}_{percent_mean}.pkl'
        elif args.method == 'Programl':
            model_dir_path = f'/cgo/prom/PROM/examples/case_study/DeviceM/save_model/Programl/'
            model_path = model_dir_path + \
                            f'{args.seed}_{percent_mean}.pkl'

        self._model_save(model_path)
        # print("Training suceessfully")

        return impoved_sp,best_speedup,model_path,percent_mean

    def valid(self, data_test,random_seed=1234):

        global test_accuracy
        data_test = self._test_data_init(data_test)

        # test
        self._test_init()
        batch_size = self.config["batch_size"]
        np.random.seed(random_seed)
        np.random.shuffle(data_test)
        batches = [
            data_test[i * batch_size : (i + 1) * batch_size]
            for i in range((len(data_test) + batch_size - 1) // batch_size)
        ]

        test_count = 0
        pre_speedup_total = []
        o_percent_all = []
        for batch in batches:
            batch_accuracy, _, baseline_speedup, oracle_percent_test = self._predict_with_batch(batch)
            test_count += batch_accuracy * len(batch)
            pre_speedup_total.append(baseline_speedup)
            o_percent_all += oracle_percent_test


        test_accuracy = test_count / len(data_test)
        percent_mean = sum(o_percent_all) / len(o_percent_all)

        valid_speed_up_geomean = np.exp(np.mean(np.log(pre_speedup_total)))/batch_size

        test_accuracy = test_count / len(data_test)
        batches_valid = batches
        return test_accuracy,valid_speed_up_geomean,batches_valid,percent_mean

    def test(self, data_test,valid_batches,random_seed=1234,mode_uq='train'):

        global test_accuracy
        data_test = self._test_data_init(data_test)

        # test
        self._test_init()
        batch_size = self.config["batch_size"]
        np.random.seed(random_seed)
        np.random.shuffle(data_test)
        batches = [
            data_test[i * batch_size : (i + 1) * batch_size]
            for i in range((len(data_test) + batch_size - 1) // batch_size)
        ]

        test_count = 0
        pre_speedup_total = []
        o_percent_all = []
        for batch in batches:
            batch_accuracy, _, baseline_speedup, oracle_percent_test = self._predict_with_batch(batch)
            test_count += batch_accuracy * len(batch)
            pre_speedup_total.append(baseline_speedup)
            o_percent_all += oracle_percent_test


        percent_mean = sum(o_percent_all) / len(o_percent_all)
        test_speed_up_geomean = np.exp(np.mean(np.log(pre_speedup_total))) / batch_size
        test_accuracy = test_count / len(data_test)
        batches_valid = batches
        return test_accuracy,test_speed_up_geomean,batches_valid,percent_mean

    def uq(self, data_train,data_cal,data_test,random_seed=1234,eva_flag=""):

        data_test = self._test_data_init(data_test)
        data_cal = self._test_data_init(data_cal)
        data_train = self._test_data_init(data_train)

        # test
        self._test_init()
        batch_size = self.config["batch_size"]
        np.random.seed(random_seed)
        np.random.shuffle(data_test)
        batches = [
            data_test[i * batch_size : (i + 1) * batch_size]
            for i in range((len(data_test) + batch_size - 1) // batch_size)
        ]

        test_count = 0
        pre_speedup_total = []
        if eva_flag == "comapre" or eva_flag == "cd":
            self._predict_uq_batch \
                (data_train, data_cal, data_test, random_seed, eva_flag=eva_flag)
            return 0
        train_batches, test_batches = self._predict_uq_batch\
            (data_train, data_cal, data_test, random_seed)
        return train_batches, test_batches

    def Incremental_train(self, train_batches, test_batches, test_percent_mean, random_seed=1234, batch_size=64):

        data_train = train_batches
        data_test = test_batches
        # data_train = self._test_data_init(data_train)
        # data_test = self._test_data_init(data_test)
        batch_size = self.config["batch_size"]
        for epoch in range(self.config["num_epochs"]):
            np.random.seed(random_seed)
            np.random.shuffle(data_train)
            train_batches = [
                data_train[i * batch_size: (i + 1) * batch_size]
                for i in range((len(data_train) + batch_size - 1) // batch_size)
            ]
            train_speedup_total = []
            valid_count = 0
            # Train
            # start_time = time.time()
            for batch in train_batches:
                train_loss, train_accuracy, baseline_speedup, oracle_percent_train = self._train_with_batch(batch)
                valid_count += train_accuracy * len(batch)
                train_speedup_total.append(baseline_speedup)

            if len(train_speedup_total) == 1:
                train_speed_up_geomean = train_speedup_total[0]
            else:
                train_speed_up_geomean = np.exp(np.mean(np.log(train_speedup_total)))

            train_accuracy = valid_count / len(data_train)

            np.random.seed(random_seed)
            np.random.shuffle(data_test)
            self._test_init()
            test_batches = [
                data_test[i * batch_size: (i + 1) * batch_size]
                for i in range((len(data_test) + batch_size - 1) // batch_size)
            ]

            test_count = 0
            pre_speedup_total = []
            o_percent_all = []
            for batch in test_batches:
                batch_accuracy, _, baseline_speedup, oracle_percent_test = self._predict_with_batch(batch)
                test_count += batch_accuracy * len(batch)
                pre_speedup_total.append(baseline_speedup)
                o_percent_all += oracle_percent_test

            if len(pre_speedup_total) == 1:
                il_speed_up = pre_speedup_total[0]
            else:
                il_speed_up = np.exp(np.mean(np.log(pre_speedup_total)))
            test_accuracy = test_count / len(data_test)
            il_percent_mean = sum(o_percent_all) / len(o_percent_all)

        plt.boxplot(o_percent_all)
        data_df = pd.DataFrame({'Data': o_percent_all})
        sns.violinplot(data=data_df, y='Data')
        seed_save = str(int(random_seed))
        plt.title('Box Plot Example ' + seed_save)
        plt.ylabel('Values')
        plt.savefig('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_IL' +
                    str(il_percent_mean) + '_' + str(seed_save) + '.png')
        data_df.to_pickle('/cgo/prom/PROM/examples/case_study/DeviceM/save_model/plot/' + 'box_plot_IL' +
                          str(il_percent_mean) + '_' + str(seed_save) + '_data.pkl')
        # plt.show()

        improved_sp = il_percent_mean - test_percent_mean
        print("The performance to the oracle is {:.2f}%, "
              "The improved speedup is {:.2f}%".format(il_percent_mean * 100, improved_sp* 100))

        return il_speed_up, improved_sp

        # for batch in batches:
        #     batch_accuracy, baseline_speedup = self._predict_uq_batch(data_train,data_cal, data_test,random_seed)
        #     test_count += batch_accuracy * len(batch)
        #     pre_speedup_total.append(baseline_speedup)
        # test_speed_up_geomean = np.exp(np.mean(np.log(pre_speedup_total))) / batch_size
        # speed_up = pre_speedup_total / len(data_test)
        # uq_accuracy = test_count / len(data_test)
        # batches_valid = batches

        # return uq_accuracy,test_speed_up_geomean,batches_valid




    def predict(self, data):
        _, pred, baseline_speedup = self._predict_with_batch(data)

        return pred, baseline_speedup

    def _train_init(self, data_train, data_valid):
        return data_train, data_valid

    def _test_data_init(self, data_test):
        return data_test

    def _test_init(self):
        pass

    def _test_load(self,model_path):
        pass

    def _train_with_batch(self, batch):
        raise NotImplementedError

    def _predict_with_batch(self, batch):
        raise NotImplementedError

    def _predict_val_batch(self, batch):
        raise NotImplementedError

    def _predict_uq_batch(self, batch, valid_batches,com_flag):
        raise NotImplementedError

    def _predict_test_batch(self, batch):
        raise NotImplementedError

    def _model_save(self,name='best_model.pkl'):
        pass
