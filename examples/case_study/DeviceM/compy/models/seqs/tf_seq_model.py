import numpy as np
import tensorflow as tf
import nni
tf.compat.v1.disable_eager_execution()
import torch
from compy.models.model import Model

from mapie.prom_classification import MapieClassifier
from mapie.metrics import (classification_coverage_score,
                           classification_mean_width_score)

class SummaryCallback(tf.keras.callbacks.Callback):
    def __init__(self, summary):
        self.__summary = summary


    def on_epoch_end(self, epoch, logs=None):
        # self.__summary["accuracy"] = logs["dense_2_accuracy"]
        # self.__summary["loss"] = logs["loss"]
        self.__summary["accuracy"] = logs.get("dense_output_accuracy")  # 使用通用的键名 "accuracy"
        self.__summary["loss"] = logs.get("dense_output_loss")

class TensorflowToMapie():
    def __init__(self, model):
        self.model= model

    def fit(self):
        return 1

    def predict_proba(self, data):
        aux_in=data[0]
        seqs=data[1]
        preds=self.model.predict(
            x=[np.array(aux_in), np.array(seqs)], batch_size=999999, verbose=False
        )[0]
        return preds

    def predict(self, data):
        aux_in = data[0]
        seqs = data[1]
        pred_proba = self.model.predict(
            x=[np.array(aux_in), np.array(seqs)], batch_size=999999, verbose=False
        )[0]
        pred = np.argmax(pred_proba, axis=1)
        return pred

class RnnTfModel(Model):
    def __init__(self, config=None, num_types=None,mode='train',model_path=None,random_seed=0):
        if not config:
            config = {
                "learning_rate": 0.001,
                "batch_size": 64,
                "num_epochs": 5,
            }
            # nni
            # tuner_params = nni.get_next_parameter()  # 这会获得一组搜索空间中的参数
            # try:
            #     config.update(tuner_params)
            # except:
            #     pass
        super().__init__(config)

        self.__num_types = num_types

        # np.random.seed(0)
        tf.random.set_seed(random_seed)
        # Language model. Takes as inputs source code sequences
        code_in = tf.keras.layers.Input(shape=(1024,), dtype="int32", name="code_in")
        x = tf.keras.layers.Embedding(
            input_dim=num_types + 1, input_length=1024, output_dim=64, name="embedding"
        )(code_in)
        x = tf.keras.layers.LSTM(
            64, implementation=1, return_sequences=True, name="lstm_1"
        )(x)
        x = tf.keras.layers.LSTM(64, implementation=1, name="lstm_2")(x)
        langmodel_out = tf.keras.layers.Dense(2, activation="sigmoid")(x)

        # Auxiliary inputs. wgsize and dsize
        auxiliary_inputs = tf.keras.layers.Input(shape=(2,))

        # Heuristic model. Takes as inputs the language model, outputs 1-hot encoded device mapping
        x = tf.keras.layers.Concatenate()([auxiliary_inputs, x])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        out = tf.keras.layers.Dense(2, activation="softmax", name="dense_output")(x)

        self.model = tf.keras.models.Model(
            inputs=[auxiliary_inputs, code_in], outputs=[out, langmodel_out]
        )
        self.model.compile(
            optimizer="adam",
            metrics=["accuracy"],
            loss=["categorical_crossentropy", "categorical_crossentropy"],
            loss_weights=[1.0, 0.2],
        )


    def _model_save(self,name='best_model.pkl'):
        torch.save(self.model, name)

    def __process_data(self, data):
        processed = {"sequences": [], "aux_in": [], "label": [],"cpu_time":[],"gpu_time":[]}
        for item in data:
            processed["sequences"].append(item["x"]["code_rep"].get_token_list())
            processed["aux_in"].append(item["x"]["aux_in"])
            processed["label"].append(item["y"])
            processed["cpu_time"].append(item["cpu_time"])
            processed["gpu_time"].append(item["gpu_time"])

        return processed

    def __process(self, data):
        # Pad sequences
        encoded = np.array(
            tf.keras.preprocessing.sequence.pad_sequences(
                data["sequences"], maxlen=1024, value=self.__num_types
            )
        )
        seqs = np.vstack([np.expand_dims(x, axis=0) for x in encoded])

        aux_in = data["aux_in"]

        # Encode labels one-hot
        ys = tf.keras.utils.to_categorical(data["label"], num_classes=2)
        cpu_time=data["cpu_time"]
        gpu_time = data["gpu_time"]

        return seqs, aux_in, ys,cpu_time,gpu_time

    def _train_with_batch(self, batch):
        seqs, aux_in, ys,cpu_time,gpu_time = self.__process(self.__process_data(batch))

        summary = {}
        callback = SummaryCallback(summary)
        # a=[np.array(aux_in), np.array(seqs)]
        # aa=[np.array(ys), np.array(ys)]
        self.model.fit(
            x=[np.array(aux_in), np.array(seqs)],
            y=[np.array(ys), np.array(ys)],
            epochs=1,
            batch_size=self.config["batch_size"],
            verbose=False,
            shuffle=True,
            callbacks=[callback],
        )
        pred = self.model.predict(
            x=[np.array(aux_in), np.array(seqs)], batch_size=999999, verbose=False
        )[0]
        """speedup"""
        baseline_speedup = []
        oracle_percent = []
        pred_label=  np.argmax(pred, axis=1)
        origin_label = [np.array(ys), np.array(ys)][0]
        origin_label= [max(sublist) for sublist in origin_label]
        # for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
        #         zip(pred_label, origin_label, cpu_time, gpu_time):
        #     if pre_label_tem == 0:
        #         baseline_speedup.append(gpu_time_tem / cpu_time_tem)
        #     if pre_label_tem == 1:
        #         baseline_speedup.append(gpu_time_tem / cpu_time_tem)
        for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
                zip(pred_label, origin_label, cpu_time, gpu_time):
            # baseline:gpu
            baseline_speedup.append(gpu_time_tem / cpu_time_tem)
            # oracle:
            if pre_label_tem == 1:
                oracle_percent.append(min(gpu_time_tem, cpu_time_tem) / gpu_time_tem)
            else:
                oracle_percent.append(min(gpu_time_tem, cpu_time_tem) / cpu_time_tem)

        baseline_speedup=np.mean(baseline_speedup)
        return summary["loss"], summary["accuracy"],baseline_speedup,oracle_percent

    def _predict_with_batch(self, batch):
        seqs, aux_in, ys,cpu_time,gpu_time = self.__process(self.__process_data(batch))

        pred = self.model.predict(
            x=[np.array(aux_in), np.array(seqs)], batch_size=999999, verbose=False
        )[0]

        valid_accuracy = np.sum(np.argmax(pred, axis=1) == np.argmax(ys, axis=1)) / len(
            pred
        )
        """ speedup"""
        """speedup"""
        baseline_speedup = []
        oracle_percent = []
        pred_label = np.argmax(pred, axis=1)
        origin_label = [np.array(ys), np.array(ys)][0]
        origin_label = [max(sublist) for sublist in origin_label]
        for pre_label_tem, ori_label_tem, cpu_time_tem, gpu_time_tem in \
                zip(pred_label, origin_label, cpu_time, gpu_time):
            baseline_speedup.append(gpu_time_tem / cpu_time_tem)
            # oracle:
            if pre_label_tem == 1:
                oracle_percent.append(min(gpu_time_tem, cpu_time_tem) / gpu_time_tem)
            else:
                oracle_percent.append(min(gpu_time_tem, cpu_time_tem) / cpu_time_tem)

        baseline_speedup = np.mean(baseline_speedup)

        return valid_accuracy, pred, baseline_speedup, oracle_percent

    def _predict_uq_batch(self, data_train,valid_batches,test_batches,random_seed):
        print("start conformal prediction")
        F1_all=[]
        Pre_all=[]
        Rec_all=[]
        Acc_all = []
        clf = TensorflowToMapie(self.model)

        method_params = {
            "lac": ("score", True),
            "top_k": ("top_k", True),
            "aps": ("cumulated_score", True),
            "raps": ("raps", True)
        }
        y_preds, y_pss, y_pred_merged,p_value = {}, {}, {},{}
        import math
        def find_alpha_range(n):
            alpha_min = max(1 / n, 0)
            alpha_max = min(1 - 1 / n, 1)
            return math.ceil(alpha_min * 100) / 100, math.floor(alpha_max * 100) / 100


        alpha_min, alpha_max = find_alpha_range(len(valid_batches))
        alphas = np.arange(alpha_min, alpha_max, 0.03)
        # for alpha in alphas:
        #     a=1 / alpha
        #     b= 1 / (1 - alpha)
        #     print("a")

        # X_cal = X_seq[valid_index]
        # one_hot_matrix = y_1hot[valid_index]
        # y_cal = np.argmax(one_hot_matrix, axis=1)
        # # y_tr = np.argmax(y_1hot[train_index], axis=1)
        # y_test = np.argmax(y_1hot[test_index], axis=1)
        # X_test = X_seq[test_index]
        seqs, aux_in, ys,cpu_time,gpu_time = self.__process(self.__process_data(valid_batches))
        X_cal=(
            aux_in, seqs
        )
        y_cal=np.argmax(ys, axis=1)

        seqs, aux_in, ys,cpu_time,gpu_time = self.__process(self.__process_data(test_batches))
        X_test=[np.array(aux_in), np.array(seqs)]
        y_test=np.argmax(ys, axis=1)

        for name, (method, include_last_label) in method_params.items():
            mapie = MapieClassifier(
                estimator=clf, method=method, cv="prefit", random_state=random_seed, classes=y_test
            )
            mapie.fit(X_cal, y_cal)
            mapie.prom_test_ncm(X_test)
            y_preds[name], y_pss[name], p_value[name] = \
                mapie.predict(X_cal, X_test, alpha=alphas, include_last_label=include_last_label)

        def count_null_set(y: np.ndarray) -> int:
            count = 0
            for pred in y[:, :]:
                if np.sum(pred) == 0:
                    count += 1
            return count

        nulls, coverages, accuracies, confidence_sizes,credibility_sizes = {}, {}, {}, {},{}
        credibility_score = {}
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        for name, (method, include_last_label) in method_params.items():
            results_array = np.zeros((len(p_value[name]), len(alphas)), dtype=bool)
            #step1, use p-value to select
            for i, alpha in enumerate(alphas):
                results_array[:, i] = np.array([value > (1 - alpha) for value in p_value[name]])
            credibility_score[name] = results_array
        for name, (method, include_last_label) in method_params.items():
            # accuracies[name] = accuracy_score(y_test, y_preds[name])
            # nulls[name] = [self.count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(self.alphas)]
            coverages[name] = [classification_coverage_score(y_test, y_pss[name][:, :, i]) for i, _ in
                               enumerate(alphas)]
            confidence_sizes[name] = [y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)]
            credibility_sizes[name] = [credibility_score[name][:, i].sum().mean() for i, _ in enumerate(alphas)]
        confidence_result = {key: min(range(len(lst)), key=lambda i: abs(lst[i] - 1)) for key, lst in
                             confidence_sizes.items()}
        credibility_result = {key: min(range(len(lst)), key=lambda i: abs(lst[i] - 1)) for key, lst in
                              credibility_sizes.items()}
        # Extract the y_ps at the closest index
        confidence_result_ps = {method: y_pss[method][:, :, confidence_result[method]] for method in y_pss}
        credibility_result_ps = {method: credibility_score[method][:, credibility_result[method]] for method in
                                 credibility_score}
        # Determine indices for each method
        index_all_tem, index_all_right_tem = {}, {}
        for (method, y_ps1), (_, y_ps2) in zip(credibility_result_ps.items(), confidence_result_ps.items()):
            for index, confidence_single in enumerate(y_ps2):
                confidence_true = sum(confidence_single)
                credibility_true = y_ps1[index]
                if method not in index_all_tem:
                    index_all_tem[method] = []
                    index_all_right_tem[method] = []
                if confidence_true == 1 and credibility_true == 1:
                # elif confidence_true == 1:
                    index_all_right_tem[method].append(index)
                else:
                    index_all_tem[method].append(index)
        index_all = list(set([idx for indices in index_all_tem.values() for idx in indices]))
        index_list = list(index_all_tem.values())
        index_list.append(index_all)
        index_all_right = list(set(range(len(y_test))) - set(index_all))
        index_list_right = list(index_all_right_tem.values())
        index_list_right.append(index_all_right)
        # # sizes里每个method最接近1的
        # result = {}  # 用于存储结果的字典
        # for key, lst in sizes.items():  # 遍历字典的键值对
        #     closest_index = min(
        #         range(len(lst)), key=lambda i: abs(lst[i] - 1)
        #     )  # 找到最接近1的数字的索引
        #     # closest_index = min(
        #     #     range(len(lst)), key=lambda i: abs(lst[i] - 1) if lst[i] != 1 else float('inf')
        #     # )  # 找到最接近1且不等于1的数字的索引
        #     result[key] = closest_index  # 将结果存入字典
        # # y_ps_90中提出来那个最接近1的位置
        # result_ps = {}
        # for method, y_ps in y_pss.items():
        #     result_ps[method] = y_ps[:, :, result[method]]
        # index_all_tem = {}
        # index_all_right_tem = {}
        # for method, y_ps in result_ps.items():
        #     for index, i in enumerate(y_ps):
        #         num_true = sum(i)
        #         if method not in index_all_tem:
        #             index_all_tem[method] = []
        #             index_all_right_tem[method] = []
        #         if num_true != 1:
        #             index_all_tem[method].append(index)
        #         elif num_true == 1:
        #             index_all_right_tem[method].append(index)
        # index_all = []
        # index_list = []
        # # 遍历字典中的每个键值对
        # for key, value in index_all_tem.items():
        #     # 使用集合对列表中的元素进行去重，并转换为列表
        #     list_length = len(value)
        #     # print(f"Length of {key}: {list_length}")
        #     # 将去重后的列表添加到新列表中
        #     index_all.extend(value)
        #     index_list.append(value)
        # index_all = list(set(index_all))
        # # print(f"Length of index_all: {len(index_all)}")
        # index_list.append(index_all)
        #
        # index_all_right = []
        # index_list_right = []
        # # 遍历字典中的每个键值对
        # for key, value in index_all_right_tem.items():
        #     # 使用集合对列表中的元素进行去重，并转换为列表
        #     list_length = len(value)
        #     # print(f"Length of {key}: {list_length}")
        #     # 将去重后的列表添加到新列表中
        #     index_all_right.extend(value)
        #     index_list_right.append(value)
        #
        # index_all_right = list(set(list(range(len(y_test)))) - set(index_all))
        # # print(f"Length of index_all: {len(index_all_right)}")
        # index_list_right.append(index_all_right)
        """ compute metircs"""
        import torch
        with torch.no_grad():
            all_pre = clf.predict(X_test)
        # all_pre=pred.max(dim=1)[1]

        different_indices = np.where(all_pre != y_test)[0]
        different_indices_right = np.where(all_pre == y_test)[0]
        # 找到的错误的： index_list
        # 找的真的错的：num_common_elements
        acc_best = 0
        F1_best = 0
        pre_best = 0
        rec_best = 0
        method_name_best = " NONE"
        method_name = {
            "naive": ("naive", False),
            # "score": ("score", False),
            # "cumulated_score": ("cumulated_score", True),
            # "random_cumulated_score": ("cumulated_score", "randomized"),
            # "top_k": ("top_k", False),
            "mixture": ("mixture", False)
        }
        for index, (single_list, single_list_right) in enumerate(zip(index_list, index_list_right)):
            common_elements = np.intersect1d(single_list, different_indices)
            num_common_elements = len(common_elements)
            common_elements_right = np.intersect1d(single_list_right, different_indices_right)
            num_common_elements_right = len(common_elements_right)
            try:
                accuracy = (num_common_elements + num_common_elements_right) / len(all_pre)
            except:
                accuracy = 0
            try:
                precision = num_common_elements / len(single_list)
            except:
                precision = 0
            try:
                recall = num_common_elements / len(different_indices)
            except:
                recall = 0
            try:
                F1 = 2 * precision * recall / (precision + recall)
            except:
                F1 = 0
            print(
                f"{list(method_name.keys())[index]} find accuracy: {accuracy * 100:.2f}%, "
                f"precision: {precision * 100:.2f}%, "
                f"recall: {recall * 100:.2f}%, "
                f"F1: {F1 * 100:.2f}%"
            )

            if F1 > F1_best:
                method_name_best = list(method_name.keys())[index]
                acc_best = accuracy
                F1_best = F1
                pre_best = precision
                rec_best = recall
        print(
            f"{method_name_best} is the best approach"
            f"best accuracy: {accuracy * 100:.2f}%, "
            f"best precision: {pre_best * 100:.2f}%, "
            f"best recall: {rec_best * 100:.2f}%, "
            f"best F1: {F1_best * 100:.2f}%"
        )
        Acc_all.append(acc_best)
        F1_all.append(F1_best)
        Pre_all.append(pre_best)
        Rec_all.append(rec_best)

        selected_count = max(int(len(y_test) * 0.05), 1)
        np.random.seed(random_seed)
        try:
            random_element = np.random.choice(common_elements, selected_count, replace=False)
        except:
            random_element = np.random.choice(range(len(y_test)), selected_count)

        # seqs, aux_in, ys, cpu_time, gpu_time = self.__process(self.__process_data(valid_batches))
        # X_cal = (
        #     aux_in, seqs
        # )
        # y_cal = np.argmax(ys, axis=1)

        sample = [test_batches[index] for index in random_element]

        train_batches = np.concatenate((data_train, sample))
        test_batches = [item for item in test_batches if item not in sample]

        return train_batches, test_batches
        # result_ps = {}
        # for method, y_ps in y_pss.items():
        #     result_ps[method] = y_ps[:, :, result[method]]
        #
        # index_all_tem = {}
        # for method, y_ps in result_ps.items():
        #     for index, i in enumerate(y_ps):
        #         num_true = sum(i)
        #         if method not in index_all_tem:
        #             index_all_tem[method] = []  # 将键的值初始化为空列表
        #         if num_true != 1:
        #             index_all_tem[method].append(index)
        # index_all = []
        # index_list = []
        # # 遍历字典中的每个键值对
        # for key, value in index_all_tem.items():
        #     # 使用集合对列表中的元素进行去重，并转换为列表
        #     list_length = len(value)
        #     print(f"Length of {key}: {list_length}")
        #     # 将去重后的列表添加到新列表中
        #     index_all.extend(value)
        #     index_list.append(value)
        # index_all = list(set(index_all))
        # print(f"Length of index_all: {len(index_all)}")
        # index_list.append(index_all)
        # """ compute metircs"""
        # # o = y[test_index]
        # # correct = p == o
        # # 所有错误的
        # import torch
        # with torch.no_grad():
        #     all_pre = clf.predict(X_test)
        # # all_pre=pred.max(dim=1)[1]
        # different_indices = np.where(all_pre != y_test)[0]
        # # 找到的错误的： index_list
        # # 找的真的错的：num_common_elements
        # F1_best = 0
        # pre_best = 0
        # rec_best = 0
        # method_name = {
        #     "naive": ("naive", False),
        #     "score": ("score", False),
        #     "cumulated_score": ("cumulated_score", True),
        #     "random_cumulated_score": ("cumulated_score", "randomized"),
        #     "top_k": ("top_k", False),
        #     "all": ("all", False),
        # }
        # for index, single_list in enumerate(index_list):
        #     common_elements = np.intersect1d(single_list, different_indices)
        #     num_common_elements = len(common_elements)
        #     try:
        #         precision = num_common_elements / len(single_list)
        #     except:
        #         precision = 0
        #     try:
        #         recall = num_common_elements / len(different_indices)
        #     except:
        #         recall = 0
        #     try:
        #         F1 = 2 * precision * recall / (precision + recall)
        #     except:
        #         F1 = 0
        #     print(
        #         list(method_name.keys())[index],
        #         "find precision为：%.2f%%" % (precision * 100),
        #     )
        #     print(list(method_name.keys())[index], "find recall：%.2f%%" % (recall * 100))
        #     print(list(method_name.keys())[index], "find F1：%.2f%%" % (F1 * 100))
        #     if F1 > F1_best:
        #         F1_best = F1
        #         pre_best = precision
        #         rec_best = recall
        # print("best precision为：%.2f%%" % (pre_best * 100))
        # print("best recall：%.2f%%" % (rec_best * 100))
        # print("best F1：%.2f%%" % (F1_best * 100))
        # F1_all.append(F1_best)
        # Pre_all.append(pre_best)
        # Rec_all.append(rec_best)
        # """ compute metircs"""
        # mean_f1 = sum(F1_all) / len(F1_all)
        # mean_pre = sum(Pre_all) / len(Pre_all)
        # mean_rec = sum(Rec_all) / len(Rec_all)
        # print("All best precision为：%.2f%%" % (mean_pre * 100))
        # print("All best recall：%.2f%%" % (mean_rec * 100))
        # print("All best F1：%.2f%%" % (mean_f1 * 100))
        # nni.report_final_result(mean_f1)
        # return mean_f1