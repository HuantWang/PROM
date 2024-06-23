# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import sys
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
import src.util as util
import os
import json
import random


class Bug_detection(util.ModelDefinition):
    def __init__(self,model=None,dataset=None,calibration_data=None,args=None):
        self.model = model
        self.calibration_data = None
        self.dataset = None

    def data_partitioning(self, dataset=r"../../../benchmark/Bug", random_seed=1234,
                          num_folders=8, calibration_ratio=0.2, args=None):
        # folder_path = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data"  # 文件夹的路径

        num_files_per_folder = 20  # 每个文件夹中需要选择的文本文件数量

        # 获取文件夹列表
        folders = [
            folder
            for folder in os.listdir(dataset)
            if os.path.isdir(os.path.join(dataset, folder))
        ]

        random.seed(random_seed)
        # 随机选择文件夹
        selected_folders = random.sample(folders, num_folders)

        # 创建数组用于存储选中的文件路径
        selected_files = []

        # 遍历选中的文件夹
        for folder in selected_folders:
            folder_dir = os.path.join(dataset, folder)
            for root, dirs, files in os.walk(folder_dir):
                for file in files:
                    if file.endswith(".txt"):
                        file_path = os.path.join(root, file)
                        selected_files.append(file_path)

                        if len(selected_files) % num_files_per_folder == 0:
                            break
                if len(selected_files) == num_folders * num_files_per_folder:
                    break

            if len(selected_files) == num_folders * num_files_per_folder:
                break
        # 输出选中的文件路径
        # print(selected_files)

        random.shuffle(selected_files)
        # file_name = []
        if os.path.exists(dataset + "/train.jsonl"):
            try:
                # 尝试删除文件
                os.remove(dataset + "/train.jsonl")
                os.remove(dataset + "/valid.jsonl")
                os.remove(dataset + "/test.jsonl")
            except:
                pass

        # for root, file in findAllFile(selected_files):
        #     if file.endswith(".txt"):
        #         name = root + '/' + file
        #         file_name.append(name)

        for i in range(len(selected_files)):
            if i < len(selected_files) * 0.6:
                with open(
                        dataset + "/train.jsonl",
                        "a",
                ) as f:
                    f.write(json.dumps(selected_files[i]) + "\n")
            if i >= len(selected_files) * 0.6 and i < len(selected_files) * 0.8:
                with open(
                        dataset + "/valid.jsonl",
                        "a",
                ) as f:
                    f.write(json.dumps(selected_files[i]) + "\n")
            if i >= len(selected_files) * 0.8:
                with open(
                        dataset + "/test.jsonl",
                        "a",
                ) as f:
                    f.write(json.dumps(selected_files[i]) + "\n")


    def predict(self, X, significant_level=0.1):
        if self.model is None:
            raise ValueError("Model is not initialized.")

        pred=self.model.predict(self, sequences='')
        probability=self.model.predict_proba(self, sequences='')
        return pred, probability

    def feature_extraction(self, srcs):
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        code_tokens = [tokenizer.tokenize(src) for src in srcs]
        seqs = [tokenizer.convert_tokens_to_ids(src) for src in code_tokens]
        # seqs = [tokenizer.tokenize(src) for src in tokens_ids]
        # pad_val = atomizer.vocab_size
        pad_val = len(seqs)
        encoded = np.array(pad_sequences(seqs, maxlen=1024, value=pad_val))
        return np.vstack([np.expand_dims(x, axis=0) for x in encoded])

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs[0]
        prob = torch.softmax(logits, dim=1)
        loss_function = nn.CrossEntropyLoss()
        if labels is not None:
            labels = torch.argmax(labels, dim=1)
            loss = loss_function(prob, labels)
            return loss, prob
        else:
            return prob

    def predict_proba(self, input_ids=None, labels=None):
        input_ids = torch.tensor(input_ids)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs[0]
        prob = torch.softmax(logits, dim=1)
        return prob.detach().numpy()

    def fit(self):
        return

    def predict(self, input_ids=None, labels=None):
        input_ids = torch.tensor(input_ids)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs[0]
        prob = torch.softmax(logits, dim=1)
        label = torch.argmax(prob, dim=1)
        return label.detach().numpy()

    # def uq(self, input_ids=None, labels=None):
    #     outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
    #     logits = outputs
    #     prob = torch.softmax(logits, dim=0)
    #     if labels is not None:
    #
    #         labels = labels.float()
    #         # loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
    #         loss = torch.log(prob[:, 0] + 1e-10)
    #         loss = -loss.mean()
    #         return loss, prob
    #     else:
    #         return prob
