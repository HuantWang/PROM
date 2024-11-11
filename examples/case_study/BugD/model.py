# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import sys


sys.path.append('./case_study/BugD')
sys.path.append('/cgo/prom/PROM')
sys.path.append('/cgo/prom/PROM/thirdpackage')
sys.path.append('/cgo/prom/PROM/src')
import src.prom.prom_util as util
import os
import json
import random
import numpy as np

class Bug_detection(util.ModelDefinition):
    """
    Class for bug detection using a pre-trained model, handling data partitioning,
    feature extraction, and making predictions.
    """
    def __init__(self,model=None,dataset=None,calibration_data=None,args=None):
        """
               Initializes the Bug_detection instance with optional model, dataset, and calibration data.

               Parameters
               ----------
               model : object, optional
                   Pre-trained model for bug detection, by default None.
               dataset : object, optional
                   Dataset for training/evaluation, by default None.
               calibration_data : object, optional
                   Calibration data for model evaluation, by default None.
               args : object, optional
                   Additional arguments for model configuration, by default None.
               """
        self.model = model
        self.calibration_data = None
        self.dataset = None

    def data_partitioning(self, dataset=r"../../../benchmark/Bug", random_seed=1234,
                          num_folders=8, calibration_ratio=0.2, args=None):
        # folder_path = "/home/huanting/model/bug_detect/CodeXGLUE-main/Defect-detection/data"
        """
        Partitions the data into training, validation, and test sets, selecting a subset of files
        from specified folders within the dataset.

        Parameters
        ----------
        dataset : str, optional
            Path to the dataset directory, by default "../../../benchmark/Bug".
        random_seed : int, optional
            Seed for randomization, by default 1234.
        num_folders : int, optional
            Number of folders to sample from, by default 8.
        calibration_ratio : float, optional
            Ratio of data to be used for calibration, by default 0.2.
        args : object, optional
            Additional arguments for data partitioning, by default None.
        """
        ...
        num_files_per_folder = 20
        try:


            folders = [
                folder
                for folder in os.listdir(dataset)
                if os.path.isdir(os.path.join(dataset, folder))
            ]
        except:
            dataset = r"../../benchmark/Bug"
            folders = [
                folder
                for folder in os.listdir(dataset)
                if os.path.isdir(os.path.join(dataset, folder))
            ]
        random.seed(random_seed)
        selected_folders = random.sample(folders, num_folders)

        selected_files = []

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
        # print(selected_files)

        random.shuffle(selected_files)
        # file_name = []
        if os.path.exists(dataset + "/train.jsonl"):
            try:
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
        """
        Makes predictions with the model for given input data.

        Parameters
        ----------
        X : list
            Input data to make predictions on.
        significant_level : float, optional
            Significance level for predictions, by default 0.1.

        Returns
        -------
        tuple
            Prediction labels and probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not initialized.")

        pred=self.model.predict(self, sequences='')
        probability=self.model.predict_proba(self, sequences='')
        return pred, probability

    def feature_extraction(self, srcs):
        """
        Extracts features from source code inputs using a tokenizer.

        Parameters
        ----------
        srcs : list
            List of source code strings for feature extraction.

        Returns
        -------
        np.array
            Encoded and padded sequence of tokens.
        """
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        code_tokens = [tokenizer.tokenize(src) for src in srcs]
        seqs = [tokenizer.convert_tokens_to_ids(src) for src in code_tokens]
        # seqs = [tokenizer.tokenize(src) for src in tokens_ids]
        # pad_val = atomizer.vocab_size
        pad_val = len(seqs)
        encoded = np.array(pad_sequences(seqs, maxlen=1024, value=pad_val))
        return np.vstack([np.expand_dims(x, axis=0) for x in encoded])

class Model(nn.Module):
    """
    A generic neural network model with an encoder for processing inputs,
    and predicting probabilities or labels.
    """
    def __init__(self, encoder, config, tokenizer, args):
        """
        Initializes the Model with a given encoder, configuration, tokenizer, and arguments.

        Parameters
        ----------
        encoder : object
            Encoder model for embedding inputs.
        config : object
            Configuration settings for the model.
        tokenizer : object
            Tokenizer to process input text.
        args : object
            Additional arguments for model configuration.
        """
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        """
        Forward pass of the model to compute logits and probability distributions.

        Parameters
        ----------
        input_ids : torch.Tensor, optional
            Tensor containing token IDs, by default None.
        labels : torch.Tensor, optional
            True labels for computing loss, by default None.

        Returns
        -------
        tuple or torch.Tensor
            Returns loss and probabilities if labels are provided, otherwise probabilities only.
        """
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
        """
        Computes the probabilities for each class based on the input IDs.

        Parameters
        ----------
        input_ids : list or np.array, optional
            List of token IDs for prediction, by default None.

        Returns
        -------
        np.array
            Probability distributions for each class.
        """
        input_ids = torch.tensor(input_ids)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs[0]
        prob = torch.softmax(logits, dim=1)
        return prob.detach().numpy()

    def fit(self):
        """
        Placeholder for fitting the model if needed.
        """
        return

    def predict(self, input_ids=None, labels=None):
        """
        Predicts class labels for the given input IDs.

        Parameters
        ----------
        input_ids : list or np.array, optional
            List of token IDs for prediction, by default None.

        Returns
        -------
        np.array
            Predicted class labels.
        """
        input_ids = torch.tensor(input_ids)
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs[0]
        prob = torch.softmax(logits, dim=1)
        label = torch.argmax(prob, dim=1)
        return label.detach().numpy()

class BiLSTMModel(nn.Module):
        """
        A bidirectional LSTM model for sequence-based classification tasks, with a linear layer for label prediction.
        """
        def __init__(self, encoder, config, tokenizer, args):
            """
                   Initializes the BiLSTMModel with embedding, LSTM, and fully connected layers.

                   Parameters
                   ----------
                   encoder : object
                       Encoder for input embeddings.
                   config : object
                       Model configuration including vocabulary and hidden size.
                   tokenizer : object
                       Tokenizer for converting text to tokens.
                   args : object
                       Additional arguments for model configuration.
                   """
            super(BiLSTMModel, self).__init__()
            self.config = config
            self.tokenizer = tokenizer
            self.args = args

            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.lstm = nn.LSTM(input_size=config.hidden_size,
                                hidden_size=config.hidden_size,
                                num_layers=2,
                                bidirectional=True,
                                batch_first=True)
            self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)  # *2 because it's bidirectional

        def forward(self, input_ids=None, labels=None):
            """
            Forward pass of the BiLSTM model to compute probabilities and, if labels are provided, loss.

            Parameters
            ----------
            input_ids : torch.Tensor, optional
                Tensor containing token IDs, by default None.
            labels : torch.Tensor, optional
                True labels for computing loss, by default None.

            Returns
            -------
            tuple or torch.Tensor
                Returns loss and probabilities if labels are provided, otherwise probabilities only.
            """
            embedded = self.embedding(input_ids)  # embedding layer
            lstm_out, _ = self.lstm(embedded)  # LSTM layer
            logits = self.fc(lstm_out[:, -1, :])  # Fully connected layer, using the output of the last time step

            prob = torch.softmax(logits, dim=1)
            loss_function = nn.CrossEntropyLoss()
            if labels is not None:
                labels = torch.argmax(labels, dim=1)
                loss = loss_function(prob, labels)
                return loss, prob
            else:
                return prob

        def predict_proba(self, input_ids=None):
            """
            Computes probability distributions over classes for given input IDs.

            Parameters
            ----------
            input_ids : list or np.array, optional
                List of token IDs for prediction, by default None.

            Returns
            -------
            np.array
                Probability distributions for each class.
            """
            input_ids = torch.tensor(input_ids)
            embedded = self.embedding(input_ids)  # embedding layer
            lstm_out, _ = self.lstm(embedded)  # LSTM layer
            logits = self.fc(lstm_out[:, -1, :])  # Fully connected layer, using the output of the last time step

            prob = torch.softmax(logits, dim=1)
            return prob.detach().numpy()

        def fit(self):
            # Define your training loop here if needed
            return

        def predict(self, input_ids=None):
            """
            Predicts class labels for the given input IDs.

            Parameters
            ----------
            input_ids : list or np.array, optional
                List of token IDs for prediction, by default None.

            Returns
            -------
            np.array
                Predicted class labels.
            """
            input_ids = torch.tensor(input_ids)
            embedded = self.embedding(input_ids)  # embedding layer
            lstm_out, _ = self.lstm(embedded)  # LSTM layer
            logits = self.fc(lstm_out[:, -1, :])  # Fully connected layer, using the output of the last time step

            prob = torch.softmax(logits, dim=1)
            label = torch.argmax(prob, dim=1)
            return label.detach().numpy()

