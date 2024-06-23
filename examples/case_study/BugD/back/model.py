# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss



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
