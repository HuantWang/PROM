from __future__ import absolute_import, division, print_function
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import nni
import numpy as np
import torch
from model import Bug_detection
from preprocess import pre
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from sklearn.naive_bayes import GaussianNB
import sys
os.environ['CURL_CA_BUNDLE'] = ''
# sys.path.append('./case_study/DeviceM')
sys.path.append('/home/huanting/PROM')
sys.path.append('/home/huanting/PROM/src')
sys.path.append('/home/huanting/PROM/thirdpackage')
sys.path.append('./case_study/BugD')
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from transformers import logging as transformers_logging

# 设置Transformers库的日志级别为ERROR，只显示错误信息
transformers_logging.set_verbosity_error()
from prom.prom_classification import MapieClassifier
from prom.metrics import (classification_coverage_score,
                           classification_mean_width_score)
from src.prom.prom_util import Prom_utils
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except:
#     from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import BiLSTMModel

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
    # GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR,  # 只显示 ERROR 及以上级别的日志消息
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_CLASSES = {
    # 'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    # 'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


import random


def cvconvert_examples_to_features(code_new, label, tokenizer, args):
    # source
    code = ' '.join(code_new.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, '6', label)


import os


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, one_hot_vectors=[], suffixes=[]):
        #
        # 提取所有不同的后缀

        self.examples = []
        with open(file_path) as ff:
            for line in ff:
                type = line.split('/')[-2]
                # pon=line.split('/')[-2]
                y = []
                gap = " "
                Graph_length = 50
                flag = "none"

                X_Code_Single = []
                X_Graph_Single = np.zeros([Graph_length, Graph_length])
                X_trace_Single = []
                X_testcase_single = []
                X_Node_Singe = []
                X_dynamic_single = []
                # a=line.split('\n')[0].split('"')[1]
                # try:
                f = open(line.split('\n')[0].split('"')[1])
                try:
                    for line in f:
                        if line == "-----label-----\n":
                            flag = "label"
                            continue
                        if line == "-----code-----\n":
                            flag = "code"
                            continue
                        if line == "-----children-----\n":
                            flag = "children"
                            continue
                        if line == "-----nextToken-----\n":
                            flag = "nextToken"
                            continue
                        if line == "-----computeFrom-----\n":
                            flag = "computeFrom"
                            continue
                        if line == "-----guardedBy-----\n":
                            flag = "guardedBy"
                            continue
                        if line == "-----guardedByNegation-----\n":
                            flag = "guardedByNegation"
                            continue
                        if line == "-----lastLexicalUse-----\n":
                            flag = "lastLexicalUse"
                            continue
                        if line == "-----jump-----\n":
                            flag = "jump"
                            continue
                        if line == "=======testcase========\n":
                            flag = "testcase"
                            continue
                        if line == "=========trace=========\n":
                            flag = "trace"
                            continue
                        if (
                                line == "-----attribute-----\n"
                                or line == "----------------dynamic----------------\n"
                        ):
                            flag = "next"
                            continue
                        if line == "-----ast_node-----\n":
                            flag = "ast_node"
                            continue
                        if line == "=======================\n":
                            break
                        if flag == "next":
                            continue
                        if flag == "label":
                            y = one_hot_vectors[suffixes.index(type)]
                            # y = line.split()
                            continue
                        if flag == "code":
                            X_Code_line = line.split("\n")[0]
                            X_Code_Single = X_Code_Single + [X_Code_line]
                            continue
                        if flag == "children":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            if num_2 < Graph_length and num_1 < Graph_length:
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            else:
                                continue
                            continue
                        if flag == "nextToken":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            continue
                        if flag == "computeFrom":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            if num_2 < Graph_length and num_1 < Graph_length:
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            else:
                                continue
                            continue
                        if flag == "guardedBy":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            if num_2 < Graph_length and num_1 < Graph_length:
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            else:
                                continue
                            continue
                        if flag == "guardedByNegation":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            if num_2 < Graph_length and num_1 < Graph_length:
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            else:
                                continue
                            continue
                        if flag == "lastLexicalUse":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            if num_2 < Graph_length and num_1 < Graph_length:
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            else:
                                continue
                            continue
                        if flag == "jump":
                            num_1 = int(line.split()[0].split(",")[0])
                            num_2 = int(line.split()[0].split(",")[1])
                            if num_2 < Graph_length and num_1 < Graph_length:
                                X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                            else:
                                continue
                            continue
                        if flag == "ast_node":
                            X_Code_line = line.split("\n")[0]
                            X_Node_Singe = X_Node_Singe + [X_Code_line]
                            continue
                        if flag == "testcase":
                            X_Code_line = line.split("\n")[0]
                            X_testcase_single = X_testcase_single + [X_Code_line]
                            X_dynamic_single = X_dynamic_single + [X_Code_line]
                        if flag == "trace":
                            X_Code_line = line.split("\n")[0]
                            X_trace_Single = X_trace_Single + [X_Code_line]
                            X_dynamic_single = X_dynamic_single + [X_Code_line]
                    f.close()
                except:
                    logging.info("please delete the file " + file_path)

                self.examples.append(
                    cvconvert_examples_to_features(gap.join(X_Code_Single), y, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def remove_duplicates_and_preserve_order(arr):
    unique_elements = []
    for item in arr:
        if item not in unique_elements:
            unique_elements.append(item)
    return unique_elements


def onehot(file_path="", seed=123):
    type_all = []
    with open(file_path) as ff:
        for line in ff:
            type = line.split('/')[-2]
            type_all.append(type)
    suffixes = remove_duplicates_and_preserve_order(type_all)
    # suffixes = list(set(type_all))
    # 创建独立的 One-Hot 向量

    one_hot_vectors = []
    for suffix in suffixes:
        one_hot = [0] * len(suffixes)
        one_hot[suffixes.index(suffix)] = 1
        one_hot_vectors.append(one_hot)
    return one_hot_vectors, suffixes


def set_seed(seed=42):
    seed = int(seed)
    # print(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def incre(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0.0
    best_f1_uq = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader),disable=True)
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):

            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate_test(args, model, tokenizer, eval_when_training=True)

                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint
                        if results['test_acc'] > best_acc:
                            best_acc = results['test_acc']
                            logger.info("  " + "*" * 20)
                            logger.info("  Best acc:%s", round(results['test_acc'], 4))
                            logger.info("  " + "*" * 20)
    return best_acc

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0.0
    best_f1_uq = 0.0
    best_acc = 0.0
    increment_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader),disable=True)
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):

            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        # print("eval results:", results)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint
                        if results['eval_acc'] > best_f1:
                            best_f1 = results['eval_acc']
                            # logger.info("  " + "*" * 20)
                            # logger.info("  Best f1:%s", round(best_f1, 4))
                            # logger.info("  Best pre:%s", round(results['eval_pre'], 4))
                            # logger.info("  Best rec:%s", round(results['eval_rec'], 4))
                            # logger.info("  Best acc:%s", round(results['eval_acc'], 4))
                            # logger.info("  " + "*" * 20)
                            print(
                                f"The current best accuracy is: {round(best_f1, 4)}")

                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix),)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                            # torch.save(model_to_save.state_dict(), output_dir)
                            # logger.info("Saving model checkpoint to %s", output_dir)
                            # print("Saving model checkpoint to {}".format(output_dir))

    return best_f1

def deploy(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    # logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dataset))
    # logger.info("  Num Epochs = %d", args.num_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_f1 = 0.0
    best_f1_uq = 0.0
    best_acc = 0.0
    increment_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader),disable=True)
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):

            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                            # Save model checkpoint
                        if results['eval_acc'] > best_f1:
                            best_f1 = results['eval_acc']
                            # logger.info("  " + "*" * 20)
                            # logger.info("  Best f1:%s", round(best_f1, 4))
                            # logger.info("  Best pre:%s", round(results['eval_pre'], 4))
                            # logger.info("  Best rec:%s", round(results['eval_rec'], 4))
                            # logger.info("  Best acc:%s", round(results['eval_acc'], 4))
                            # logger.info("  " + "*" * 20)
                            print(
                                f"The current accuracy is: {round(best_f1, 4)}")

                            checkpoint_prefix = 'checkpoint-best-acc'
                            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix),)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                            torch.save(model_to_save.state_dict(), output_dir)
                            # logger.info("Saving model checkpoint to %s", output_dir)
                            print("Saving model checkpoint to {}".format(output_dir))

                        if results['eval_acc'] >= 0.1 and results['eval_acc'] <= 0.9:
                            print(
                                f"The current accuracy is: {round(best_f1, 4)}")
                            results_uq, incre_index = conformal_prediction(args, model, tokenizer)
                            if results_uq['find_f1'] > best_f1_uq:
                                best_f1_uq = results_uq['find_f1']
                                # # logger.info("  " + "*" * 20)
                                # # logger.info("model prediction best f1:%s", round(results_uq['eval_acc'], 4))
                                # logger.info("find accuracy：%.2f%%" % (results_uq['find_acc'] * 100))
                                # logger.info("find precision为：%.2f%%" % (results_uq['find_pre'] * 100))
                                # logger.info("find recall为：%.2f%%" % (results_uq['find_rec'] * 100))
                                # logger.info("find F1：%.2f%%" % (best_f1_uq * 100))
                                # logger.info("  " + "*" * 20)
                                # 将 logger.info 语句改为单行的 print 语句
                                # print(
                                #     f"find accuracy：{results_uq['find_acc'] * 100:.2f}%, "
                                #     f"find precision为：{results_uq['find_pre'] * 100:.2f}%, "
                                #     f"find recall为：{results_uq['find_rec'] * 100:.2f}%, "
                                #     f"find F1：{best_f1_uq * 100:.2f}%")
                                # print("  " + "*" * 20)

                                # nni.report_intermediate_result(best_f1_uq)
                                checkpoint_prefix = 'checkpoint-best-acc'
                                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                                torch.save(model_to_save.state_dict(), output_dir)
                                logger.info("Saving the retrained model checkpoint to %s", output_dir)
                                # 将 logger.info 语句改为 print 语句
                                # print(f"Saving the retrained model checkpoint to {output_dir}")
                                """increment learning"""
                                # train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.one_hot_vectors,
                                #                             args.suffixes)
                                # test_dataset = TextDataset(tokenizer, args, args.test_data_file, args.one_hot_vectors,
                                #                            args.suffixes)
                                # incre_data = [test_dataset[i] for i in incre_index]
                                #
                                # train_dataset=train_dataset.append(incre_data)
                                # test_dataset = [item for index, item in enumerate(test_dataset) if
                                #                index not in incre_index]
                                print("Incremental Learning...")
                                selected_content = []
                                old_train = []
                                new_test = []
                                with open(args.test_data_file, 'r') as jsonl_file:
                                    for idx, line in enumerate(jsonl_file):
                                        if idx in incre_index:
                                            selected_content.append(json.loads(line))
                                        else:
                                            new_test.append(json.loads(line))
                                with open(args.train_data_file, 'r') as jsonl_file:
                                    for idx, line in enumerate(jsonl_file):
                                        old_train.append(json.loads(line))

                                new_train = old_train + selected_content
                                with open('../../../benchmark/Bug/new_train.jsonl', 'w') as selected_file:
                                    for item in new_train:
                                        json.dump(item, selected_file)
                                        selected_file.write('\n')
                                with open('../../../benchmark/Bug/new_test.jsonl', 'w') as selected_file:
                                    for item in new_test:
                                        json.dump(item, selected_file)
                                        selected_file.write('\n')

                                checkpoint_prefix = 'checkpoint-best-acc/model.bin'
                                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                                model.load_state_dict(torch.load(output_dir))
                                model.to(args.device)

                                train_dataset = TextDataset(tokenizer, args, '../../../benchmark/Bug/new_train.jsonl',
                                                            args.one_hot_vectors,
                                                            args.suffixes)
                                results_origin = evaluate_test(args, model, tokenizer)
                                retrained_acc = incre(args, train_dataset, model, tokenizer)
                                increment_acc_single = retrained_acc - results_origin["test_acc"]
                                print(f"The retrained accuracy is: {retrained_acc * 100:.2f}%, "
                                      f"The increment accuracy is: {increment_acc_single * 100:.2f}% "
                                      f"The best increment accuracy is: {increment_acc * 100:.2f}%")
                                print("*" * 60)
                                if increment_acc < increment_acc_single:
                                    increment_acc = increment_acc_single
                                # if retrained_acc < increment_acc_single:
                                #     increment_acc = increment_acc_single
                                # if increment_acc < increment_acc_single:
                                #     increment_acc = increment_acc_single
    return increment_acc


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, args.one_hot_vectors, args.suffixes)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0,
                                 pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # 混淆矩阵
    # preds = logits[:, 0] > 0.5
    from sklearn.metrics import confusion_matrix, classification_report
    all_labels = []
    all_predictions = []
    for label, predict in zip(labels, logits):
        # 前向传播并获取预测结果
        label = torch.argmax(torch.tensor(label)).item()
        predict = torch.argmax(torch.tensor(predict)).item()
        # top_values, top_indices = torch.topk(torch.tensor(predict), k=3)
        # 将真实标签和预测标签加入列表中
        all_labels.append(label)
        all_predictions.append(predict)
    confusion_mat = confusion_matrix(all_labels, all_predictions)
    #
    # print(confusion_mat)
    # from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score
    # eval_f1 = f1_score(all_labels, all_predictions, average='weighted')
    # eval_acc = accuracy_score(all_labels, all_predictions)
    # eval_rec = recall_score(all_labels, all_predictions, average='weighted')
    # eval_loss = eval_loss / nb_eval_steps
    # perplexity = torch.tensor(eval_loss)
    # result = {
    #     "eval_loss": float(perplexity),
    #     "eval_acc": round(eval_acc, 4),
    #     "eval_rec": round(eval_rec, 4),
    #     "eval_f1": round(eval_f1, 4),
    # }
    # return result
    TP = 0
    # FP=0
    # FN=0
    TN = 0
    for i, j in zip(labels, logits):
        # original positive
        _, i = torch.topk(torch.tensor(i), k=1)
        _, j = torch.topk(torch.tensor(j), k=1)
        if i in j:
            # 预测正确
            TP = TP + 1
        else:
            # 预测错误
            TN = TN + 1
        # if i == 1 and j == 1:
        #     TP=TP+1
        # elif i == 1 and j == 0:
        #     FN=FN+1
        # elif i == 0 and j == 1:
        #     FP = FP + 1
        # elif i == 0 and j == 0:
        #     TN=TN+1

    eval_acc = TP / (TP + TN)
    # try:
    #     eval_pre = TP/(TP+FP)
    # except:
    #     eval_pre= 0
    # try:
    #     eval_rec = TP/(TP+FN)
    # except:
    #     eval_rec = 0
    # try:
    #     eval_f1 = 2 * eval_pre*eval_rec/(eval_pre+eval_rec)
    # except:
    #     eval_f1 = 0

    # eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        # "eval_pre": round(eval_pre, 4),
        # "eval_rec": round(eval_rec, 4),
        # "eval_f1": round(eval_f1, 4),
    }
    return result


def conformal_prediction(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    print("Start the conformal prediction...")
    global find_pre, find_recall
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, args.one_hot_vectors, args.suffixes)
    # if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #     os.makedirs(eval_output_dir)
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0,
                                 pin_memory=True)
    # Eval!
    model.eval()
    labels = []
    x_val = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        labels.append(label.cpu().numpy())
        x_val.append(inputs.cpu().numpy())
        # with torch.no_grad():
        #     lm_loss, logit = model(inputs, label)
        #     labels.append(label.cpu().numpy())
        #     x_val.append(logit.cpu().numpy())

    y_cal_single = np.concatenate(labels, 0)
    X_cal = np.concatenate(x_val, 0)
    y_cal = []
    for label in y_cal_single:
        label = torch.argmax(torch.tensor(label)).item()
        y_cal.append(label)

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    test_dataset = TextDataset(tokenizer, args, args.test_data_file, args.one_hot_vectors, args.suffixes)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # TEST!

    x_test = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        labels.append(label.cpu().numpy())
        x_test.append(inputs.cpu().numpy())
        # with torch.no_grad():
        #     logit = model(inputs)
        #     logits.append(logit.cpu().numpy())
        #     labels.append(label.cpu().numpy())

    X_test = np.concatenate(x_test, 0)
    y_test_single = np.concatenate(labels, 0)
    y_test = []
    for label in y_test_single:
        # 前向传播并获取预测结果
        label = torch.argmax(torch.tensor(label)).item()
        # 将真实标签和预测标签加入列表中
        y_test.append(label)
    # for label in y_test_single:
    #     # 前向传播并获取预测结果
    #     label = torch.argmax(torch.tensor(label)).item()
    #     # 将真实标签和预测标签加入列表中
    #     y_test.append(label)
    ###########UQ
    # clf = GaussianNB().fit(X_cal, y_cal)
    clf = model
    # mapie_score = MapieClassifier(estimator=clf, cv="prefit", method=args.method, random_state=42)
    # raps cumulated_score naive
    method_params = {
        "lac": ("score", True),
        "top_k": ("top_k", True),
        "aps": ("cumulated_score", True),
        # "raps": ("raps", True)
    }
    Prom_thread = Prom_utils(clf, method_params, task="bug")

    calibration_data = X_cal
    cal_y= y_cal
    test_x= X_test
    test_y= y_test
    all_pre = clf.predict(X_test)
    y_preds, y_pss, p_value = Prom_thread.conformal_prediction(
        cal_x=calibration_data, cal_y=cal_y, test_x=test_x, test_y=test_y, significance_level="auto")

    # Evaluate conformal prediction
    print("Detect the drifting samples...")
    # Prom_thread.evaluate_mapie \
    #     (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y_test,
    #      significance_level=0.05)
    #
    # Prom_thread.evaluate_rise \
    #     (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y_test,
    #      significance_level=0.05)

    index_all_right, index_list_right, Acc_all, F1_all, Pre_all, Rec_all,index_list,common_elements \
        = Prom_thread.evaluate_conformal_prediction \
        (y_preds=y_preds, y_pss=y_pss, p_value=p_value, all_pre=all_pre, y=y_test,significance_level='auto')

    # Increment learning
    # print("Finding the most valuable instances for incremental learning...")
    #
    # print("a")
    # # y_preds, y_pss = {}, {}
    # # alphas = np.arange(0.1, 1, 0.1)
    # # # alphas=[0.1]
    # # for name, (method, include_last_label) in method_params.items():
    # #     mapie = MapieClassifier(estimator=clf, method=method, cv="prefit", random_state=42)
    # #     mapie.fit(X_cal, y_cal)
    # #     y_preds[name], y_pss[name] = mapie.predict(X_test, alpha=alphas, include_last_label=include_last_label)
    #
    # ##########
    # # def count_null_set(y: np.ndarray) -> int:
    # #     """
    # #     Count the number of empty prediction sets.
    # #
    # #     Parameters
    # #     ----------
    # #     y: np.ndarray of shape (n_sample, )
    # #
    # #     Returns
    # #     -------
    # #     int
    # #     """
    # #     count = 0
    # #     for pred in y[:, :]:
    # #         if np.sum(pred) == 0:
    # #             count += 1
    # #     return count
    #
    # # nulls, coverages, accuracies, sizes = {}, {}, {}, {}
    # # for name, (method, include_last_label) in method_params.items():
    # #     accuracies[name] = accuracy_score(y_test, y_preds[name])
    # #     nulls[name] = [
    # #         count_null_set(y_pss[name][:, :, i]) for i, _ in enumerate(alphas)
    # #     ]
    # #     coverages[name] = [
    # #         classification_coverage_score(
    # #             y_test, y_pss[name][:, :, i]
    # #         ) for i, _ in enumerate(alphas)
    # #     ]
    # #     sizes[name] = [
    # #         y_pss[name][:, :, i].sum(axis=1).mean() for i, _ in enumerate(alphas)
    # #     ]
    # # # sizes里每个method最接近1的
    # # result = {}  # 用于存储结果的字典
    # # for key, lst in sizes.items():  # 遍历字典的键值对
    # #     closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - 1))  # 找到最接近1的数字的索引
    # #     result[key] = closest_index  # 将结果存入字典
    # # # y_ps_90中提出来那个最接近1的位置
    # # result_ps = {}
    # # for method, y_ps in y_pss.items():
    # #     result_ps[method] = y_ps[:, :, result[method]]
    # #
    # # index_all_tem = {}
    # # index_all_right_tem = {}
    # # for method, y_ps in result_ps.items():
    # #     for index, i in enumerate(y_ps):
    # #         num_true = sum(i)
    # #         if method not in index_all_tem:
    # #             index_all_tem[method] = []
    # #             index_all_right_tem[method] = []
    # #         if num_true != 1:
    # #             index_all_tem[method].append(index)
    # #         elif num_true == 1:
    # #             index_all_right_tem[method].append(index)
    # # index_all = []
    # # index_list = []
    # # # 遍历字典中的每个键值对
    # # for key, value in index_all_tem.items():
    # #     # 使用集合对列表中的元素进行去重，并转换为列表
    # #     list_length = len(value)
    # #     # print(f"Length of {key}: {list_length}")
    # #     # 将去重后的列表添加到新列表中
    # #     index_all.extend(value)
    # #     index_list.append(value)
    # # index_all = list(set(index_all))
    # # # print(f"Length of index_all: {len(index_all)}")
    # # index_list.append(index_all)
    # #
    # # index_all_right = []
    # # index_list_right = []
    # # # 遍历字典中的每个键值对
    # # for key, value in index_all_right_tem.items():
    # #     # 使用集合对列表中的元素进行去重，并转换为列表
    # #     list_length = len(value)
    # #     # print(f"Length of {key}: {list_length}")
    # #     # 将去重后的列表添加到新列表中
    # #     index_all_right.extend(value)
    # #     index_list_right.append(value)
    #
    # index_all_right = list(set(list(range(len(y_test)))) - set(index_all))
    # # print(f"Length of index_all: {len(index_all_right)}")
    # index_list_right.append(index_all_right)
    """metric"""
    # acc_best = 0
    # F1_best = 0
    # pre_best = 0
    # rec_best = 0
    # # 投票
    # method_name = {
    #     # "naive": ("naive", False),
    #     "score": ("score", False),
    #     "cumulated_score": ("cumulated_score", True),
    #     "random_cumulated_score": ("cumulated_score", "randomized"),
    #     "top_k": ("top_k", False),
    #     "mixture": ("mixture", False)
    # }
    # # 合并三个数组
    # find_recall = 0
    # find_pre = 0
    # find_acc = 0
    # random_element = []
    # for index_all, method_name_single in zip(index_list, method_name):
        # # 被错误分类的index
        # find_right_wrong = []
        # all_wrong = []
        # # 模型预测不准确的并且是(beigien)de
        # find_neg_right = []
        # index_tem = 0
        # predict_labels = model.predict(X_test)
        # for label, predict in zip(y_test_single, predict_labels):
        #     _, true_label = torch.topk(torch.tensor(label), k=1)
        #     pred_label = predict
        #     if index_tem in index_all:
        #         if true_label != pred_label:
        #             # 不确定的index中被错误分类的index（找到的不对的）
        #             find_right_wrong.append(index_tem)
        #             all_wrong.append(index_tem)
        #         # 找到的对的
        #     else:
        #         if true_label != pred_label:
        #             # 被错误分类的index（找到的不对的）
        #             all_wrong.append(index_tem)
        #         if true_label == pred_label:
        #             find_neg_right.append(index_tem)
        #     index_tem += 1
        # find_acc = (len(find_right_wrong) + len(find_neg_right)) / len(y_test_single)
        # # 找对了多少
        # if index_all == []:
        #     find_pre = 0
        # else:
        #     find_pre = len(find_right_wrong) / len(index_all)
        # # 有多少被找出来了（找到的不对的/所有不对的）
        # if len(all_wrong) == 0:
        #     break
        # find_recall = len(find_right_wrong) / len(all_wrong)
        # try:
        #     F1_find = 2 * find_pre * find_recall / (find_pre + find_recall)
        # except:
        #     F1_find = 0
        # if F1_best < F1_find:
        #     F1_best = F1_find
        #     acc_best = find_acc
        #     pre_best = find_pre
        #     rec_best = find_recall
        # logger.info(f"{method_name_single} find accuracy：{find_acc * 100:.2f}% "
        #             f"find precision为：{find_pre * 100:.2f}% "
        #             f"find recall为：{find_recall * 100:.2f}% "
        #             f"find F1：{F1_find * 100:.2f}%")
    """"IL"""

    selected_count = max(int(len(y_test) * 0.05), 1)
    np.random.seed(args.seed)
    try:
        random_element = np.random.choice(common_elements, selected_count, replace=False)
    except:
        random_element = np.random.choice(range(len(y_test_single)), selected_count)

    # logger.info(f"Best find ACC_best：{acc_best * 100:.2f}% "
    #             f"Best find pre_best：{pre_best * 100:.2f}% "
    #             f"Best find rec_best：{rec_best * 100:.2f}% "
    #             f"Best find F1_best：{F1_best * 100:.2f}%")
    # F1_find = F1_best
    #
    # Acc_all, F1_all, Pre_all, Rec_all
    result_find = {
        "find_acc": round(Acc_all[0], 4),
        "find_pre": round(Pre_all[0], 4),
        "find_rec": round(Rec_all[0], 4),
        "find_f1": round(F1_all[0], 4),
    }

    return result_find, random_element


def evaluate_test(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, "../../../benchmark/Bug/new_train.jsonl", args.one_hot_vectors, args.suffixes)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0,
                                 pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # 混淆矩阵
    # preds = logits[:, 0] > 0.5
    from sklearn.metrics import confusion_matrix, classification_report
    all_labels = []
    all_predictions = []
    for label, predict in zip(labels, logits):
        # 前向传播并获取预测结果
        label = torch.argmax(torch.tensor(label)).item()
        predict = torch.argmax(torch.tensor(predict)).item()
        # top_values, top_indices = torch.topk(torch.tensor(predict), k=3)
        # 将真实标签和预测标签加入列表中
        all_labels.append(label)
        all_predictions.append(predict)
    confusion_mat = confusion_matrix(all_labels, all_predictions)
    # print(confusion_mat)
    # from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score
    # eval_f1 = f1_score(all_labels, all_predictions, average='weighted')
    # eval_acc = accuracy_score(all_labels, all_predictions)
    # eval_rec = recall_score(all_labels, all_predictions, average='weighted')
    # eval_loss = eval_loss / nb_eval_steps
    # perplexity = torch.tensor(eval_loss)
    # result = {
    #     "eval_loss": float(perplexity),
    #     "eval_acc": round(eval_acc, 4),
    #     "eval_rec": round(eval_rec, 4),
    #     "eval_f1": round(eval_f1, 4),
    # }
    # return result
    TP = 0
    # FP=0
    # FN=0
    TN = 0
    for i, j in zip(labels, logits):
        # original positive
        _, i = torch.topk(torch.tensor(i), k=1)
        _, j = torch.topk(torch.tensor(j), k=1)
        if i in j:
            # 预测正确
            TP = TP + 1
        else:
            # 预测错误
            TN = TN + 1
        # if i == 1 and j == 1:
        #     TP=TP+1
        # elif i == 1 and j == 0:
        #     FN=FN+1
        # elif i == 0 and j == 1:
        #     FP = FP + 1
        # elif i == 0 and j == 0:
        #     TN=TN+1

    eval_acc = TP / (TP + TN)
    # try:
    #     eval_pre = TP/(TP+FP)
    # except:
    #     eval_pre= 0
    # try:
    #     eval_rec = TP/(TP+FN)
    # except:
    #     eval_rec = 0
    # try:
    #     eval_f1 = 2 * eval_pre*eval_rec/(eval_pre+eval_rec)
    # except:
    #     eval_f1 = 0

    # eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "test_loss": float(perplexity),
        "test_acc": round(eval_acc, 4),
        # "eval_pre": round(eval_pre, 4),
        # "eval_rec": round(eval_rec, 4),
        # "eval_f1": round(eval_f1, 4),
    }
    return result


def epoch_test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file, args.one_hot_vectors, args.suffixes)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader),disable=True):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)[0]
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    all_labels = []
    all_predictions = []
    for label, predict in zip(labels, logits):
        # 前向传播并获取预测结果
        label = torch.argmax(torch.tensor(label)).item()
        predict = torch.argmax(torch.tensor(predict)).item()

        # 将真实标签和预测标签加入列表中
        all_labels.append(label)
        all_predictions.append(predict)
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
    test_f1 = f1_score(all_labels, all_predictions, average='weighted')
    test_acc = accuracy_score(all_labels, all_predictions)
    test_rec = recall_score(all_labels, all_predictions, average='weighted')
    perplexity = torch.tensor(eval_loss)
    result = {
        "test_acc": round(test_acc, 4),
        "test_rec": round(test_rec, 4),
        "test_f1": round(test_f1, 4),
    }
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return result
    # preds = logits[:, 0] > 0.5
    # with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
    #     for example, pred in zip(eval_dataset.examples, preds):
    #         if pred:
    #             f.write(example.idx + '\t1\n')
    #         else:
    #             f.write(example.idx + '\t0\n')

    # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    # checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # model.load_state_dict(torch.load(output_dir))
    # model.to(args.device)
    # results = evaluate_test(args, model, tokenizer)
    # logger.info("***** Eval results *****")
    # # a=result['eval_f1']
    # for key in sorted(results.keys()):
    #     logger.info("  %s = %s", key, str(round(results[key], 4)))
    # results=epoch_uq(args, model, tokenizer)
    # nni.report_final_result(best_f1_uq)

    # if args.do_test and args.local_rank in [-1, 0]:
    # checkpoint_prefix = 'checkpoint-best-acc/model.bin'
    # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    # model.load_state_dict(torch.load(output_dir))
    # model.to(args.device)
    # aates(args, model, tokenizer)

    # return increment_acc


def model_initial():
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "learning_rate": 0.002,
            "alpha": 0.1,
            "epoch": 3,
            "seed": 4046,
            "method": 'top_k'
        }

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=32, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=params['learning_rate'], type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=params['epoch'],
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--method', type=str, default=params['method'])
    parser.add_argument('--mode', choices=['train', 'deploy'], help="Mode to run: train or deploy")
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size
    args.per_gpu_eval_batch_size = args.eval_batch_size
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set the random seed
    args.seed = int(args.seed)
    # print("seed:", args.seed)
    set_seed(args.seed)
    # 多分类数量
    labels_num = 8
    # Data Preprocess


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    #
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = labels_num
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)
    return model, config, tokenizer, args

def codebert_train(model_pre, config, tokenizer, args):
    model = BiLSTMModel(model_pre, config, tokenizer, args)
    prom_loop = Bug_detection(model=model)

    # dataset partition
    print("dataset partition...")
    prom_loop.data_partitioning(dataset=r'../../../benchmark/Bug', random_seed=args.seed, num_folders=8)
    args.one_hot_vectors, args.suffixes = onehot(args.train_data_file, args.seed)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        print("Extracting features...")
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.one_hot_vectors, args.suffixes)
        if args.local_rank == 0:
            torch.distributed.barrier()
        print("Training the underlying model...")
        best_acc = train(args, train_dataset, model, tokenizer)
    nni.report_final_result(best_acc)

def codebert_deploy(model_pre, config, tokenizer, args):
    model = BiLSTMModel(model_pre, config, tokenizer, args)
    prom_loop = Bug_detection(model=model)

    # dataset partition
    print("dataset partition...")
    prom_loop.data_partitioning(dataset=r'../../../benchmark/Bug', random_seed=args.seed, num_folders=8)
    args.one_hot_vectors, args.suffixes = onehot(args.train_data_file, args.seed)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        print("Extracting features...")
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.one_hot_vectors, args.suffixes)
        if args.local_rank == 0:
            torch.distributed.barrier()
        print("Training the underlying model...")
        increment_acc = deploy(args, train_dataset, model, tokenizer)
        # increment_acc = train(args, train_dataset, model, tokenizer)
        print("The best incremental accuracy is: ", increment_acc)
    nni.report_final_result(increment_acc)

if __name__ == "__main__":
    # initial the model parameters
    print("initial the model parameters...")
    model_pre, config, tokenizer, args = model_initial()
    if args.mode == 'train':
        codebert_train(model_pre, config, tokenizer, args)
    elif args.mode == 'deploy':
        codebert_deploy(model_pre, config, tokenizer, args)
    # codebert_train(model_pre, config, tokenizer, args)
    codebert_deploy(model_pre, config, tokenizer, args)
    """
    --output_dir=./saved_models     --model_type=roberta     --tokenizer_name=microsoft/codebert-base     --model_name_or_path=microsoft/codebert-base   --do_train  --do_eval     --do_test     --train_data_file=../../../benchmark/Bug/train.jsonl     --eval_data_file=../../../benchmark/Bug/valid.jsonl     --test_data_file=../../../benchmark/Bug/test.jsonl --evaluate_during_training
    """


