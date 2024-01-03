import ast

import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import random
import os
from torch.utils.data import Dataset
import gc
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from tqdm import tqdm
from transformers import BartTokenizer, AutoConfig,BartConfig
import glob
import logging
from transformers import AutoTokenizer, BartForConditionalGeneration
import re
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import pandas as pd
import time
from assistant import is_pure_dialogue
from assistant import get_part_words_string
import assistant
from eval.evaluate import *

from data_process import CleanData
from data_process import save_to_excel
import data_process as dp

from sklearn import metrics
from assistant import diff_num_speaker
from assistant import interval_dev


random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

cfg = BartConfig.from_pretrained("facebook/bart-large")
cfg.gradient_checkpointing = True

Bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large", config=cfg)

Bart.to(device)

model = Bart
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(stream_handler)


def get_candidate_list(fiction_name, data_file, sheet_name="dev"):
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    fictions_df = dict(tuple(df.groupby('fiction_name')))
    candidate_list = fictions_df[fiction_name]["speaker"].tolist()
    candidate_list = list(set(candidate_list))

    return candidate_list


def cal_prob_batch(target_text: list, input_text: list, model, tokenizer):

    encodings = tokenizer(input_text, return_tensors="pt", max_length=512, padding='max_length', add_special_tokens=True,
                          truncation=True,)

    encodings = {k: v.to(device) for k, v in encodings.items()}
    labels = tokenizer(target_text, return_tensors="pt", max_length=20, padding='max_length', add_special_tokens=True,
                       truncation=True,)['input_ids'].to(device)

    decoder_input_ids = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=-1).to(device)


    with torch.no_grad():
        outputs = model(**encodings, labels=labels, decoder_input_ids=decoder_input_ids)

    logits = outputs["logits"].detach()

    logits_softmax = torch.softmax(logits, dim=-1)
    # logits_softmax = logits

    labels_token_prob_list = [logits_softmax[i, range(labels.shape[-1]), labels[i, :]] for i in
                              range(labels.shape[0])]

    labels_token_prob_list = torch.stack(labels_token_prob_list)
    # The generation probability of the location of the special token is set to 0
    labels_token_prob_list[labels == 0] = 0
    labels_token_prob_list[labels == 1] = 0
    labels_token_prob_list[labels == 2] = 0

    # Calculate the probability of generating each label and sum the probabilities of all tokens in labels_token_prob_list
    non_zero_counts = torch.count_nonzero(labels_token_prob_list, dim=1)

    row_sums = torch.sum(labels_token_prob_list, dim=1)
    labels_prob_list = row_sums / non_zero_counts.float()

    return labels_prob_list


class MyDataset2(Dataset):
    def __init__(self, data, is_train=True, candidates='', alias_dict=None, only_speaker=False, only_address=False, normal_setting=False):
        super(MyDataset2, self).__init__()
        self.episode_id = []
        self.quoteText = []

        self.labels = []
        self.global_attention_mask = []
        self.index = []
        self.speaker = []
        self.above = []
        self.below = []
        self.tokenizer = tokenizer
        self.quoteText = []
        self.is_person = []
        self.quoteType = []
        self.addressee = []
        self.weight = []
        self.aliases = []
        self.address_aliases = []
        self.address_num = []
        self.only_speaker = only_speaker
        self.only_address = only_address
        data.sample(frac=1)

        if normal_setting:
            for sample in data.index:
                if data.loc[sample].values[1] == "Explicit":
                    self.quoteText.append(data.loc[sample].values[0])
                    self.quoteType.append(data.loc[sample].values[1])  # quoteType
                    self.addressee.append(data.loc[sample].values[2])
                    self.above.append(data.loc[sample].values[3])
                    self.below.append(data.loc[sample].values[4])
                    self.weight.append(data.loc[sample].values[5])  # weight of quotation
                    self.speaker.append(data.loc[sample].values[6])  # speaker
                    self.aliases.append(data.loc[sample].values[7])  # aliases
                    self.address_aliases.append(data.loc[sample].values[8])
                    self.address_num.append(data.loc[sample].values[9])

        else:
            for sample in data.index:

                if data.loc[sample].values[1] != "Explicit":
                    self.quoteText.append(data.loc[sample].values[0])
                    self.quoteType.append(data.loc[sample].values[1])
                    self.addressee.append(data.loc[sample].values[2])
                    self.above.append(data.loc[sample].values[3])
                    self.below.append(data.loc[sample].values[4])
                    self.weight.append(data.loc[sample].values[5])
                    self.speaker.append((data.loc[sample].values[6]))
                    self.aliases.append((data.loc[sample].values[7]))
                    self.address_aliases.append(data.loc[sample].values[8])
                    self.address_num.append(data.loc[sample].values[9])


    def __getitem__(self, item):
        aliases = self.aliases[item]
        address_aliases = self.address_aliases[item]
        addressee = self.addressee[item]
        above = self.above[item]
        above = get_part_words_string(above, is_before=False)
        speaker = self.speaker[item]
        below = self.below[item]
        below = get_part_words_string(below, is_before=True)
        quotetext = self.quoteText[item]
        weight = torch.tensor(self.weight[item], dtype=torch.long)

        if self.only_speaker:
            source = above + quotetext + " " + 'replied by' + '<mask>' + " " + below
            target = "Speaker: " + speaker
        elif self.only_address:
            source = above + quotetext + " listened by " + "<mask>" + " ." + below
            target = "Addressees: " + addressee

        else:
            source = above + quotetext + ' which replied by ' + '<mask>' + " is listened by " + "<mask>" + " ." + below
            target = "replied by : " + speaker + " " + "listened by: " + addressee

        return {"source": source, "speaker_label": speaker, "address_label": addressee, "weight": weight,"aliases": aliases}

    def __len__(self):
        return len(self.speaker)

    @staticmethod
    def collate_fn(data):
        speaker = [item["speaker_label"] for item in data]
        address = [item["address_label"] for item in data]
        source = [item["source"] for item in data]
        aliases = [item["aliases"] for item in data]
        weight = [item["weight"] for item in data]
        weight = torch.stack(weight, dim=0)
        return {"source": source, "speaker_label": speaker, "address_label": address, "weight": weight, "aliases": aliases}


def chinese_dev_classify(data_dir, model_dir, dev_fiction: list, dev_sheet="dev"):
    data_dir = data_dir

    model_dict = torch.load(model_dir)
    model.load_state_dict(model_dict)
    model.eval()
    bsz = 4
    total = 0
    correct = 0
    error = 0
    error_in_context = 0
    error_not_in_context = 0
    error_in_answer_in_context = 0
    error_in_answer_not_in_context = 0
    error_not_in_answer_not_in_context = 0
    error_not_in_answer_in_context = 0
    correct_answer_in = 0
    correct_topk = 0
    data_path = data_dir

    dev_data = CleanData(data_path,
                         save_punctuations=True,
                         save_stopwords=True,
                         sheet_name=dev_sheet)

    dev_cleaned_data = dev_data.clean_address_data()
    # dev_cleaned_data = dp.avoid_long_tail(dev_cleaned_data)

    fictions_df_dict = dict(tuple(dev_cleaned_data.groupby('fiction_name')))  # Split the test data by novel name
    for fiction_name, fiction_df in fictions_df_dict.items():
        val_outputs = []
        val_targets = []
        if fiction_name in dev_fiction:
            fiction_total = 0
            fiction_correct = 0
            fiction_correct_topk = 0
            candidates_list = fiction_df["speaker"].tolist()

            candidates_list = list(set(candidates_list))
            k = 5
            if len(candidates_list) < k:
                k = len(candidates_list)

            candidates_template_list = ["s" for _ in range(len(candidates_list))]
            for i, candidates in enumerate(candidates_list):
                candidates_template_list[i] = "Speaker: " + candidates

            dev_set = MyDataset2(fiction_df, is_train=False, normal_setting=True)
            dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                                     batch_size=bsz,
                                                     collate_fn=MyDataset2.collate_fn,
                                                     shuffle=True,
                                                     drop_last=True
                                                     )

            for index, value_dict in enumerate(dev_loader):
                input_text = [x for x in value_dict["source"] for _ in
                              range(len(candidates_list))]  # [ source1 * bsz, source2 * 2]
                target_list = candidates_template_list * bsz  # [The number of candidates * bsz]

                weights = value_dict["weight"]  # [weight1,weight2,...]
                speakers = value_dict["speaker_label"]  # [speaker1,speaker2,...]
                addressees = value_dict["address_label"]
                aliases = value_dict["aliases"]
                text_prob = cal_prob_batch(target_list, input_text, model, tokenizer)
                text_probs = torch.chunk(text_prob, bsz)
                for i, prob in enumerate(text_probs):
                    max_prob = prob.argmax(dim=-1)
                    value, indice = prob.topk(k, dim=-1, largest=True, sorted=True)
                    val_outputs.append(max_prob.item() % len(candidates_list))
                    val_targets.append(candidates_list.index(speakers[i]))
                    answer = candidates_list[max_prob.item() % len(candidates_list)]
                    total += weights[i]
                    fiction_total += weights[i]
                    if answer == speakers[i] or answer in aliases[i]:
                        correct += weights[i]
                        fiction_correct += weights[i]

                        if answer in value_dict["source"][i]:
                            correct_answer_in += 1

                    else:
                        error += 1
                        if answer in value_dict["source"][i]:
                            error_in_context += 1
                            if speakers[i] in value_dict["source"][i]:
                                error_in_answer_in_context += 1
                            else:
                                error_in_answer_not_in_context += 1

                        else:
                            error_not_in_context += 1
                            if speakers[i] in value_dict["source"][i]:
                                error_not_in_answer_in_context += 1
                            else:
                                error_not_in_answer_not_in_context += 1

                    topk_answer = [candidates_list[j % len(candidates_list)] for j in indice.cpu().numpy().tolist()]

                    if speakers[i] in topk_answer:
                        correct_topk += weights[i]
                        fiction_correct_topk += weights[i]



def dev_classify(data_dir, model_dir, dev_fiction: list, dev_sheet="PDNC"):
    data_dir = data_dir

    model_dict = torch.load(model_dir)
    model.load_state_dict(model_dict)
    model.eval()

    model_dict = torch.load(model_dir)
    model.load_state_dict(model_dict)
    model.eval()
    bsz = 4
    total = 0
    correct = 0
    error = 0
    error_in_context = 0
    error_not_in_context = 0
    error_in_answer_in_context = 0
    error_in_answer_not_in_context = 0
    error_not_in_answer_not_in_context = 0
    error_not_in_answer_in_context = 0
    correct_answer_in = 0
    correct_topk = 0
    data_path = data_dir

    dev_data = CleanData(data_path,
                         save_punctuations=True,
                         save_stopwords=True,
                         sheet_name=dev_sheet)

    dev_cleaned_data = dev_data.clean_address_data()
    # dev_cleaned_data = dp.avoid_long_tail(dev_cleaned_data)

    fictions_df_dict = dict(tuple(dev_cleaned_data.groupby('fiction_name')))  # Split the test data by novel name
    for fiction_name, fiction_df in fictions_df_dict.items():
        val_outputs = []
        val_targets = []
        if fiction_name in dev_fiction:
            fiction_total = 0
            fiction_correct = 0
            fiction_correct_topk = 0
            candidates_list = fiction_df["speaker"].tolist()

            candidates_list = list(set(candidates_list))
            k = 5
            if len(candidates_list) < k:
                k = len(candidates_list)

            candidates_template_list = ["s" for _ in range(len(candidates_list))]
            for i, candidates in enumerate(candidates_list):
                candidates_template_list[i] = "Speaker: " + candidates

            dev_set = MyDataset2(fiction_df, is_train=False, normal_setting=False)
            dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                                     batch_size=bsz,
                                                     collate_fn=MyDataset2.collate_fn,
                                                     shuffle=True,
                                                     drop_last=True
                                                     )

            for index, value_dict in enumerate(dev_loader):
                input_text = [x for x in value_dict["source"] for _ in
                              range(len(candidates_list))]  # [ source1 * bsz, source2 * 2]
                target_list = candidates_template_list * bsz  # [The number of candidates * bsz]

                weights = value_dict["weight"]  # [weight1,weight2,...]
                speakers = value_dict["speaker_label"]  # [speaker1,speaker2,...]
                addressees = value_dict["address_label"]
                aliases = value_dict["aliases"]
                text_prob = cal_prob_batch(target_list, input_text, model, tokenizer)
                text_probs = torch.chunk(text_prob, bsz)
                for i, prob in enumerate(text_probs):
                    max_prob = prob.argmax(dim=-1)
                    value, indice = prob.topk(k, dim=-1, largest=True, sorted=True)
                    val_outputs.append(max_prob.item() % len(candidates_list))
                    val_targets.append(candidates_list.index(speakers[i]))
                    answer = candidates_list[max_prob.item() % len(candidates_list)]
                    total += weights[i]
                    fiction_total += weights[i]
                    if answer == speakers[i] or answer in aliases[i]:
                        correct += weights[i]
                        fiction_correct += weights[i]

                        if answer in value_dict["source"][i]:
                            correct_answer_in += 1

                    else:
                        error += 1
                        if answer in value_dict["source"][i]:
                            error_in_context += 1
                            if speakers[i] in value_dict["source"][i]:
                                error_in_answer_in_context += 1
                            else:
                                error_in_answer_not_in_context += 1

                        else:
                            error_not_in_context += 1
                            if speakers[i] in value_dict["source"][i]:
                                error_not_in_answer_in_context += 1
                            else:
                                error_not_in_answer_not_in_context += 1

                    topk_answer = [candidates_list[j % len(candidates_list)] for j in indice.cpu().numpy().tolist()]

                    if speakers[i] in topk_answer:
                        correct_topk += weights[i]
                        fiction_correct_topk += weights[i]


    print(("Accuracy: {}".format(correct / total)))
    print("Top5 Accuracy: {}".format(correct_topk / total))

    print("The probability of the correct answer occurring in the context：{}".format(correct_answer_in / correct))

    print("------------------")
    print("When error: ")
    print("The wrong answer can be found in the context: {}".format(error_in_context / error))
    print("Both incorrect and correct answers are found in the context: {}".format(error_in_answer_in_context / error))
    print("Only wrong answer can be found in the context: {}".format(error_in_answer_not_in_context / error))
    print("Only correct answer can be found: {}".format(error_not_in_answer_in_context / error))
    print("Both incorrect and correct answers can't be found in the context:{}".format(error_not_in_answer_not_in_context / error))



