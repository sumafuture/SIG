import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import json
import random
import os
from torch.utils.data import Dataset
import time
import config
import numpy as np
import copy
from tqdm import tqdm
from transformers import BartTokenizer, AutoConfig,BartConfig
from transformers import BertTokenizer
import glob
import logging
from transformers import AutoTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

import re
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import pandas as pd

from utils.training_control import *
from utils.load_data import *
from evaluate import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

random.seed(1)

tokenizer = BertTokenizer.from_pretrained(config.bart_model_dir)

cfg = BartConfig.from_pretrained(config.bart_model_dir)
cfg.gradient_checkpointing = True

model = BartForConditionalGeneration.from_pretrained(config.bart_model_dir, config=cfg)
#model = CPTForConditionalGeneration.from_pretrained(config.cpt_model_dir)

if config.is_resume:
    model.load_state_dict(torch.load(os.path.join(config.resume_dir, 'sig.pt'), map_location='cpu')['model'])
    print("resume from" + config.resume_dir)

model.to(device)

# training log
LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%m:%s %a'




class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.episode_id = []
        self.context = []
        self.question = []
        self.answer = []
        self.utter_text = []
        self.lower_text = []
        self.listener = []
        data.sample(frac=1)

        self.only_speaker = config.is_only_speaker if hasattr(config, 'is_only_speaker') else True
        for sample in data.index:
            self.episode_id.append(data.loc[sample].values[0])
            self.context.append(data.loc[sample].values[1])
            self.question.append(data.loc[sample].values[2])
            self.answer.append(data.loc[sample].values[3])
            self.utter_text.append(data.loc[sample].values[5])

            self.lower_text.append(data.loc[sample].values[6])
            self.listener.append(data.loc[sample].values[7])

    def __getitem__(self, item):
        speaker = self.answer[item]
        above = self.utter_text[item]

        below = self.lower_text[item]
        listener = self.listener[item]
        quotetext = self.question[item]

        source_template = config.source_template
        target_template = config.target_template
        if not config.is_only_speaker:
            try:
                context = above + "[SEP]" + quotetext + source_template["speaker_prompt_template"] + source_template["addressee_prompt_template"] + "[SEP]" + below
                question = "[CLS]" + quotetext + source_template["speaker_prompt_template"] + '[SEP]' + source_template["addressee_prompt_template"]
                answer = target_template["speaker_prompt_template"] + speaker + '[SEP]' + target_template["addressee_prompt_template"] + listener

            except TypeError:

                context = str(above) + "[SEP]" + str(quotetext) + source_template["speaker_prompt_template"] + source_template["addressee_prompt_template"] + "[SEP]" + str(
                    below)
                question = "[CLS]" + quotetext + source_template["speaker_prompt_template"] + '[SEP]' + source_template[
                    "addressee_prompt_template"]
                answer = target_template["speaker_prompt_template"] + speaker + '[SEP]' + target_template[
                    "addressee_prompt_template"] + listener

        else:
            try:
                context = above + "[SEP]" + source_template["speaker_prompt_template_prefix"] + quotetext + "[SEP]" + below
                question = "[CLS]" + quotetext + source_template["speaker_prompt_template"]
                answer = target_template["speaker_prompt_template"]  + speaker + '[SEP]'

            except TypeError:

                context = str(above) + "[SEP]" + source_template["speaker_prompt_template_prefix"] + quotetext + "[SEP]" + str(below)
                question = "[CLS]" + str(quotetext) + source_template["speaker_prompt_template"]
                answer = target_template["speaker_prompt_template"]  + str(speaker) + '[SEP]'



        if config.is_add_question:
            source = tokenizer(context + question,
                               max_length=config.input_max_length,
                               add_special_tokens=True,
                               truncation=True,
                               padding="max_length",
                               return_tensors="pt"
                               )

        else:
            source = tokenizer(context,
                               max_length=config.input_max_length,
                               add_special_tokens=True,
                               truncation=True,
                               padding="max_length",
                               return_tensors="pt"
                               )



        source_i = source["input_ids"]
        source_m = source["attention_mask"]

        target = tokenizer(answer,
                           max_length=20,  # The value cannot be lower than the maximum label length after the target template is added
                           add_special_tokens=True,
                           truncation=True,
                           return_tensors="pt",
                           padding="max_length"
                           )

        target_i = target["input_ids"]
        target_m = target["attention_mask"]

        trg_labels = target_i * target_m.int() + (-100) * (1 - target_m.int())

        trg_labels = trg_labels[..., 1:].contiguous()  # text</s>
        trg_input_ids = target_i[..., : -1].contiguous()  # <s>text
        target_mask = target_m[..., : -1].contiguous()

        return source_i.to(device), source_m.to(device), trg_labels.to(device), trg_input_ids.to(device), \
               target_mask.to(device)

    def __len__(self):
        return len(self.episode_id)


def collate_fn(data):
    src_input_ids = [dataitem[0] for dataitem in data]
    src_input_ids = pad_sequence(src_input_ids, batch_first=True,
                                 padding_value=1)
    src_input_ids = torch.squeeze(src_input_ids)

    src_attention_mask = [dataitem[1] for dataitem in data]
    src_attention_mask = pad_sequence(src_attention_mask, batch_first=True, padding_value=0)
    src_attention_mask = torch.squeeze(src_attention_mask)

    trg_labels = [dataitem[2] for dataitem in data]

    trg_labels = pad_sequence(trg_labels, batch_first=True, padding_value=-100)
    trg_labels = torch.squeeze(trg_labels)

    trg_input_ids = [dataitem[3] for dataitem in data]
    trg_input_ids = pad_sequence(trg_input_ids, batch_first=True, padding_value=1)
    trg_input_ids = torch.squeeze(trg_input_ids)

    trg_m = [dataitem[4] for dataitem in data]
    trg_m = pad_sequence(trg_m, batch_first=True, padding_value=0)
    trg_m = torch.squeeze(trg_m)

    return src_input_ids, src_attention_mask, trg_labels, trg_input_ids, trg_m




scaler = GradScaler()


def train(model, epochs=config.train_epoch, train_sample_num=-1, accumulation_steps=config.accumulation_steps, dev_epoch=config.dev_epoch, lr=2e-5, save=config.is_save,only_speaker=False):
    bsz = config.batch_size

    patience = 0
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if config.is_resume:
        optimizer.load_state_dict(
            torch.load(os.path.join(config.resume_dir, 'sig.pt'))['optimizer'])
    loss_item_list = []
    epochs_loss = []
    train_loss = 0
    train_size = 0
    acc_list = []
    dev_acc = -1.0
    fiction_data = CleanData(config.train_dir,
                             save_punctuations=True,
                             save_stopwords=True,
                             sheet_name='data')
    fiction_cleaned_data = fiction_data.unsplit_data()
    train_set = MyDataset(fiction_cleaned_data
                                     )

    for epoch in range(epochs):
        #  Load data
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=bsz,
                                                   collate_fn=collate_fn,
                                                   shuffle=True,
                                                   drop_last=True
                                                   )

        if train_sample_num < 0:
            train_size += len(train_loader)

        else:
            train_size += train_sample_num


        for i, (
                src_input_ids, src_attention_mask, trg_labels, trg_input_ids, trg_mask) in enumerate(
            train_loader):
            with autocast():
                out = model(input_ids=src_input_ids,
                            attention_mask=src_attention_mask,
                            decoder_input_ids=trg_input_ids,
                            decoder_attention_mask=trg_mask,
                            labels=trg_labels,
                            output_hidden_states=True,
                            return_dict=True
                            )

                # last_hidden_state = out["encoder_last_hidden_state"]
                # ave_last_hidden_state = torch.mean(last_hidden_state, dim=1)
                logits = out["logits"]
                loss = out["loss"]

                loss = loss.float()

            loss.requires_grad_(True)
            scaler.scale(loss).backward()
            # fiction_output_embedding.append(ave_last_hidden_state.cpu().detach().numpy().tolist())

            if loss != loss:
                raise Exception('NaN in loss, crack!')
            train_loss += loss.cpu().detach().numpy().tolist()

            epochs_loss.append(loss.item())

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)  # Update parameter
                optimizer.zero_grad()
                scaler.update()

                loss_item_list.append(np.mean(epochs_loss))
                epochs_loss = []

            if i == train_sample_num:
                break
                #torch.save(model.state_dict(),  os.path.join(checkpoint_dir, 'sig.pt'))

        # adjust learning rate after each epoch
        adjust_learning_rate(optimizer, config.lr_decay)

        print("Training set average error：{}".format(train_loss / train_size))

        if epoch % dev_epoch == 0:
            dev_outcome = dev_generation(config.dev_dir, model=model, tokenizer=tokenizer, dev_sheet="data",
                                            )

            recent_dev_acc = dev_outcome["accuracy"]
            #recent_dev_topk_acc = dev_outcome["topk_accuracy"]
            if recent_dev_acc > dev_acc:
                dev_acc = recent_dev_acc
                patience = 0

                if save:
                    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    # checkpoint
                    checkpoint_dir = os.path.join(config.checkpoint_dir,
                                                  os.path.join(config.model_name, timestamp))
                    try:
                        save_checkpoint({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            {

                                'training_loss': train_loss / train_size,
                                'dev_acc': round(recent_dev_acc, 4),
                                # 'tokenizer_dir': config.bart_model_dir,
                                # "model_dir": config.bart_model_dir,
                                # "cfg_dir": config.bart_model_dir

                            },
                            checkpoint_dir)
                    except Exception as e:
                        print(e)
            else:
                if recent_dev_acc < dev_acc - config.tolerate_threshold:
                    patience += 1
                    if patience >= config.patience:
                        print()
                        print("Early stop")
                        break


if __name__ == '__main__':
    train(model=model)