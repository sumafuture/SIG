import os
import torch
import sys
# from typing import Sequence
sys.path.insert(0, os.getcwd())

data_dir = os.getcwd() + '/data/WP2021/'  # dataset (train, dev. test)
bart_model_dir = 'pretrain_model/bart-large-chinese/'

stopwords_dir = os.getcwd() + '/data/stopwords.txt'
file_list = ['train_unsplit.xlsx', 'dev_unsplit.xlsx', 'test_unsplit.xlsx']
train_dir = data_dir + file_list[0]
dev_dir = data_dir + file_list[1]
test_dir = data_dir + file_list[2]

checkpoint_dir = "checkpoint/SIG"
log_dir = checkpoint_dir
candidate_dir = os.getcwd() + '/data/WP2021/name_list.txt'  # candidate_list of WP2021
is_save = False
is_resume = True
is_only_speaker = False
resume_dir = r""  # checkpoint load address
model_name = "WP2021"

# Hyperparameters
lr = 1e-10
batch_size = 4
lr_decay = 0.95
patience = 3
tolerate_threshold = 0.01  # Tolerance threshold for model performance worse than last evaluation
dev_epoch = 2
max_dev_sample = 500  # Maximum number of evaluation samples
accumulation_steps = 4
topk = 2
input_max_length = 612

# Because the data set currently used by SIG is fixed to labels, in order to correspond generatef answer to labels, common speaker suffixes are recorded for fuzzy matching.

normal_chinese_suffix = ["先生", '小姐', '书记', '主任', '老', '长官', '长', '阿姨', '叔', '叔叔', '爷', '奶奶', '医生','大夫',
                         '护士', '师兄', '师姐', '支书', "师傅", "师父"]

