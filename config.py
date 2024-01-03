import os
import torch
import sys
# from typing import Sequence
sys.path.insert(0,os.getcwd())

data_dir = os.getcwd() + '/data/WP2021/'  # dataset (train, dev. test)
bart_model_dir = 'pretrain_model/bart-large-chinese/'
cpt_model_dir = 'pretrain_model/cpt-large/'
stopwords_dir = os.getcwd() + '/data/stopwords.txt'
file_list = ['train(4).xlsx', 'dev(4).xlsx', 'test(4).xlsx']
train_dir = data_dir + file_list[0]
dev_dir = data_dir + file_list[1]
test_dir = data_dir + file_list[2]

checkpoint_dir = "checkpoint/SIG"
log_dir = checkpoint_dir

is_save = False
is_resume = True
is_only_speaker = False
resume_dir = r"D:\speaker identification baseline\Project SIG(chinese)\checkpoint\SIG\WP2021\20231101120357"
model_name = "WP2021"

# Hyperparameters
lr = 1e-10
batch_size = 4
lr_decay = 0.95
patient = 3  # 允许模型表现较差的次数
tolerate_threshold = 0.01  # 对于模型表现差于上一次评估的容忍阈值
dev_step = 2  # 每几轮进行一次评估
max_dev_sample = 500  # 最大评估样本数
accumulation_steps = 4
topk = 2


normal_chinese_suffix = ["先生", '小姐', '书记', '主任', '老', '长官', '长', '阿姨', '叔', '叔叔', '爷', '奶奶', '医生','大夫',
                         '护士', '师兄', '师姐', '支书', "师傅", "师父"]