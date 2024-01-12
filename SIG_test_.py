# -*- coding: utf-8 -*-
import torch

import random
import os

import re
import config

from transformers import BartTokenizer, AutoConfig,BartConfig

from transformers import AutoTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

from transformers import BertTokenizer
import jieba
import pandas as pd
from utils.load_data import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

random.seed(1)

tokenizer = BertTokenizer.from_pretrained(config.bart_model_dir)

cfg = BartConfig.from_pretrained(config.bart_model_dir)
cfg.gradient_checkpointing = True



def read_file(file_name):
    fp = open(file_name, "r", encoding="utf-8")
    content_lines = fp.readlines()
    fp.close()
    # Remove the newline character at the end of the line, otherwise it will interfere with the stop word matching process
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")

    return content_lines


def test_generation():
    model = BartForConditionalGeneration.from_pretrained(config.bart_model_dir, config=cfg)
    # model = CPTForConditionalGeneration.from_pretrained(config.cpt_model_dir)
    model.load_state_dict(torch.load(os.path.join(config.resume_dir, 'sig.pt'), map_location='cpu')['model'])
    model.to(device)
    model.eval()
    text2text_generator = Text2TextGenerationPipeline(model, tokenizer, device=0)
    paragraph = """
    “这四人上了岸，只怕泄漏了我此番南来的机密。”欧阳锋一直冷眼旁观，看到他大刺刺的神情早就心中大是不忿，暗想瞧你张三这副落汤鸡般的狼狈模样。
    “这四人上了岸，只怕泄漏了我此番南来的机密。” 的说话者是[MASK]。
    """


    generated_text = text2text_generator(paragraph, max_length=150, do_sample=False)[0]['generated_text']
    output = generated_text.split("：")
    output = re.sub(" ", '', output[1])


    print(output)


def get_candidate_list(data_file, sheet_name="data"):
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    df["novel"] = df["id"].apply(lambda x: x.split('_')[0])
    candidate_list = df["answer"].tolist()
    candidate_list = list(set(candidate_list))

    return candidate_list


def get_candidate_list_from_chinese_context(context:str, stopwords:list):

    #pattern = r'[\u201c\u300c].*?[\u201d\u300d]'
    assert "“" in context
    context = '“' + context + '”'

    #result = re.sub(pattern, '', context)
    seg_list = jieba.lcut(
        context,
        cut_all=False)
    pre_index = -1
    suf_index = -2
    remove_index_list = []

    for seg_index, seg_word in enumerate(seg_list):

        if seg_word == "“":
            pre_index = seg_index
        elif seg_word == "”":
            suf_index = seg_index
            if pre_index != -1:
                for remove_index in range(pre_index, suf_index + 1):
                    remove_index_list.append(remove_index)
                pre_index = -1

    final_list = list(set([v for i, v in enumerate(seg_list) if v not in stopwords and len(v) > 1 and i not in remove_index_list]))

    return final_list


def cal_prob_batch(target_text: list, input_text: list, model, tokenizer):

    encodings = tokenizer(input_text, return_tensors="pt", max_length=config.input_max_length, padding='max_length', add_special_tokens=True,
                          truncation=True,)

    encodings = {k: v.to(device) for k, v in encodings.items()}
    labels = tokenizer(target_text, return_tensors="pt", max_length=20, padding='max_length', add_special_tokens=True,
                       truncation=True,)['input_ids'].to(device)

    decoder_input_ids = torch.cat([torch.zeros_like(labels[:, :1]), labels[:, :-1]], dim=-1).to(device)

    with torch.no_grad():
        outputs = model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"],
                        labels=labels, decoder_input_ids=decoder_input_ids)

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
    # labels_token_prob_list[labels == 20517] = 0
    # labels_token_prob_list[labels == 20494] = 0
    # labels_token_prob_list[labels == 17520] = 0
    # labels_token_prob_list[labels == 25832] = 0
    # Calculate the probability of generating each label and sum the probabilities of all tokens in labels_token_prob_list
    non_zero_counts = torch.count_nonzero(labels_token_prob_list, dim=1)

    row_sums = torch.sum(labels_token_prob_list, dim=1)
    labels_prob_list = row_sums / non_zero_counts.float()

    return labels_prob_list


def test_classify():

    random.seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = config.test_dir


    dev_data = CleanData(data_path,
                         save_punctuations=True,
                         save_stopwords=True,
                         sheet_name='data')

    dev_cleaned_data = dev_data.unsplit_data()

    candidates_list = dev_cleaned_data["answer"].tolist()

    candidates_list = list(set(candidates_list))
    cfg = BartConfig.from_pretrained(config.bart_model_dir)
    cfg.gradient_checkpointing = True

    model = BartForConditionalGeneration.from_pretrained(config.bart_model_dir, config=cfg)
    # model = CPTForConditionalGeneration.from_pretrained(config.cpt_model_dir)
    model.load_state_dict(torch.load(os.path.join(config.resume_dir, 'sig.pt'), map_location='cpu')['model'])
    model.to(device)
    model.eval()
    k = 10
    paragraph = """
    "“你别多事！我偏要问她个明白。”陆无双向耶律齐瞪了一眼，道：“狗咬吕洞宾，将来有得苦头给你吃的。”耶律齐脸上一红，心知陆无双已瞧出自己对郭英生了情意，这句话是说，这姑娘如此蛮不讲理，只怕你后患无穷。郭芙瞥见耶律齐突然脸红，疑心大起，追问：“你也疑心我不是爹爹、妈妈的亲生女儿？” 的说话者是[MASK]。耶律齐道：“不是，不是，咱们走罢，别理会她了。”陆无双抢着道：“他自然疑心啊，否则何以要你快走？”郭芙满脸通红，按剑不语。,
    “你也疑心我不是爹爹、妈妈的亲生女儿？” 的说话者是[MASK]。
    """


    stopwords = read_file(config.stopwords_dir)
    final_list = get_candidate_list_from_chinese_context(paragraph, stopwords)
    print(list(final_list))
    added_candidates = ['张三', '李四', '王五', '赵六','欧阳锋', '郭芙']


    final_list.extend(added_candidates)

    candidates_list = final_list
    candidates_template_list = ["s" for _ in range(len(candidates_list))]
    for i, candidates in enumerate(candidates_list):
        candidates_template_list[i] = "说话者是: " + candidates

    print(len(candidates_template_list))
    input_text = [paragraph for _ in
                  range(len(candidates_list))]
    text_prob, top5_begin_p = cal_prob_batch(candidates_template_list, input_text, model, tokenizer)  # 计算各个候选者的生成概率

    max_prob = text_prob.argmax(dim=-1)
    value, indice = text_prob.topk(k, dim=-1, largest=True, sorted=True)
    answer = candidates_list[max_prob.item() % len(candidates_list)]
    print(answer)
    topk_answer = [candidates_list[j % len(candidates_list)] for j in indice.cpu().numpy().tolist()]
    print(topk_answer)
    print([candidates_list[j % len(candidates_list)] for j in top5_begin_p.cpu().numpy().tolist()])


if __name__ == '__main__':
    test_generation()
