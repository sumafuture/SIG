
from torch.utils.data import Dataset

import logging
import torch
from utils.training_control import *
from utils.load_data import *
import config
from fastprogress import master_bar, progress_bar
from tqdm import tqdm
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
from modeling_cpt import CPTForConditionalGeneration

random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(stream_handler)


def generation_process(answer: str):
    try:
        output = answer.split("：")
        output = re.sub(" ", '', output[1])
        content_after_colon = output.strip()
    except:
        output = re.sub(" ", '', answer)
        content_after_colon = output.strip()
    return content_after_colon


def judge_equal(answer:str, label: str):
    """

    Args:
        answer: The answer of model
        label: The golden answer

    Returns: Whether the answer and label has the same meaning

    In a chinese paragraph, 王明 can be referred to as 王先生 or simply 明. These calls all refer to the same person.
    """
    has_normal_suffix = False
    answer = "".join(list(answer))
    label = "".join(list(label))
    if any(x in answer or x in label for x in config.normal_chinese_suffix):
        has_normal_suffix = True
    if answer in label or label in answer or answer == label:
        return True

    correct = 0.0
    answer2list = list(answer)
    label2list = list(label)
    for l in label2list:
        if l in answer2list:
            correct += 1
        if correct / len(label2list) >= 0.5 and label2list[-1] == answer2list[-1]:
            return True
    if label2list[0] == answer2list[0] and has_normal_suffix:
        return True
    return False


def get_candidate_list(data_file, sheet_name="data"):
    df = pd.read_excel(data_file, sheet_name=sheet_name)

    candidate_list = df["answer"].tolist()
    candidate_list = list(set(candidate_list))

    return candidate_list


def get_candidate_list_from_chinese_context(context: str, stopwords_list: list):
    # 定义正则表达式模式，匹配中文引号括住的内容 \u201c 对应中文的左双引号（“），\u201d 对应中文的右双引号（”），\u300c 对应中文的左书名号（「），\u300d 对应中文的右书名号（」）
    # pattern = r'[\u201c\u300c].*?[\u201d\u300d]'
    if '“' in context:
        context = '“' + context + '”'
    # 使用 re 模块的 sub 方法替换匹配到的内容为空字符串
    # result = re.sub(pattern, '', context)
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

    final_list = list(set([v for i, v in enumerate(seg_list) if
                           v not in stopwords_list and len(v) > 1 and i not in remove_index_list]))

    return final_list



def cal_prob_batch(target_text: list, input_text: list, model, tokenizer):

    encodings = tokenizer(input_text, return_tensors="pt", max_length=612, padding='max_length', add_special_tokens=True,
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

    # Calculate the probability of generating each label and sum the probabilities of all tokens in labels_token_prob_list
    non_zero_counts = torch.count_nonzero(labels_token_prob_list, dim=1)

    row_sums = torch.sum(labels_token_prob_list, dim=1)
    labels_prob_list = row_sums / non_zero_counts.float()

    return labels_prob_list


class MyDataset2(Dataset):
    def __init__(self, data, is_train=True, candidates='', alias_dict=None,):
        super(MyDataset2, self).__init__()

        self.episode_id = []
        self.context = []
        self.question = []
        self.answer = []
        self.utter_text = []
        self.lower_text = []
        data.sample(frac=1)

        for sample in data.index:
            self.episode_id.append(data.loc[sample].values[0])
            self.context.append(data.loc[sample].values[1])
            self.question.append(data.loc[sample].values[2])
            self.answer.append(data.loc[sample].values[3])
            self.utter_text.append(data.loc[sample].values[5])

            self.lower_text.append(data.loc[sample].values[6])

    def __getitem__(self, item):
        speaker = self.answer[item]
        above = self.utter_text[item]

        below = self.lower_text[item]

        quotetext = self.question[item]
        try:
            context = above + "[SEP]" + "[MASK] 说："+ quotetext + "[SEP]" + below
            quetion = "[CLS]" + quotetext + "说话者为[MASK]。" + '[SEP]'
            answer = "说话者为：" + speaker
        except TypeError:
            print(speaker, above, below, quotetext, item)
            context = str(above) + "[SEP]" + str(quotetext) + "说话者为[MASK]" + "[SEP]" + str(below)
            quetion = "[CLS]" + str(quotetext) + "说话者为[MASK]。" + '[SEP]'
            answer = "说话者为：" + str(speaker)

        return {"source": str(context)+str(quetion), "speaker_label": speaker, "context": str(above) + str(below)}

    def __len__(self):
        return len(self.answer)

    @staticmethod
    def collate_fn(data):
        speaker = [item["speaker_label"] for item in data]

        source = [item["source"] for item in data]
        context = [item["context"] for item in data]
        return {"source": source, "speaker_label": speaker, "context": context}


def chinese_dev_classify(data_dir, model, tokenizer,  dev_sheet="data", topk=2, candidate_from_context=False, max_dev_sample=10000):
    model.eval()
    bsz = 4
    total = 0

    correct_answer_in = 0
    correct_topk = 0
    data_path = data_dir

    correct = 0
    error = 0
    error_in_context = 0
    error_not_in_context = 0
    dev_data = CleanData(data_path,
                         save_punctuations=True,
                         save_stopwords=True,
                         sheet_name=dev_sheet)

    dev_cleaned_data = dev_data.unsplit_data()
    # dev_cleaned_data = dp.avoid_long_tail
    val_outputs = []
    val_targets = []

    candidates_list = dev_cleaned_data["answer"].tolist()

    candidates_list = list(set(candidates_list))

    stopwords_list = load_stopwords(config.stopwords_dir)

    k = topk
    if len(candidates_list) < k:
        k = len(candidates_list)

    candidates_template_list = ["s" for _ in range(len(candidates_list))]
    for i, candidates in enumerate(candidates_list):
        candidates_template_list[i] = "说话者是: " + candidates

    dev_set = MyDataset2(dev_cleaned_data, is_train=False)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                             batch_size=bsz,
                                             collate_fn=MyDataset2.collate_fn,
                                             shuffle=True,
                                             drop_last=True
                                             )

    for index, value_dict in tqdm(enumerate(dev_loader), desc="Evaluating in classify"):
        if index * config.batch_size >= max_dev_sample:
            break
        if candidate_from_context:
            cache_target_list = []
            target_list = []
            candidates_nums = []
            speakers = value_dict["speaker_label"] # [speaker1,speaker2,...]
            search_answer_target_list = []
            for context in value_dict["context"]:

                candidates_list_from_context = get_candidate_list_from_chinese_context(context=context, stopwords_list=stopwords_list)
                candidates_nums.append(len(candidates_list_from_context))
                cache_target_list.extend(candidates_list_from_context)
                search_answer_target_list.append(candidates_list_from_context)
            for target_i, target_v in enumerate(cache_target_list):
                target_list.append("说话者是：" + target_v)

            # [source0 * candidates_nums[0], source2 * candidates_nums[2],...]
            input_text = [item for items in zip(value_dict["source"], candidates_nums) for item in [items[0]] * items[1]]
            text_prob = cal_prob_batch(target_list, input_text, model, tokenizer)
            text_probs = torch.split(text_prob, split_size_or_sections=candidates_nums, dim=0)
            for i, prob in enumerate(text_probs):
                max_prob = prob.argmax(dim=-1)
                value, indice = prob.topk(k, dim=-1, largest=True, sorted=True)
                answer = search_answer_target_list[i][max_prob.item()]
                total += 1
                if judge_equal(answer=answer, label=speakers[i]):
                    correct += 1

                    if answer in value_dict["source"][i]:
                        correct_answer_in += 1
                else:
                    error += 1
                    #print("answer:" + answer, "label:" + speakers[i], index)
                    if answer in value_dict["source"][i]:
                        error_in_context += 1

                    else:
                        error_not_in_context += 1

                topk_answers = [search_answer_target_list[i][j] for j in indice.cpu().numpy().tolist()]

                for topk_answer in topk_answers:
                    if judge_equal(answer=topk_answer, label=speakers[i]):
                        correct_topk += 1
                    break


        else:

            input_text = [x for x in value_dict["source"] for _ in
                          range(len(candidates_list))]  # [ source1 * bsz, source2 * bsz]

            target_list = candidates_template_list * bsz  # [The number of candidates * bsz]

            speakers = value_dict["speaker_label"]  # [speaker1,speaker2,...]
            text_prob = cal_prob_batch(target_list, input_text, model, tokenizer)
            text_probs = torch.chunk(text_prob, bsz)
            for i, prob in enumerate(text_probs):
                max_prob = prob.argmax(dim=-1)
                value, indice = prob.topk(k, dim=-1, largest=True, sorted=True)
                val_outputs.append(max_prob.item() % len(candidates_list))
                val_targets.append(candidates_list.index(speakers[i]))
                answer = candidates_list[max_prob.item() % len(candidates_list)]
                total += 1

                if judge_equal(answer=answer, label=speakers[i]):
                    correct += 1

                    if answer in value_dict["source"][i]:
                        correct_answer_in += 1
                else:
                    error += 1

                    if answer in value_dict["source"][i]:
                        error_in_context += 1

                    else:
                        error_not_in_context += 1

                topk_answers = [candidates_list[j % len(candidates_list)] for j in indice.cpu().numpy().tolist()]

                for topk_answer in topk_answers:
                    if judge_equal(answer=topk_answer, label=speakers[i]):
                        correct_topk += 1
                    break

    print("Accuracy: %4f" % (correct/total))
    print("Top2_Accuracy: %4f" % (correct_topk/total))
    return {"accuracy": correct/total, "topk_accuracy": correct_topk/total}


def chinese_dev_generation(data_dir, model, tokenizer,  dev_sheet="data",max_dev_sample=500):
    model.eval()
    model.to("cpu")
    bsz = 4
    total = 0

    correct_answer_in = 0
    correct_topk = 0
    data_path = data_dir
    correct = 0
    error = 0

    dev_data = CleanData(data_path,
                         save_punctuations=True,
                         save_stopwords=True,
                         sheet_name=dev_sheet)

    dev_cleaned_data = dev_data.unsplit_data()

    text2text_generator = Text2TextGenerationPipeline(model, tokenizer)

    dev_set = MyDataset2(dev_cleaned_data, is_train=False)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set,
                                             batch_size=bsz,
                                             collate_fn=MyDataset2.collate_fn,
                                             shuffle=True,
                                             drop_last=True
                                             )
    for index, value_dict in tqdm(enumerate(dev_loader), desc="Evaluating in generation"):
        if index * config.batch_size >= max_dev_sample:
            break
        labels = value_dict["speaker_label"]
        contexts = value_dict["source"]
        try:
            answers = text2text_generator(contexts, max_length=300, do_sample=False)

            for i, answer in enumerate(answers):
                total += 1
                answer = generation_process(answer['generated_text'])
                if judge_equal(answer, labels[i]):
                    correct += 1
                else:
                    error += 1
                    print("answer:" + answer, "label:" + labels[i])
        except:
            continue

    print("Accuracy: %4f" % (correct / total))
    model.to(device)
    return {"accuracy": correct / total}


if __name__ == '__main__':
    model = BartForConditionalGeneration.from_pretrained(config.bart_model_dir)
    model.load_state_dict(torch.load(os.path.join(config.resume_dir, 'sig.pt'), map_location='cpu')['model'])
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(config.bart_model_dir)
    #chinese_dev_classify(data_dir=config.test_dir, model=model, tokenizer=tokenizer, candidate_from_context=False)
    print("++++++++++++++")
    #chinese_dev_classify(data_dir=config.test_dir, model=model, tokenizer=tokenizer, candidate_from_context=True)
    chinese_dev_generation(data_dir=config.test_dir, model=model, tokenizer=tokenizer,max_dev_sample=50)



