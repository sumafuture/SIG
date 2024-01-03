import spacy
import pandas as pd
import config
from utils.load_data import *
import ast
nlp = spacy.load("zh_core_web_sm")

stopwords_list = load_stopwords(config.stopwords_dir)
names_and_aliases = load_names(r"D:\speaker identification baseline\Project SIG(chinese)\data\WP2021\name_list.txt")


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


def get_listen(src_file, output_file):
    names_and_aliases_list =[]

    def get_listener(context, names_list):
        # 定义正则表达式模式，匹配中文双引号内的内容
        pattern = r'“.*?”'

        # 提取没有被引号括住的文本
        non_quoted_text = re.sub(pattern, '', context)

        people_names = []
        for n in names_list:
            if n in non_quoted_text:
                people_names.append(n)

        return list(set(people_names))

    for name, aliases in names_and_aliases:
        names_and_aliases_list.append(name)
        names_and_aliases_list.extend(aliases)
    data_df = pd.read_excel(src_file, sheet_name='data')
    df = data_df.loc[:,
         ["id", "context", "question", "answer", "answer_start", 'upper_text', 'lower_text']]


    df["persons"] = df["context"].apply(lambda x:get_listener(x,names_list=names_and_aliases_list))
    listeners = []
    for index, row in df.iterrows():
        speaker = row["answer"]

        listener = []
        for a in row["persons"]:
            if not judge_equal(speaker, a):

                listener.append(a)
        filtered_list = [item for item in listener if
                         not any(item in mom_item and mom_item != item for mom_item in listener)]

        ans = []
        for f in filtered_list:
            is_listener = False
            # 检查听者是否为上下文的speaker
            for j in range(max(0, index - 5), index):
                if judge_equal(f, df.loc[j, 'answer']):
                    ans.append(f)
                    is_listener = True
                    break
            if not is_listener:
                for j in range(index + 1, min(index + 6, len(df))):
                    if judge_equal(f, df.loc[j, 'answer']):
                        ans.append(f)
                        break

        listeners.append(ans)

    df["listeners"] = listeners

    with pd.ExcelWriter(output_file) as writer:

        df.to_excel(writer, sheet_name="data", index=False)

"""
data_df = pd.read_excel(r"D:\speaker identification baseline\Project SIG(chinese)\data\WP2021\train(4).xlsx", sheet_name='data')
for index, row in data_df.iterrows():

    try:
        a = ast.literal_eval(row["listeners"])
        if ast.literal_eval(row["listeners"]) == []:
            print(index)
        
        if ast.literal_eval(row["listeners"]) != []:
            print(ast.literal_eval(row["listeners"])[0])
            
        if index == 5000:
            break

    except:
        print(index)
"""


get_listen(r"D:\speaker identification baseline\Project SIG(chinese)\data\WP2021\test(2).xlsx",
           r"D:\speaker identification baseline\Project SIG(chinese)\data\WP2021\test(4).xlsx")