import copy
import numpy as np
import pandas as pd
import re
import ast
import io

import random
from collections import Counter

def is_quote(sentence):
    return sentence.startswith('“') and sentence.endswith('”')


def is_pure_dialogue(sw, xw, threshold=80):

    context = sw + xw


    quoted_parts = re.findall(r'"(.*?)"', context)


    quoted_length = sum(len(part) for part in quoted_parts)


    percentage = (quoted_length / len(context)) * 100

    return percentage > threshold



def calculate_total_occurrences(dictionary, long_string):
    """

    :param dictionary: speaker：alias
    :param long_string: context
    :return: ({speaker: number},
    """
    total_occurrences = {}
    total_speakers = 0
    for key, short_strings in dictionary.items():
        occurrences = sum(long_string.count(short_string) for short_string in short_strings)
        total_speakers += occurrences
        total_occurrences[key] = occurrences

    return {"dict": total_occurrences,
            "number": total_speakers}


def calculate_speaker(alias_list, context):

    n = 0
    if isinstance(alias_list, str):
        alias_list = ast.literal_eval(alias_list)
    for alias in alias_list:
        n += context.count(alias)
        n += context.count(alias.lower())

    return n


def get_alias(str_list):

    list_obj = ast.literal_eval(str_list)
    for i, v in enumerate(list_obj):
        list_obj[i] = v.replace("\'", "")

    return list_obj


def find_key_by_value(dictionary, value):

    for key, lst in dictionary.items():

        if value in lst:
            return key
    return -1


def get_part_words_string(input_string, cut_length=None, is_before=True):

    words_list = input_string.split()


    half_length = len(words_list) // 2

    if cut_length:
        if is_before:

            half_words_list = words_list[:cut_length]
        else:
            half_words_list = words_list[cut_length:]

    else:

        if is_before:

            half_words_list = words_list[:half_length]
        else:
            half_words_list = words_list[half_length:]

    half_words_string = ' '.join(half_words_list)

    return half_words_string


def diff_num_speaker(data_dir, sample_count, sheet_name="train", is_print=False):
    result = {}
    df = pd.read_excel(data_dir, sheet_name=sheet_name)
    speaker_number = df["speaker"].value_counts()
    for index in range(len(sample_count) - 1):
        speakers = speaker_number[
            (speaker_number > sample_count[index]) & (speaker_number <= sample_count[index + 1])].index.tolist()
        count = sample_count[index]
        if count in result:
            result[count].extend(speakers)
        else:
            result[count] = speakers

    if is_print:
        for count, speakers in result.items():
            print(f"Speaker whose sample number bigger than {count}：")
            print(speakers)
            print("___")

    return result


def interval_dev(interval_dict: dict, data_dict: dict, mark: str, correct: bool):

    key = find_key_by_value(interval_dict, mark)
    if key not in data_dict:
        if correct:
            data_dict[key] = [1 for _ in range(2)]
        else:
            data_dict[key] = [0, 1]  # [correct, total]

    else:
        if correct:
            data_dict[key] = [i+1 for i in data_dict[key]]
        else:

            data_dict[key] = [num+1 if index % 2 == 1 else num for index, num in enumerate(list(data_dict[key]))]

    return data_dict


def cal_acc(data_dict: dict):
    acc_dict = {}
    correct = 0
    total = 0
    for key, value in data_dict.items():
        acc_dict[key] = data_dict[key][0] / data_dict[key][1]
        correct += data_dict[key][0]
        total += data_dict[key][1]
    acc_dict[999] = correct / total
    return acc_dict


def find_speaker_name(speaker_id, merged_df, fiction_name: str):

    result = merged_df.loc[(merged_df['speaker_id'] == speaker_id) & (merged_df['fiction_name'] == fiction_name), 'speaker']
    if not result.empty:
        return result.iloc[0]
    else:
        return None


def vote_and_random_pick(list_of_lists):

    list_length = len(list_of_lists[0])


    result = []


    for i in range(list_length):
        votes = Counter([sub_list[i] for sub_list in list_of_lists])
        max_vote = max(votes.values())
        max_vote_elements = [element for element, count in votes.items() if count == max_vote]


        if len(max_vote_elements) > 1:
            result.append(random.choice(max_vote_elements))
        else:
            result.append(max_vote_elements[0])

    return result


if __name__ == '__main__':
    print(vote_and_random_pick([[1,2,3], [2,3,3], [2,1,3]]))



