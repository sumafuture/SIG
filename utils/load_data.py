
import pandas as pd

import config
from collections import defaultdict, Counter

import json
import random

import numpy as np

import re
import ast
import jieba
import time
from nltk.corpus import stopwords

from collections import OrderedDict



class CleanData:

    def __init__(self, src_file, save_punctuations=True, save_stopwords=True, lower=False, sheet_name=None,
                 is_avoid_longtail=False, id_unify=False):
        """

        :param src_file: data file
        :param save_punctuations: Whether to save punctuation marks. Defaults to True
        :param save_stopwords: Whether to save stop words. Defaults to True
        :param lower: Regardless of whether all letters are lowercase. Defaults to False
        """
        super().__init__()
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.stop_words = set(stopwords.words('english'))
        self.src_file = src_file
        self.save_punctuations = save_punctuations
        self.save_stopwords = save_stopwords
        self.lower = lower
        self.sheet_name = sheet_name
        self.is_avoid_longtail = is_avoid_longtail
        self.id_unify = id_unify

    @staticmethod
    def all_lower(text):
        text = str(text).lower()
        return text

    @staticmethod
    def remove_extra_spaces(text):
        text = re.sub(' +', ' ', str(text)).strip()
        return text

    @staticmethod
    def append_quotation_mark(text):
        text = f'"{text}"'
        return text

    @staticmethod
    def clear_nan(text):
        text = re.sub('nan', '', str(text)).strip()


    def remove_punctuations(self, text):
        text = re.sub('[%s]' % re.escape(self.punctuation), " ", str(text))
        return text

    def remove_stopwords(self, text):
        return " ".join([word for word in str(text).split() if word not in self.stop_words])

    @staticmethod
    def avoid_long_tail(df):

        counts = df['answer'].value_counts()

        to_drop = df[df['answer'].map(counts) <= 10]

        df = df.drop(to_drop.index)

        return df

    @staticmethod
    def get_addressees(str_list):

        list_obj = ast.literal_eval(str_list)
        for i, v in enumerate(list_obj):
            list_obj[i] = v.replace("\'", "")
        if len(list_obj) == 0:
            addressees = "no one"
        else:
            addressees = list_obj[0]

            if random.random() < 0.5:
                for j in range(1, len(list_obj)):
                    addressees = addressees + " and " + list_obj[j]

            elif random.random() < 1.0:
                for j in range(1, len(list_obj)):
                    addressees = addressees + ", " + list_obj[j]
            elif random.random() < 0.75:
                for j in range(1, len(list_obj)):
                    addressees = addressees + "  " + list_obj[j]

            else:
                for j in range(1, len(list_obj)):
                    addressees = addressees + "  " + list_obj[j]
        return addressees

    @staticmethod
    def get_addressees_id(str_list):
        list_obj = ast.literal_eval(str_list)
        d = {}
        addressees_id = [0 for i in range(1)]

        return addressees_id

    @staticmethod
    def split_text_around_sentence(row):
        sentence = row["question"]
        text = row["context"]

        sentence_index = text.find(sentence)

        if sentence_index == -1:
            return None, None

        upper_text = text[:sentence_index]
        lower_text = text[sentence_index + len(sentence):]

        return pd.Series([upper_text, lower_text])


    def unsplit_data(self):
        data_df = pd.read_excel(self.src_file, sheet_name='data')
        df = data_df.loc[:,
            ["id", "context", "question", "answer", "answer_start",'upper_text', 'lower_text', 'listeners']]

        df['listeners'] = df['listeners'].apply(lambda x: " ，".join(ast.literal_eval(x)))
        if self.id_unify:
            df["id"] = df["id"].apply(lambda x: x.split('_')[0])
        if self.is_avoid_longtail:
            df = self.avoid_long_tail(df)
        return df


def load_stopwords(file_name) -> list:
    fp = open(file_name, "r", encoding="utf-8")
    content_lines = fp.readlines()
    fp.close()
    # Remove the newline character at the end of the line, otherwise it will interfere with the stop word matching process
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")

    return content_lines


def load_names(file_name):

    names_and_aliases = []

    # Open the txt file for reading
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:

            parts = line.strip().split()
            name = parts[1]  # name of speaker
            aliases = parts[2:]  # aliases of speaker

            # Add the (name and aliases) to the list
            names_and_aliases.append((name, aliases))

    return names_and_aliases
