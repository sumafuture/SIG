
import pandas as pd


from collections import defaultdict, Counter

import json
import random

import numpy as np

import re
import ast

import time
from nltk.corpus import stopwords

from collections import OrderedDict
#bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
#encoder = bart.get_encoder()


class CleanData:

    def __init__(self, src_file, save_punctuations=True, save_stopwords=True, lower=False,sheet_name=None):
        """

        :param src_file: data file
        :param save_punctuations: Whether to save punctuation marks. Defaults to True
        :param save_stopwords: Whether to save stop words. Defaults to True
        :param lower: Whether or not all letters are lowercase. Defaults to False
        """
        super().__init__()
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.stop_words = set(stopwords.words('english'))
        self.src_file = src_file
        self.save_punctuations = save_punctuations
        self.save_stopwords = save_stopwords
        self.lower = lower
        self.sheet_name = sheet_name

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

        counts = df['speaker'].value_counts()

        to_drop = df[df['speaker'].map(counts) <= 10]

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

    def clean_data(self):
        data_df = pd.read_excel(self.src_file, sheet_name=self.sheet_name)

        #df = data_df.loc[:, ["quoteText", "quoteTextDecoder", "quoteType", "speaker", "shangwen", "xiawen", "referringExpression"]]
        df = data_df.loc[:,
             ["quoteTextEncoder", "quoteTextDecoder", "quoteType", "speaker", "above", "below", "speaker_id","quote_num","referringExpression"]]



        df["quoteTextEncoder"] = df[
            "quoteTextEncoder"].apply(self.remove_extra_spaces)
        df["source"] = df["above"] + " " + df["quoteTextEncoder"] + " " + df["below"]  # contact
        df["target"] = df["above"] + " " + df["quoteTextDecoder"] + " " + df["below"]

        if not self.save_punctuations:
            df["target"] = df[
                "target"].apply(self.remove_punctuations)

            df["source"] = df[
                "source"].apply(self.remove_punctuations)

        if not self.save_stopwords:
            df["target"] = df["target"].apply(self.remove_stopwords)
            df["source"] = df["source"].apply(self.remove_stopwords)

        if self.lower:
            df["target"] = df["target"].apply(self.all_lower)
            df["source"] = df["source"].apply(self.all_lower)

        df["target"] = df[
            "target"].apply(self.remove_extra_spaces)

        df["source"] = df[
            "source"].apply(self.remove_extra_spaces)

        df = self.avoid_long_tail(df)

        return df

    def mul_task_data(self, has_test=False):
        train_df = pd.read_excel(self.src_file, sheet_name='train')
        dev_df = pd.read_excel(self.src_file, sheet_name='dev')
        #test_df = pd.read_excel(self.src_file, sheet_name='test')
        cache_df = [train_df]
        if has_test is True:
            test_df = pd.read_excel(self.src_file, sheet_name='test')
            cache_df = [train_df, dev_df, test_df]
        data_df = []

        for d_f in cache_df:
            df = d_f.loc[:, ["quoteTextEncoder", "quoteTextDecoder", "quoteType", "speaker", "above", "below",
                              "fiction_name","speaker_id", "quote_num","referringExpression"]]



            # df["referringExpression"] = df["referringExpression"].apply(self.clear_nan)
            df["quoteTextEncoder"] = df[
                "quoteTextEncoder"].apply(self.append_quotation_mark)
            df["source"] = df["above"] + " " + df["quoteTextEncoder"] + " " + df["below"]  # contact
            df["target"] = df["above"] + " " + df["quoteTextDecoder"] + " " + df["below"]

            if not self.save_punctuations:
                df["target"] = df[
                    "target"].apply(self.remove_punctuations)

                df["source"] = df[
                    "source"].apply(self.remove_punctuations)

            if not self.save_stopwords:
                df["target"] = df["target"].apply(self.remove_stopwords)
                df["source"] = df["source"].apply(self.remove_stopwords)

            if self.lower:
                df["target"] = df["target"].apply(self.all_lower)
                df["source"] = df["source"].apply(self.all_lower)

            df["target"] = df[
                "target"].apply(self.remove_extra_spaces)

            df["source"] = df[
                "source"].apply(self.remove_extra_spaces)

            data_df.append(df)
        return data_df

    def clean_address_data(self):
        data_df = pd.read_excel(self.src_file, sheet_name=self.sheet_name)

        df = data_df.loc[:,
             ["quoteText", "quoteType", "addressees", "above", "below", "quote_num","speaker","Aliases",'addressees_aliases',"addressees_num","fiction_name","speaker_id"]]


        df["quoteText"] = df[
            "quoteText"].apply(self.remove_extra_spaces)
        df["quoteText"] = df["quoteText"].apply(self.append_quotation_mark)
        df["addressees"] = df["addressees"].apply(self.get_addressees)
        df["Aliases"] = df["Aliases"].apply(lambda x: ast.literal_eval(x))
        df['addressees_aliases'] = df['addressees_aliases'].apply(lambda x: ast.literal_eval(x))

        return df


    def test_data(self):
        test_df = pd.read_excel(self.src_file, sheet_name='Sheet1')
        df = test_df.loc[:,
             ["Quotation", "Speaker", "Speaker_id", "Above", "Below", "fiction_name"]]
        df.rename(columns={"Above": 'above', 'Below': 'below', "Speaker": "speaker", "Quotation": "quoteText",
                           "Speaker_id": "speaker_id"}, inplace=True)

        df["quoteText"] = df[
            "quoteText"].apply(self.remove_extra_spaces)
        df["quoteText"] = df["quoteText"].apply(self.append_quotation_mark)

        #df = self.avoid_long_tail(df)
        return df


def save_to_excel(data, csv_file, column_name=0):
    """

    :param data: Lists and sublists that hold data
    :param csv_file: The file path to store the data
    :param column_name: Name of columns
    :return: None
    """
    sheet_name = 'Sheet1'
    if column_name == 0:
        column_name = ["The answer of model", "The top5 answer of model", "golden label"]

    csv_data = data
    field_widths = {"The answer of model": 30, "The top5 answer of model": 150, "golden label": 60}
    df = pd.DataFrame(columns=column_name, data=csv_data)

    try:
        with pd.ExcelWriter(csv_file) as writer:
            df.to_excel(writer, sheet_name=sheet_name, encoding='utf-8')
            worksheet = writer.sheets[sheet_name]

            for k, v in field_widths.items():
                if k in df.columns:
                    i = df.columns.to_list().index(k) + 1
                    worksheet.set_column(i, i, v)


    except:
        print(f'Unable to write {csv_file}, if opened, close the file。。。')
        time.sleep(3)



def rearrange_dfs(df):

    traindf_init = df[df['quoteType'] == 'Explicit']
    testdf_init = df[df['quoteType'] != 'Explicit']
    trainy_init = traindf_init['speaker'].tolist()
    testy_init = testdf_init['speaker'].tolist()  # ['Alice', 'Alice', 'Alice', 'The White Rabbit',...]
    traincounter = Counter(trainy_init)
    testcounter = Counter(testy_init)  # {a: 4, b: 5, ...}

    add_train_inds = []

    for s in traincounter:
        if 0 < traincounter[s] < 5:
            diff = 5 - traincounter[s]
            sinds = [ind for ind, val in enumerate(trainy_init) if val == s]

            choices = np.random.choice(sinds, size=diff, replace=True)
            add_train_inds.extend(choices)

    newtrainrows = traindf_init.values.tolist()

    newtrainy = trainy_init[:]

    for ai in add_train_inds:
        newtrainrows.append(traindf_init.iloc[ai].tolist())
        newtrainy.append(trainy_init[ai])

    newtraindf = pd.DataFrame(newtrainrows, columns=traindf_init.columns)

    return newtraindf, testdf_init


# excludes the examples whose speakers with less than 10 annotated quotations in each task, in order to avoid the long tail of minor characters
def avoid_long_tail(df):

    counts = df['speaker'].value_counts()

    to_drop = df[df['speaker'].map(counts) <= 10]

    df = df.drop(to_drop.index)

    return df


def get_speakers_set(data_df):
    """

    :param data_df: speaker DataFrame
    :return: DataFrame{speaker, "speaker_input_ids", "speaker_attention_mask"}
    """

    speakers = list(data_df["speaker"])
    speakers = list(OrderedDict.fromkeys(speakers).keys())

    f = [[speakers[i]]for i in range(len(speakers))]

    df = pd.DataFrame(f, columns=["speaker"])

    return df


def get_dev_speaker(df):
    df = avoid_long_tail(df)
    speakers = []
    for sample in df.index:
        if df.loc[sample].values[6] != "Explicit":
            speakers.append(df.loc[sample].values[4])

    speakers = list(set(speakers))
    return speakers


def json2excel(json_file, excel_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
        column_name = []  # column name: ["quoteID", ...]
        v = []
        for key in params[0].keys():
            column_name.append(key)
        for sample in params:  # Iterate over each sample of the file

            sample_values = []
            for value in sample.values():
                sample_values.append(value)

            v.append(sample_values)

    df = pd.DataFrame(v, columns=column_name)  # Process into a dataframe and store it in a list to be loaded into excel

    with pd.ExcelWriter(excel_file) as writer:

        df.to_excel(writer, sheet_name="data", index=False)




