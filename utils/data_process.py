
import pandas as pd
from collections import defaultdict, Counter
import json
import random
import numpy as np
import re
import ast
import time
from nltk.corpus import stopwords





class CleanData:

    def __init__(self, src_file, save_punctuations=True, save_stopwords=True, lower=False,sheet_name=None):
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

    def process_data(self):
        data_df = pd.read_excel(self.src_file, sheet_name=self.sheet_name)

        df = data_df.loc[:,
             ["id", "context", "question", "answer", "answer_start"]]

        df["context"] = df[
            "context"].apply(self.remove_extra_spaces)
        df["context"] = df[
            "context"].apply(lambda x: re.sub(r'Instance index: \d+', '', x))
        df["question"] = df["question"].apply(self.remove_extra_spaces)


        df[['upper_text', 'lower_text']] = df.apply(self.split_text_around_sentence, axis=1)

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

    while True:
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
        else:
            # print(f'{fname} success！')
            break
            pass
    return


def rearrange_dfs(df):
    # Remove examples where the answer appears infrequently, if necessary
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


def answer2txt(excel_file=None, txt_dir=None):
    """

    Args:
        excel_file: Excel file including data from multiple novels
        txt_dir: The directory to save candidate_list of each novel

    Returns:

    """
    excel_file = excel_file
    df = pd.read_excel(excel_file,sheet_name="data")

    df["novel"] = df["id"].apply(lambda x: x.split('_')[0])
    grouped = df.groupby('novel')
    for group_name, group_data in grouped:
        index = 0
        output_file = txt_dir + f'{group_name}_test_answers.txt'
        answers = group_data["answer"].tolist()
        answers_counts = Counter(answers)
        with open(output_file, 'w', encoding='utf-8') as f:
            for element, count in answers_counts.items():
                f.write(f'{index} {element}\n')
                index += 1


def alias2excel(txt_file, excel_file):
    excel_data = pd.read_excel(excel_file,sheet_name="data")
    answers = excel_data['answer']
    ids = excel_data['id']


    alias_dict = defaultdict(list)

    # Read each line from the text file, find the corresponding answer according to the serial number, and add the answer to the alias list
    with open(txt_file, 'r') as txt_file:
        for line in txt_file:
            parts = line.strip().split()
            if len(parts) >= 3:
                id, name, *aliases = parts

                alias_dict[name].append(name)
                alias_dict[name].extend(aliases)

    # Save the alias dictionary to an Excel file or other format
    alias_data = pd.DataFrame(alias_dict.items(), columns=['Name', 'Aliases'])
    alias_data.to_excel('output_alias_file.xlsx', index=False)


def json2excel(json_file, excel_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
        column_name = ["id", "context", "question", "answer", "answer_start"]  # column name: ["quoteID", ...]
        v = []
        data = params["data"]
        for sample in data:

            paragraphs = sample["paragraphs"]
            for paragraph in paragraphs:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    sample_values = [0 for _ in range(len(column_name))]
                    sample_values[0] = qa["id"]  # id
                    sample_values[1] = context  # context
                    sample_values[2] = qa["question"]  # question
                    sample_values[3] = qa["answers"][0]["text"]  # answer
                    sample_values[4] = qa["answers"][0]["answer_start"]  # answer_start

                    v.append(sample_values)

    df = pd.DataFrame(v, columns=column_name)  # Process into a dataframe and store it in a list to be loaded into excel

    with pd.ExcelWriter(excel_file) as writer:
        df.to_excel(writer, sheet_name="data", index=False)










