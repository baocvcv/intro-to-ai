""" Base model class """

from os import listdir
from os.path import join
import io
import gc
from collections import defaultdict
import re
import json
# from string import punctuation as eng_punct

# from zhon.hanzi import punctuation as ch_punct

class BaseModel:
    ' Base class for all models  '

    def __init__(self, table_path='../pinyin_table', file_path=''):
        ' Constructor '
        # text file path
        self.file_path = file_path
        # pinyin table path
        self.table_path = table_path
        # all the Chinese characters in a string
        self.all_words = ""
        # words -> index in self.all_words
        self.word_dict = defaultdict(lambda: -1)
        # pinyin -> list of words
        self.pinyin_dict = defaultdict(list)
        self.load_pinyin_table()

    def train(self, file_path: str):
        ' Use the file to train the model '

    def save_model(self, filename: str):
        ' Save the model params '

    def load_model(self, filename: str):
        ' Load saved model '

    def translate(self, pinyin_input: str) -> str:
        ' Translate the input pinyin to words '

    def load_pinyin_table(self):
        ' Load pinyin table '
        print("[Info] Loading pinyin dictionary...")
        pinyin_list = io.open(join(self.table_path, "pinyin_dict.txt"),
                              mode='r',
                              encoding='utf-8').readlines()
        self.all_words = {}
        for line in pinyin_list:
            words = line.split()
            self.pinyin_dict[words[0]] = words[1:]
            for w in words[1:]:
                if w not in self.all_words:
                    self.all_words[w] = len(self.all_words)
        print("[Info] Loading finished!")
        gc.collect()

    def generate_data(self) -> str:
        ' Load training data '
        if self.file_path == '':
            print("[Error] Training file path not set!")
            exit()
        # re to filter out punctuations
        # punc = re.compile(r"[%s%s0-9a-zA-Z]+" % (ch_punct, eng_punct))
        punc = re.compile(r"[^\u4e00-\u9fff]+")
        for filename in listdir(self.file_path):
            if 'README' in filename or 'readme' in filename:
                continue
            print("[Info] Going through ", filename)
            file_content = io.open(join(self.file_path, filename),
                                   mode='r',
                                   encoding='utf-8').readlines()
            for line in file_content:
                line_content = punc.sub(' ', json.loads(line)['html'])
                for l in line_content.split():
                    if l != '':
                        yield l
