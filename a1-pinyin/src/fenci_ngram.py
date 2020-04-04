''' N-gram model improved by Jieba '''

import gc
from os import makedirs
from os.path import join, isdir, exists
from math import log
from collections import defaultdict, Counter, OrderedDict
import time
import re

import dill as pickle

from .ngram import NGramModel

class XNGramModel(NGramModel):
    ' N-gram model with Jieba '

    def __init__(self,
                 n=2,
                 table_path='../pinyin_table',
                 file_path='',
                 model_path='models/n-gram'):
        ' Constructor '
        super().__init__(n, table_path, file_path, model_path)

    def train(self, force=False):
        ' Train '
        # train phrase based 2-gram
        self.train_jieba()

    def train_jieba(self):
        ' Train the 2-gram model with jieba '
        # the index of phrases in phrase_freq & conditional_pro
        # {'abc': 0, ...}
        index = {}
        # The freq of phrases, arranged according to index
        phrase_freq = []
        # total phrase count
        total_phrase_cnt = 0
        # see definition in @train_n
        conditional_pro = []
        lambdas = []

        for line in self.generate_data():
            phrases = jieba.lcut(line)
            _len = len(phrases)
            total_phrase_cnt += _len
            for i in range(_len-1):
                ' count phrases '
                p = phrases[i]
                if p in index:
                    idx = index[p]
                    phrase_freq[idx] += 1
                else:
                    idx = index[p] = len(index)
                    conditional_pro.append(defaultdict(float))
                    phrase_freq.append(1)
                ' conditional occurrences '
                p_cur = phrases[i+1]
                if p_cur in index:
                    idx_cur = index[p_cur]
                else:
                    idx_cur = index[p_cur] = len(index)
                conditional_pro[idx][idx_cur] += 1

            # count the last phrase of the sentence
            if _len > 0:
                p = phrases[i]
                if p in index:
                    idx = index[p]
                    phrase_freq[idx] += 1
                else:
                    idx = index[p] = len(phrase_freq)
                    phrase_freq.append(1)

        # calc probability
        for p, idx in index.items():
            cnt = phrase_freq[idx]
            cp = conditional_pro[idx]
            for w in cp:
                cp[w] = (cp[w] - self.D_VALUE) / cnt
            lambdas.append(self.D_VALUE / cnt)
            phrase_freq[idx] /= total_phrase_cnt
        print("[Info] Training finished!")
        print("[Info] Saving model...")
        self.save_model(index, 'jieba_index.p')
        self.save_model(phrase_freq, 'jieba_freq.p')
        self.save_model(lambdas, 'jieba_lambdas.p')
        self.save_model(conditional_pro, 'jieba_prob.p')
        print("[Info] Model saved.")
        gc.collect()

    def load_model(self):
        ' Load saved model '
        # jieba
        if self.use_jieba:
            self.jieba_prob = pickle.load(
                open(join(self.model_dir, 'jieba_prob.p'), 'rb'))
            self.jieba_lambdas = pickle.load(
                open(join(self.model_dir, 'jieba_lambdas.p'), 'rb'))
            self.jieba_index = pickle.load(
                open(join(self.model_dir, 'jieba_index.p'), 'rb'))
            self.jieba_freq = pickle.load(
                open(join(self.model_dir, 'jieba_freq.p'), 'rb'))
        print("[Info] Loaded model from ", self.model_dir)

 


