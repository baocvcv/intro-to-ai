""" N-gram model """

import gc
from os import makedirs
from os.path import join, isdir, exists
from math import log
from collections import defaultdict, Counter
import time
import re
from operator import itemgetter

import dill as pickle

from basemodel import BaseModel
from config import DEBUG, D_VALUE, MAX_PHRASE_LEN, MAX_PROCESS_NUM

class NGramModel(BaseModel):
    ' The N-Gram model '

    def __init__(self,
                 n=2,
                 table_path='pinyin_table',
                 file_path='',
                 model_path='models/n-gram'):
        ' Constructor '
        super().__init__(table_path, file_path)
        self.n = n
        self.model_dir = model_path
        if not isdir(model_path):
            makedirs(model_path)
        self.load_pinyin_table()

    def train(self, levels=[]):
        ' Train '
        print("[Info] Training the model...")
        time_d = time.time()
        # train word based n-gram model
        for i in levels:
            print("[Info] Running with n = %d" % i)
            self.train_n(i)
        time_d = round(time.time()-time_d, 3)
        print("[Info] Training took %ss" % time_d)

    def train_n(self, nn):
        ' Train '
        # nn is the length of the prefix
        # use 2-d dictionary to count the conditional probability
        # {'a': {'b': 0.1, 'c': 0.2, ...}
        conditional_pro = []
        # lambdas used for discounting
        # {'a':0.1, 'b':0.2, ...}
        lambdas = []
        # phrase -> index in conditional_pro and lambdas
        index = {}

        # loop through and count
        phrase_counter = Counter()
        for line in self.generate_data():
            _len = len(line['text'])
            for i in range(_len-nn):
                ' count number of different phrases '
                p = line['text'][i : i+nn]
                ' add to index '
                idx = -1
                if p not in index:
                    idx = index[p] = len(index)
                    conditional_pro.append(defaultdict(float))
                else:
                    idx = index[p]
                phrase_counter[idx] += 1

                ' count conditional occurrences '
                idx_word = -1
                word = line['text'][i+nn]
                if word in self.all_words:
                    idx_word = self.all_words[word]
                else:
                    idx_word = self.all_words[word] = len(self.all_words)
                conditional_pro[idx][idx_word] += 1
            if _len-nn >= 0:
                p = line['text'][_len-nn : _len]
                idx = -1
                if p not in index:
                    idx = index[p] = len(index)
                    conditional_pro.append(defaultdict(float))
                else:
                    idx = index[p]
                phrase_counter[idx] += 1


        # print("华|新: ", conditional_pro[index['新']][self.all_words['华']])
        # print("新：", phrase_counter['新'])

        # calculate probability
        for _, idx in index.items(): # for each phrase
            cnt = phrase_counter[idx]
            cp = conditional_pro[idx]
            for w in cp: # each word
                cp[w] = (cp[w]-D_VALUE) / cnt
            lambdas.append(D_VALUE / cnt)
        print("[Info] Dict size: ", len(phrase_counter))
        print("[Info] Training finished!")
        # print("P(华|新): ", conditional_pro[index['新']][self.all_words['华']])
        print("[Info] Saving model...")
        self.save_model(index, 'index%d.p' % nn)
        self.save_model(lambdas, 'lambdas%d.p' % nn)
        self.save_model(conditional_pro, 'prob%d.p' % nn)
        print("[Info] Model saved.")
        gc.collect()

    def save_model(self, data, name: str):
        ' Save the model params '
        pickle.dump(data,
                    open(join(self.model_dir, name), 'wb'))
        print("[Info] Saved %s to " % name, self.model_dir)

    def load_model(self):
        ' Load saved model '
        self._load_model()

    def _load_model(self):
        ' Load saved model '
        print("[Info] Loading model...")
        # word ngram
        self.lambdas = []
        self.conditional_pro = []
        self.index = []
        for i in range(self.n):
            self.conditional_pro.append(pickle.load(
                open(join(self.model_dir, 'prob%d.p' % i), 'rb')))
            self.lambdas.append(pickle.load(
                open(join(self.model_dir, 'lambdas%d.p' % i), 'rb')))
            self.index.append(pickle.load(
                open(join(self.model_dir, 'index%d.p' % i), 'rb')))
        print("[Info] Loaded model from ", self.model_dir)

        ' used for debug '
        # for idx, p in enumerate(self.conditional_pro):
        #     print("---------{}--------".format(idx))
        #     for i, x in enumerate(p):
        #         for j, y in enumerate(p[x]):
        #             print("%s|%s" % (y, x), "=>", p[x][y])
        #             if j > 10:
        #                 break
        #         if i > 20:
        #             break

    def translate(self, pinyin_input: str) -> str:
        ' Translate the input pinyin to words '
        result = self._translate(pinyin_input)
        # sort result
        result = list(result.items())
        result.sort(key=itemgetter(1), reverse=True)
        # print(result)
        if len(result) == 0:
            print("[Error] Please check your input.")
            return ''
        return result[0][0]

    def _translate(self,
                   pinyin_input: str,
                   prior: dict = {'': 0}) -> str:
        ' Translate the input pinyin to words list '
        pinyin_input = re.split(r'\s+', pinyin_input.lower().strip())

        old_sentences = prior
        for _len, syllable in enumerate(pinyin_input):
            # For each pinyin, get candidate words
            # Calculate conditional probability, record history
            # print('[%d]: ' % _len, old_sentences)
            new_sentences = {}
            for w in self.pinyin_dict[syllable]:
                best_sentence = ''
                max_prob = float('-inf')
                for sentence in old_sentences:
                    w_prev = sentence[-self.n+1:]
                    cur_prob = old_sentences[sentence] \
                        + log(self.get_probability(w, w_prev))
                    if cur_prob > max_prob:
                        max_prob = cur_prob
                        best_sentence = sentence + w
                new_sentences[best_sentence] = max_prob
            old_sentences = new_sentences
        return old_sentences

    def get_probability(self, w, w_prev):
        ' Retrieve probability of P(w | w_prev) '
        if w_prev == '':
            # the index for '' is 0
            idx_word = self.all_words[w]
            if idx_word in self.conditional_pro[0][0]:
                return self.conditional_pro[0][0][idx_word]
            else:
                return self.lambdas[0][0]

        l = len(w_prev)
        if w_prev in self.index[l]:
            idx = self.index[l][w_prev]
            idx_word = self.all_words[w]
            p1 = self.conditional_pro[l][idx][idx_word]
            p2 = self.get_probability(w, w_prev[1:])
            return p1 + self.lambdas[l][idx] * p2 * len(self.conditional_pro[l][idx])
        else:
            return self.get_probability(w, w_prev[1:])
