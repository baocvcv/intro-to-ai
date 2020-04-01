""" N-gram model """

import gc
from os import makedirs
from os.path import join, isdir
from math import log
from collections import defaultdict, Counter, OrderedDict
# import pickle
import dill as pickle
from .basemodel import BaseModel

#TODO: change prob dict to using integer index and 8-bit probability
#TODO: change training to working on a single length at a time

class OrderedCounter(Counter, OrderedDict):
    ' Counter with order '

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
         return self.__class__, (OrderedDict(self),)

class NGramModel(BaseModel):
    ' The N-Gram model '
    # LAMBDA = 0.95
    # use absolute discounting
    D_VALUE_1 = 0.5
    D_VALUE = 0.75

    def __init__(self, n=2, table_path='../pinyin_table', file_path=''):
        ' Constructor '
        super().__init__(table_path, file_path)
        self.n = n

    def train(self):
        ' Train '
        print("[Info] Training the model...")
        # use 2-d dictionary to count the conditional probability
        # [ {'a': {'b': 0.1, 'c': 0.2, ...}, ...}, ...]
        self.conditional_pro = [defaultdict(lambda: defaultdict(float)) \
                                for i in range(self.n)]
        # lambdas used for discounting
        # [ {'': 0.1}, {'a':0.1, 'b':0.2, ...}]
        self.lambdas = [{} for i in range(self.n)]

        # loop through and count
        phrase_counter = [Counter() for i in range(self.n)]
        for line in self.generate_data():
            ' count number of different phrases '
            for i in range(self.n):
                for j in range(len(line)-i+1):
                    phrase_counter[i][line[j : j+i]] += 1

            ' count conditional occurrences '
            # First n-1 chars in the line
            len1 = min(self.n-1, len(line))
            for i in range(len1): # line[i]
                for l in range(self.n): # prefix length of l
                    s = max(i-l, 0)
                    p = line[s : i]
                    if p in phrase_counter[l]: # if p has been counted???
                        self.conditional_pro[l][p][line[i]] += 1
            # Rest of the chars
            for i in range(self.n-1, len(line)):
                for l in range(self.n):
                    p = line[i-l : i]
                    self.conditional_pro[l][p][line[i]] += 1

        # print("华|新: ", self.conditional_pro[1]['新']['华'])
        # print("新：", phrase_counter[1]['新'])
        # print("' '：", phrase_counter[0][''])
        # print("total: ", total_num_phrase)

        # calculate probability
        for l, cp in enumerate(self.conditional_pro): # each phrase length
            # value for discounting
            for p in cp: # for each phrase
                cnt = phrase_counter[l][p]
                for w in cp[p]: # each word
                    cp[p][w] = (cp[p][w]-self.D_VALUE) / cnt
                self.lambdas[l][p] = self.D_VALUE / cnt
        print("[Info] Training finished!")
        # print("P(华|新): ", self.conditional_pro[1]['新']['华'])
        # print("P(新|''): ", self.conditional_pro[0]['']['新'])
        gc.collect()

    def save_model(self, toDir: str ='models/2-gram'):
        ' Save the model params '
        if not isdir(toDir):
            makedirs(toDir)
        pickle.dump(self.lambdas, open(join(toDir, 'lambdas.p'), 'wb'))
        for i in range(self.n):
            pickle.dump(self.conditional_pro[i],
                        open(join(toDir, 'prob%d.p' % i), 'wb'))
        print("[Info] Saved model to ", toDir)

    def load_model(self, fromDir: str = 'models/2-gram'):
        ' Load saved model '
        self.lambdas = pickle.load(open(join(fromDir, 'lambdas.p'), 'rb'))
        self.conditional_pro = []
        for i in range(self.n):
            fin = open(join(fromDir, 'prob%d.p' % i), 'rb')
            self.conditional_pro.append(pickle.load(fin))
            fin.close()
        print("[Info] Loaded model from ", fromDir)

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
        pinyin_input = pinyin_input.lower()

        old_sentences = defaultdict(float)
        old_sentences[''] = .0
        for _len, syllable in enumerate(pinyin_input.split(' ')):
            # For each pinyin, get candidate words
            # Calculate conditional probability, record history
            # print('[%d]: ' % _len, old_sentences)
            new_sentences = defaultdict(float)
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

        # get max and return
        result = list(old_sentences.items())
        result.sort(key=lambda r: r[1], reverse=True)
        print(result[:5])
        return result[0][0]

    def get_probability(self, w, w_prev):
        ' Retrieve probability of P(w | w_prev) '
        if w_prev == '':
            if w in self.conditional_pro[0]['']:
                return self.conditional_pro[0][''][w]
            else:
                return self.lambdas[0]['']

        l = len(w_prev)
        if w_prev in self.conditional_pro[l]:
            p1 = self.conditional_pro[l][w_prev][w]
            p2 = self.get_probability(w, w_prev[1:])
            return p1 + self.lambdas[l][w_prev] * p2
        else:
            return self.get_probability(w, w_prev[1:])


