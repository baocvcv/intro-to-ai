''' N-gram model improved by Jieba '''

import gc
from os import makedirs
from os.path import join, isdir, exists
from math import log
from collections import defaultdict, Counter, OrderedDict
import time
import re

import dill as pickle

from .ngram_zhuyin import NGramPYModel

DEBUG = False

class XNGramModel(NGramPYModel):
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
        # word based n-gram should be trained already
        print("[Info] Training the model...")
        time_d = time.time()
        self.train_jieba()
        time_d = round(time.time()-time_d, 3)
        print("[Info] Training took %ss" % (self.n, time_d))

    def train_jieba(self):
        ' Train the 2-gram model with jieba '
        # the index of phrases in phrase_freq & conditional_pro
        # {'abc': 1, ...}
        index = {}
        # phrase dict: {'wo men': ['我们', '我门' ...]}
        phrase_dict = defaultdict(list)
        # The freq of phrases, arranged according to index
        phrase_freq = []
        # total phrase count
        total_phrase_cnt = 0
        # see definition in @train_n
        conditional_pro = []
        lambdas = []
        # max phrase len
        max_len = 0
        max_p = ''

        def count_phrase(p, py, index, phrase_freq, conditional_pro, phrase_dict):
            if p in index:
                idx = index[p]
                phrase_freq[idx] += 1
            else:
                idx = index[p] = len(index)
                conditional_pro.append(defaultdict(float))
                phrase_freq.append(1)
                phrase_dict[py].append(p)
            return idx

        for line in self.generate_data():
            phrases = line['fc']
            _len = len(phrases)
            total_phrase_cnt += _len
            # i = 0
            p = phrases[0]
            count_phrase(p, ' '.join(line['py'][:len(p)]), index,
                         phrase_freq, conditional_pro, phrase_dict)
            pos = len(p)
            if len(p) > max_len:
                max_len = len(p)
                max_p = p
            for i in range(1, _len):
                ' count phrases & conditional occurrences '
                p = phrases[i]
                idx = count_phrase(p, ' '.join(line['py'][pos : pos+len(p)]), index,
                                   phrase_freq, conditional_pro, phrase_dict)
                conditional_pro[index[phrases[i-1]]][idx] += 1
                pos += len(p)
                if len(p) > max_len:
                    max_len = len(p)
                    max_p = p
            # TODO: count sentence stop sign
        print("[Info] Max phrase length: ", max_len, max_p)

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
        self.save_model(phrase_dict, 'jieba_dict.p')
        self.save_model(conditional_pro, 'jieba_prob.p')
        print("[Info] Model saved.")
        gc.collect()

    def load_model(self):
        ' Load saved model '
        # jieba
        self.jieba_prob = pickle.load(
            open(join(self.model_dir, 'jieba_prob.p'), 'rb'))
        self.jieba_lambdas = pickle.load(
            open(join(self.model_dir, 'jieba_lambdas.p'), 'rb'))
        self.jieba_index = pickle.load(
            open(join(self.model_dir, 'jieba_index.p'), 'rb'))
        self.jieba_freq = pickle.load(
            open(join(self.model_dir, 'jieba_freq.p'), 'rb'))
        self.jieba_dict = pickle.load(
            open(join(self.model_dir, 'jieba_dict.p'), 'rb'))
        super().load_model()
 
    def translate(self, pinyin_input: str) -> str:
        ' Translate '
        time_d = time.time()
        result = self._translate(pinyin_input)
        time_d = round(time.time()-time_d, 5)
        print("Used %fs", time_d)
        # sort result
        # result = [(s, result[s][0]) for s in result]
        # result.sort(key=lambda r: r[1], reverse=True)
        # print(result)
        if len(result) == 0:
            print("[Error] Please check your input.")
            return ''
        return result

    def _translate(self, pinyin_input: str) -> str:
        ' Translate '
        pinyin_input = pinyin_input.lower().strip()
        pinyin_input = re.split(r'\s+', pinyin_input)
        splits = self.cut_n(len(pinyin_input))

        all_best_sentence = ''
        all_max_prob = float('-inf')
        for split in splits:
            input_split = [' '.join(pinyin_input[split[i-1] : split[i]]) for i in range(1, len(split))]
            if DEBUG:
                print("Work on split: ", input_split)

            old_sentences = {'': (.0, [])}
            for _len, phrase_py in enumerate(input_split):
                if DEBUG:
                    print('[%d]: ' % _len, old_sentences)
                new_phrase_len = len(phrase_py.split())
                new_sentences = {}
                if phrase_py in self.jieba_dict:
                    # update sentence using jieba
                    for p in self.jieba_dict[phrase_py]:
                        best_sentence = ''
                        best_fenci = []
                        max_prob = float('-inf')
                        for sentence, tup in old_sentences.items():
                            p_prev = tup[1][-1] if len(tup[1]) > 0 else ''
                            cur_prob = tup[0] + log(self.get_probability_jb(p, p_prev))
                            if cur_prob > max_prob:
                                max_prob = cur_prob
                                best_sentence = sentence + p
                                best_fenci = tup[1] + [p]
                        new_sentences[best_sentence] = (max_prob, best_fenci)
                if phrase_py not in self.jieba_dict or new_phrase_len == 1:
                    # update sentence using n-gram
                    simple_os = {}
                    for s, tup in old_sentences.items():
                        simple_os[s] = tup[0]
                    simple_ns = super()._translate(phrase_py, simple_os)
                    for s, p in simple_ns.items():
                        os = s[:-new_phrase_len]
                        if s not in new_sentences:
                            new_sentences[s] = (p, old_sentences[os][1]+[s[-new_phrase_len:]])
                old_sentences = new_sentences
            
            if DEBUG:
                print('finish split: ', old_sentences)
            best = list(old_sentences.items())
            best.sort(key=lambda r: r[1][0], reverse=True)
            if best[0][1][0] > all_max_prob:
                (all_best_sentence, all_max_prob) = (best[0][0], best[0][1][0])

        return all_best_sentence
            
    def get_probability_jb(self, p: str, p_prev: str):
        ' Return P(w | w_prev) '
        if DEBUG:
            print("Get prob: ", p, p_prev)
        idx = self.jieba_index[p]
        if p_prev == '' or p_prev not in self.jieba_index:
            return self.jieba_freq[idx]
        # if p_prev in self.jieba_index:
        idx_prev = self.jieba_index[p_prev]
        p1 = self.jieba_prob[idx_prev][idx]
        p2 = self.jieba_freq[idx]
        return p1 + self.jieba_lambdas[idx] * p2

    def cut_n(self, input_len):
        ' Cut input to various length segments '
        res = []
        tmp = [[0]]
        for _ in range(input_len):
            new_tmp = []
            for l in tmp:
                cur_sum = l[-1]
                pos_max = min(input_len+1, cur_sum+7)
                for t in range(cur_sum + 1, pos_max):
                    if t < input_len:
                        new_tmp.append(l + [t])
                    else:
                        res.append(l + [input_len])
            tmp = new_tmp
        return res


