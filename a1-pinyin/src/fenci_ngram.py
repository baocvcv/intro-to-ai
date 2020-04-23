''' N-gram model improved by Jieba '''

import gc
from os import makedirs
from os.path import join, isdir, exists
from math import log
from collections import defaultdict, Counter, OrderedDict
import time
import re
import multiprocessing

import dill as pickle

from ngram_zhuyin import NGramPYModel
from config import DEBUG, D_VALUE, MAX_PHRASE_LEN, MAX_PROCESS_NUM

class XNGramModel(NGramPYModel):
    ' N-gram model with Jieba '
    LAMBDA = 0.9

    def __init__(self,
                 n=2,
                 table_path='pinyin_table',
                 file_path='',
                 model_path='models/n-gram',
                 zhuyin=False):
        ' Constructor '
        super().__init__(n, table_path, file_path, model_path)
        self.zhuyin = zhuyin

    def train(self, force=False):
        ' Train '
        # train phrase based 2-gram
        # word based n-gram should be trained already
        print("[Info] Training the model...")
        time_d = time.time()
        self.train_jieba()
        time_d = round(time.time()-time_d, 3)
        print("[Info] Training took %ss" % time_d)

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
                lambdas.append(0)
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
        # print("[Info] Max phrase length: ", max_len, max_p)
        # some statistics
        # print("total phrase cnt =", total_phrase_cnt)
        # print('no of different phrases =', len(index))
        # f_out = open('rare_words.txt', 'w')
        # cnt = [0] * 6
        # for p, idx in index.items():
        #     cnt_p = phrase_freq[idx]
        #     if cnt_p >= 100:
        #         cnt[5] += 1
        #     else:
        #         cnt[int(cnt_p / 20)] += 1
        #     if cnt_p < 20:
        #         f_out.write(p + '\n')
        # f_out.close()
        # for i in range(5):
        #     print('[%d-%d]:' % (i*20, i*20+20), cnt[i])
        # print('[100-]', cnt[5])
        # return
        
        # filter out some words
        new_index = {}
        new_phrase_dict = defaultdict(list)
        valid_idx_set = set()
        for py, phrases in phrase_dict.items():
            for p in phrases:
                idx = index[p]
                if phrase_freq[idx] >= 40 or len(p) >= 4:
                    new_index[p] = idx
                    new_phrase_dict[py].append(p)
                    valid_idx_set.add(idx)
                else:
                    total_phrase_cnt -= phrase_freq[idx]

        # calc probability
        for p, idx in new_index.items():
            cnt = phrase_freq[idx]
            cp = conditional_pro[idx]
            for w in cp:
                cp[w] = (cp[w] - D_VALUE) / cnt
            lambdas[idx] = D_VALUE / cnt
            phrase_freq[idx] /= total_phrase_cnt
        print("[Info] Training finished!")
        print("[Info] Saving model...")
        self.save_model(new_index, 'jieba_index.p')
        self.save_model(phrase_freq, 'jieba_freq.p')
        self.save_model(lambdas, 'jieba_lambdas.p')
        self.save_model(new_phrase_dict, 'jieba_dict.p')
        self.save_model(conditional_pro, 'jieba_prob.p')
        print("[Info] Model saved.")
        gc.collect()

    def load_model(self):
        ' Load saved model '
        print("[Info] Loading model...")
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
        if self.zhuyin:
            self._load_zymodel()
        else:
            self._load_model()
 
    def translate(self, pinyin_input: str) -> str:
        ' Translate '
        result = self._fc_translate(pinyin_input)
        if len(result) == 0:
            print("[Error] Please check your input.")
            return ''
        return result

    def _fc_translate(self, pinyin_input: str) -> str:
        ' Translate '
        pinyin_input = pinyin_input.lower().strip()
        self.pinyin_input = re.split(r'\s+', pinyin_input)
        splits = self.cut_n(len(self.pinyin_input))
        if len(self.pinyin_input) < 30:
            return self._fc_do_translate(splits)[0]
        else:
            args = []
            l = len(splits)
            chunksize = int(l / MAX_PROCESS_NUM)
            remainder = l % MAX_PROCESS_NUM
            for i in range(0, remainder*chunksize, chunksize+1):
                args.append(splits[i : i+1+chunksize])
            for i in range(remainder*(chunksize+1), l, chunksize):
                args.append(splits[i : i+chunksize])
            with multiprocessing.Pool(MAX_PROCESS_NUM) as pool:
                res = pool.map(self._fc_do_translate, args)
            best_sentence = ''
            max_prob = float('-inf')
            for r in res:
                if r[1] > max_prob:
                    max_prob = r[1]
                    best_sentence = r[0]
            return best_sentence

    def _fc_do_translate(self, splits: list):
        all_best_sentence = ''
        all_max_prob = float('-inf')
        for split in splits:
            input_split = [' '.join(self.pinyin_input[split[i-1] : split[i]]) for i in range(1, len(split))]
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
                    if DEBUG:
                        print('--- Using jieba ---')
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
                    # parse old_sentences to the format used by word-based ngram model
                    if DEBUG:
                        print('--- Using n-gram ---')
                    simple_os = {}
                    for s, tup in old_sentences.items():
                        simple_os[s] = tup[0]
                    # send data to be processed
                    if self.zhuyin:
                        simple_ns = self._zy_translate(phrase_py, simple_os)
                    else:
                        simple_ns = self._translate(phrase_py, simple_os)
                    # parse back
                    for s, p in simple_ns.items():
                        os = s[:-new_phrase_len]
                        if s not in new_sentences:
                            p_old = old_sentences[os][0]
                            p_new = p_old + (p - p_old) * self.LAMBDA
                            new_sentences[s] = (p_new, old_sentences[os][1]+[s[-new_phrase_len:]])
                old_sentences = new_sentences
            
            if DEBUG:
                print('finish split: ', old_sentences)
            best = list(old_sentences.items())
            best.sort(key=lambda r: r[1][0], reverse=True)
            if best[0][1][0] > all_max_prob:
                (all_best_sentence, all_max_prob) = (best[0][0], best[0][1][0])
                if DEBUG:
                    print('[BEST!!]', all_best_sentence, all_max_prob)

        return (all_best_sentence, all_max_prob)
            
    def get_probability_jb(self, p: str, p_prev: str):
        ' Return P(w | w_prev) '
        # if DEBUG:
            # print("Get prob: ", p, p_prev)
        idx = self.jieba_index[p]
        if p_prev == '' or p_prev not in self.jieba_index:
            return self.jieba_freq[idx]
        # if p_prev in self.jieba_index:
        idx_prev = self.jieba_index[p_prev]
        p1 = self.jieba_prob[idx_prev][idx]
        p2 = self.jieba_freq[idx]
        return p1 + self.jieba_lambdas[idx] * p2 * len(self.jieba_prob[idx_prev])

    def cut_n(self, input_len):
        ' Cut input to various length segments '
        res = []
        tmp = [[0]]
        for _ in range(input_len):
            new_tmp = []
            for l in tmp:
                cur_sum = l[-1]
                pos_max = min(input_len+1, cur_sum+1+MAX_PHRASE_LEN)
                for t in range(cur_sum + 1, pos_max):
                    if t < input_len:
                        new_tmp.append(l + [t])
                    else:
                        res.append(l + [input_len])
            tmp = new_tmp
        return res


