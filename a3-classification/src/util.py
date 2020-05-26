# coding=utf-8
import os
from os.path import join
import time
from datetime import timedelta
import pickle as pkl
from collections import defaultdict
import argparse

import torch
import numpy as np
from tqdm import tqdm

# max length for vocab dict
MAX_VOCAB_SIZE = 50000
UNK, PAD = '<UNK>', '<PAD>'  # unknown，padding
# label dict
LABEL = {'感动':0, '同情':1, '无聊':2, '愤怒':3,
         '搞笑':4, '难过':5, '新奇':6, '温馨':7}

def train_valid_split(input_file, saveTo, ratio=0.1):
    ''' split file into train and valid dataset '''
    train = []
    valid = []
    with open(input_file, encoding='UTF-8', mode='r') as f:
        valid_step = int(1 / ratio)
        for i, line in enumerate(f):
            if i % valid_step == 0:
                valid.append(line)
            else:
                train.append(line)
    with open(join(saveTo, 'train.txt'), 'w') as f:
        f.writelines(train)
    with open(join(saveTo, 'valid.txt'), 'w') as f:
        f.writelines(valid)

def inspect_train(train_file):
    ''' inpect training file '''
    from collections import Counter
    label_cnt = Counter()
    num_bins = 20
    bins = [0] * num_bins
    step = 50
    with open(train_file, 'r') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue

            _, label, content = lin.split('\t')
            label = label.split(' ')[1:]
            label = [l.split(':') for l in label]
            label = sorted(label, key=lambda x: x[1], reverse=True)[0]
            label = label[0]
            label_cnt[label] += 1

            length = len(content.split(' '))
            if length > (step * (num_bins-1)):
                bins[-1] += 1
            else:
                bins[int(length / step)] += 1
    total_cnt = 0
    for label in label_cnt:
        total_cnt += label_cnt[label]
    for label in label_cnt:
        print(label, '=', '%4d' % label_cnt[label],
              '%.3f' % (label_cnt[label]/total_cnt))
    cnt_cumulative = 0
    for i, cnt in enumerate(bins):
        cnt_cumulative += cnt
        print('%3d' % (i*step), '=', '%5d' % cnt, '%.3f' % (cnt_cumulative/total_cnt))
    print("total_cnt =", total_cnt)


def build_vocab(input_file_path, tokenizer, max_size, min_freq):
    ''' Build vocab dictionary '''
    vocab_freq = defaultdict(int)
    with open(input_file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[2]
            for word in tokenizer(content):
                vocab_freq[word] += 1
        vocab_list = sorted([v for v in vocab_freq.items() if v[1] >= min_freq],
                            key=lambda x: x[1],
                            reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, params, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # word-based
    else:
        tokenizer = lambda x: [y for y in x]  # char-based
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=5)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    unk = vocab.get(UNK)
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                _, label, content = lin.split('\t')
                # parse label
                label = label.split(' ')[1:]
                label = [l.split(':') for l in label]
                label = sorted(label, key=lambda x: int(x[1]), reverse=True)[0]
                label = LABEL[label[0]]
                # parse content
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, unk))
                contents.append((words_line, label, seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, params['pad_size'])
    valid = load_dataset(config.valid_path, params['pad_size'])
    test = load_dataset(config.test_path, params['pad_size'])
    return vocab, train, valid, test


# TODO: modify this!
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    ''' get time '''
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


parser = argparse.ArgumentParser(description='Parse training data with word vectors')
parser.add_argument('--input', '-i', type=str, default="../data/sina/sinanews.train",
                    help='training data file')
parser.add_argument('--vocab', '-v', type=str, default="../data/sina/vocab_dict.pkl",
                    help='vocabulary dictionary')
parser.add_argument('--embedding', '-e', type=str,
                    default="../word_vectors/merge_sgns_bigram_char300.txt", help='word vector file')
parser.add_argument('--output', '-o', type=str,
                    default="../data/sina/sinanews_embedding_merge_all", help='output file')
parser.add_argument('--ratio', '-r', type=float, default=0.1, help='ratio of valid')
parser.add_argument('cmd', type=str, choices=['split', 'check', 'build'])

if __name__ == "__main__":
    ''' extract embedding '''
    args = parser.parse_args()
    if args.cmd == 'build':
        train_file = args.input
        vocab_file = args.vocab
        pretrain_file = args.embedding
        train_embedded = args.output
        # load vocabulary dict
        if os.path.exists(vocab_file):
            word_to_id = pkl.load(open(vocab_file, 'rb'))
        else:
            tokenizer = lambda x: x.split(' ')  # word-based
            # tokenizer = lambda x: [y for y in x]  # char-based
            word_to_id = build_vocab(train_file , tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(word_to_id, open(vocab_file, 'wb'))

        # generate vocab encodings
        with open(pretrain_file, "r", encoding='UTF-8') as f:
            l = f.readline().strip().split(' ')
            no_word, emb_dim = (int(l[0]), int(l[1]))
            embeddings = np.random.rand(len(word_to_id), emb_dim)
            embedding_hit = 0
            for i, line in enumerate(f):
                lin = line.strip().split(" ") # split to words
                if lin[0] in word_to_id:
                    idx = word_to_id[lin[0]]
                    emb = [float(x) for x in lin[1:301]]
                    embeddings[idx] = np.asarray(emb, dtype='float32')
                    embedding_hit += 1
        print("vocab_dict_size =", len(word_to_id))
        print("embedding_hit=", embedding_hit)
        np.savez_compressed(train_embedded , embeddings=embeddings)
    elif args.cmd == 'split':
        train_valid_split(args.input,
                          saveTo=join(*(args.input.split('/')[:-1])),
                          ratio=args.ratio)
    elif args.cmd == 'check':
        inspect_train(args.input)
