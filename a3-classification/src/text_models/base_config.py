# coding=utf-8
from os.path import join

import torch
import numpy as np

class BaseConfig(object):
    """ Model configs """

    def __init__(self, dataset, embedding):
        ''' all the parameters '''
        self.model_name = 'model'

        ''' paths and devices '''
        self.dataset = dataset
        self.dataset_path = join(
            '/home/fred/External/Codes/intro-to-ai/a3-classification/data/',
            dataset
        )
        self.dataset_path = join(
            '/home/mengxy/bh/intro-to-ai/a3-classification/data/',
            dataset
        )
        self.train_path = join(self.dataset_path, 'train.txt.boost')
        self.valid_path = join(self.dataset_path, 'valid.txt')
        self.test_path = join(self.dataset_path, 'sinanews.test')
        self.class_list = [x.strip() for x in open(
            join(self.dataset_path, 'classes.txt'), encoding='utf-8').readlines()]
        self.vocab_path = join(self.dataset_path, 'vocab_dict.pkl')
        self.save_path = join(self.dataset_path,
                              'saved_model/',
                              self.model_name + '.ckpt')
        self.log_path = join(self.dataset_path, 'log', self.model_name)
        self.embedding = embedding
        if embedding != 'random':
            self.embedding_pretrained = torch.tensor(
                np.load(join(self.dataset_path, embedding))["embeddings"].astype('float32'))
        else:
            self.embedding_pretrained = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        ''' general training params '''
        # if error rate does not improve over #require_improvment epochs, stop training
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        # vocab dict length
        self.n_vocab = 0
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.use_word = True
        self.num_epochs = 200
        self.batch_size = 16
        self.output_int = 32
