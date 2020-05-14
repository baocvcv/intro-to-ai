from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_config import BaseConfig

# class Config(object):

#     """配置参数"""
#     def __init__(self, dataset, embedding):
#         self.model_name = 'tCNN'
#         self.dataset_path = join('../data/', dataset)
#         self.train_path = join(self.dataset_path, 'train.txt')
#         self.dev_path = join(self.dataset_path, 'validate.txt')
#         self.test_path = join(self.dataset_path, 'sinanews.test')
#         self.class_list = [x.strip() for x in open(
#             join(self.dataset_path, 'classes.txt'), encoding='utf-8').readlines()]
#         self.vocab_path = join(self.dataset_path, 'vocab_dict.pkl')
#         self.save_path = join(self.dataset_path,
#                               'saved_model/',
#                               self.model_name + '.ckpt')
#         self.log_path = join(self.dataset_path, 'log', self.model_name)
#         if embedding != 'random':
#             self.embedding_pretrained = torch.tensor(
#                 np.load(join(self.dataset_path, embedding))["embeddings"].astype('float32'))
#         else:
#             self.embedding_pretrained = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

#         ''' general training params '''
#         self.dropout = 0.5
#         # if error rate does not improve over #require_improvment epochs, stop training
#         self.require_improvement = 1000
#         self.num_classes = len(self.class_list)
#         # vocab dict length
#         self.n_vocab = 0
#         self.num_epochs = 10 #TODO for debug
#         self.batch_size = 32
#         # sentence length
#         self.pad_size = 512
#         self.learning_rate = 1e-3
#         self.embed = self.embedding_pretrained.size(1)\
#             if self.embedding_pretrained is not None else 300

#         ''' model specific params '''
#         # kernel sizes for the first layer
#         self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
#         # kernel number for the first layer
#         self.num_filters = 256                                          # 卷积核数量(channels数)

class Config(BaseConfig):
    """ Model configs """
    
    def __init__(self, dataset, embedding):
        super().__init__(dataset, embedding)

        ''' override training params '''
        self.num_epochs = 40 #TODO for debug
        self.batch_size = 32
        self.output_int = 30
        # sentence length
        self.pad_size = 512
        self.learning_rate = 1e-3

        ''' model params '''
        # kernel sizes for the first layer
        # TODO: ???
        self.filter_sizes = (2, 3, 4, 8, 16)                                   # 卷积核尺寸
        # kernel number for the first layer
        self.num_filters = 256                                          # 卷积核数量(channels数)

'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
