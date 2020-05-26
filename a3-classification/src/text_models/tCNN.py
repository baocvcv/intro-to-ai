# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_config import BaseConfig


class Config(BaseConfig):
    """ Model configs """

    def __init__(self, dataset, embedding):
        super().__init__(dataset, embedding)
        self.model_name = 'tCNN'

        ''' override training params '''
        self.dropout = 0.6
        self.num_epochs = 100
        self.batch_size = 16
        self.output_int = 50
        # sentence length
        self.pad_size = 400
        self.learning_rate = 1e-3

        ''' model params '''
        # kernel sizes for the first layer
        self.filter_sizes = (2, 3, 4, 8, 16)                 # 卷积核尺寸
        # kernel number for the first layer
        self.num_filters = 256 # 卷积核数量(channels数)


class Model(nn.Module):
    '''Convolutional Neural Networks for Sentence Classification'''

    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
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
