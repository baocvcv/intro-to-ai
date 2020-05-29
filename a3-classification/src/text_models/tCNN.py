# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
import numpy as np

params = {
    'model': 'tCNN',
    'tuning': False,
    'dropout': 0.43,
    'pad_size': 350,
    'lr': 5e-4,
    'weight_decay': 3e-3,
    'filter_sizes': (2, 4, 8),
    'num_filters': 150,
}

params_tune = {
    #'model': 'tCNN',
    #'tuning': True,
    #'dropout': tune.sample_from(lambda spec: np.random.uniform(0.2, 0.8)),
    #'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
    #'pad_size': tune.sample_from(lambda spec: np.random.randint(16, 512)),
    #'lr': [1e-4, 5e-4, 1e-3, 1e-2, 1e-1],#tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    #'weight_decay': tune.sample_from(lambda spec: np.random.uniform(.0, 0.01)),
    #'filter_sizes': tune.choice([(2, 3, 4), (2, 4, 8), (2, 3, 6, 12)]),
    #'filter_sizes': [(2,3,4), (2,4,8), (2,3,6,12)],
    #'num_filters': tune.choice([50, 100, 150, 200]),
    #'num_filters': [50, 100, 150, 200],
    'pad_size': [50, 100, 200, 300, 500],

    #'dropout': 0.43,
    #'pad_size': 350,
    #'lr': 4e-4,
    #'weight_decay': 3e-3,
    #'filter_sizes': (2, 4, 8),
    #'num_filters': 150,
}


class Model(nn.Module):
    '''Convolutional Neural Networks for Sentence Classification'''

    def __init__(self, params, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, params['num_filters'], (k, config.embed)) for k in params['filter_sizes']])
        self.dropout = nn.Dropout(params['dropout'])
        self.fc = nn.Linear(params['num_filters'] * len(params['filter_sizes']), config.num_classes)

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
