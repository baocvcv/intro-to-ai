# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
import numpy as np

params = {
    'model': 'mlp',
    'tuning': False,
    'dropout': 0.5,
    'pad_size': 64,
    'lr': 5e-4,
    'weight_decay': 3e-3,
    'hidden_layer': 100,
}

params_tune = {
    #'model': 'tCNN',
    #'tuning': True,
    #'dropout': tune.sample_from(lambda spec: np.random.uniform(0.2, 0.8)),
    #'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
    #'pad_size': tune.sample_from(lambda spec: np.random.randint(16, 512)),
    'pad_size': [50, 100, 200, 300, 500]
    #'lr': [1e-4, 5e-4, 1e-3, 1e-2, 1e-1],#tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    #'weight_decay': tune.sample_from(lambda spec: np.random.uniform(.0, 0.01)),
    #'hidden_layer': tune.choice([50, 100, 150, 200]),
    #'hidden_layer': [20, 30, 40 ,50],
}


class Model(nn.Module):
    '''Multilayer Perceptron Model for Sentence Classification'''

    def __init__(self, params, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.fc1 = nn.Linear(config.embed * params['pad_size'], params['hidden_layer'])
        self.dropout = nn.Dropout(params['dropout'])
        self.fc2 = nn.Linear(params['hidden_layer'], config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.view(out.shape[0], out.shape[1] * out.shape[2])
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
