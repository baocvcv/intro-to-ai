# coding=utf-8
import torch.nn as nn
from ray import tune
import numpy as np

params = {
    'model': 'tRNN',
    'tuning': False,
    'dropout': 0.6,
    'pad_size': 350,
    'lr': 1e-4,
    'weight_decay': 3e-3,
    'hidden_size': 32,
    'num_layers': 1,
}

params_tune = {
    'model': 'tRNN',
    'tuning': True,
    'dropout': tune.sample_from(lambda spec: np.random.uniform(0.2, 0.8)),
    'pad_size': tune.sample_from(lambda spec: np.random.randint(193, 512)),
    'lr': tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    'weight_decay': tune.sample_from(lambda spec: np.random.uniform(.0, 0.01)),
    'hidden_size': tune.choice([32, 64, 128, 192]),
    'num_layers': tune.choice([1, 2, 4]),
}

class Model(nn.Module):
    '''Recurrent Neural Network for Text Classification with Multi-Task Learning'''

    def __init__(self, params, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, int(params['hidden_size']), int(params['num_layers']),
                            bidirectional=True, batch_first=True, dropout=params['dropout'])
        self.fc = nn.Linear(int(params['hidden_size']) * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # hidden state
        return out
