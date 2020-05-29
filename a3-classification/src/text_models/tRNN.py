# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
import numpy as np

params = {
    'model': 'tRNN',
    'tuning': False,
    'dropout': 0.4,
    'pad_size': 300,
    'lr': 5e-4,
    'weight_decay': 3e-3,
    'hidden_size': 128,
    'num_layers': 1,
}

params_tune = {
    #'model': 'tRNN',
    #'tuning': True,
    #'dropout': tune.sample_from(lambda spec: np.random.uniform(0.2, 0.8)),
    #'pad_size': tune.sample_from(lambda spec: np.random.randint(193, 512)),
    #'lr': tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    #'weight_decay': tune.sample_from(lambda spec: np.random.uniform(.0, 0.01)),
    #'hidden_size': tune.choice([32, 64, 128, 192]),
    #'num_layers': tune.choice([1, 2, 4]),
    #'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
    #'lr': [1e-4, 5e-4, 1e-3, 1e-2, 1e-1],#tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    #'hidden_size': [32, 64, 128, 192],
    #'num_layers': [1, 2, 4],
    'pad_size': [50, 100, 200, 300, 500],
}

class Model(nn.Module):
    '''Recurrent Neural Network for Text Classification with Multi-Task Learning'''

    def __init__(self, params, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                    config.embedding_pretrained, freeze=False
                    )
        else:
            self.embedding = nn.Embedding(config.n_vocab,
                    config.embed, padding_idx=config.n_vocab - 1)
        if params['num_layers'] == 1:
            self.rnn = nn.GRU(config.embed, int(params['hidden_size']),
                    int(params['num_layers']), bidirectional=True,
                    batch_first=True)
        else:
            self.rnn = nn.LSTM(config.embed, int(params['hidden_size']),
                    int(params['num_layers']), bidirectional=True,
                    batch_first=True, dropout=params['dropout'])
        self.W2 = nn.Linear(
                2 * int(params['hidden_size']) + config.embed,
                2 * int(params['hidden_size'])
                )
        self.fc = nn.Linear(int(params['hidden_size']) * 2, config.num_classes)
        self.dropout = nn.Dropout(params['dropout'])

    def forward(self, x):
        x, _ = x # [batch_size, seq_len]
        embed = self.dropout(self.embedding(x)) # [batch_size, seq_len, embeding]
        out, _ = self.rnn(embed) # [batch_size, seq_len, hidden_size*2]
        x = torch.cat((out, embed), 2)  # [batch, seq, embed+hidden*2]
        x = torch.tanh(self.W2(x)).permute(0, 2, 1)  # [batch, hidden*2, seq]
        x = F.max_pool1d(x, x.size()[2]).squeeze(2)
        #out = self.fc(out[:, -1, :])  # hidden state
        return self.fc(x)
