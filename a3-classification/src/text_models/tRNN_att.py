# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray import tune
import numpy as np

params = {
    'model': 'tRNN_att',
    'tuning': False,
    'dropout': 0.43,
    'pad_size': 334,
    'lr': 4e-4,
    'weight_decay': 3e-3,
    'hidden_size': 128,
    'num_layers': 2,
    'hidden_size2': 64,
}

class Model(nn.Module):
    '''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''

    def __init__(self, params, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, params['hidden_size'], params['num_layers'],
                            bidirectional=True, batch_first=True, dropout=params['dropout'])
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(params['hidden_size'] * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(params['hidden_size'] * 2, params['hidden_size2'])
        self.fc = nn.Linear(params['hidden_size2'], config.num_classes)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
