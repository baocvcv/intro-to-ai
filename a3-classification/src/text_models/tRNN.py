# coding=utf-8
import torch.nn as nn
from ray import tune
import numpy as np

params = {
    'model': 'tRNN',
    'tuning': False,
    'dropout': 0.43,
    'pad_size': 334,
    'lr': 4e-4,
    'weight_decay': 3e-3,
    'hidden_size': 128,
    'num_layers': 4,
}

params_tune = {
    'model': 'tRNN',
    'tuning': True,
    'dropout': tune.sample_from(lambda spec: np.random.uniform(0.2, 0.8)),
    'pad_size': tune.sample_from(lambda spec: np.random.randint(16, 512)),
    'lr': tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
    'weight_decay': tune.sample_from(lambda spec: np.random.uniform(.0, 0.01)),
    'hidden_size': 128,
    'num_layers': 4,
}

class Model(nn.Module):
    '''Recurrent Neural Network for Text Classification with Multi-Task Learning'''

    def __init__(self, params, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, params['hidden_size'], params['num_layers'],
                            bidirectional=True, batch_first=True, dropout=params['dropout'])
        self.fc = nn.Linear(params['hidden_size'] * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # hidden state
        return out

    ''' Flexible length RNN '''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
