from os.path import join

import torch
import torch.nn as nn
import numpy as np

from .base_config import BaseConfig

# class Config(object):
#     """ Model configs """

#     def __init__(self, dataset, embedding):
#         ''' all the parameters '''

#         ''' paths and devices '''
#         self.model_name = 'tRNN'
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
#         self.hidden_size = 128                                          # lstm隐藏层
#         self.num_layers = 2                                             # lstm层数

class Config(BaseConfig):
    """ Model configs """
    
    def __init__(self, dataset, embedding):
        super().__init__(dataset, embedding)

        ''' override training params '''
        self.num_epochs = 4 #TODO for debug
        self.batch_size = 64
        self.output_int = 20
        # sentence length
        self.pad_size = 512
        self.learning_rate = 1e-3

        ''' model params '''
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数

'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
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
