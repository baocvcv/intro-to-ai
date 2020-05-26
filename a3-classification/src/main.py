# coding=utf-8
import argparse
import time
from importlib import import_module

import torch
import numpy as np

from train_eval import train, init_network, test
from util import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description="Text Classification")
parser.add_argument("--model", "-m", type=str, required=True,
                    choices=['tCNN', 'tRNN'],
                    help="choose a model from tCNN, tRNN")
parser.add_argument("cmd", type=str, default="test",
                    choices=['train', 'test', 'tune'],
                    help="train or test the model")

if __name__ == '__main__':
    args = parser.parse_args()
    module = import_module('text_models.' + args.model)

    if args.cmd == 'train':
        params = module.params
        train(params)
    elif args.cmd == 'test':
        params = module.params
        from text_models.base_config import BaseConfig
        config = BaseConfig
        vocab_dict, _, _, d_test = build_dataset(config, use_word=config.use_word)
        test_iter = build_iterator(d_test, config)
        config.n_vocab = len(vocab_dict)
        model = module.Model(params, config).to(config.device)
        test(config, model, test_iter)
    elif args.cmd == 'tune':
        pass

        

    
