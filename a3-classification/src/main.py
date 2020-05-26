# coding=utf-8
import argparse
import time
from importlib import import_module

import torch
import numpy as np

from train_eval import train, init_network
from util import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description="Text Classification")
parser.add_argument("--model", "-m", type=str, required=True,
                    choices=['tCNN', 'tRNN'],
                    help="choose a model from tCNN, tRNN")
parser.add_argument("cmd", type=str, default="test",
                    choices=['train', 'test'],
                    help="train or test the model")

if __name__ == '__main__':
    args = parser.parse_args()

    dataset = 'sina'
    embedding = 'sinanews_embedding_merge_all.npz'

    module = import_module('text_models.' + args.model)
    config = module.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # load training data
    time0 = time.time()
    print("[Info] Start training " + args.model + " with " + dataset)
    vocab_dict, d_train, d_valid, d_test = build_dataset(config, use_word=True)
    d_train = build_iterator(d_train, config)
    d_valid = build_iterator(d_valid, config)
    d_test = build_iterator(d_test, config)
    print("[Info] took ", get_time_dif(time0))

    # training
    config.n_vocab = len(vocab_dict)
    model = module.Model(config).to(config.device)
    init_network(model)
    print("[Info] Model parameters: ")
    print(model.parameters)
    print("[Info] Training starts ...")
    train(config, model, d_train, d_valid, d_test)
