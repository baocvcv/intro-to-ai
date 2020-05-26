# coding=utf-8
import argparse
import time
from importlib import import_module

# import torch
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

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
        np.random.seed()
        # torch.manual_seed(1)
        # torch.cuda.manual_seed_all(1)
        # torch.backends.cudnn.deterministic = True
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
        params = module.params_tune
        ray.init(local_mode=True)
        analysis = tune.run(
            train,
            num_samples=50,
            scheduler=ASHAScheduler(metric='mean_accuracy', mode='max'),
            stop={
                'mean_accuracy': 0.9,
                'training_iteration': 100
            },
            config=params,
            fail_fast=True
        )
        print('Best config is:', analysis.get_best_config(metrix='mean_accuracy'))
        

    
