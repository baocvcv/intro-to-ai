# coding=utf-8
import argparse
import time
from importlib import import_module
from os.path import join

# import torch
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from train_eval import train, init_network, test
from util import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description="Text Classification")
parser.add_argument("--model", "-m", type=str, required=True,
                    choices=['tCNN', 'tRNN', 'dpCNN', 'tRNN_att'],
                    help="choose a model")
parser.add_argument("cmd", type=str, default="test",
                    choices=['train', 'test', 'tune', 'log'],
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
        analysis = tune.run(
            train,
            #num_samples=5,
            scheduler=ASHAScheduler(metric='mean_accuracy', mode='max'),
            stop={
                'mean_accuracy': 0.9,
                'training_iteration': 100
            },
            config=params,
            fail_fast=True,
            resources_per_trial={"gpu": 6}
        )
        print('Best config is:', analysis.get_best_config(metric='mean_accuracy'))
        df = analysis.trial_dataframes
        df.to_csv(r'/home/mengxy/bh/intro-to-ai/a3-classification/tune/'+args.model+'.csv')
        '''
        print(df)
        ax = None
        for d in df.values():
            ax = d.mean_accuracy.plot(ax=ax, legend=False)
        ax.savefig('/home/mengxy/bh/intro-to-ai/result.png')
        '''
    elif args.cmd == 'log':
        from copy import copy
        from collections import defaultdict
        params = module.params
        params_tune = module.params_tune
        name_list = []
        logs = {}
        acc_log = defaultdict(list)
        for name in params_tune:
            for val in params_tune[name]:
                params_new = copy(params)
                params_new[name] = val
                log, acc = train(params_new)
                logs[name + '=' + str(val)] = log
                acc_log[name].append((val, acc))
        print('Param Test_Acc')
        for name in acc_log:
            print(name)
            for val, acc in acc_log[name]:
                print(val, '&' , acc, r'\\')
        with open(join('log', params['model'] + '.csv'), 'w') as f:
            for name in logs:
                f.write(name + '\n')
                f.write('training_loss,')
                f.write(','.join(logs[name]['training_loss']))
                f.write('\n')
                f.write('validation_loss,')
                f.write(','.join(logs[name]['validation_loss']))
                f.write('\n')





