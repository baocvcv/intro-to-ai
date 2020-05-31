# coding=utf-8
import time
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import pearsonr
from tensorboardX import SummaryWriter
from ray import tune
from ray.tune import track

from util import get_time_dif, build_dataset, build_iterator, calc_weight
from text_models.base_config import BaseConfig

dataset = 'sina'
embedding = 'sinanews_embedding_merge_all.npz'
#TODO: grid search for parameters, train-validation

# init weights, use xavier by default
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(params):
    config = BaseConfig(dataset, embedding)

    # load training data
    print("[Info] Start training " + params['model'] + " with " + config.dataset)
    weights = torch.from_numpy(calc_weight(config.train_path)).float().to(config.device)
    vocab_dict, d_train, d_valid, d_test = build_dataset(config, params, use_word=config.use_word)
    train_iter = build_iterator(d_train, config)
    valid_iter = build_iterator(d_valid, config)
    test_iter = build_iterator(d_test, config)
    config.n_vocab = len(vocab_dict)

    # load model
    module = import_module('text_models.' + params['model'])
    model = module.Model(params, config).to(config.device)
    init_network(model)
    print("[Info] Model parameters: ")
    print(model.parameters)

    # start training
    print("[Info] Using device ", config.device)
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'],
                                 weight_decay=params['weight_decay'])

    # decrease lr each epoch by a factor of gamma
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    valdation_best_loss = float('inf')
    cur_batch = 0  # current batch
    last_significant_batch = 0  # the last batch with improvement
    flag = False  # true if there is no improvement over a long period
    log = {'training_loss':[], 'validation_loss':[]}
    writer = SummaryWriter(
        log_dir=config.get_log_path(params['model']) + '/' + time.strftime('%m-%d_%H.%M', time.localtime())
    )
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        #scheduler.step() # 学习率衰减
        for i, (trains, labels, _) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels, weight=weights)
            loss.backward()
            optimizer.step()
            if cur_batch % config.output_int == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                validation_acc, valdation_loss = evaluate(config, model, valid_iter)
                if valdation_loss < valdation_best_loss:
                    valdation_best_loss = valdation_loss
                    torch.save(model.state_dict(), config.get_save_path(params['model']))
                    improve = '*'
                    last_significant_batch = cur_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(cur_batch, loss.item(), train_acc, valdation_loss, validation_acc, time_dif, improve))
                log['training_loss'].append(str(loss.item()))
                log['validation_loss'].append(str(valdation_loss.item()))
                writer.add_scalar("loss/train", loss.item(), cur_batch)
                writer.add_scalar("loss/valdation", valdation_loss, cur_batch)
                writer.add_scalar("acc/train", train_acc, cur_batch)
                writer.add_scalar("acc/valdation", validation_acc, cur_batch)
                model.train()
            cur_batch += 1
            if cur_batch - last_significant_batch > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag and not params['tuning']:
            break
        # valid after each epoch
        if params['tuning']:
            validation_acc, valdation_loss = evaluate(config, model, valid_iter)
            track.log(mean_accuracy=validation_acc)
    writer.close()
    acc = test(config, model, config.get_save_path(params['model']), test_iter)
    return log, acc


def test(config, model, model_path, test_iter):
    # test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, pearson = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    print("Pearson correlation = %.5f" % pearson)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return test_acc


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    pearson_sum = 0
    with torch.no_grad():
        for texts, labels, dist_labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            # calculate pearson
            outputs = outputs.cpu().numpy()
            dist_labels = dist_labels.cpu().numpy()
            for i in range(outputs.shape[0]):
                pearson_sum += pearsonr(outputs[i], dist_labels[i])[0]

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        #pearson = pearsonr(labels_all, predict_all)[0]
        pearson = pearson_sum / len(data_iter) / config.batch_size
        return acc, loss_total / len(data_iter), report, confusion, pearson
    return acc, loss_total / len(data_iter)
