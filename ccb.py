from torch import nn
from utils import *
import numpy as np
import argparse
import torch
import json
import os


class DataIterator:

    def __init__(self, data, cuda):
        self.X, self.y, self.s = data
        self.batch_size = 1
        self.cuda = cuda
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.X.shape[0]:
            self.idx = 0
            raise StopIteration()
        X = torch.as_tensor(
            self.X[self.idx:(self.idx + self.batch_size),:])
        X = X.type(torch.float)
        y = torch.as_tensor(
            self.y[self.idx:(self.idx + self.batch_size)])
        y = y.type(torch.float)
        s = self.s[self.idx]
        self.idx += X.shape[0]
        if self.cuda:
            X, y = X.cuda(), y.cuda()
        return X, y, s


class DataReader:

    def __init__(self, options):
        self.data = np.load(options['data_fn'], allow_pickle=True)
        self.cuda = options['cuda']

    def groups(self):
        return np.unique(self.data['s_train'])

    def n_features(self):
        return self.data['X_train'].shape[1]

    def n_predictions(self):
        return len(np.unique(self.data['y_train']))

    def data_iterator_for_group(self, group, data_category='train'):
        if data_category == 'train':
            inds = self.data['train_indices']
            X = self.data['X_train'][inds,:]
            y = self.data['y_train'][inds]
            s = self.data['s_train'][inds]
        elif data_category == 'val':
            inds = self.data['val_indices']
            X = self.data['X_train'][inds,:]
            y = self.data['y_train'][inds]
            s = self.data['s_train'][inds]
        elif data_category == 'test':
            X = self.data['X_test']
            y = self.data['y_test']
            s = self.data['s_test']
        X = X[s == group,:]
        y = y[s == group]
        s = s[s == group]
        data = (X, y, s)
        return DataIterator(data, self.cuda)

    def data_iterator(self, data_category='train'):

        if data_category == 'train':
            inds = self.data['train_indices']
            X = self.data['X_train'][inds,:]
            y = self.data['y_train'][inds]
            s = self.data['s_train'][inds]
        elif data_category == 'val':
            inds = self.data['val_indices']
            X = self.data['X_train'][inds,:]
            y = self.data['y_train'][inds]
            s = self.data['s_train'][inds]
        elif data_category == 'test':
            X = self.data['X_test']
            y = self.data['y_test']
            s = self.data['s_test']

        # for test only
        # X = X[:100,:]
        # y = y[:100]
        # s = s[:100]

        data = (X, y, s)
        return DataIterator(data, self.cuda)


class GradientContextualBandits(nn.Module):
    '''
    Contextual Bandits with Policy Gradient
    '''

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.Softmax(dim=1))

    def forward(self, X):
        return self.net(X)


class Predictor:

    def __init__(self, cuda=False):
        self.cuda = cuda

    def predict(self, model, data):
        trues, preds, ss = [], [], []
        for X, y, s in data:
            with torch.no_grad():
                cur_preds = model[s](X)
            if self.cuda:
                cur_preds = cur_preds.cpu()
                y = y.cpu()
            cur_preds = cur_preds.numpy()
            cur_preds = np.argmax(cur_preds, axis=1)
            preds.extend(cur_preds.flatten().tolist())
            trues.extend(y.numpy().tolist())
            ss.append(s)
        trues, preds = np.asarray(trues), np.asarray(preds)
        ss = np.asarray(ss)
        return trues, preds, ss


class Evaluator:

    def __init__(self, predictor, knns_fn, k=5):
        self.predictor = predictor
        self.k = k

        with open(knns_fn, 'r') as f:
            lines = f.read().strip().split('\n')
        knns = {}
        for idx, line in enumerate(lines):
            neighbors = line.split(',')
            neighbors = [int(neighbor) for neighbor in neighbors]
            knns[idx] = neighbors[:k]
        self.knns = knns

    def compute_consistency(self, y_pred, knns):
        diff = 0.0
        for idx in range(y_pred.shape[0]):
            diff += np.abs(y_pred[idx] - 
                        np.mean(y_pred[knns[idx][:self.k]]))
        return 1 - diff / y_pred.shape[0]

    def evaluate(self, model, data, eval_consistency=False):
        trues, preds, s = self.predictor.predict(model, data)
        acc, discri = performance_evaluate(trues, preds, s)

        if eval_consistency:
            consistency = self.compute_consistency(preds, self.knns)
            return acc, discri, consistency
        return acc, discri


class Trainer:

    def __init__(self, options):
        self.options = options

        self.data_reader = DataReader(options)
        self.in_dim = self.data_reader.n_features()
        self.hidden_dim = options['hidden_dim']
        self.out_dim = self.data_reader.n_predictions()
        self.gcbs = []
        self.optimizers = []
        for group in self.data_reader.groups():
            gcb = GradientContextualBandits(self.in_dim,
                                            self.hidden_dim,
                                            self.out_dim)
            if options['cuda']:
                gcb = gcb.cuda()
            self.gcbs.append(gcb)
            optimizer = torch.optim.Adam(gcb.parameters(),
                                         lr=options['lr'])
            self.optimizers.append(optimizer)
        self.predictor = Predictor(options['cuda'])
        self.evaluator = Evaluator(self.predictor, options['knns_fn'],
                                   k=options['k'])
        
        self.max_epoch = options['max_epoch']
        self.val_freq = options['val_freq']
        self.model_selection_cri = options['model_selection_cri']
        self.cuda = options['cuda']
        self.beta = options['beta']

    def copy_models(self):
        new_gcbs = []
        for gcb in self.gcbs:
            new_gcb = GradientContextualBandits(self.in_dim,
                                                self.hidden_dim,
                                                self.out_dim)
            new_gcb.load_state_dict(gcb.state_dict())
            new_gcbs.append(new_gcb)
        return new_gcbs

    def take_action(self, pred):
        if self.cuda:
            pred = pred.cpu()
        pred = pred.detach().numpy().flatten()
        return np.random.choice(pred.shape[0], p=pred)

    def kl_divergence(self, pred, other_pred):
        other_pred += 1e-8
        return torch.sum(pred * torch.log(pred / other_pred))

    def compute_reward(self, X, y, pred, action, s):
        unfairness = 0.0
        for idx, gcb in enumerate(self.gcbs):
            if idx == s:
                continue
            with torch.no_grad():
                other_pred = gcb(X)
                unfairness += self.kl_divergence(pred, other_pred)
        if y[0] == action:
            reward = 1
        else:
            reward = 0
        return reward - unfairness * self.beta

    def train(self, logger):
        train_data = self.data_reader.data_iterator(
                        data_category='train')
        val_data = self.data_reader.data_iterator(
                        data_category='val')
        test_data = self.data_reader.data_iterator(
                        data_category='test')
        
        idx = 0
        best_rslt = float('-inf')
        best_models = None
        for epoch in range(self.max_epoch):
            for X, y, s in train_data:
                idx += 1
                self.optimizers[s].zero_grad()
                pred = self.gcbs[s](X)

                action = self.take_action(pred)
                reward = self.compute_reward(X, y, pred, action, s)

                policy_loss = -1 * torch.log(pred[0, action]) * reward
                policy_loss.backward()
                self.optimizers[s].step()

                if idx % self.val_freq == 0 and epoch > 0:
                    acc, discri = self.evaluator.evaluate(self.gcbs, 
                        val_data, eval_consistency=False)
                    if self.model_selection_cri == 'discri':
                        val_score = -discri
                    elif self.model_selection_cri == 'delta':
                        val_score = acc - discri
                    if val_score > best_rslt:
                        best_rslt = val_score
                        best_models = self.copy_models()
                    log_info = 'Epoch: {} acc = {} discri = {}'.format(
                                epoch, acc, discri)
                    logger.info(log_info)

        log_info = 'Val criterion = {} Val score = {}'.format(
                    self.model_selection_cri, best_rslt)
        logger.info(log_info)
        acc, discri, consist = self.evaluator.evaluate(best_models, test_data,
                    eval_consistency=True)
        log_info = 'Final acc = {} discri = {} consist = {}'.format(
                    acc, discri, consist)
        logger.info(log_info)
        return best_models


def save_models(models, trainer, out_fn):
    model_state_dicts = []
    for model in models:
        model_state_dicts.append(model.state_dict())
    torch.save({
        'model_state_dicts': model_state_dicts,
        'in_dim': trainer.in_dim,
        'hidden_dim': trainer.hidden_dim,
        'out_dim': trainer.out_dim
        }, out_fn)


def load_models(in_fn):
    data = torch.load(in_fn)
    gcbs = []
    for sd in data['model_state_dicts']:
        gcb = GradientContextualBandits(data['in_dim'],
                                        data['hidden_dim'],
                                        data['out_dim'])
        gcb.load_state_dict(sd)
        gcbs.append(gcb)
    return gcbs


if __name__ == '__main__':
    options = {'data_fn': './data/adult/adult.npz',
               'knns_fn': './data/adult/knns.txt',
               'cuda': False,
               'max_epoch': 2,
               'lr': 0.0001,
               'hidden_dim': 20,
               'beta': 10,
               'model_selection_cri': 'discr',
               'val_freq': 1000}

    log_fn = 'ccb_test.log'
    logger = get_logger(log_fn)
    trainer = Trainer(options)
    best_models = trainer.train(logger)
