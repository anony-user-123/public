from fairml import ridge_fair, beta
from utils import *
import numpy as np
import argparse
import json
import os


class DataIterator:

    def __init__(self, data):
        self.X, self.y, self.s = data
        self.batch_size = 1
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.X.shape[0]:
            self.idx = 0
            raise StopIteration()
        X = np.asarray(
            self.X[self.idx:(self.idx + self.batch_size),:])
        X = X.astype(np.float)
        y = np.asarray(
            self.y[self.idx:(self.idx + self.batch_size)])
        y = y.astype(np.float)
        s = self.s[self.idx]
        self.idx += X.shape[0]
        return X, y, s


class DataReader:

    def __init__(self, options):
        self.data = np.load(options['data_fn'], allow_pickle=True)
        self.dataset = options['dataset']
        self.max_epoch = options['epoch']

    def groups(self):
        return np.unique(self.data['s_train'])

    def n_features(self):
        return self.data['X_train'].shape[1]

    def n_predictions(self):
        return len(np.unique(self.data['y_train']))

    def _make_train_data(self, X, y, s):
        '''
        Parameters:
        -----------
        X: (n_samples, d)
        y: (n_samples,)
        s: (n_samples,)
        
        Returns:
        --------
        X_train: shape (T, k, d)
        Y_train: shape (T, k)
        K: the number of arms
        d: the number of features
        '''
        T = X.shape[0] * self.max_epoch
        K = np.unique(s).shape[0]
        d = X.shape[1]
        X_train = np.zeros((T, K, d))
        Y_train = np.zeros((T, K), dtype=int)
        for k in range(K):
            idx = 0
            t = 0
            while t < T:
                idx = idx % X.shape[0]
                if s[idx] == k:
                    X_train[t, k, :] = X[idx, :]
                    Y_train[t, k] = y[idx]
                    t += 1
                idx += 1
        return X_train, Y_train, K, d

    def data_iterator(self, data_category='train'):
        '''
        Returns:
        --------
        '''
        if data_category == 'train':
            inds = self.data['train_indices']
            X = self.data['X_train'][inds,:]
            y = self.data['y_train'][inds]
            s = self.data['s_train'][inds]
            X, Y, K, d = self._make_train_data(X, y, s)
            return X, Y, K, d
        elif data_category == 'val':
            inds = self.data['val_indices']
            X = self.data['X_train'][inds,:]
            y = self.data['y_train'][inds]
            s = self.data['s_train'][inds]
        elif data_category == 'test':
            X = self.data['X_test']
            y = self.data['y_test']
            s = self.data['s_test']
        # else:
        #     if data_category == 'train':
        #         inds = self.data['train_indices'][0]
        #         X = self.data['X'][inds, :]
        #         y = self.data['y'][inds]
        #         s = self.data['s'][inds]
        #         X, Y, K, d = self._make_train_data(X, y, s)
        #         return X, Y, K, d
        #     elif data_category == 'val':
        #         inds = self.data['val_indices'][0]
        #         X = self.data['X'][inds, :]
        #         y = self.data['y'][inds]
        #         s = self.data['s'][inds]
        #     elif data_category == 'test':
        #         inds = self.data['te_indices'][0]
        #         X = self.data['X'][inds, :]
        #         y = self.data['y'][inds]
        #         s = self.data['s'][inds]
        data = (X, y, s)
        return DataIterator(data)


class Predictor:

    def __init__(self):
        pass

    def predict(self, model, data):
        '''
        Parameters:
        -----------
        model : ndarray (k, d)
        data : DataIterator

        Returns:
        --------
        trues, preds, ss
        '''
        trues, preds, ss = [], [], []
        for X, y, s in data:
            pred = np.dot(X, model[[s]].T)
            preds.extend(pred.flatten().tolist())
            trues.extend(y.tolist())
            ss.append(s)
        preds = [1 if pred > 0.5 else 0 for pred in preds]
        trues = np.asarray(trues)
        preds = np.asarray(preds)
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
        # fix this, only compute neighbors in the test set
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
        # k, d, c: n_arms, n_fts, scale_ft_weights
        self.k = self.data_reader.groups().shape[0]
        self.d = self.data_reader.n_features()
        # self.model = beta(k, d, options['c']) # shape (k, d)

        self.predictor = Predictor()
        self.evaluator = Evaluator(self.predictor,
                                   options['knns_fn'], k=options['k'])

        # confidence parameter
        self.delta = options['delta']
        self.lambda_ = options['lambda_']
        self.val_freq = options['val_freq']
        self.model_selection_cri = options['model_selection_cri']

    def train(self, logger):
        X_train, Y_train, _, _ = self.data_reader.\
            data_iterator(data_category='train')
        val_data = self.data_reader.data_iterator(data_category='val')
        test_data = self.data_reader.data_iterator(data_category='test')

        T = X_train.shape[0]
        model, best_rslt = ridge_fair(X_train, Y_train, self.k, self.d, 
                   self.delta, T, self.lambda_, self.val_freq,
                   val_data, self.model_selection_cri, logger,
                   self.evaluator, _print_progress=True)
        log_info = 'Val criterion = {} Val score = {}'.format(
            self.model_selection_cri, best_rslt)
        logger.info(log_info)
        acc, discri, consist = self.evaluator.evaluate(model, 
            test_data, eval_consistency=True)
        log_info = 'Final acc = {} discri = {} consist = {}'.format(
            acc, discri, consist)
        logger.info(log_info)
        return model


if __name__ == '__main__':
    options = {'data_fn': './data/adult/adult.npz',
               'knns_fn': './data/adult/knns.txt',
               'dataset': 'adult',
               'k': 5,
               'epoch': 2,
               'delta': 0.5,
               'lambda_': 10.0,
               'val_freq': 10,
               'model_selection_cri': 'discri'}

    log_fn = 'rawlsian_test.log'
    logger = get_logger(log_fn)
    trainer = Trainer(options)
    trainer.train(logger)
