from scipy import optimize as optim
from numba.decorators import jit
from utils import *
import numpy as np
import argparse
import json
import os


@jit
def distances(X, v, alpha, N, P, k):
    dists = np.zeros((N, P))
    for i in range(N):
        for p in range(P):
            for j in range(k):    
                dists[i, j] += (X[i, p] - v[j, p]) * \
                               (X[i, p] - v[j, p]) * alpha[p]
    return dists

@jit
def M_nk(dists, N, k):
    M_nk = np.zeros((N, k))
    exp = np.zeros((N, k))
    denom = np.zeros(N)
    for i in range(N):
        for j in range(k):
            exp[i, j] = np.exp(-1 * dists[i, j])
            denom[i] += exp[i, j]
        for j in range(k):
            if denom[i]:
                M_nk[i, j] = exp[i, j] / denom[i]
            else:
                M_nk[i, j] = exp[i, j] / 1e-6
    return M_nk

@jit    
def M_k(M_nk, N, k):
    M_k = np.zeros(k)
    for j in range(k):
        for i in range(N):
            M_k[j] += M_nk[i, j]
        M_k[j] /= N
    return M_k

@jit        
def x_n_hat(X, M_nk, v, N, P, k):
    x_n_hat = np.zeros((N, P))
    L_x = 0.0
    for i in range(N):
        for p in range(P):
            for j in range(k):
                x_n_hat[i, p] += M_nk[i, j] * v[j, p]
            L_x += (X[i, p] - x_n_hat[i, p]) * (X[i, p] - x_n_hat[i, p])
    return x_n_hat, L_x

@jit
def yhat(M_nk, y, w, N, k):
    yhat = np.zeros(N)
    L_y = 0.0
    for i in range(N):
        for j in range(k):
            yhat[i] += M_nk[i, j] * w[j]
        yhat[i] = 1e-6 if yhat[i] <= 0 else yhat[i]
        yhat[i] = 0.999 if yhat[i] >= 1 else yhat[i]
        L_y += -1 * y[i] * np.log(yhat[i]) - (1.0 - y[i]) * \
               np.log(1.0 - yhat[i])
    return yhat, L_y


# @jit
def LFR(params, data_sensitive, data_nonsensitive, y_sensitive, 
        y_nonsensitive, k=10, A_x = 1e-4, A_y = 0.1, A_z = 1000, 
        results=0):
    
    # LFR.iters += 1 
    Ns, P = data_sensitive.shape
    Nns, _ = data_nonsensitive.shape
    
    alpha0 = params[:P]
    alpha1 = params[P : 2 * P]
    w = params[2 * P : (2 * P) + k]
    v = np.matrix(params[(2 * P) + k:]).reshape((k, P))
        
    dists_sensitive = distances(data_sensitive, v, alpha1, Ns, P, k)
    dists_nonsensitive = distances(data_nonsensitive, v, 
                                   alpha0, Nns, P, k)

    M_nk_sensitive = M_nk(dists_sensitive, Ns, k)
    M_nk_nonsensitive = M_nk(dists_nonsensitive, Nns, k)
    
    M_k_sensitive = M_k(M_nk_sensitive, Ns, k)
    M_k_nonsensitive = M_k(M_nk_nonsensitive, Nns, k)
    
    L_z = 0.0
    for j in range(k):
        L_z += abs(M_k_sensitive[j] - M_k_nonsensitive[j])

    x_n_hat_sensitive, L_x1 = x_n_hat(data_sensitive, 
                                      M_nk_sensitive, v, Ns, P, k)
    x_n_hat_nonsensitive, L_x2 = x_n_hat(data_nonsensitive, 
                                    M_nk_nonsensitive, v, Nns, P, k)
    L_x = L_x1 + L_x2

    yhat_sensitive, L_y1 = yhat(M_nk_sensitive, y_sensitive, w, Ns, k)
    yhat_nonsensitive, L_y2 = yhat(M_nk_nonsensitive, y_nonsensitive,
                                   w, Nns, k)
    L_y = L_y1 + L_y2

    criterion = A_x * L_x + A_y * L_y + A_z * L_z

    # if LFR.iters % 250 == 0:
    #     print(LFR.iters, criterion)
      
    if results:
        return (yhat_sensitive, yhat_nonsensitive,
                M_nk_sensitive, M_nk_nonsensitive)
    else:
        return criterion
# LFR.iters = 0


class DataReader:

    def __init__(self, options):
        self.data = np.load(options['data_fn'], allow_pickle=True)

    def n_features(self):
        return self.data['X_train'].shape[1]

    def get_data(self, data_category='train'):
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
        sen_idx = (s == 1)
        nosen_idx = (s == 0)
        X_sen = X[sen_idx, :]
        X_nosen = X[nosen_idx, :]
        y_sen = y[sen_idx]
        y_nosen = y[nosen_idx]
        return X_sen, X_nosen, y_sen, y_nosen


class Predictor:

    def __init__(self, options):
        self.k = options['k']
        self.A_x = options['A_x']
        self.A_y = options['A_y']
        self.A_z = options['A_z']

    def predict(self, weights, data):
        X_sen, X_nosen, y_sen, y_nosen = data
        # TODO
        yhat_sen, yhat_nosen, _, _ = LFR(weights, X_sen, X_nosen, 
                                         y_sen, y_nosen, self.k, 
                                         self.A_x, self.A_y, self.A_z,
                                         results=1)
        trues = np.concatenate((y_sen, y_nosen))
        preds = np.concatenate((yhat_sen, yhat_nosen))
        preds = (preds > 0.5).astype(np.int)
        ss = np.asarray([1] * y_sen.shape[0] + [0] * y_nosen.shape[0])
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


def evaluate(weights):
    evaluate.count += 1
    if evaluate.count % evaluate.eval_freq != 0:
        return
    acc, discri = evaluate.evaluator.evaluate(
                weights, evaluate.val_data, eval_consistency=False)
    if evaluate.model_selection_cri == 'discri':
        val_score = -discri
    elif evaluate.model_selection_cri == 'delta':
        val_score = acc - discri
    if val_score > evaluate.best_rslt:
        evaluate.best_rslt = val_score
        evaluate.best_models = weights.copy()
    log_info = 'Epoch: {} acc = {} discri = {}'.format(evaluate.count,
            acc, discri)
    evaluate.logger.info(log_info)
evaluate.count = 0


class LFRTrainer:

    def __init__(self, options):
        self.options = options

        self.data_reader = DataReader(options)
        self.predictor = Predictor(options)
        self.evaluator = Evaluator(self.predictor, options['knns_fn'])
        
        self.max_iter = options['max_iter']
        self.k = options['k']
        self.A_x = options['A_x']
        self.A_y = options['A_y']
        self.A_z = options['A_z']
        self.val_freq = options['val_freq']
        self.epsilon = options['epsilon'] # learning rate
        self.model_selection_cri = options['model_selection_cri']

    def _initialize(self):
        n_fts = self.data_reader.n_features()
        weights = np.random.uniform(size=(n_fts * 2 + self.k + 
                                          n_fts * self.k))
        bnd = []
        for i, _ in enumerate(weights):
            if i < n_fts * 2 or i >= n_fts * 2 + self.k:
                bnd.append((None, None))
            else:
                bnd.append((0, 1))
        return weights, bnd

    def train(self, logger):
        train_data = self.data_reader.get_data(data_category='train')
        val_data = self.data_reader.get_data(data_category='val')
        test_data = self.data_reader.get_data(data_category='test')

        evaluate.eval_freq = self.val_freq
        evaluate.evaluator = self.evaluator
        evaluate.val_data = val_data
        evaluate.model_selection_cri = self.model_selection_cri
        evaluate.best_rslt = float('-inf')
        evaluate.best_models = None
        evaluate.logger = logger

        weights, bnd = self._initialize()
        results = optim.fmin_l_bfgs_b(
                LFR,
                x0=weights,
                epsilon=self.epsilon,
                args=train_data + \
                    (self.k, self.A_x, self.A_y, self.A_z, 0),
                bounds=bnd,
                approx_grad=True,
                maxiter=self.max_iter,
                callback=evaluate)
        weights = results[0]

        log_info = 'Val criterion = {} Val score = {}'.format(
                self.model_selection_cri, evaluate.best_rslt)
        logger.info(log_info)
        acc, discri, consist = self.evaluator.evaluate(
            evaluate.best_models, test_data, eval_consistency=True)
        log_info = 'Final acc = {} discri = {} consist = {}'.format(
            acc, discri, consist)
        logger.info(log_info)
        return evaluate.best_models


if __name__ == '__main__':
    options = {'data_fn': './data/adult/adult.npz',
               'knns_fn': './data/adult/knns.txt',
               'k': 10,
               'max_iter': 2,
               'A_x': 1.0,
               'A_y': 1.0,
               'A_z': 1.0,
               'val_freq': 2,
               'epsilon': 1e-4,
               'model_selection_cri': 'discri',}

    log_fn = 'lfr_test.log'
    logger = get_logger(log_fn)
    trainer = LFRTrainer(options)
    trainer.train(logger)
