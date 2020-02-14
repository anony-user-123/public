from sklearn.metrics import accuracy_score
import numpy as np
import logging
import os


def make_possible_dirs(fns):
    for fn in fns:
        if os.path.dirname(fn) != '':
            os.makedirs(os.path.dirname(fn), exist_ok=True)


def get_logger(log_fn, mode='w'):
    make_possible_dirs([log_fn])
    logger = logging.getLogger('mc_logger')
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_fn, mode=mode)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


def performance_evaluate(y_true, y_pred, s):
    acc = accuracy_score(y_true, y_pred)
    
    ys1 = y_pred[s == 1]
    ys0 = y_pred[s == 0]
    discri = np.abs(np.mean(ys1) - np.mean(ys0))
    return acc, discri


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
