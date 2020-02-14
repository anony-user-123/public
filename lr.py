from sklearn.linear_model import LogisticRegression
from utils import performance_evaluate
from multiprocessing import Pool
import numpy as np


class Trainer:

    def __init__(self, options):
        self.options = options
        self.model = LogisticRegression(max_iter=options['max_iter'],
                                        C=options['C'],
                                        solver='lbfgs')

    def train(self):
        data = np.load(self.options['data_fn'], allow_pickle=True)
        train_indices = data['train_indices']
        X_train = data['X_train'][train_indices]
        y_train = data['y_train'][train_indices].astype(np.int)

        self.model.fit(X_train, y_train)
        return self.model


class Evaluator:

    def __init__(self, data_fn, knns_fn, k=5):
        self.data = np.load(data_fn, allow_pickle=True)
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

    def evaluate(self, model, data='val', eval_consistency=False):
        if data == 'val':
            val_indices = self.data['val_indices']
            X_test = self.data['X_train'][val_indices]
            y_test = self.data['y_train'][val_indices].astype(np.int)
            s_test = self.data['s_train'][val_indices].astype(np.int)
        elif data == 'test':
            X_test = self.data['X_test']
            y_test = self.data['y_test'].astype(np.int)
            s_test = self.data['s_test'].astype(np.int)

        y_pred = model.predict(X_test)
        acc, discri = performance_evaluate(y_test, y_pred, s_test)
        
        if eval_consistency:
            consistency = self.compute_consistency(y_pred, self.knns)
            return acc, discri, consistency
        return acc, discri


class Reporter:

    def __init__(self, options):
        self.trainer = Trainer(options)
        self.evaluator = Evaluator(options['data_fn'],
                                   options['knns_fn'],
                                   k=options['k'])
        self.model_selection_cri = options['model_selection_cri']

    def report(self):
        model = self.trainer.train()
        val_rslt = self.evaluator.evaluate(model, data='val', 
                                           eval_consistency=False)
        test_rslt = self.evaluator.evaluate(model, data='test', 
                                            eval_consistency=True)
        if self.model_selection_cri == 'discri':
            val_score = -val_rslt[1]
        elif self.model_selection_cri == 'delta':
            val_score = val_rslt[0] - val_rslt[1]

        return {'val_score': val_score,
                'test_acc': test_rslt[0],
                'test_discri': test_rslt[1],
                'test_consist': test_rslt[2]}


def get_report(model_selection_cri, C):
    options = {'max_iter': 500,
               'data_fn': '../data/health/health.npz',
               'knns_fn': '../data/health/knns.txt',
               'k': 5,
               'model_selection_cri': model_selection_cri,
               'C': C}
    rslt = Reporter(options).report()
    return rslt


def result_of_one_criterion(cri):
    # Cs = [0.0001, 0.0003, 0.0009, 0.001, 0.003, 0.009,
    #       0.01, 0.03, 0.09, 0.1, 0.3, 0.9]
    Cs = [0.3, 0.6, 1.0]
    # Cs = [1.0]
    paras = []
    for C in Cs:
        paras += [[cri, C]]
    pool = Pool()
    rslts = pool.starmap(get_report, paras)
    report = []
    for rslt in rslts:
        report += [[rslt['val_score'], rslt['test_acc'], 
                    rslt['test_discri'], rslt['test_consist']]]
    report = sorted(report, reverse=False)
    output_format = ('Val score = {}. Test acc = {}, discri = {} '
                     'consist = {}')
    outputs = []
    for rslt in report:
        output = output_format.format(*rslt)
        outputs += [output]
    return '\n'.join(outputs)


def main():
    discri_output = result_of_one_criterion('discri')
    delta_output = result_of_one_criterion('delta')
    output = ('Discrimination Criterion Results:\n {}\n\n'
              'Delta Criterion Results:\n {}').format(discri_output,
                                                      delta_output)
    out_fn = 'lr_result.txt'
    with open(out_fn, 'w') as f:
        f.write(output)


if __name__ == '__main__':
    main()
