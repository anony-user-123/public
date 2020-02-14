from sklearn.metrics import accuracy_score
from torch import nn
from utils import *
import numpy as np
import argparse
import torch
import json
import os


class Encoder(nn.Module):
    
    def __init__(self, data_dim, arch, hidden_dim):
        super().__init__()
        
        layers = []
        arch = [data_dim] + arch + [hidden_dim]
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i != len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    
    def __init__(self, hidden_dim, arch, data_dim):
        super().__init__()
        
        layers = []
        arch = [hidden_dim + 1] + arch + [data_dim]
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i != len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, a):
        z = torch.cat([z, a], dim=1)
        return self.net(z)


class Classifier(nn.Module):
    
    def __init__(self, hidden_dim, arch):
        super().__init__()
        
        layers = []
        arch = [hidden_dim] + arch + [1]
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i != len(arch) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    
    def __init__(self, hidden_dim, arch):
        super().__init__()
        
        layers = []
        arch = [hidden_dim] + arch + [1]
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i != len(arch) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.net(z)


class DataIterator:

    def __init__(self, data, batch_size, cuda):
        self.X, self.y, self.s = data
        self.batch_size = batch_size
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
        s = torch.as_tensor(
            self.s[self.idx:(self.idx + self.batch_size)])
        s = s.type(torch.float)
        s = s.view(-1, 1)
        self.idx += X.shape[0]
        if self.cuda:
            X, y, s = X.cuda(), y.cuda(), s.cuda()
        return X, y, s


class DataReader:

    def __init__(self, options):
        self.data = np.load(options['data_fn'], allow_pickle=True)
        self.batch_size = options['batch_size']
        self.cuda = options['cuda']

    def n_features(self):
        return self.data['X_train'].shape[1]

    def data_iteraotr(self, data_category='train'):
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

        data = (X, y, s)
        return DataIterator(data, self.batch_size, self.cuda)


class ClsRecAdvLoss(nn.Module):
    
    def __init__(self, alpha, beta, gamma):
        super(ClsRecAdvLoss, self).__init__()
        
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.bce_criterion1 = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        self.bce_criterion2 = nn.BCELoss()
        
    def forward(self, y_hat, y, x_hat, x, a_hat, a):
        return self.alpha * self.bce_criterion1(y_hat, y) + \
               self.beta * self.mse_criterion(x_hat, x) - \
               self.gamma * self.bce_criterion2(a_hat, a)


class AdvLoss(nn.Module):
    
    def __init__(self, gamma):
        super(AdvLoss, self).__init__()
        
        self.gamma = gamma
        self.bce_criterion = nn.BCELoss()
        
    def forward(self, a_hat, a):
        return self.bce_criterion(a_hat, a)


class Predictor:

    def __init__(self, cuda=False):
        self.cuda = cuda

    def predict(self, model, data):
        '''
        model : tuple (encoder, classifier)
        '''
        encoder, clsfr = model
        trues, preds, ss = [], [], []
        for X, y, s in data:
            with torch.no_grad():
                cur_preds = clsfr(encoder(X))
            if self.cuda:
                cur_preds = cur_preds.cpu()
                y = y.cpu()
                s = s.cpu()
            # cur_preds = cur_preds.numpy()
            cur_preds = (cur_preds > 0.5).type(torch.int)
            trues.extend(y.numpy().flatten().tolist())
            preds.extend(cur_preds.numpy().flatten().tolist())
            ss.extend(s.numpy().flatten().tolist())
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
        self.data_reader = DataReader(options)

        self.data_dim = self.data_reader.n_features()
        self.arch_dim = options['arch_dim']
        self.hidden_dim = options['hidden_dim']
        self.encoder = Encoder(self.data_dim, [self.arch_dim], 
                               self.hidden_dim)
        self.decoder = Decoder(self.hidden_dim, [self.arch_dim], 
                               self.data_dim)
        self.clsfr = Classifier(self.hidden_dim, [self.arch_dim])
        self.disc = Discriminator(self.hidden_dim, [self.arch_dim])
        if options['cuda']:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.clsfr = self.clsfr.cuda()
            self.disc = self.disc.cuda()
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                            lr=options['lr'])
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),
                                            lr=options['lr'])
        self.clsfr_opt = torch.optim.Adam(self.clsfr.parameters(),
                                          lr=options['lr'])
        self.disc_opt = torch.optim.Adam(self.disc.parameters(),
                                         lr=options['lr'])
        
        self.predictor = Predictor(options['cuda'])
        self.evaluator = Evaluator(self.predictor, options['knns_fn'],
                                   k=options['k'])
        self.max_epoch = options['max_epoch']
        self.val_freq = options['val_freq']
        self.model_selection_cri = options['model_selection_cri']
        self.cuda = options['cuda']
        self.alpha = options['alpha']
        self.beta = options['beta']
        self.gamma = options['gamma']

    def copy_models(self):
        encoder = Encoder(self.data_dim, [self.arch_dim],
                          self.hidden_dim)
        decoder = Decoder(self.hidden_dim, [self.arch_dim],
                          self.data_dim)
        clsfr = Classifier(self.hidden_dim, [self.arch_dim])
        disc = Discriminator(self.hidden_dim, [self.arch_dim])
        encoder.load_state_dict(self.encoder.state_dict())
        decoder.load_state_dict(self.decoder.state_dict())
        clsfr.load_state_dict(self.clsfr.state_dict())
        disc.load_state_dict(self.disc.state_dict())
        return {'encoder': encoder, 'decoder': decoder,
                'clsfr': clsfr, 'disc': disc}

    def train(self, logger):
        train_data = self.data_reader.data_iteraotr(
                        data_category='train')
        val_data = self.data_reader.data_iteraotr(
                        data_category='val')
        test_data = self.data_reader.data_iteraotr(
                        data_category='test')

        cls_rec_adv_criterion = ClsRecAdvLoss(self.alpha, self.beta,
                                              self.gamma)
        adv_criterion = AdvLoss(self.gamma)

        idx = 0
        best_rslt = float('-inf')
        best_models = None
        for epoch in range(self.max_epoch):
            for x, y, s in train_data:
                idx += 1
                # train encoder, decoder, clsfr
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                self.clsfr_opt.zero_grad()

                z = self.encoder(x)
                x_hat = self.decoder(z, s)
                y_hat = self.clsfr(z)
                s_hat = self.disc(z)

                cls_rec_adv_loss = cls_rec_adv_criterion(y_hat, y,
                                    x_hat, x, s_hat, s)
                cls_rec_adv_loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()
                self.clsfr_opt.step()

                # train discriminator
                self.disc_opt.zero_grad()
                z = self.encoder(x)
                s_hat = self.disc(z)

                adv_loss = adv_criterion(s_hat, s)
                adv_loss.backward()
                self.disc_opt.step()

                if idx % self.val_freq == 0:
                    acc, discri = self.evaluator.evaluate(
                                    (self.encoder, self.clsfr),
                                    val_data,
                                    eval_consistency=False)
                    if self.model_selection_cri == 'discri':
                        val_score = -discri
                    elif self.model_selection_cri == 'delta':
                        val_score = acc - discri
                    if val_score > best_rslt:
                        best_rslt = val_score
                        best_model = self.copy_models()
                    log_info = 'Epoch : {} acc = {} discri = {}'.format(
                            epoch, acc, discri)
                    logger.info(log_info)
        log_info = 'Val criterion = {} Val score = {}'.format(
                    self.model_selection_cri, best_rslt)
        logger.info(log_info)
        best_eval_model = (best_model['encoder'], best_model['clsfr'])
        acc, discri, consist = self.evaluator.evaluate(best_eval_model,
                    test_data, eval_consistency=True)
        log_info = 'Final acc = {} discri = {} consist = {}'.format(
                    acc, discri, consist)
        logger.info(log_info)
        return best_model


if __name__ == '__main__':
    options = {'data_fn': './data/adult/adult.npz',
               'knns_fn': './data/adult/knns.txt',
               'batch_size': 32,
               'cuda': False,
               'arch_dim': 20,
               'hidden_dim': 10,
               'lr': 1e-3,
               'k': 5,
               'max_epoch': 2,
               'val_freq': 10,
               'model_selection_cri': 'discri',
               'alpha': 1.0,
               'beta': 0.0,
               'gamma': 1.0,}

    log_fn = 'alfr_test.log'
    logger = get_logger(log_fn)
    trainer = Trainer(options)
    trainer.train(logger)
