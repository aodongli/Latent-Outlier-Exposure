""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

from abc import ABC, abstractmethod
from model import BaseNet
from dataloader import BaseADDataset

class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset, net: BaseNet) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, dataset, net: BaseNet):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass

from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0,
                 oe=False,
                 oe_loss='weighted'):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Outlier exposure
        self.oe = oe
        self.oe_loss = oe_loss

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Set optimizer (Adam optimizer for now)
        params = list(net.parameters())
        params.append(self.c)
        optimizer = optim.Adam(params, lr=self.lr, 
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        train_size = dataset.train_set.__len__()
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            reg_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, gt_labels, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                
                pos_loss = dist
                if self.oe:
                    loss_accept, idx_accept = torch.topk(
                        dist, int(dist.shape[0] * 0.9), largest=False, sorted=False)
                    loss_reject, idx_reject = torch.topk(
                        dist, int(dist.shape[0] * 0.1), largest=True, sorted=False)
                    
                    neg_loss = 1. / torch.sum((outputs - self.c) ** 2, dim=1)
                    
                    if self.oe_loss == 'weighted':
                        _loss = torch.cat(
                            [pos_loss[idx_accept],0.5*pos_loss[idx_reject]+0.5*neg_loss[idx_reject]], 0)
                    elif self.oe_loss == 'radical':
                        _loss = torch.cat([pos_loss[idx_accept], neg_loss[idx_reject]], 0)
                    elif self.oe_loss == 'refine':
                        _loss = pos_loss[idx_accept]
                    elif self.oe_loss == 'known_gt':
                        _loss = torch.cat([pos_loss[gt_labels==0], neg_loss[gt_labels==1]],0)
                    else:
                        raise NotImplementedError
            
                else:
                    _loss = pos_loss
                
                if self.objective == 'soft-boundary':
                    raise NotImplementedError
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(_loss)

                # add L2 regularization
                l2_reg = 0
                for w in net.parameters():
                    l2_reg += (self.weight_decay * (w**2)).sum()


                # have the same scale with the likelihood
                l2_reg /= train_size

                total_loss = loss + l2_reg
                
                total_loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                reg_epoch += l2_reg.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}   Time: {:.3f}    Loss: {:.8f}    Reg: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches, reg_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def compute_obs_var(self, dataset, net: BaseNet):
        '''Use covariance matrix as a metric matrix. 

        We assume covariance matrix is diagonal.

        Previously, it assumes identity covariance. 
        We now weight each dimension by its estimated variance.

        1/n * (phi(x) - mu)**2
        '''
        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        print('Center:', self.c)

        # estimate variance
        # assume each dimension is independent
        net.eval()
        with torch.no_grad():
            sample_var = 0.
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                outputs = net(inputs)
                sample_var += torch.sum((outputs - self.c) ** 2, 0)

            train_size = dataset.train_set.__len__()
            sample_var /= train_size

        return sample_var


    def test(self, dataset, net: BaseNet, ellipsoid=False):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            if ellipsoid:
                sample_var = self.compute_obs_var(dataset, net)
                print(sample_var.size(), sample_var)

            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)

                if ellipsoid:
                    dist = torch.sum(((outputs - self.c) ** 2) / sample_var, dim=1)
                else:
                    dist = torch.sum(((outputs - self.c) ** 2), dim=1)
                    
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        c = nn.Parameter(torch.Tensor(1).to(self.device))
        nn.init.normal_(c, 0, 1)
        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
