""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from torch_rbf import basis_func_dict, RBFManualCentroid

class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
    

class RBFNetManualCentroid(BaseNet):
    def __init__(self, device='cuda'):
        super().__init__()

        self.rep_dim = 1
        
        self.layers = nn.Sequential(
          RBFManualCentroid(2, 3, basis_func_dict()['gaussian'], device=device),
        )
        
        self.fc = nn.Linear(3, self.rep_dim, bias=True)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x