""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

import random
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
from deep_svdd import DeepSVDD
from dataloader import LoadContaminatedData
from model import RBFNetManualCentroid
from trainer import DeepSVDDTrainer

def run(oe=True, oe_loss='refine'):
    # generate data
    rng = np.random.RandomState(seed=1234)

    # normal
    xs = []

    # true class: binary Gaussian mixture
    data_size = 90

    comp1_prob = 1.0

    mean_1 = [1, 1]
    cov_1 = np.asarray([[1,0],[0,1]]) * 0.07
    mean_2 = [-1, 1]
    cov_2 = np.asarray([[1,0],[0,1]]) * 0.05

    xs = []
    sample_id = rng.binomial(1, comp1_prob, data_size)
    for i in sample_id:
        if i:
            x = rng.multivariate_normal(mean=mean_1, cov=cov_1)
        else:
            x = rng.multivariate_normal(mean=mean_2, cov=cov_2)
        xs.append(x)
        
    xs = np.asarray(xs)

    # abnormal
    abnormal_size = 5
    mean = [-0.25, 2.5]
    cov = [[0.03, 0.0],[0.0, 0.03]]
    o1 = rng.multivariate_normal(mean=mean, cov=cov, size=abnormal_size)

    mean = [-1., 0.5]
    cov = [[0.03, 0.0],[0.0, 0.03]]
    o2 = rng.multivariate_normal(mean=mean, cov=cov, size=abnormal_size)

    o = np.concatenate([o1, o2], axis=0)

    trans_o = o
    trans_xs = xs

    # train
    settings = {}
    settings['optimizer_name'] = 'adam'
    settings['lr'] = 1e-2
    settings['n_epochs'] = 200
    settings['lr_milestone'] = [100]
    settings['batch_size'] = 25
    settings['weight_decay'] = 1e-1
    settings['n_jobs_dataloader'] = 1
    settings['nu'] = 0.1

    settings['oe'] = oe
    settings['oe_loss'] = oe_loss

    # FullyConn, FullyConnManualFeature, LeNet1D, RBFNet
    settings['net_name'] = 'RBFNetManualCentroid'
    net_dict = {
        'RBFNetManualCentroid': RBFNetManualCentroid,
    }

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Print configuration
    logger.info('Deep SVDD objective: %s' % 'one-class')

    # Set seed
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    logger.info('Set seed to %d.' % 1234)

    # Default device to 'cpu' if cuda is not available
    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)

    n_jobs_dataloader = settings['n_jobs_dataloader']
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataloader = LoadContaminatedData(np.expand_dims(trans_xs, axis=1), np.expand_dims(trans_o, axis=1))
    dataset = dataloader()

    # # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD('one-class', settings['nu'])
    deep_SVDD.set_network(net_dict[settings['net_name']], device=device)

    # Log training details
    logger.info('Training optimizer: %s' % settings['optimizer_name'])
    logger.info('Training learning rate: %g' % settings['lr'])
    logger.info('Training epochs: %d' % settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (settings['lr_milestone'],))
    logger.info('Training batch size: %d' % settings['batch_size'])
    logger.info('Training weight decay: %g' % settings['weight_decay'])

    # Train model on dataset and re-use deep_SVDD at each stage
    deep_SVDD.train(dataset,
                    optimizer_name=settings['optimizer_name'],
                    lr=settings['lr'],
                    n_epochs=settings['n_epochs'],
                    lr_milestones=settings['lr_milestone'],
                    batch_size=settings['batch_size'],
                    weight_decay=settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    oe=settings['oe'],
                    oe_loss=settings['oe_loss'])

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    import matplotlib.cm as cm

    # title map
    titles = {
        'known_gt': 'G-truth', 
        'weighted': 'LOE$_S$ (ours)',
        'radical': 'LOE$_H$ (ours)',
        'refine': 'Refine',
        'none': 'Blind'
    }

    # plot anomaly score contours
    nx, ny = 100, 100
    mesh_x = np.linspace(-2, 2, nx)
    mesh_y = np.linspace(-0.5, 3.5, ny)
    xv, yv = np.meshgrid(mesh_x, mesh_y)
    xy = np.concatenate([np.expand_dims(xv, -1), np.expand_dims(yv, -1)], axis=-1)
    xy = xy.astype(np.float32)

    res = np.zeros(xv.shape)

    with torch.no_grad():
        trans_xy = xy
        c_pred = deep_SVDD.net(torch.tensor(trans_xy).float().view(-1, 1, trans_xy.shape[-1]).to(device))
        loss = (c_pred - deep_SVDD.trainer.c).square().sum(-1).sqrt()
        loss = loss.view(100, 100).to('cpu').numpy()

    hf = plt.contourf(mesh_x, mesh_y, loss, levels=10, vmin=0., vmax=2.25)
    h = plt.contour(mesh_x, mesh_y, loss, levels=10, colors='gray', vmin=0., vmax=2.25)
    plt.clabel(h, h.levels[::2], inline=True, fontsize=12, colors='white')

    plt.axis('off')
    plt.colorbar(hf)
    plt.axis('scaled')

    plt.scatter(xs[:,0], xs[:,1], color='C0', edgecolor='silver', label='Normality')
    plt.scatter(o[:,0], o[:,1], color='C1', edgecolor='silver', label='Anomaly')

    plt.title(titles[settings['oe_loss']])
    plt.legend(loc='upper right')

    if settings['oe_loss'] == 'refine':
        plt.savefig('./toy_example_RBFNet_Refine.pdf', dpi=96, bbox_inches='tight')
    elif settings['oe_loss'] == 'known_gt':
        plt.savefig('./toy_example_RBFNet_latentOE_KnownGT.pdf', dpi=96, bbox_inches='tight')
    elif settings['oe_loss'] == 'radical':
        plt.savefig('./toy_example_RBFNet_latentOE_Radical.pdf', dpi=96, bbox_inches='tight')
    elif settings['oe_loss'] == 'weighted':
        plt.savefig('./toy_example_RBFNet_latentOE_Weighted.pdf', dpi=96, bbox_inches='tight')
    elif settings['oe_loss'] == 'none':
        plt.savefig('./toy_example_RBFNet_Blind.pdf', dpi=96, bbox_inches='tight')

    plt.clf()


if __name__ == '__main__':

    # Blind
    run(oe=False, oe_loss='none')

    # Refine
    run(oe=True, oe_loss='refine')

    # LOE-S
    run(oe=True, oe_loss='weighted')

    # LOE-H
    run(oe=True, oe_loss='radical')

    # G-truth
    run(oe=True, oe_loss='known_gt')