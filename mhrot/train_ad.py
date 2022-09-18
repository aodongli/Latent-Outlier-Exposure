""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

import argparse
import numpy as np
from data_loader import Data_Loader

import sys
import os
from datetime import datetime
import logging

import multitask_transformations as mt_ts
import opt_tc_multitask as mt_tc


def print_log(*args, **kwargs):
    print("[{}]".format(datetime.now()), *args, **kwargs)
    sys.stdout.flush()


def multitask_transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data, mt_trans_inds = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, mt_trans_inds

def load_multitask_trans_data(args, trans):
    foldername = f'./data/{args.dataset}/multihead/{args.type_trans}_{args.class_ind}_{args.c_percent}/'

    if os.path.exists(foldername):
        # load data
        x_train_trans = np.load(foldername + '/x_train_trans.npy')
        mt_train_labels = np.load(foldername + '/mt_train_labels.npy')
        y_train = np.load(foldername + '/y_train.npy')
        x_test_trans = np.load(foldername + '/x_test_trans.npy')
        mt_test_labels = np.load(foldername + '/mt_test_labels.npy')
        y_test = np.load(foldername + '/y_test.npy')

        print_log('Training size=%d, test size=%d' % (len(x_train_trans), len(x_test_trans)))
        print_log(f'Data dimension: {x_train_trans.shape}')
    else:
        # process data
        dl = Data_Loader()
        x_train, y_train, x_test, y_test = dl.get_dataset(args.dataset, true_label=args.class_ind, c_percent=args.c_percent)
        y_train = np.repeat(y_train, trans.n_transforms)
        print_log('Finish loading data.')
        x_train_trans, mt_train_labels = multitask_transform_data(x_train, trans)
        print_log('Finish transforming train data.')
        x_test_trans, mt_test_labels = multitask_transform_data(x_test, trans)
        print_log('Finish transforming test data.')
        print_log('Training size=%d, test size=%d' % (len(x_train), len(x_test)))
        print_log(f'Data dimension: {x_train.shape}')
        x_test_trans, x_train_trans = x_test_trans.transpose(0, 3, 1, 2), x_train_trans.transpose(0, 3, 1, 2)
        # y_test = np.array(y_test) == args.class_ind

        # save data
        os.makedirs(foldername)
        for name, arr in zip(["x_train_trans", "mt_train_labels", "y_train", "x_test_trans", "mt_test_labels", "y_test"], 
                             [x_train_trans, mt_train_labels, y_train, x_test_trans, mt_test_labels, y_test]):
            np.save(foldername + f'/{name}.npy', arr)
    return x_train_trans, mt_train_labels, y_train, x_test_trans, mt_test_labels, y_test


def train_multitask_anomaly_detector(args):

    transformer = mt_ts.get_transformer(args.type_trans)
    x_train, mt_train_labels, y_train, x_test, mt_test_labels, y_test = load_multitask_trans_data(args, transformer)
    auc = 0.5
    latent_gauss_auc = 0.5
    while auc == 0.5:
        # safeguard for unstable training
        tc_obj = mt_tc.TransClassifier(transformer.n_transforms, 
                                    transformer.n_transforms_per_task, 
                                    args)
        auc, latent_gauss_auc = tc_obj.fit_trans_classifier(x_train, mt_train_labels, y_train, x_test, mt_test_labels, y_test)
    return auc, latent_gauss_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=16, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    # Training options
    parser.add_argument('--batch_size', default=360, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=16, type=int)

    # Trans options
    parser.add_argument('--type_trans', default='medium', type=str)

    # CT options
    parser.add_argument('--lmbda', default=0.0, type=float)
    parser.add_argument('--m', default=0.1, type=float)
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    parser.add_argument('--class_ind', default=1, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str) # cifar10 fmnist
    parser.add_argument('--c_percent', default=0.1, type=float)

    parser.add_argument('--foldername', default='cifar10/', type=str) # cifar10 fmnist
    parser.add_argument('--repeat', default=5, type=int)

    # New enhancements
    parser.add_argument('--oe', default=False, type=bool)
    parser.add_argument('--oe_method', default='max_ent', type=str)
    parser.add_argument('--oe_loss', default='weighted', type=str)
    parser.add_argument('--oe_rank', default='latent_gauss', type=str)
    parser.add_argument('--oe_weight', default=0.5, type=float)

    parser.add_argument('--ad_temp', default=1.0, type=float)

    parser.add_argument('--lr_decay', default=False, type=bool)
    parser.add_argument('--sgd_opt', default=False, type=bool)

    ######### multitask softmax headers ########
    # [Hendrycks et al., ss-ood, 2019]
    parser.add_argument('--multitask', default=True, type=bool)
    # parser.add_argument('--multitask', default=False, type=bool)
    parser.add_argument('--epoch_lr_decay', default=False, type=bool)


    args = parser.parse_args()
    print(args)

    repeat_aucs = []
    repeat_aucs_latent_gauss = []
    for _repeat in range(args.repeat):

        dataset_auc = 0.
        dataset_auc_latent_gauss = 0.
        for i in range(10):            
            args._foldername = (args.foldername + '/' +
                           f'{(args.oe_method + "_" + args.oe_loss + "_" + args.oe_rank +  "_") if args.oe else ""}contaminated_ratio{args.c_percent}/' +
                           f'{_repeat}/' + 
                           f'class{i}/')
        
            os.makedirs(args._foldername,
                        exist_ok=True)

            fileh = logging.FileHandler(args._foldername + 'training.log', 'a')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fileh.setFormatter(formatter)

            log = logging.getLogger()  # root logger
            for hdlr in log.handlers[:]:  # remove all old handlers
                log.removeHandler(hdlr)
            log.addHandler(fileh)      # set the new handler
            log.setLevel(logging.INFO)

            args.class_ind = i
            if args.dataset == 'fmnist':
                args.reg = False
            print("Dataset: %s" % args.dataset)
            print("True Class:", args.class_ind)

            if args.multitask:
                _auc, _auc_latent_gauss = train_multitask_anomaly_detector(args)
                dataset_auc += _auc
                dataset_auc_latent_gauss += _auc_latent_gauss
            else:
                raise NotImplementedError

        ave_dataset_auc = dataset_auc/10
        ave_dataset_auc_latent_gauss = dataset_auc_latent_gauss/10
        print_log(f'repeat {_repeat} training objective anomaly score: ', ave_dataset_auc)
        print_log(f'repeat {_repeat} latent gauss anomaly score: ', ave_dataset_auc_latent_gauss)

        repeat_aucs.append(ave_dataset_auc)
        repeat_aucs_latent_gauss.append(ave_dataset_auc_latent_gauss)

    print_log('AUC stats:', np.mean(repeat_aucs), '+/-', np.std(repeat_aucs))
    print_log('AUC stats (latent Gauss):', np.mean(repeat_aucs_latent_gauss), '+/-', np.std(repeat_aucs_latent_gauss))
