""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

import scipy.io
import numpy as np
import pandas as pd
import torchvision.datasets as dset
import os

class Data_Loader:

    def __init__(self, n_trains=None):
        self.n_train = n_trains
        self.rng = np.random.RandomState(123)

    def norm(self, data, mu=1):
        return 2 * (data / 255.) - mu

    def get_dataset(self, dataset_name, c_percent=None, true_label=1):
        if dataset_name == 'cifar10':
            return self.load_data_CIFAR10(true_label, c_percent)
        if dataset_name == 'fmnist':
            return self.load_data_FMNIST(true_label, c_percent)


    def load_data_CIFAR10(self, true_label, c_percent):
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        trainset = dset.CIFAR10(root, train=True, download=True)
        train_data = trainset.data
        train_labels = np.asarray(trainset.targets)

        testset = dset.CIFAR10(root, train=False, download=True)
        test_data = testset.data
        test_labels = np.asarray(testset.targets)

        tr_id_data = train_data[np.where(train_labels == true_label)]
        tr_ood_data = train_data[np.where(train_labels != true_label)]
        te_id_data = test_data[np.where(test_labels == true_label)]
        te_ood_data = test_data[np.where(test_labels != true_label)]

        tr_x, tr_y = self.contaminate_images(tr_id_data, 
                                             tr_ood_data, 
                                             c_percent)

        te_ood_x, _ = self.subsample(te_ood_data, len(te_id_data))
        te_x = np.concatenate([te_id_data, te_ood_x], 0)
        te_y = np.zeros(len(te_x))
        te_y[len(te_id_data):] = 1
        
        tr_x = self.norm(np.asarray(tr_x, dtype='float32'))
        te_x = self.norm(np.asarray(te_x, dtype='float32'))
        return tr_x, tr_y, te_x, te_y


    def load_data_FMNIST(self, true_label, c_percent):
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        trainset = dset.FashionMNIST(root, train=True, download=True)
        train_data = np.expand_dims(trainset.data, axis=-1)
        train_labels = trainset.targets

        testset = dset.FashionMNIST(root, train=False, download=True)
        test_data = np.expand_dims(testset.data, axis=-1)
        test_labels = testset.targets

        tr_id_data = train_data[np.where(train_labels == true_label)]
        tr_ood_data = train_data[np.where(train_labels != true_label)]
        te_id_data = test_data[np.where(test_labels == true_label)]
        te_ood_data = test_data[np.where(test_labels != true_label)]

        print(f'tr_id_data: {len(tr_id_data)}\n',
              f'tr_ood_data: {len(tr_ood_data)}\n'
              f'te_id_data: {len(te_id_data)}\n'
              f'te_ood_data: {len(te_ood_data)}\n')

        tr_x, tr_y = self.contaminate_images(tr_id_data, 
                                             tr_ood_data, 
                                             c_percent)

        # te_ood_x = te_ood_data
        te_ood_x, _ = self.subsample(te_ood_data, len(te_id_data))
        te_x = np.concatenate([te_id_data, te_ood_x], 0)
        te_y = np.zeros(len(te_x))
        te_y[len(te_id_data):] = 1
        
        tr_x = self.norm(np.asarray(tr_x, dtype='float32'))
        te_x = self.norm(np.asarray(te_x, dtype='float32'))
        return tr_x, tr_y, te_x, te_y


    def subsample(self, data, subsample_size, label=None):
        ori_size = len(data)

        if subsample_size >= ori_size:
            return data, label
        
        sample_idx = self.rng.choice(ori_size, subsample_size, replace=False)

        data = np.asarray(data)[sample_idx]
        if label is not None:
            label = np.asarray(label)[sample_idx]

        return data, label


    def _contaminate_images(self, in_x, ood_x, c_percent):
        # replacement
        replace_num = int(len(in_x) * c_percent)

        # select random in and ood samples
        in_idx = self.rng.choice(len(in_x), replace_num, replace=False)
        ood_idx = self.rng.choice(len(ood_x), replace_num, replace=False)

        # update data
        in_x[in_idx] = ood_x[ood_idx]

        # update targets
        labels = np.zeros(len(in_x))
        labels[in_idx] = 1

        return in_x, labels


    def contaminate_images(self, in_x, ood_x, c_percent):
        # add up
        replace_num = int(len(in_x) * c_percent / (1 - c_percent))

        # select ood samples randomly
        ood_idx = self.rng.choice(len(ood_x), replace_num, replace=False)

        # update data
        in_x = np.concatenate([in_x, ood_x[ood_idx]], 0)

        # update targets
        labels = np.zeros(len(in_x))
        labels[-replace_num:] = 1

        return in_x, labels