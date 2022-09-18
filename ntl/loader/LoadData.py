""" This code is shared for review purposes only. Do not copy, reproduce, share, publish,
or use for any purpose except to review our ICML submission. Please delete after the review process.
The authors plan to publish the code deanonymized and with a proper license upon publication of the paper. """

from scipy import io
from .utils import *
import torch
import random
import pandas as pd
import scipy.io
from scipy.io import arff
import sys
import zipfile
import os
import torchvision.datasets as dset
import numpy as np

def synthetic_contamination(norm,anorm,contamination_rate):
    num_clean = norm.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    try:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=False)
    except:
        idx_contamination = np.random.choice(np.arange(anorm.shape[0]),num_contamination,replace=True)
    train_contamination = anorm[idx_contamination]
    train_contamination = train_contamination + np.random.randn(*train_contamination.shape)*np.std(anorm,0,keepdims=True)
    train_data = np.concatenate([norm,train_contamination],0)
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1
    return train_data,train_labels
    
def Load_Tabular(dataset_name,contamination_rate=0.0,root='DATA/Tabular/'):
    abs_file_path = root+dataset_name
    mat_files=['annthyroid','breastw','cardio','forest_cover','glass','ionosphere','letter','lympho','mammography','mnist','musk',
               'optdigits','pendigits','pima','satellite','satimage','shuttle','speech','vertebral','vowels','wbc','wine']
    if dataset_name in mat_files:
        print ('generic mat file')
        return build_train_test_generic_matfile(abs_file_path,contamination_rate)

    if dataset_name == 'seismic':
        print('seismic')
        return build_train_test_seismic(abs_file_path+'.arff',contamination_rate)

    if dataset_name == 'mulcross':
        print('mullcross')
        return build_train_test_mulcross(abs_file_path+'.arff',contamination_rate)

    if dataset_name == 'abalone':
        print('abalone')
        return build_train_test_abalone(abs_file_path+'.data',contamination_rate)

    if dataset_name == 'ecoli':
        print('ecoli')
        return build_train_test_ecoli(abs_file_path+'.data',contamination_rate)

    if dataset_name == 'kdd':
        print ('kdd')
        return build_train_test_kdd(root+'/kddcup.data_10_percent_corrected.zip',contamination_rate)

    if dataset_name == 'kddrev':
        print ('kddrev')
        return build_train_test_kdd_rev(root+'/kddcup.data_10_percent_corrected.zip',contamination_rate)

    sys.exit ('No such dataset!')

def build_train_test_generic_matfile(name_of_file,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
    dataset = scipy.io.loadmat(name_of_file)
    X = dataset['X']
    classes = dataset['y']
    dim = X.shape[1]
    #jointXY = torch.cat((torch.tensor(X,dtype=torch.double), torch.tensor(classes,dtype=torch.double)), dim=1)
    jointXY = np.concatenate((X, classes,), 1)
    normals=jointXY[jointXY[:,-1]==0]
    anomalies=jointXY[jointXY[:,-1]==1]
    #normals = normals[torch.randperm(normals.shape[0])]
    # rng= np.random.RandomState(123)
    normals = normals[np.random.permutation(normals.shape[0])]
    train = normals[:int(normals.shape[0] / 2) + 1]
    test_norm = normals[int(normals.shape[0] / 2) + 1:]

    test = np.concatenate((test_norm, anomalies),0)
    test_labels = test[:, -1]
    test = test[:, :dim]
    if contamination_rate>0:
        train,train_labels = synthetic_contamination(train[:, :dim],anomalies[:, :dim],contamination_rate)
    else:
        train = train[:, :dim]
        train_labels = np.zeros(train.shape[0])

    return train, train_labels,test, test_labels


def build_train_test_seismic(name_of_file,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
    dataset, meta = arff.loadarff(name_of_file)
    dataset = pd.DataFrame(dataset)
    classes = dataset.iloc[:, -1]
    dataset = dataset.iloc[:, :-1]
    dataset = pd.get_dummies(dataset.iloc[:, :-1])
    dataset = pd.concat((dataset, classes), axis=1)
    normals = dataset[dataset.iloc[:, -1] == b'0'].values
    anomalies = dataset[dataset.iloc[:, -1] == b'1'].values
    normals = normals[np.random.permutation(normals.shape[0])]
    normals = normals[:, :-1].astype('float32')
    anomalies = anomalies[:, :-1].astype('float32')
    train = normals[:int(normals.shape[0] / 2) + 1]
    test_norm = normals[int(normals.shape[0] / 2) + 1:]
    test = np.concatenate((test_norm, anomalies),0)

    test_labels = np.zeros(test.shape[0])
    test_labels[test_norm.shape[0]:]=1
    if contamination_rate>0:
        train,train_labels = synthetic_contamination(train,anomalies,contamination_rate)
    else:
        train_labels = np.zeros(train.shape[0])
    return train, train_labels,test, test_labels

def build_train_test_mulcross(name_of_file,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
    dataset, _ = arff.loadarff(name_of_file)
    dataset = pd.DataFrame(dataset)
    normals = dataset[dataset.iloc[:, -1] == b'Normal'].values
    anomalies = dataset[dataset.iloc[:, -1] == b'Anomaly'].values
    normals = normals[np.random.permutation(normals.shape[0])]
    normals = normals[:, :-1].astype('float32')
    anomalies = anomalies[:, :-1].astype('float32')
    train = normals[:int(normals.shape[0] / 2) + 1]
    test_norm = normals[int(normals.shape[0] / 2) + 1:]
    test = np.concatenate((test_norm, anomalies),0)

    test_labels = np.zeros(test.shape[0])
    test_labels[test_norm.shape[0]:]=1
    if contamination_rate>0:
        train,train_labels = synthetic_contamination(train,anomalies,contamination_rate)
    else:
        train_labels = np.zeros(train.shape[0])

    return train, train_labels,test, test_labels

def build_train_test_ecoli(name_of_file,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
    dataset = pd.read_csv(name_of_file, header=None, sep='\s+')
    dataset = dataset.iloc[:, 1:]
    anomalies = np.array(
        dataset[(dataset.iloc[:, 7] == 'omL') | (dataset.iloc[:, 7] == 'imL') | (dataset.iloc[:, 7] == 'imS')])[:,
                :-1]
    normals = np.array(dataset[(dataset.iloc[:, 7] == 'cp') | (dataset.iloc[:, 7] == 'im') | (
                dataset.iloc[:, 7] == 'pp') | (dataset.iloc[:, 7] == 'imU') | (dataset.iloc[:, 7] == 'om')])[:, :-1]
    normals = normals.astype('double')
    anomalies = anomalies.astype('double')

    normals = normals[np.random.permutation(normals.shape[0])]
    train = normals[:int(normals.shape[0] / 2) + 1]
    test_norm = normals[int(normals.shape[0] / 2) + 1:]
    test = np.concatenate((test_norm, anomalies),0)

    test_labels = np.zeros(test.shape[0])
    test_labels[test_norm.shape[0]:]=1
    if contamination_rate>0:
        train,train_labels = synthetic_contamination(train,anomalies,contamination_rate)
    else:
        train_labels = np.zeros(train.shape[0])
    return train, train_labels,test, test_labels

def build_train_test_abalone(path,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test

    data = pd.read_csv(path, header=None, sep=',')
    data = data.rename(columns={8: 'y'})
    data['y'].replace([8, 9, 10], -1, inplace=True)
    data['y'].replace([3, 21], 0, inplace=True)
    data.iloc[:, 0].replace('M', 0, inplace=True)
    data.iloc[:, 0].replace('F', 1, inplace=True)
    data.iloc[:, 0].replace('I', 2, inplace=True)
    test = data[data['y'] == 0]
    normal = data[data['y'] == -1].sample(frac=1)
    num_normal_samples_test = normal.shape[0] // 2
    test_data = np.concatenate((test.drop('y', axis=1), normal[:num_normal_samples_test].drop('y', axis=1)), axis=0)
    train = normal[num_normal_samples_test:]
    train_data = train.drop('y', axis=1).values
    test_labels = np.concatenate((test['y'], normal[:num_normal_samples_test]['y'].replace(-1, 1)), axis=0)
    for i in range(test_labels.shape[0]):
        if test_labels[i] == 0:
            test_labels[i] = 1
        else:
            test_labels[i] = 0
    if contamination_rate>0:
        train_data,train_labels = synthetic_contamination(train_data,test_data[test_labels==1],contamination_rate)
    else:
        train_labels = np.zeros(train_data.shape[0])
    return train_data, train_labels,test_data, test_labels

def build_train_test_kdd(name_of_file,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
    zf = zipfile.ZipFile(name_of_file)
    kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
    entire_set = np.array(kdd_loader)
    revised_pd = pd.DataFrame(entire_set)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix='new1')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix='new2')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix='new3')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix='new6')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix='new11')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix='new21')), axis=1)
    revised_pd.drop(revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1)
    new_columns = [0, 'new1_icmp', 'new1_tcp', 'new1_udp', 'new2_IRC', 'new2_X11', 'new2_Z39_50', 'new2_auth',
                   'new2_bgp',
                   'new2_courier', 'new2_csnet_ns', 'new2_ctf', 'new2_daytime', 'new2_discard', 'new2_domain',
                   'new2_domain_u', 'new2_echo', 'new2_eco_i', 'new2_ecr_i', 'new2_efs', 'new2_exec', 'new2_finger',
                   'new2_ftp', 'new2_ftp_data', 'new2_gopher', 'new2_hostnames', 'new2_http', 'new2_http_443',
                   'new2_imap4',
                   'new2_iso_tsap', 'new2_klogin', 'new2_kshell', 'new2_ldap', 'new2_link', 'new2_login', 'new2_mtp',
                   'new2_name', 'new2_netbios_dgm', 'new2_netbios_ns', 'new2_netbios_ssn', 'new2_netstat', 'new2_nnsp',
                   'new2_nntp', 'new2_ntp_u', 'new2_other', 'new2_pm_dump', 'new2_pop_2', 'new2_pop_3', 'new2_printer',
                   'new2_private', 'new2_red_i', 'new2_remote_job', 'new2_rje', 'new2_shell', 'new2_smtp',
                   'new2_sql_net',
                   'new2_ssh', 'new2_sunrpc', 'new2_supdup', 'new2_systat', 'new2_telnet', 'new2_tftp_u', 'new2_tim_i',
                   'new2_time', 'new2_urh_i', 'new2_urp_i', 'new2_uucp', 'new2_uucp_path', 'new2_vmnet', 'new2_whois',
                   'new3_OTH', 'new3_REJ', 'new3_RSTO', 'new3_RSTOS0', 'new3_RSTR', 'new3_S0', 'new3_S1', 'new3_S2',
                   'new3_S3', 'new3_SF', 'new3_SH', 4, 5, 'new6_0', 'new6_1', 7, 8, 9, 10, 'new11_0', 'new11_1', 12, 13,
                   14,
                   15, 16, 17, 18, 19, 'new21_0', 'new21_1', 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                   35, 36, 37, 38, 39, 40, 41]
    revised_pd = revised_pd.reindex(columns=new_columns)
    revised_pd.loc[revised_pd[41] != 'normal.', 41] = 0.0
    revised_pd.loc[revised_pd[41] == 'normal.', 41] = 1.0
    kdd_normal = np.array(revised_pd.loc[revised_pd[41] == 0.0], dtype=np.double)
    kdd_anomaly = np.array(revised_pd.loc[revised_pd[41] == 1.0], dtype=np.double)
    # kdd_normal = torch.tensor(kdd_normal)
    # kdd_anomaly = torch.tensor(kdd_anomaly)
    kdd_normal = kdd_normal[
        np.random.permutation(kdd_normal.shape[0])]
    # kdd_anomaly = kdd_anomaly[torch.randperm(kdd_anomaly.shape[0])]
    train = kdd_normal[:int(kdd_normal.shape[0] / 2) + 1]
    test_norm = kdd_normal[int(kdd_normal.shape[0] / 2) + 1:]

    test = np.concatenate((test_norm, kdd_anomaly),0)

    dim = test.shape[1] - 1
    test_labels = test[:, -1]
    test = test[:, :dim]
    if contamination_rate>0:
        train,train_labels = synthetic_contamination(train[:, :dim],kdd_anomaly[:, :dim],contamination_rate)
    else:
        train = train[:, :dim]
        train_labels = np.zeros(train.shape[0])
    return train, train_labels,test, test_labels


def build_train_test_kdd_rev(name_of_file,contamination_rate):  # takes a matrice that contains the entire dataset, and creates a trainset with 50% of the data of all normals, and the rest are test
    zf = zipfile.ZipFile(name_of_file)
    kdd_loader = pd.read_csv(zf.open('kddcup.data_10_percent_corrected'), delimiter=',')
    entire_set = np.array(kdd_loader)
    revised_pd = pd.DataFrame(entire_set)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 1], prefix='new1')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 2], prefix='new2')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 3], prefix='new3')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 6], prefix='new6')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 11], prefix='new11')), axis=1)
    revised_pd = pd.concat((revised_pd, pd.get_dummies(revised_pd.iloc[:, 21], prefix='new21')), axis=1)
    revised_pd.drop(revised_pd.columns[[1, 2, 3, 6, 11, 20, 21]], inplace=True, axis=1)
    new_columns = [0, 'new1_icmp', 'new1_tcp', 'new1_udp', 'new2_IRC', 'new2_X11', 'new2_Z39_50', 'new2_auth',
                   'new2_bgp',
                   'new2_courier', 'new2_csnet_ns', 'new2_ctf', 'new2_daytime', 'new2_discard', 'new2_domain',
                   'new2_domain_u', 'new2_echo', 'new2_eco_i', 'new2_ecr_i', 'new2_efs', 'new2_exec', 'new2_finger',
                   'new2_ftp', 'new2_ftp_data', 'new2_gopher', 'new2_hostnames', 'new2_http', 'new2_http_443',
                   'new2_imap4',
                   'new2_iso_tsap', 'new2_klogin', 'new2_kshell', 'new2_ldap', 'new2_link', 'new2_login', 'new2_mtp',
                   'new2_name', 'new2_netbios_dgm', 'new2_netbios_ns', 'new2_netbios_ssn', 'new2_netstat', 'new2_nnsp',
                   'new2_nntp', 'new2_ntp_u', 'new2_other', 'new2_pm_dump', 'new2_pop_2', 'new2_pop_3', 'new2_printer',
                   'new2_private', 'new2_red_i', 'new2_remote_job', 'new2_rje', 'new2_shell', 'new2_smtp',
                   'new2_sql_net',
                   'new2_ssh', 'new2_sunrpc', 'new2_supdup', 'new2_systat', 'new2_telnet', 'new2_tftp_u', 'new2_tim_i',
                   'new2_time', 'new2_urh_i', 'new2_urp_i', 'new2_uucp', 'new2_uucp_path', 'new2_vmnet', 'new2_whois',
                   'new3_OTH', 'new3_REJ', 'new3_RSTO', 'new3_RSTOS0', 'new3_RSTR', 'new3_S0', 'new3_S1', 'new3_S2',
                   'new3_S3', 'new3_SF', 'new3_SH', 4, 5, 'new6_0', 'new6_1', 7, 8, 9, 10, 'new11_0', 'new11_1', 12, 13,
                   14,
                   15, 16, 17, 18, 19, 'new21_0', 'new21_1', 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                   35, 36, 37, 38, 39, 40, 41]
    revised_pd = revised_pd.reindex(columns=new_columns)

    revised_pd.loc[revised_pd[41] != 'normal.', 41] = 1.0
    revised_pd.loc[revised_pd[41] == 'normal.', 41] = 0.0
    kdd_normal = np.array(revised_pd.loc[revised_pd[41] == 0.0], dtype=np.double)
    kdd_anomaly = np.array(revised_pd.loc[revised_pd[41] == 1.0], dtype=np.double)

    kdd_anomaly = kdd_anomaly[
        np.random.permutation(kdd_anomaly.shape[0])]
    test_anomaly = kdd_anomaly[:int(kdd_normal.shape[0] / 4)]
    rest_anomaly = kdd_anomaly[int(kdd_normal.shape[0] / 4):]

    kdd_normal = kdd_normal[
        np.random.permutation(kdd_normal.shape[0])]
    train = kdd_normal[:int(kdd_normal.shape[0] / 2) + 1]
    test_norm = kdd_normal[int(kdd_normal.shape[0] / 2) + 1:]

    test = np.concatenate((test_norm, test_anomaly),0)

    dim = test.shape[1] - 1
    test_labels = test[:, -1]
    test = test[:, :dim]

    if contamination_rate>0:
        train,train_labels = synthetic_contamination(train[:, :dim],test_anomaly[:, :dim],contamination_rate)
    else:
        train = train[:, :dim]
        train_labels = np.zeros(train.shape[0])
    return train, train_labels,test, test_labels
    
def CIFAR10_feat(normal_class,root='DATA/cifar10_features/',contamination_rate=0.0):
    trainset = torch.load(root+'trainset_2048.pt')
    train_data,train_targets = trainset
    testset = torch.load(root+'testset_2048.pt')
    test_data,test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    
    train_clean = train_data[train_targets==normal_class]
    train_contamination = train_data[train_targets!=normal_class]
    num_clean = train_clean.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]    
    train_data = torch.cat((train_clean,train_contamination),0)
        
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1

    return train_data,train_labels,test_data,test_labels

def FMNIST_feat(normal_class,root='DATA/FashionMNIST/',contamination_rate=0.0):
    trainset = torch.load(root+'trainset_2048.pt')
    train_data,train_targets = trainset
    testset = torch.load(root+'testset_2048.pt')
    test_data,test_targets = testset
    test_labels = np.ones_like(test_targets)
    test_labels[test_targets==normal_class]=0
    
    train_clean = train_data[train_targets==normal_class]
    train_contamination = train_data[train_targets!=normal_class]
    num_clean = train_clean.shape[0]
    num_contamination = int(contamination_rate/(1-contamination_rate)*num_clean)

    idx_contamination = np.random.choice(np.arange(train_contamination.shape[0]),num_contamination,replace=False)
    train_contamination = train_contamination[idx_contamination]    
    train_data = torch.cat((train_clean,train_contamination),0)
        
    train_labels = np.zeros(train_data.shape[0])
    train_labels[num_clean:]=1

    return train_data,train_labels,test_data,test_labels



def load_data(data_name,cls,contamination_rate=0.0):
    tabular_dataset = ['annthyroid','breastw','cardio','glass','ionosphere','letter','lympho','mammography',
                       'mnist','musk','optdigits','pendigits','pima','satellite','satimage','shuttle',
                       'speech','vertebral','vowels','wbc','wine','ecoli','abalone','seismic','mulcross','forest_cover','kddrev','kdd']
    ## normal data with label 0, anomalies with label 1
    if data_name in tabular_dataset:
        train, train_label, test, test_label = Load_Tabular(data_name,contamination_rate=contamination_rate)
    elif data_name == 'thyroid':
        train, train_label, test, test_label = Thyroid_train_valid_data(contamination_rate)
    elif data_name == 'arrhythmia':
        train, train_label, test, test_label = Arrhythmia_train_valid_data(contamination_rate)
    elif data_name == 'cifar10_feat':
        train, train_label, test, test_label = CIFAR10_feat(cls,contamination_rate=contamination_rate)
    elif data_name == 'fmnist_feat':
        train, train_label, test, test_label = FMNIST_feat(cls,contamination_rate=contamination_rate)
    # elif data_name == 'mvtec_feat':
    #     train, train_label, test, test_label = MVTEC_feat(cls,contamination_rate=contamination_rate)

    trainset = CustomDataset(train,train_label)
    testset = CustomDataset(test,test_label)
    return trainset,testset,testset

def Thyroid_train_valid_data(contamination_rate):
    data = io.loadmat("DATA/thyroid.mat")
    samples = data['X']  # 3772
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    norm_samples = samples[labels == 0]  # 3679 norm
    anom_samples = samples[labels == 1]  # 93 anom

    n_train = len(norm_samples) // 2
    train = norm_samples[:n_train]  # 1839 train
    # train_label = np.zeros(train.shape[0])
    val_real = norm_samples[n_train:]
    val_fake = anom_samples
    val = np.concatenate([val_real,val_fake],0)
    # train,val = norm_data(train,val)
    val_label = np.zeros(val.shape[0])
    val_label[val_real.shape[0]:]=1
    train,train_label = synthetic_contamination(train,anom_samples,contamination_rate)
    return train,train_label,val,val_label

def Arrhythmia_train_valid_data(contamination_rate):
    data = io.loadmat("DATA/arrhythmia.mat")
    samples = data['X']  # 518
    labels = ((data['y']).astype(np.int32)).reshape(-1)

    norm_samples = samples[labels == 0]  # 452 norm
    anom_samples = samples[labels == 1]  # 66 anom

    n_train = len(norm_samples) // 2
    train = norm_samples[:n_train]  # 226 train
    # train_label = np.zeros(train.shape[0])
    val_real = norm_samples[n_train:]
    val_fake = anom_samples
    val = np.concatenate([val_real,val_fake],0)
    # train,val = norm_data(train,val)
    val_label = np.zeros(val.shape[0])
    val_label[val_real.shape[0]:]=1
    train, train_label = synthetic_contamination(train, anom_samples, contamination_rate)
    return train,train_label,val,val_label



