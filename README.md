This repository provides additional PyTorch implementation of **Latent Outlier Exposure for Anomaly Detection with Contaminated Data**(<https://arxiv.org/abs/2202.08088>). See also <https://github.com/boschresearch/LatentOE-AD>.

# Toy Example

To get toy experiment (Figure 1) in the paper, please run the following command. The plots will saved to the local folder.

```
cd toy_example
python main.py
```

# NTL Experiment

To run the experiment with NTL on image data: CIFAR-10 and F-MNIST, on tabular data: Thyroid and Arrhythmia, please run the command and replace \$# with available options (see below): 

```
cd ntl
python Launch_Exps.py --config-file $1 --dataset-name $2 --true_contamination $3 --est_contamination $4 --oe_est $5
```

config-file: config_thyroid.yml; config_arrhy.yml; config_cifar10_feat.yml; config_fmnist_feat.yml

dataset-name: thyroid; arrhythmia; cifar10_feat; fmnist_feat

true_contamination: float value between 0 and 1

est_contamination: float value between 0 and 1

oe_est: blind, refine, hard, soft

# MHRot Experiment

To run the experiment with MHRot on contaminated CIFAR-10 and F-MNIST with $\alpha_0=0.1$ , please run the following command with different options. Please note the code will automatically download CIFAR-10 and F-MNIST datasets, and the data augmentations make take some time.

```
cd mhrot

## CIFAR-10 ##
# Blind
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --oe_rank=latent_gauss --foldername=cifar10/
# Refine
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --oe_rank=latent_gauss --foldername=cifar10/ --oe=True --oe_loss=refine
# Soft LOE
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --oe_rank=latent_gauss --foldername=cifar10/ --oe=True --oe_loss=weighted
# Hard LOE
python train_ad.py --dataset=cifar10 --epochs=16 --lr=1e-3 --oe_rank=latent_gauss --foldername=cifar10/ --oe=True --oe_loss=radical

## F-MNIST ##
# Blind
python train_ad.py --dataset=fmnist --epochs=3 --lr=1e-4 --oe_rank=training_obj --foldername=fmnist/
# Refine
python train_ad.py --dataset=fmnist --epochs=3 --lr=1e-4 --oe_rank=training_obj --foldername=fmnist/ --oe=True --oe_loss=refine
# Soft LOE
python train_ad.py --dataset=fmnist --epochs=3 --lr=1e-4 --oe_rank=training_obj --foldername=fmnist/ --oe=True --oe_loss=weighted
# Hard LOE
python train_ad.py --dataset=fmnist --epochs=3 --lr=1e-4 --oe_rank=training_obj --foldername=fmnist/ --oe=True --oe_loss=radical
```
