""" This code is shared for review purposes only. Do not copy, reproduce, share,
publish, or use for any purpose except to review our ICML submission. Please
delete after the review process. The authors plan to publish the code
deanonymized and with a proper license upon publication of the paper. """

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import numpy as np


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__
    

class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader
    
    
from torch.utils.data import Dataset


class LabelledImgDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx, ...]
        label = self.targets[idx]
        return image, label, idx
    
    
class ProductDataloader(TorchvisionDataset):
    def __init__(self, tr, ad_te):
        super().__init__(root='./data')

        self.train_set = tr
        self.test_set = ad_te
    

class LoadData:
    def __init__(self, x_normal, x_abnormal):
        
        # for __call__ usage
        self.tr = None
        self.ad_te = None

        t_normal = np.zeros(x_normal.shape[0])
        tr = LabelledImgDataset(x_normal.astype(np.float32), t_normal)

        _ad_te = np.concatenate([x_normal.astype(np.float32), x_abnormal.astype(np.float32)], axis=0)
        t_ad_te = np.concatenate([t_normal, np.ones(x_abnormal.shape[0])])
        ad_te = LabelledImgDataset(_ad_te, t_ad_te)

        self.tr = tr
        self.ad_te = ad_te

    def __call__(self):
        product_dataloader = ProductDataloader(self.tr, self.ad_te)

        return product_dataloader
    
    
class LoadContaminatedData:
    def __init__(self, x_normal, x_abnormal):
        
        # for __call__ usage
        self.tr = None
        self.ad_te = None

        x = np.concatenate([x_normal.astype(np.float32), x_abnormal.astype(np.float32)], axis=0)
        t_contam = np.zeros(x_normal.shape[0] + x_abnormal.shape[0])
        t_contam[x_normal.shape[0]:] = 1
        tr = LabelledImgDataset(x, t_contam)

        _ad_te = np.concatenate([x_normal.astype(np.float32), x_abnormal.astype(np.float32)], axis=0)
        t_ad_te = np.concatenate([np.zeros(x_normal.shape[0]), np.ones(x_abnormal.shape[0])])
        ad_te = LabelledImgDataset(_ad_te, t_ad_te)

        self.tr = tr
        self.ad_te = ad_te

    def __call__(self):
        product_dataloader = ProductDataloader(self.tr, self.ad_te)

        return product_dataloader