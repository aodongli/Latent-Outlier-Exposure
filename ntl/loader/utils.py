""" This code is shared for review purposes only. Do not copy, reproduce, share, publish,
or use for any purpose except to review our ICML submission. Please delete after the review process.
The authors plan to publish the code deanonymized and with a proper license upon publication of the paper. """

import numpy as np
from torch.utils.data import Dataset

def norm_data(train, val):
    mus = train.mean(0)
    sds = train.std(0)
    sds[sds == 0] = 1

    def get_norm(xs, mu, sd):
        return np.array([(x - mu) / sd for x in xs])

    train = get_norm(train, mus, sds)
    val = get_norm(val, mus, sds)
#    val_fake = get_norm(val_fake, mus, sds)
    return train,val

class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        data = {"sample": sample, "label": label}
        return data

def norm(data, mu=1):
    return 2 * (data / 255.) - mu