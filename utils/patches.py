import random
import matplotlib
import numpy as np
import pylab as plt

import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

from utils.dataloader import make_weights_for_balanced_classes


def coor2patches(x: np.ndarray, coor: list, min_len=(32, 32)):
    """

    :param x: dim=(N,TS)
    :param coor: (2, N)
    :param min_len: (h, w)
    :return:
    """
    coor = list(coor)
    coor[0] -= coor[0].min()
    coor[1] -= coor[1].min()
    max_length = max(coor[0].max()+1, min_len[0]), max(coor[1].max()+1, min_len[1])
    img = np.zeros((max_length[0], max_length[1], x.shape[-1]), dtype='float32') - np.abs(x.min())
    for k in range(x.shape[-1]):
        img[coor[0], coor[1], k] = x[:, k]
    return img


def coor2patches_whole_input(input_data, set_name='train'):
    patches = []
    for x, coor in zip(input_data[f'data_{set_name}'], input_data[f'roi_coors_{set_name}']):
        patches.append(coor2patches(x, coor))
    return patches


def show_patches(patch):
    if patch.ndim > 2:
        patch = patch.max(axis=-1)

    plt.imshow(patch, cmap='gray')
    plt.show()


def normalize(data, stats=None, norm_type='robust'):
    # data dim: NCHW (C should be 7)
    # stats: len == data.shape[1]-1
    # norm_type: robust / 0mean / 01
    i = 0  # no other channels  to consider
    mask = data != 0
    stats_recorded = []

    v = data[mask != 0]
    if norm_type == '0mean':
        if stats is None:
            stats1, stats2 = v.mean(), v.std()
        else:
            stats1, stats2 = stats[i]
        data = (data - stats1) / stats2
    if norm_type == 'robust':
        if stats is None:
            stats1, stats2 = np.percentile(v, 25).mean(), np.percentile(v, 75)
        else:
            stats1, stats2 = stats[i]
        data = (data - stats1) / (stats2 - stats1)
    if norm_type == '01':
        if stats is None:
            stats1, stats2 = v.min(), v.max()
        else:
            stats1, stats2 = stats[i]
        data = (data - stats1) / (stats2 - stats1)
    stats_recorded.append((stats1, stats2))

    data *= mask  # mask the background

    if stats is None:
        return data, stats_recorded
    else:
        return data


def preprocess(input_data):
    """"""
    norm_type = '0mean'
    input_data['data_train'], stats = normalize(input_data['data_train'], stats=None, norm_type=norm_type)
    input_data['data_val'] = normalize(input_data['data_val'], stats, norm_type=norm_type)
    input_data['data_test'] = normalize(input_data['data_test'], stats, norm_type=norm_type)
    return input_data


class WholeCores(Dataset):
    def __init__(self, data, label, transform, min_inv=.0, num_ts=20):

        total_ts = data.shape[-1]
        assert total_ts % num_ts == 0
        self.data = data[(label >= min_inv) | (label == 0)].transpose([0, 3, 1, 2])
        self.data = self.data.reshape((self.data.shape[0] * self.data.shape[1],) + self.data.shape[2:])
        self.data = self.data.reshape((-1, num_ts) + self.data.shape[1:])

        self.label = torch.tensor(label[(label >= min_inv) | (label == 0)] > 0).long()
        self.label = self.label.tile((total_ts//num_ts, 1)).T.flatten()

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        return data, self.label[idx]


class WholeCores3D(WholeCores):
    def __init__(self, *args, **kwargs):
        super(WholeCores3D, self).__init__(*args, **kwargs)
        self.data = np.expand_dims(self.data, 1)


class TestDataset(Dataset):
    def __init__(self, data, label, transform, input_channels=2):
        if input_channels == 4:
            self.data = data[..., 2:-1]
        else:
            self.data = data[..., :input_channels]
        self.input_channels = input_channels
        self.label = torch.tensor(label).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx])
        return data, self.label[idx]


class Augment:
    def __init__(self):
        pass

    def __call__(self, data):
        if random.random() > .5:
            data = np.flip(data, axis=-2)
        if random.random() > .5:
            data = np.flip(data, axis=-1)
        # data = ndimage.rotate(data, random.randint(0, 360), reshape=False, order=1, axes=(-2, -1))
        return data  # .astype('float32')


class Raw:
    def __init__(self):
        pass

    def __call__(self, data):
        return data  # .astype('float32')


def create_loader(dataset, batch_size=128, jobs=0, add_sampler=False, shuffle=False, drop_last=False):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    # For unbalanced dataset we create a weighted sampler
    sampler = None
    if add_sampler:
        shuffle = False
        # Compute samples weight (each sample should get its own weight)
        weights = make_weights_for_balanced_classes(dataset.label)
        # weights = make_weights_for_balanced_classes(dataset)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=jobs,
                            pin_memory=True, drop_last=drop_last)
    return dataloader

