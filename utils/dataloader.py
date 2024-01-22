import torch
import numpy as np
from torch.utils.data import DataLoader
# from .dataset import DatasetV1, to_categorical
# from tslearn.clustering import TimeSeriesKMeans


def make_weights_for_balanced_classes(target):
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    try:
        return torch.tensor([weight[t] for t in target])
    except IndexError as E:
        return torch.zeros_like(target)


# def make_weights_for_balanced_classes(dataset, nclasses=2):
#     count = [0] * nclasses
#     for l in dataset.label:
#         count[l[1]] += 1
#     weight_per_class = [0.] * nclasses
#     N = float(sum(count))
#     for i in range(nclasses):
#         weight_per_class[i] = N / float(count[i])
#     weight = [0] * len(dataset)
#     for idx, l in enumerate(dataset.label):
#         weight[idx] = weight_per_class[l[1]]
#     return torch.tensor(weight)


def create_loaders_test(*datasets, bs=128, jobs=0, data_loader=None):
    """Wraps the datasets returned by create_datasets function with data loaders."""
    dataloaders = []
    for dataset in datasets:
        if data_loader is None:
            data_loader = DataLoader
        dataloaders.append(data_loader(dataset, batch_size=bs, shuffle=False, num_workers=jobs, pin_memory=False))
    return dataloaders


def create_loader(dataset, bs=4086, jobs=0, add_sampler=False, shuffle=False, aug_lib='self_time',
                  get_sampler_only=False, weight_by_inv=False):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    # For unbalanced dataset we create a weighted sampler
    # sampler = ImbalancedDatasetSampler(dataset)
    sampler = None
    if add_sampler:
        shuffle = False
        # Compute samples weight (each sample should get its own weight)
        if weight_by_inv:
            label = np.digitize(dataset.inv, [0, .4, .6, .8]) - 1
        else:
            if hasattr(dataset, 'label'):
                label = dataset.label.argmax(1) if dataset.label.ndim == 2 else dataset.label
            else:
                label = dataset.data.y[dataset.indices()].int()
        weights = make_weights_for_balanced_classes(torch.tensor(label))
        # weights = make_weights_for_balanced_classes(dataset)
        batch_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        if aug_lib == 'tsaug':
            sampler = torch.utils.data.sampler.BatchSampler(
                batch_sampler,
                batch_size=bs // jobs if jobs > 0 else bs,
                drop_last=True if jobs > 0 else False)
            bs = jobs if jobs > 0 else None
            # jobs = 0
        else:
            sampler = batch_sampler
    if get_sampler_only:
        return sampler
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=sampler, num_workers=jobs,
                            pin_memory=False, drop_last=True)
    return dataloader


def create_loader_unsup(dataset, bs=4086, jobs=0, shuffle=True):
    if dataset is None:
        return None
    bs = int(bs * dataset.cfg.unsup_ratio)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=jobs, pin_memory=True)
