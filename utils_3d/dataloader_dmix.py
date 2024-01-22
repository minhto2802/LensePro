import torch
from torch.utils.data import DataLoader

from .dataset import PatchLabeledDataset
from utils_3d.dataloader import create_loader
from utils_3d.dataset import CropFixSize
from training_strategy.self_supervised_learning.augmentations import *


class PatchLabeledDatasetDMix(PatchLabeledDataset):
    def __init__(self, *args, r=0, pred=[], probability=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.probability = probability
        self.filter_by_pred(pred)
        if self.transform is not None:
            self.transform.n_outputs = 2

    @staticmethod
    def get_pred_idx(pred):
        return pred.nonzero()[0]

    def filter_by_pred(self, pred):
        pred_idx = self.get_pred_idx(pred)
        self.files = self.files[pred_idx]
        self.label = self.label[pred_idx]
        self.probability = [self.probability[i] for i in pred_idx]

    def __getitem__(self, idx):
        data = np.load(self.files[idx], mmap_mode='c').astype('float32')
        if (data.shape[0] > 1) and (data.ndim == 3):
            data = data.mean(axis=0)

        data = (self.transform(data), self.transform(data))

        data = tuple(self.norm_data(d) for d in data)
        data = [(d - np.median(d)) / (np.percentile(d, 75) - np.percentile(d, 25)) for d in data]

        if self.label is not None:
            label = self.label[idx]
            return data[0], data[1], label, self.probability[idx]
        else:
            return data[0], data[1]


class PatchUnlabeledDatasetDMix(PatchLabeledDatasetDMix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = None

    @staticmethod
    def get_pred_idx(pred):
        return (1 - pred).nonzero()[0]


class PatchDatasetDMix(PatchLabeledDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_idx = True
        if self.transform is not None:
            self.transform.n_outputs = 1


class PatchDataloader:
    def __init__(self, args):
        self.args = args
        aug = OneCropTransform if not args.time_series else JustCropTransform
        transform = aug(not args.random_crop, in_channels=args.in_channels)
        self.kwargs = dict(inv_range=(0.7, 1), pid_range=(0, 100), gs_range=(7, 10),
                           transform=transform, return_idx=False, slide_idx=None,
                           stats=None, norm=args.norm, time_series=args.time_series,)
        self.kwargs_loader = dict(batch_size=args.batch_size, num_workers=args.workers, drop_last=False,
                                  pin_memory=False)

    def run(self, mode, pred=None, prob=None):
        kwargs_loader = self.kwargs_loader.copy()
        args = self.args

        if mode == 'warmup':
            trn_ds = PatchDatasetDMix(args.data_root,
                                      # oversampling_cancer=True,
                                      **self.kwargs)
            train_sampler = create_loader(trn_ds, args.batch_size * 2, jobs=args.workers, add_sampler=True,
                                          get_sampler_only=True, weight_by_inv=False)
            train_loader = torch.utils.data.DataLoader(trn_ds, sampler=train_sampler, **kwargs_loader)
            return train_loader

        elif mode == 'eval_train':
            self.kwargs['transform'] = CropFixSize()
            trn_ds = PatchDatasetDMix(args.data_root, **self.kwargs)
            kwargs_loader['drop_last'] = False
            train_loader = torch.utils.data.DataLoader(trn_ds, shuffle=False, **kwargs_loader)
            return train_loader

        elif mode == 'train':
            assert (pred is not None) and (prob is not None)
            # self.kwargs['oversampling_cancer'] = False
            labeled_trn_ds = PatchLabeledDatasetDMix(args.data_root, pred=pred, probability=prob,
                                                     return_prob=True,
                                                     **self.kwargs)
            train_sampler = create_loader(labeled_trn_ds, args.batch_size, jobs=args.workers, add_sampler=True,
                                          get_sampler_only=True, weight_by_inv=False)
            labeled_train_loader = DataLoader(labeled_trn_ds, sampler=train_sampler, **kwargs_loader)
            # labeled_train_loader = DataLoader(labeled_trn_ds, shuffle=True, **kwargs_loader)

            unlabeled_trn_ds = PatchUnlabeledDatasetDMix(args.data_root, pred=pred, probability=prob,
                                                         **self.kwargs)
            unlabeled_train_loader = torch.utils.data.DataLoader(unlabeled_trn_ds, shuffle=True, **kwargs_loader)
            return labeled_train_loader, unlabeled_train_loader

        elif mode == 'test':
            raise NotImplementedError()
