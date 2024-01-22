import re
import os
from glob import glob
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import scipy.stats as ss
from scipy.ndimage import rotate

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader

# from utils.augmentations import Grid
from PIL import Image


class PatchDataset(Dataset):
    def __init__(self, data_dir, transform, pid_range=(0, np.Inf), norm=True, return_idx=True, stats=None,
                 slide_idx=-1, time_series=False, pid_excluded=None, return_prob=False,
                 tta=False, *args, **kwargs):
        super(PatchDataset, self).__init__()
        # self.files = glob(f'{data_dir}/*/*/*/*.npy')
        data_dir = data_dir.replace('\\', '/')
        self.files = glob(f'{data_dir}/*/*.npy')
        self.transform = transform
        self.pid_range = pid_range
        self.pid_excluded = pid_excluded
        self.norm = norm
        self.pid, self.cid, self.label = [], [], None
        self.attrs = ['files', 'pid', 'cid']
        self.stats = stats
        self.slide_idx = slide_idx
        self.time_series = time_series
        self.return_idx = return_idx
        self.return_prob = return_prob
        self.probability = None
        self.tta = tta  # test time augmentation

    def extract_metadata(self):
        for file in self.files:
            self.pid.append(int(re.findall('/Patient(\d+)/', file)[0]))
            self.cid.append(int(re.findall('/core(\d+)', file)[0]))
        for attr in self.attrs:  # convert to array
            setattr(self, attr, np.array(getattr(self, attr)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], mmap_mode='c').astype('float32')
        # if self.queens_data:
        #     data = data.mean(axis=0)
        # if data.shape[-1] == 3:
        #     data = (((data - data.min()) / (data.max() - data.min())) * 255).astype('uint8')
        #     data = Image.fromarray(data)

        # if data.ndim == 3:
        #     # data = data[..., -100:]
        #     slide_idx = self.slide_idx if self.slide_idx is not None else np.random.choice(data.shape[-1])
        #     data = data[..., slide_idx]

        if self.transform is not None:
            data = self.transform(data)
        if self.time_series:
            data = F.avg_pool2d(torch.tensor(data), kernel_size=(8, 8), stride=8).flatten(1).T
            if self.norm:
                data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
        if self.norm and not self.time_series:
            if isinstance(data, tuple) or isinstance(data, list):
                data = tuple(self.norm_data(d) for d in data)
            else:
                data = self.norm_data(data)
            data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))

        if self.tta:
            data = np.concatenate([data, np.flip(data, axis=-1)], axis=0)

        if self.label is not None:
            label = self.label[idx]
            if self.return_prob:
                assert self.probability is not None
                return data, label, self.probability[idx]
            if self.return_idx:
                return data, label, idx
            return data, label

        return data[0], data[1]

    def norm_data(self, data):
        if self.stats is not None:
            data = (data - self.stats[0]) / self.stats[1]
        else:
            data = (data - data.mean()) / data.std()
            # data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
        return data

    def filter_by_pid(self):
        idx = np.logical_and(self.pid >= self.pid_range[0], self.pid <= self.pid_range[1])
        if self.pid_excluded is not None:
            idx[np.isin(self.pid, self.pid_excluded)] = False
        self.filter_by_idx(idx)

    def filter_by_idx(self, idx):
        for attr in self.attrs:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[idx])


class PatchLabeledDataset(PatchDataset):
    def __init__(self, data_dir, transform=None, pid_range=(0, np.Inf), inv_range=(0, 1), gs_range=(7, 10),
                 queens_data=False, file_idx=None, oversampling_cancer=False, *args, **kwargs):
        super().__init__(data_dir, transform, pid_range, *args, **kwargs)
        self.inv_range = inv_range
        self.gs_range = gs_range
        self.attrs.extend(['label', 'gs', 'location', 'id', 'inv'])
        self.label, self.inv, self.gs, self.location, self.id = [[] for _ in range(5)]
        self.queens_data = queens_data

        oversampling_dict = {0.8: 22, 0.7: 17, 0.6: 11, 0.5: 7, 0.4: 6}
        # oversampling_dict = {0.7: 17, 0.6: 19, 0.5: 11, 0.4: 10}
        if oversampling_cancer:
            oversampling_rate = oversampling_dict[min(self.inv_range)]
            # the oversampling rate (17) is calculated based on the class ratio after all filtering steps
            oversampled_files = []
            for file in self.files:
                if '_cancer' in file:
                    oversampled_files.extend([file for _ in range(oversampling_rate)])
            self.files += oversampled_files

        self.extract_metadata()
        self.filter_by_pid()
        self.filter_by_inv()
        self.filter_by_gs()
        if file_idx is not None:
            self.filter_by_idx(file_idx)

    def extract_metadata(self):
        for file in self.files:
            folder_name = os.path.basename(os.path.dirname(file))
            self.label.append(0) if folder_name.split('_')[1] == 'benign' else self.label.append(1)
            self.location.append(folder_name.split('_')[-2])
            self.inv.append(float(re.findall('_inv([\d.[0-9]+)', file)[0]))
            self.gs.append(int(re.findall('_gs(\d+)', file)[0]))
            self.pid.append(int(re.findall('/Patient(\d+)/', file)[0]))
            self.cid.append(int(re.findall('/core(\d+)_', file)[0]))
            self.id.append(int(folder_name.split('_')[-1][2:]))
        for attr in self.attrs:  # convert to array
            setattr(self, attr, np.array(getattr(self, attr)))

    def filter_by_inv(self):
        idx = np.logical_and(self.inv >= self.inv_range[0], self.inv <= self.inv_range[1])
        idx = np.logical_or(idx, self.inv == 0)
        self.filter_by_idx(idx)

    def filter_by_gs(self):
        idx = np.logical_and(self.gs >= self.gs_range[0], self.gs <= self.gs_range[1])
        idx = np.logical_or(idx, self.gs == 0)
        self.filter_by_idx(idx)


class PatchLabeledDatasetILR(PatchLabeledDataset):
    def __init__(self, *args, **kwargs):
        super(PatchLabeledDatasetILR, self).__init__(*args, **kwargs)

    def get_correcting_mask(self, core_len, predictions, true_involvement, predicted_involvement,
                            correcting_params):
        inv_dif_thr, prob_thr = correcting_params
        inv_dif = np.abs(np.subtract(predicted_involvement, true_involvement))
        inv_dif[np.array(true_involvement) == 0] = 1  # no correction for benign labels
        correcting_mask = np.array(inv_dif <= inv_dif_thr)
        correcting_mask = np.concatenate([np.repeat(_, cl) for (_, cl) in zip(correcting_mask, core_len)])
        # print(correcting_mask.sum(), predictions.max(), predictions[correcting_mask].max())
        correcting_mask[predictions.max(1) < prob_thr] = False

        return correcting_mask, self.files, predictions

    def correct_labels(self, correcting_mask, source_files, predictions, verbose=False):
        if not correcting_mask.sum():
            return 0
        target = np.isin(source_files, self.files, assume_unique=True)
        # extra data (not currently exist in the training set)
        correcting_mask_ext = correcting_mask[~target]
        predictions_ext = predictions[~target][correcting_mask_ext]
        source_files_ext = source_files[~target][correcting_mask_ext]
        # filter source files & correcting mask & predictions based on target files
        correcting_mask = correcting_mask[target]
        predictions = predictions[target]
        source_files = source_files[target]
        # map source files & correcting mask & predictions to target files
        _, map_idx = np.where(source_files[:, None] == self.files)
        correcting_mask = correcting_mask[map_idx]
        predictions = predictions[map_idx]
        source_files = source_files[map_idx]  # only for match checking with target files
        # sort source files
        source_files, uniq_idx = np.unique(source_files, return_index=True)
        correcting_mask = correcting_mask[uniq_idx]
        predictions = predictions[uniq_idx]
        # sort current files
        self.files, uniq_idx = np.unique(self.files, return_index=True)
        self.label = self.label[uniq_idx]
        assert np.all(source_files == self.files)

        # correcting target labels
        old_label = np.zeros_like(self.label) + self.label
        self.label[correcting_mask] = predictions[correcting_mask].argmax(1)
        n_corrected = np.abs(old_label - self.label).sum()
        # add extra data
        # self.label = np.concatenate([self.label, predictions_ext.argmax(1)])
        # self.files = np.concatenate([self.files, source_files_ext])
        n_extra_files = len(self.files) - len(source_files)

        if verbose:
            # add new time-series if available
            print(f'Cls_ratio: Old = {old_label.sum() / len(self.label):.3f}')
            print(f'Cls_ratio: New = {self.label.sum() / len(self.label):.3f}')
            print(f'Correcting amount: {n_corrected} patches ({100 * n_corrected / len(correcting_mask):.5f}%)')
            print(f'Extra data: {len(self.files) - len(source_files)} patches')
            print(np.unique(self.inv))
        # return n_corrected + n_extra_files
        return True


class PatchUnlabeledDataset(PatchDataset):
    def __init__(self, data_dir, transform=None, pid_range=(0, np.Inf), stats=None, norm=True, *args, **kwargs):
        super(PatchUnlabeledDataset, self).__init__(data_dir, transform, pid_range, norm=norm)
        # Note: cid: per patient core id; id: absolute core id
        self.extract_metadata()
        self.filter_by_pid()


class NumpyDataset(Dataset):
    def __init__(self, files, label, indices=None, ts_len=50, inv=None, total_ts_len=200,
                 random_crop_ts=False, input_dim=3, norm=False, augment=False,
                 drop_range: (float, float) = (0.0, 0.0), stats=None, transform=None, is_train=False,
                 patch_info=(24, 32), metadata=None, shift_perc=(0, 0), min_inv=0, max_inv=1.,
                 shift_range=(-240, 240, -50, 50), n_views=1, use_anchor=False, grid=None,
                 self_train=False, return_info=False, exclude_benign=False, re_extract=False, ts_start=None):
        self.files = files
        self.label = label
        self.indices = indices
        self.ts_len = ts_len
        self.ts_start = ts_start if ts_start is not None else 0
        self.total_ts_len = total_ts_len
        self.random_crop_ts = random_crop_ts if ts_len < total_ts_len else False
        self.input_dim = input_dim
        self.augment = augment
        self.norm = norm
        self.drop_range = drop_range
        self.stats = stats
        self.transform = transform
        self.signals = None
        self.is_train = is_train
        self.patch_info = patch_info  # TODO: temporary only
        self.metadata = metadata
        self.shift_perc = shift_perc
        self.inv = [md['Involvement'] for md in self.metadata]
        self.shift_range = shift_range
        self.min_inv = min_inv
        self.max_inv = max_inv
        self.use_anchor = use_anchor
        self.n_views = n_views
        self.self_train = self_train
        self.re_extract = re_extract
        self.exclude_benign = exclude_benign
        if grid is not None and grid.use:
            self.grid = Grid(grid.d1, grid.d2, grid.rotate, grid.ratio, grid.mode, grid.max_prob)
        else:
            self.grid = None
        self.filter_by_inv()
        self.return_info = return_info
        # if is_train:
        #     self.inv = np.array(self.inv) * 2 - 1

    @property
    def pid(self):
        return [md['PatientId'] for md in self.metadata]

    @property
    def cid(self):
        return [md['CoreId'] for md in self.metadata]

    @property
    def gs(self):
        return [md['GleasonScore'] for md in self.metadata]

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        # label
        label = self.label[index]
        label = int(label[1]) if self.is_train else label

        if self.augment:
            # data
            ts_start = np.random.randint(0, self.total_ts_len - self.ts_len + 1) if self.random_crop_ts else 0
            # extract patches with random shifting
            if self.self_train:
                data = gen_views_self_train(self.files[index], patch_info=self.patch_info, num_ts=200,
                                            inv=self.inv[index], shift_range=self.shift_range, n_views=self.n_views)
            else:
                data = gen_views(self.files[index], patch_info=self.patch_info, num_ts=200, inv=self.inv[index],
                                 shift_range=self.shift_range, n_views=self.n_views, use_anchor=self.use_anchor,
                                 grid=self.grid)
        else:
            ts_start = self.ts_start
            if self.re_extract:
                data = gen_views(self.files[index], patch_info=self.patch_info, num_ts=200, inv=self.inv[index],
                                 shift_range=None, n_views=1, use_anchor=False, random_scale=None,
                                 grid=False, self_train=False)
            else:
                data = np.load(self.files[index], mmap_mode='c')

        if self.input_dim == 1:
            # data_h = self.random_remove_channels(data)
            # data_w = self.random_remove_channels(data.transpose([0, 2, 1, 3]))
            # data = np.array(data) if isinstance(data, list) else data
            # data_t = self.reduce_dim_v0(data.transpose([0, 3, 1, 2]))[:, start_ts:start_ts + self.ts_len]
            data_t = self.reduce_dim(data)[:, ts_start:ts_start + self.ts_len]
            # data = np.concatenate([data_h, data_w, data_t], axis=-1)
            data = data_t
        else:
            data = np.array(data).transpose([0, -1, 1, 2])

        # l = [np.argmax(label) if len(label) == 2 else label, self.inv[index]] + np.min(data, axis=1).tolist()
        # for i, v in enumerate(l):
        #     if i == (len(l) - 1):
        #         print(np.round(v, 3), end='\n')
        #     else:
        #         print(np.round(v, 3), end=" ")

        if self.transform is not None:
            data = self.transform[0](data.T).T.astype('float32')

        if self.self_train:
            if self.n_views < 2:
                raise ValueError("For now, n_views has to be at least 2 when self_train is True")

        if self.norm:
            if self.n_views > 1:
                data = list(data.reshape([self.n_views, self.patch_info[0], -1]))
                for i, d in enumerate(data):
                    data[i] = (d - np.median(d)) / (np.percentile(d, 75) - np.percentile(d, 25))
            else:
                data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))

        if self.stats[0] is not None:
            data = (data - self.stats[0]) / self.stats[1]

        if self.return_info:
            gs = self.gs[index] if isinstance(self.gs[index], int) else 0
            label = (label, gs, *[_[index] for _ in [self.inv, self.pid, self.cid]])
            return data, label

        return data, label, self.inv[index]

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.files)

    def initialize_stats(self, num_workers):
        self.stats = np.empty((len(self.files), 3), dtype='float32')
        dataset_tmp = NumpyDatasetForStats(files=self.files)
        dataloader = DataLoader(dataset_tmp, batch_size=8, num_workers=num_workers, shuffle=False)

        with tqdm(dataloader, unit="batch") as t_epoch:
            for batch in t_epoch:
                index, _stats = batch
                self.stats[index] = _stats.numpy()

    def gather_signals(self, num_workers):
        self.signals = []
        dataset_tmp = NumpyDatasetForSignals(files=self.files)
        dataloader = DataLoader(dataset_tmp, batch_size=8, num_workers=num_workers, shuffle=False)

        with tqdm(dataloader, unit="batch") as t_epoch:
            for batch in t_epoch:
                self.signals.append(batch.numpy())
        self.signals = np.concatenate(self.signals, axis=0)

    def reduce_dim_v0(self, data):
        drop_range = self.drop_range
        data = data.reshape((data.shape[0], data.shape[1], -1))
        if not self.grid:
            if sum(drop_range):  # Pixel-level masking
                num_drop = int(np.random.uniform(drop_range[0], drop_range[1], 1) * data.shape[-1])
                keep_matrix = np.ones(data.shape, dtype='bool')
                for i in range(keep_matrix.shape[0]):
                    keep_matrix[i, :, np.random.permutation(data.shape[-1])[:num_drop]] = 0
                # data = data[keep_matrix].reshape((data.shape[0], data.shape[1], -1))
                data *= keep_matrix
                # data = data.mean(axis=-1)  # Average signals
                data = data.sum(axis=-1) / (data.shape[-1] - num_drop)
        else:  # Use the grid mask for masking
            data_reduced = np.zeros([*data.shape[:2]])  # current data shape: Num patches x Num TS x H x W
            for i in range(data.shape[0]):
                num_remaining = np.sum(data[i, 0] == 1)
                data_reduced[i] = data[i, :].sum(axis=-1) / num_remaining
            return data_reduced
        # data = trim_mean(data, 0.1, axis=-1)

        print(data.sum())
        exit()
        return data

    def reduce_dim(self, patches):
        """
        Reduce H, W and keep only Time dimension
        :param patches: a list of N numbers of 3D arrays (size HWT)
        :return: 2D arrays (size NT)
        """

        def random_remove_pix(_patch, _num_drop=None):
            if _num_drop is None:
                _num_drop = int(np.random.uniform(self.drop_range[0], self.drop_range[1], 1) * 1024)
            _patch = _patch.reshape([-1, _patch.shape[-1]])
            if self.grid:  # grid masking
                keep_matrix = _patch != 0
            else:  # Pixel-level masking
                keep_matrix = np.ones(_patch.shape, dtype='bool')
                keep_matrix[np.random.permutation(_patch.shape[0])[:_num_drop]] = 0
                _patch *= keep_matrix
            if (_patch.shape[0] - _num_drop) == 0:
                print(_num_drop)
            _patch = _patch.sum(axis=0, keepdims=True) / (_patch.shape[0] - _num_drop)
            return _patch

        if self.grid or sum(self.drop_range):
            num_drop = int(np.random.uniform(self.drop_range[0], self.drop_range[1], 1) * 1024)
            patches = [random_remove_pix(patch, num_drop) for patch in patches]
        else:
            patches = [patch.mean(axis=(0, 1))[np.newaxis] for patch in patches]
        return np.concatenate(patches)

    def filter_by_inv(self):
        # inv = [] + self.inv
        attrs = ['files', 'label', 'indices', 'signals', 'metadata', 'inv']

        self.inv = np.array(self.inv)
        idx = np.logical_and(self.inv >= self.min_inv, self.inv <= self.max_inv)
        if not self.exclude_benign:
            idx = np.logical_or(idx, self.inv == 0)
        else:
            idx[self.inv == 0] = False
        for attr in attrs:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[idx])

        # for attr in attrs:
        #     if getattr(self, attr) is not None:
        #         setattr(self, attr, np.array([_ for (_, _inv) in zip(getattr(self, attr), inv)
        #                                       if (self.min_inv <= _inv <= self.max_inv) or (_inv == 0)]))


class NumpyDatasetForStats(Dataset):
    def __init__(self, files):
        super(NumpyDatasetForStats, self).__init__()
        self.files = files

    def __getitem__(self, index):
        data = np.load(self.files[index]).transpose([0, 3, 1, 2])
        stats = np.array([np.median(data), np.percentile(data, 75), np.percentile(data, 25)])
        return index, stats

    def __len__(self):
        return len(self.files)

    def filter_by_inv(self):
        # inv = [] + self.inv
        attrs = ['files', 'label', 'indices', 'signals', 'metadata', 'inv']

        self.inv = np.array(self.inv)
        idx = np.logical_and(self.inv >= self.min_inv, self.inv <= self.max_inv)
        if not self.exclude_benign:
            idx = np.logical_or(idx, self.inv == 0)
        else:
            idx[self.inv == 0] = False
        for attr in attrs:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[idx])


class NumpyDatasetForSignals(NumpyDatasetForStats):
    def __getitem__(self, index):
        data = np.load(self.files[index]).transpose([0, 3, 1, 2])
        return data.mean(axis=(2, 3))


class NumpyDatasetExtractedOnDisk:
    def __init__(self, extract_root, extract_dirs, ts_len=50, transform=None, patch_info=(24, 32), min_inv=0,
                 max_inv=1., ts_start=None):
        self.extract_root = extract_root
        self.extract_dirs = extract_dirs
        self.ts_len = ts_len
        self.transform = transform
        self.patch_info = patch_info
        self.min_inv = min_inv
        self.max_inv = max_inv
        self.data, self.metadata, self.label, self.inv, self.gs = [None] * 5
        self.ts_start = 0 if not ts_start else ts_start
        self.files = self.get_files()
        self.inv = self.get_inv_from_files(self.files)
        self.label = (self.inv > 0).astype('int')

    def get_files(self):
        files = []
        for extract_dir in self.extract_dirs:
            files.extend(glob('/'.join([self.extract_root, extract_dir, '*', '*.npy'])))
        inv = self.get_inv_from_files(files)
        idx = np.where(((self.min_inv <= inv) & (inv <= self.max_inv)) | (inv == 0))[0]
        return [files[i] for i in idx]

    @staticmethod
    def get_inv_from_files(files):
        import re
        return np.array([re.findall('_i([\d.[0-9]+)', file)[0] for file in files], dtype='float32')

    def __getitem__(self, idx):
        return np.load(self.files[idx])[..., self.ts_start:self.ts_start + self.ts_len]

    def __len__(self):
        return len(self.files)


class NumpyDatasetExtracted:
    def __init__(self, extract_root, extract_dirs, ts_len=50, transform=None, patch_info=(24, 32), min_inv=0,
                 max_inv=1., grid=None, dif_learning=False, cut_mix_1d=False, rand_erase=False, exclude_benign=False,
                 ts_start=None, data_suffix='', stats=(None, None), norm=True):
        self.extract_root = extract_root
        self.extract_dirs = extract_dirs
        self.ts_len = ts_len
        self.transform = transform
        self.patch_info = patch_info
        self.min_inv = min_inv
        self.max_inv = max_inv
        self.data, self.metadata, self.label, self.inv, self.gs = [None] * 5
        self.grid = None
        self.dif_learning = dif_learning
        self.cut_mix_1d = cut_mix_1d
        self.random_erase = rand_erase
        self.exclude_benign = exclude_benign
        self.ts_start = ts_start
        self.load_dataset(data_suffix)
        self.filter_by_inv()
        self.random_crop_ts = False if ts_len == self.data.shape[-1] else True

        self.stats = None, None
        if norm:
            if stats[0] is None:
                self.stats = self.data.mean(), self.data.std()
            else:
                self.stats = stats
            # # # self.mean, self.std = None, None
            # # # self.mean, self.std = 0.24337529,
            self.data = (self.data - self.stats[0]) / self.stats[1]

        self.benign_indices = np.where(self.label == 0)[0] if not exclude_benign else []
        self.cancer_indices = np.where(self.label == 1)[0]
        self.min_swap = [0.6, 0.4, 0.4]  # benign, cancer

    def load_dataset(self, data_suffix):
        """
        Load the dataset to RAM
        :return:
        """
        # self.data, self.metadata = [], []
        # for extract_dir in self.extract_dirs:
        #     folder = f'{self.extract_root}/{extract_dir}'
        #     self.data.append(np.load(f'{folder}/data_merged_{self.patch_info[0]}x{self.patch_info[1]}.npy',
        #                              mmap_mode='c'))
        #     self.metadata.append(np.load(f'{folder}/metadata_merged_{self.patch_info[0]}x{self.patch_info[1]}.npy'))
        extract_dir = self.extract_dirs[0]
        folder = f'{self.extract_root}/{extract_dir}'
        self.data = np.load(f'{folder}/data_merged_{self.patch_info[0]}x{self.patch_info[1]}{data_suffix}'
                            f'.npy', mmap_mode='c')
        self.metadata = np.load(f'{folder}/metadata_merged_{self.patch_info[0]}x{self.patch_info[1]}{data_suffix}'
                                f'.npy')
        # self.data = np.concatenate(self.data, axis=0)
        # self.metadata = np.concatenate(self.metadata, axis=0)
        self.label = np.array(self.metadata[:, 2], dtype='int')
        self.inv = np.array(self.metadata[:, -3], dtype='float32')
        if np.any(self.inv < 0):
            self.inv = (self.inv + 1) / 2
        self.gs = np.array(self.metadata[:, 3], dtype='float32').astype('int')

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        inv = self.inv[index]
        # if self.transform is not None:
        #     data = self.transform[0](data.T).T.astype('float32')

        # if np.random.rand() > 0:
        if self.cut_mix_1d:
            data, inv = self.cut_mix(data, inv)
            label = int(inv != 0)  # fix label if the inv is changed
        if self.random_erase:
            data = random_erase(data)
        # data = self.transform(data) if self.transform else data

        # differential learning
        if self.dif_learning:
            if label == 0:
                # index_dif = np.random.choice(np.arange(len(self.data)))
                index_dif = np.random.choice(self.cancer_indices)
            else:
                index_dif = np.random.choice(self.benign_indices)

                # indices = np.where(np.logical_and(self.inv >= inv - 0.4, self.inv <= inv - 0.2))[0]
                # if not len(indices):
                #     index_dif = np.random.choice(self.benign_indices)
                # else:
                #     index_dif = np.random.choice(indices)

                # from scipy.stats import skewnorm
                # import pylab as plt
                # a = -100
                # indices = np.where(self.inv < (inv-0.2))[0]
                # indices = indices[self.inv[indices].argsort()][::-1]
                # p = np.linspace(skewnorm.ppf(0.99, a), skewnorm.ppf(0.01, a), len(indices))
                # pdf = skewnorm.pdf(p, a)
                # index_dif = np.random.choice(indices, p=pdf / pdf.sum())
                # indices = np.where(np.logical_and(self.inv >= inv-0.4, self.inv <= inv-0.2))[0]

            data_dif, inv_dif = self.data[index_dif], self.inv[index_dif]
            if self.cut_mix_1d:
                data_dif, _ = self.cut_mix(data_dif, inv_dif, allow_b2c=False, source_list=self.benign_indices)
            data = (data, data_dif)

        if self.random_crop_ts:
            if self.ts_start is None:
                ts_start = np.random.randint(0, data.shape[-1] - self.ts_len + 1)
            else:
                ts_start = self.ts_start

            if isinstance(data, tuple):
                data = tuple([_[:, ts_start:ts_start + self.ts_len] for _ in data])
            else:
                data = data[:, ts_start: ts_start + self.ts_len]

        return data, label, inv

    def cut_mix(self, data, inv, allow_b2c=True, source_list=None):
        if inv == 0:
            if np.random.rand() > .05:
                data = self.benign_cut_mix(data)
            elif allow_b2c:  # flip label of the benign core
                inv = np.random.randint(50, 100) / 100
                data = self.b2c_cut_mix(data, inv)
                inv = 0.4  # return minimum inv (since the source cores (though w high inv) may contain benign signals)
        elif inv >= 0.5:
            data = self.high_inv_cut_mix(data, inv, source_list)
        else:
            pass
        # elif 0.5 <= inv < 0.7:
        #     data = self.low_inv_cut_mix(data, inv)
        return data, inv

    def filter_by_inv(self):
        if (self.min_inv <= min(self.inv)) and (self.max_inv >= max(self.inv)):
            return
        attrs = ['label', 'data', 'metadata', 'gs', 'inv']
        idx = np.logical_and(self.inv >= self.min_inv, self.inv <= self.max_inv)
        if not self.exclude_benign:
            idx = np.logical_or(idx, self.inv == 0)
        else:
            idx[self.inv == 0] = False
        for attr in attrs:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr)[idx])

    def __len__(self):
        return self.data.shape[0]

    def benign_cut_mix(self, x, *args):
        """
        Randomly replacing a random number signals in a benign core with those from different benign cores
        :param x: time-series of a benign core (HT)
        :return: mixed multivariate time-series (likely to have lower cancer involvement than the original core)
        """
        return cut_mix(x, self.data, self.benign_indices, 1 - self.min_swap[0], x.shape[0])

    def high_inv_cut_mix(self, x, inv, source_list=None):
        """
        Randomly replacing a random number signals in a high-inv core with those from benign cores
        :param x: time-series of a high-involvement core (HT)
        :param inv: involvement of cancer in x
        :param source_list:
        :return: mixed multivariate time-series (likely to have lower cancer involvement than the original core)
        """
        if source_list is None:
            # source_list = self.benign_indices if np.random.rand() > 0.5 else self.cancer_indices
            source_list = np.arange(len(self.data))
        return cut_mix(x, self.data, source_list, inv - self.min_swap[1], x.shape[0], )

    def low_inv_cut_mix(self, x, inv, min_inv_thr=0.8):
        """
        Randomly replacing a random number signals in a high-inv core with those from benign cores
        :param x: time-series of a high-involvement core (HT)
        :param inv: involvement of cancer in x
        :param min_inv_thr: minimum cancer involvement of the source
        :return: mixed multivariate time-series (likely to have lower cancer involvement than the original core)
        """
        source_list = self.cancer_indices[self.inv[self.cancer_indices] >= min_inv_thr]
        return cut_mix(x, self.data, source_list, inv - self.min_swap[2], x.shape[0], )

    def b2c_cut_mix(self, x, inv, min_inv_thr=0.8):
        source_list = self.cancer_indices[self.inv[self.cancer_indices] >= min_inv_thr]
        return cut_mix(x, self.data, source_list, inv, x.shape[0], )


def cut_mix(target, source, source_list, swap_percentage, num_channels):
    """
    Randomly replacing a random number signals in a benign/cancer core with those from different benign cores
    :param target:
    :param source:
    :param swap_percentage:
    :param source_list:
    :param num_channels:
    :return:
    """
    num_swap = max(int(np.ceil(0.1 * num_channels)), int(np.random.randint(swap_percentage * 100) / 100 * num_channels))
    swap_instance_indices = np.random.choice(source_list, num_swap)
    target_indices = np.random.choice(num_channels, num_swap)
    source_indices = np.random.choice(num_channels, num_swap)
    target_new = target.copy()
    for swap_idx, target_idx, source_idx in zip(swap_instance_indices, target_indices, source_indices):
        target_new[target_idx] = source[swap_idx][source_idx]
    return target_new


def random_erase(x, ratio=(30, 70)):
    """

    :param x: (HT)
    :param ratio: tuple indicates the range of erasing length
    :return:
    """
    e_ts_idx = np.random.randint(x.shape[0])
    e_ratio = np.random.randint(*ratio) * 0.01
    e_len = int(x.shape[1] * e_ratio)
    e_start = np.random.randint(e_len) - 1
    e_val = np.random.randint(100) / 100 * x.var()
    x[e_ts_idx, e_start:e_start + e_len] = e_val
    return x


def load_raw_npy(filename, patch_info=(24, 32)):
    roi_file_name = filename.replace(f'_patch3D_{patch_info[0]}_{patch_info[1]}', '_roi')
    roi = np.load(roi_file_name, mmap_mode='c')
    rf_file_name = filename.replace(f'_patch3D_{patch_info[0]}_{patch_info[1]}', '_rf_whole_200')
    rf = np.load(rf_file_name, mmap_mode='c')
    return rf, roi


def keep_largest_blob(input_mask):
    from skimage import measure
    labels_mask = measure.label(input_mask)
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
        labels_mask[labels_mask != 0] = 1
        return labels_mask
    return input_mask


def extract_patches(rf, roi_info, patch_info=(24, 32), num_ts=200, inv=None,
                    shift_perc=(0.5, 0.1), shift_range=(-240, 240, -50, 50), random_scale=(0.7, 1.), grid=None,
                    return_mask=False):
    roi_shape, rows, cols = roi_info

    num_patches, patch_size_org = patch_info
    patch_roi = None
    if return_mask:
        patch_roi = np.zeros((num_patches,) + roi_shape, dtype='uint8')
    # patches = np.zeros((num_patches, patch_size, patch_size, num_ts), dtype='float32')
    patches = []

    # Generate random steps along the needle
    steps = gen_steps(rows, num_patches, random_scale)

    # Extract patches along the core
    start_rows = np.unique(rows)[::steps][:num_patches]
    start_cols = np.array(
        [int(np.mean(cols[rows == min(r, max(rows))]) - patch_size_org / 2) for r in start_rows + int(
            patch_size_org /
            2)])
    start_cols[start_cols < 0] = 0
    patch_h, patch_w = patch_size_org, patch_size_org
    shifts = []
    for j in range(num_patches):
        # patch_h = patch_size_org + np.random.randint(int(patch_size_org * 0.25))
        # patch_w = patch_size_org + np.random.randint(int(patch_size_org * 0.25))
        if shift_range is not None:
            shift_row, shift_col = np.random.randint(*shift_range[0:2]), np.random.randint(*shift_range[2:])
        else:
            shift_row, shift_col = 0, 0
        start_row, start_col = max(start_rows[j] + shift_row, 0), max(start_cols[j] + shift_col, 0)
        shifts.append([start_row, patch_h, start_col, patch_w])
        if return_mask:
            patch_roi[j, start_row:start_row + patch_h, start_col: start_col + patch_w] = 1
        # if grid is not None:
        #     patches[j] = grid(patches[j])
    # shifts = [shifts[i] for i in np.argsort([_[0] for _ in shifts])]
    for shift in shifts:
        patches.append(deepcopy(rf[shift[0]:shift[0] + shift[1], shift[2]:shift[2] + shift[3], :]).astype('float32'))
    if return_mask:
        return patches, patch_roi
    return patches


def visualize_rolling_rois(rf, roi, rows=None, cols=None, fig_num=1):
    if rows is None or cols is None:
        rows, cols = np.where(roi == 1)

    import pylab as plt
    from utils_3d.visualize import make_analytical
    x_vis = make_analytical(np.squeeze(rf[..., 199]))
    plt.figure(fig_num)
    plt.imshow(x_vis, cmap='gray')
    plt.contour(roi, colors='blue')
    for i in range(20):
        shift_col = np.random.randint(max(cols) - min(cols), roi.shape[1] - max(cols))
        shift_row = np.random.randint(0, 150)
        roi_rolled = np.roll(roi, (shift_row, shift_col), axis=[0, 1])
        plt.contour(roi_rolled, colors='orange')
    plt.show()


def rotate_by_roi_center(roi, rows=None, cols=None):
    """Created a rotated version of ROI with ROI center as the rotation center"""
    if rows is None or cols is None:
        rows, cols = np.where(roi == 1)
    # if min(cols) == 0:
    #     return roi

    roi_patch = roi[min(rows): max(rows), min(cols): max(cols)]
    max_angle = np.floor(np.degrees(np.arctan((roi_patch.shape[1]) / roi_patch.shape[0])))
    angle = np.random.randint(-max_angle, max_angle)
    roi_patch_rotated = rotate(roi_patch, angle=angle, reshape=False, order=0)
    roi_rotated = np.zeros_like(roi)
    roi_rotated[min(rows): max(rows), min(cols): max(cols)] = roi_patch_rotated
    return keep_largest_blob(roi_rotated)


def gen_views(filename, patch_info=(24, 32), num_ts=200, inv=None,
              shift_perc=(0.5, 0.1), shift_range=(-240, 240, -50, 50), random_scale=(0.7, 1.), n_views=1,
              use_anchor=False, grid=None, self_train=False, viz_rolling=False, return_mask=False, to_rotate=False):
    """"""
    rf, roi = load_raw_npy(filename, patch_info)
    if to_rotate:
        roi = rotate_by_roi_center(roi)
    else:
        roi = keep_largest_blob(roi)
    rows, cols = np.where(roi == 1)

    shift_ranges = [shift_range for _ in range(n_views)]

    # patch_info = [24, 32]  # TODO: TEMPORARY HACK

    views = []
    for i in range(n_views):
        views.extend(extract_patches(rf, (roi.shape, rows, cols),
                                     patch_info, num_ts, inv, shift_perc, shift_ranges[i], random_scale, grid=grid,
                                     return_mask=return_mask))
    if use_anchor:  # anchors are stored in the last element of the list if used
        views.extend(extract_patches(rf, (roi.shape, rows, cols),
                                     patch_info, num_ts, inv, shift_perc, [-25, 25, -5, 5], random_scale, grid=grid))
    return views


def gen_views_self_train(filename, patch_info=(24, 32), num_ts=200, inv=None,
                         shift_perc=(0.5, 0.1), shift_range=(-240, 240, -50, 50), random_scale=(0.7, 1.), n_views=1,
                         viz_rolling=False, return_mask=False, to_rotate=False):
    """"""
    rf, roi = load_raw_npy(filename, patch_info)
    if to_rotate:
        roi = rotate_by_roi_center(roi)
    else:
        roi = keep_largest_blob(roi)
    rows, cols = np.where(roi == 1)

    if viz_rolling:
        visualize_rolling_rois(rf, roi)
    shift_cols = np.arange(0, roi.shape[1] // 2, n_views)
    views = []
    for i, shift_col in zip(range(n_views), shift_cols):
        shift_row = np.random.randint(0, 20)
        roi = np.roll(roi, (shift_row, shift_col), axis=[0, 1])
        rows, cols = np.where(roi == 1)
        shift_ranges = [shift_range for _ in range(n_views)]
        # patch_info = [24, 32]  # TODO: TEMPORARY HACK
        views.extend(extract_patches(rf, (roi.shape, rows, cols),
                                     patch_info, num_ts, inv, shift_perc, shift_ranges[i], random_scale,
                                     return_mask=return_mask))
    return views


def gen_steps(rows, num_patches, random_scale=None):
    row_max, row_min = rows.max(), rows.min()
    if random_scale is not None:
        scale_factor = np.sqrt(np.random.uniform(*random_scale))
        row_max, row_min = row_max * scale_factor, row_min * np.abs(2 - scale_factor)
    steps = int((row_max - row_min) / num_patches)
    return steps


def gen_shift_vals_normal_dist(shift_range, num_vals):
    x = np.arange(shift_range[0], shift_range[1] + 1)
    x_u, x_l = x + 0.5, x - 0.5
    prob = ss.norm.cdf(x_u, scale=np.abs(shift_range[0]) // 3) - ss.norm.cdf(x_l, scale=np.abs(shift_range[1]) // 3)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    return np.random.choice(x, size=num_vals, p=prob)


def describe(ds_train, ds_val, ds_test):
    width0, width1 = 20, 12
    print(f"{' ': <{width0}}{'Train': <{width1}}{'Val': <{width1}}{'Test': <{width1}}")

    def print_lines(title, n1, n2, n3):
        print(f"{title: <{width0}}{n1: <{width1}}{n2: <{width1}}{n3: <{width1}}")

    print_lines('n files', len(ds_train.files), len(ds_val.files), len(ds_test.files))
    print_lines('n patients', len(np.unique(ds_train.pid)), len(np.unique(ds_val.pid)), len(np.unique(ds_test.pid)))
    print_lines('n cores', len(np.unique(ds_train.id)), len(np.unique(ds_val.id)), len(np.unique(ds_test.id)))
    print_lines('n benign', sum(ds_train.label == 0), sum(ds_val.label == 0), sum(ds_test.label == 0))
    print_lines('n cancer', sum(ds_train.label == 1), sum(ds_val.label == 1), sum(ds_test.label == 1))


def try_PatchLabeledDataset():
    data_root = '/home/minh/PycharmProjects/Preparation/DataPreparation/dummy/snapshots_in_use_Amoon_2/patches' \
                '/patch_48x48_str16_avg/*/*_core/'

    ds_train = PatchLabeledDataset(data_root, inv_range=(0.4, 1), pid_range=(0, 100), gs_range=(7, 10),
                                   transform=CropFixSize(), norm=False)
    ds_val = PatchLabeledDataset(data_root, inv_range=(0.4, 1), pid_range=(101, 130), gs_range=(7, 10),
                                 transform=CropFixSize(), norm=False)
    ds_test = PatchLabeledDataset(data_root, inv_range=(0.4, 1), pid_range=(131, 200), gs_range=(7, 10),
                                  transform=CropFixSize(), norm=False)

    ds = ds_train
    print(len(ds))
    for attr in ds.attrs:
        print(attr, np.unique(getattr(ds, attr)))
    print(ds[0])
    print(ds[0][0].shape, ds[0][0].dtype)

    describe(ds_train, ds_val, ds_test)

    import torch
    save_dir = r'supplementary/pictures'
    mean = 0
    std = 0
    os.makedirs(save_dir, )
    set_names = ['train', 'val', 'test']
    for _ds, name in zip([ds_train, ds_val, ds_test], set_names):
        loader = torch.utils.data.DataLoader(_ds, batch_size=128, num_workers=8, pin_memory=True)
        for i, (data, label) in enumerate(loader):
            mean += sum(data.squeeze().mean([1, 2]))
            std += sum(data.squeeze().std([1, 2]))
        print(name, mean / len(_ds), std / len(_ds))
    return


class CropFixSize:
    def __init__(self, sz=32, in_channels=1):
        import imgaug.augmenters as iaa
        self.in_channels = in_channels
        self.seq = iaa.CenterCropToFixedSize(sz, sz)

    def __call__(self, sample):

        if (sample.ndim > 2) and (sample.shape[-1] > 1):
            assert sample.shape[-1] >= self.in_channels
            sample = sample[..., :self.in_channels]

        x1 = self.seq(image=sample)

        if x1.ndim > 2:
            return x1.transpose([2, 0, 1])
        return x1[np.newaxis]


def try_patch_UnlabeledDataset():
    data_root = '/home/minh/PycharmProjects/Preparation/DataPreparation/dummy/snapshots_in_use_Amoon_2/patches' \
                '/patch_48x48_str32_avg/*/*_entire_prostate/'
    from training_strategy.self_supervised_learning.augmentations import OneCropTransform
    ds = PatchUnlabeledDataset(data_root, OneCropTransform(), pid_range=(0, 100))
    print(len(ds))
    # for i in range(len(ds)):
    #     x = ds[i]
    #     if (x[0].shape != (1, 32, 32)) or (x[1].shape != (1, 32, 32)):
    #         print(i, ds.files[i])
    #         os.remove(ds.files[i])

    # import torch
    # loader = torch.utils.data.DataLoader(ds, batch_size=512, num_workers=10, pin_memory=False, drop_last=False)
    # print(len(loader))
    # for i, data in tqdm(enumerate(loader)):
    #     continue

    return


def main():
    # try_patch_UnlabeledDataset()
    try_PatchLabeledDataset()


if __name__ == '__main__':
    main()
