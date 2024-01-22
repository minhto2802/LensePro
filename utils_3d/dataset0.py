import re

import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset, Dataset, DataLoader


class NumpyDataset(Dataset):
    def __init__(self, files, label, indices=None, ts_len=50, inv=None, total_ts_len=200,
                 random_crop_ts=False, input_dim=3, norm=False, augment=False,
                 drop_range: (float, float) = (0.0, 0.0), stats=None, transform=None, is_train=False,
                 patch_info=(24, 32), metadata=None, shift_perc=(0, 0)):
        self.files = files
        self.label = label
        self.indices = indices
        self.ts_len = ts_len
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

    @property
    def pid(self):
        return [md['PatientId'] for md in self.metadata]

    @property
    def cid(self):
        return [md['CoreId'] for md in self.metadata]

    @property
    def gs(self):
        return [md['GleasonScore'] for md in self.metadata]

    def assert_correct_file(self, index):
        pid, cid = int(re.findall('Patient(\d+)', self.files[index])[0]), int(re.findall('Core(\d+)', self.files[
            index])[0])
        assert (pid == self.pid[index]) and (cid == self.cid[index])

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        start_ts = np.random.randint(0, self.total_ts_len - self.ts_len + 1) if self.random_crop_ts else 0
        # data = extract_patches(self.files[index], patch_info=self.patch_info, num_ts=200, inv=self.inv[index],
        #                        # augment=True, shift_perc=((.4, .5), (0, 0)))  # self.augment)

        # #                        augment=self.augment, shift_perc=((.0, -.5), (-1, .1)))  # self.augment)
        self.assert_correct_file(index)

        if not self.augment:
            shift_perc = ([0, 0], [0, 0])
        else:
            if self.inv[index]:
                shift_perc = ([0, 0.1], [0., 0.05])
            elif not self.label[self.pid == self.pid[index]][:, 1].sum():  # Benign patient
                shift_perc = ([0, 0.5], [-0.1, 0.1])
                # shift_perc = ([-0.2, 0.5], [-0.5, 1])
            else:
                shift_perc = ([0, .3], [-0.1, .05])
        if self.augment:
            data = extract_patches(self.files[index], patch_info=self.patch_info, num_ts=200, inv=self.inv[index],
                                   augment=True,
                                   # augment=False,
                                   shift_perc=shift_perc,
                                   # shift_perc=((-.1, .1), (-.05, .05)),
                                   # shift_perc=((0, 0), (0, 0)),
                                   # augment=True, shift_perc=((.0, .5), (0, 0)))  # self.augment)
                                   )
        else:
            data = np.load(self.files[index], mmap_mode='c')

        # data = np.load(self.files[index], mmap_mode='c')

        if self.input_dim == 1:
            # if self.norm:
            #     if self.stats is None:
            #         stats = (np.median(data), np.percentile(data, 75), np.percentile(data, 25))
            #     else:
            #         stats = self.stats[index]
            #     data = (data - stats[0]) / (stats[1] - stats[2])
            # Reshape for dropping out signals
            # data = data.reshape((data.shape[0], data.shape[1], -1))  # WRONG ONE AT FIRST
            # data = data.reshape((data.shape[0], -1, data.shape[-1])).transpose([0, 2, 1])
            # if self.augment:
            # if sum(self.drop_range):
            #     num_drop = int(np.random.uniform(self.drop_range[0], self.drop_range[1], 1) * data.shape[-1])
            #     keep_matrix = np.ones(data.shape, dtype='bool')
            #     for i in range(keep_matrix.shape[0]):
            #         keep_matrix[i, :, np.random.permutation(data.shape[-1])[:num_drop]] = 0
            #     data = data[keep_matrix].reshape((data.shape[0], data.shape[1], -1))
            # data_h = self.random_remove_channels(data)
            # data_w = self.random_remove_channels(data.transpose([0, 2, 1, 3]))
            data_t = self.random_remove_channels(data.transpose([0, 3, 1, 2]))[:, start_ts:start_ts + self.ts_len]
            # data = np.concatenate([data_h, data_w, data_t], axis=-1)
            data = data_t
            # if self.norm:
            # if not self.is_train:
            #     data = (data - np.mean(data)) / np.std(data)
            # data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))
            # data = (data - np.mean(data)) / np.std(data)

        # if self.augment:
        #     data = data[np.random.permutation(self.patch_info[0])]

        return data, self.label[index]  # , self.inv[index]

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

    def random_remove_channels(self, data):
        drop_range = self.drop_range
        data = data.reshape((data.shape[0], data.shape[1], -1))
        if sum(drop_range):
            num_drop = int(np.random.uniform(drop_range[0], drop_range[1], 1) * data.shape[-1])
            keep_matrix = np.ones(data.shape, dtype='bool')
            for i in range(keep_matrix.shape[0]):
                keep_matrix[i, :, np.random.permutation(data.shape[-1])[:num_drop]] = 0
            data = data[keep_matrix].reshape((data.shape[0], data.shape[1], -1))

        data = data.mean(axis=-1)  # Average signals
        if self.norm:
            data = (data - np.median(data)) / (np.percentile(data, 75) - np.percentile(data, 25))

            # data = (data - np.median(data, axis=1, keepdims=True)) / (np.percentile(data, 75, keepdims=True, axis=1
            #                                                                    ) - np.percentile(
            #     data, 25, keepdims=True, axis=1))
        return data.astype('float32')

    def plot_signals(self, index, data, start_ts):
        if self.inv[index]:
            data2 = np.load(self.files[index], mmap_mode='c')
            data2 = self.random_remove_channels(data2.transpose([0, 3, 1, 2]))[:, start_ts:start_ts + self.ts_len]

            import matplotlib
            matplotlib.use('TkAgg')
            import pylab as plt
            ax = plt.subplots(1, 2)[1]
            ax[0].plot(data2.T)
            ax[1].plot(data.T)
            plt.show()


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


class NumpyDatasetForSignals(NumpyDatasetForStats):
    def __getitem__(self, index):
        data = np.load(self.files[index]).transpose([0, 3, 1, 2])
        return data.mean(axis=(2, 3))


def extract_patches(filename, patch_info=(24, 32), num_ts=200, inv=None,
                    shift_perc=((0.0, 0.0), (0, 0)), augment=False):
    """

    :param filename:
    :param patch_info:
    :param num_ts:
    :param inv:
    :param augment:
    :param shift_perc: not used if inv is not None
    :return:
    """

    roi_file_name = filename.replace(f'_patch3D_{patch_info[0]}_{patch_info[1]}', '_roi')
    roi = np.load(roi_file_name, mmap_mode='c')
    rf_file_name = filename.replace(f'_patch3D_{patch_info[0]}_{patch_info[1]}', '_rf_whole_200')
    rf = np.load(rf_file_name, mmap_mode='c')

    rows, cols = np.where(roi == 1)
    max_shift_range = [None, None]
    num_rows, num_cols = rows.max() - rows.min(), cols.max() - cols.min()

    shift_row, shift_col = 0, 0
    # if augment:
    #     # pass
    #     if inv:
    #         shift_perc = ([-0.001, 0.001], [-0.001, 0.001])
    #
    #     #     shift_perc = ((-inv * 0.25, inv * 0.25), shift_perc[1])
    #         # shift_perc = ((-inv * 0.1, inv * 0.1), shift_perc[1])
    #     # if gs == '-':
    #     # shift_perc = ((np.random.randint(4, 8)/10 * 0.25, np.random.randint(7, 11)/10 * 0.35), shift_perc[1])
    #     # shift_perc = ([0, 0], [0, 0])
    #     # shift_perc = ((inv * 100, inv * 101), shift_perc[1])
    #     # print(shift_perc)
    # else:
    #     shift_perc = ([0, 0], [0, 0])
    #    #     print('------', shift_perc)
    # max_shift_range[0] = sorted([int(shift_perc[0][0] * num_rows), int(shift_perc[0][1] * num_rows)])
    # max_shift_range[1] = sorted([int(shift_perc[1][0] * num_cols), int(shift_perc[1][1] * num_cols)])

    # make the shift range symmetric
    # max_shift_range = [(-max_shift_range[0], max_shift_range[0]), (-max_shift_range[1], max_shift_range[1])]

    num_patches, patch_size = patch_info  # 48, 32  # patch_info  # TODO: temporary only
    patch_mask = np.zeros((num_patches,) + roi.shape, dtype='uint8')
    patches = np.zeros((num_patches, patch_size, patch_size, num_ts), dtype='float32')

    steps = int((rows.max() - rows.min()) / num_patches)
    # Extract patches along the core
    start_rows = np.unique(rows)[::steps][:num_patches]
    start_cols = np.array(
        [int(np.mean(cols[rows == r]) - patch_size / 2) for r in start_rows + int(patch_size / 2)])
    start_cols[start_cols < 0] = 0
    for j in range(num_patches):
        # # shift_row = np.random.randint(max_shift_range[0][0], 0) if np.any(max_shift_range[0]) else 0
        # # shift_col = np.random.randint(max_shift_range[1][0], 0) if np.any(max_shift_range[1]) else 0
        # # Random shifting
        # shift_row = np.random.randint(*max_shift_range[0]) if np.any(max_shift_range[0]) else 0
        # shift_col = np.random.randint(*max_shift_range[1]) if np.any(max_shift_range[1]) else 0
        # # Extracting patches
        # # try:
        # start_row = max(start_rows[j] + shift_row, 0)
        # start_col = min(max(start_cols[j] + shift_col, 0), rf.shape[2] - patch_size)

        shift_row, shift_col = np.random.randint(-40, 40), np.random.randint(-20, 20)
        start_row, start_col = max(start_rows[j] + shift_row, 0), max(start_cols[j] + shift_col, 0)
        patch_mask[j, start_row:start_row + patch_size, start_col: start_col + patch_size] = 1
        patches[j] = rf[start_row:start_row + patch_size, start_col: start_col + patch_size, :]
    return patches
