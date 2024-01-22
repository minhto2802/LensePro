import os
import mat73
import random
from typing import Union
from copy import deepcopy

import torch
import numpy as np
import torch.multiprocessing
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Dataset
# from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve, AddNoise

torch.multiprocessing.set_sharing_strategy('file_system')

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from self_time.optim.pretrain import get_transform
from utils.misc import load_pickle, robust_norm, robust_norm_per_core
from utils.patches import coor2patches_whole_input
from preprocessing.s02b_create_unsupervised_dataset import load_datasets as load_unlabelled_datasets
from utils.split_data_v0 import merge_split, merge_split_train_val, merge_train_val


def preproc_input(x, norm_per_signal, condition=None):
    """Preprocess training or test data, filter data by condition"""
    if condition is not None:
        x = x[condition]

    x = np.array([norm01_rf(d, per_signal=norm_per_signal) for d in x])
    return x


def norm01_rf(x, per_signal=True):
    """Normalize RF signal magnitude to [0 1] (per signal or per core)"""
    ax = 1 if per_signal else (0, 1)
    mi = x.min(axis=ax)
    ma = x.max(axis=ax)
    rfnorm = (x - mi) / (ma - mi)
    return rfnorm


def to_categorical(values):
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


def to_categorical_v0(y):
    """Deprecated"""
    n_classes = np.max(y) + 1
    y_c = np.zeros((len(y), np.int(n_classes)))
    for i in range(len(y)):
        y_c[i, np.int(y[i])] = 1
    return y_c


def create_datasets_cores(ftrs_train, inv_train, corelen_train, ftrs_val, inv_val, corelen_val):
    counter = 0
    signal_train = []
    for i in range(len(corelen_train)):
        temp = ftrs_train[counter:(counter + corelen_train[i])]
        signal_train.append(temp)
        counter += corelen_train[i]

    counter = 0
    signal_val = []
    for i in range(len(corelen_val)):
        temp = ftrs_val[counter:(counter + corelen_val[i])]
        signal_val.append(temp)
        counter += corelen_val[i]

    label_train = to_categorical(inv_train > 0)
    label_val = to_categorical(inv_val > 0)
    return signal_train, label_train, inv_train, signal_val, label_val, inv_val


class DatasetV1(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, label, location, inv, core_id, patient_id, ts_id, core_len, roi_coors,
                 # neighbors,
                 # stn,
                 aug_type: Union[list, str] = ('G2',), initial_min_inv=.4, n_neighbor=0,
                 n_views=1, transform_prob=.2, degree=1, aug_lib='self_time', n_duplicates=1,
                 stn_alpha=2, ts_len=200, input_channels=-1, channel_len=25):
        """"""
        if input_channels > 1:
            # The length of the time-series must be divisible by the number of input channels
            assert (not (data.shape[1] % channel_len))
            data = data.reshape((data.shape[0], -1, channel_len))
            if input_channels > 0:
                data = data[:, :input_channels]
        ts_cid = np.concatenate([np.arange(cl) for cl in core_len])

        self.data_dict = {
            'data': data,
            'label': label,
            'location': location,
            'inv': inv,
            'core_id': core_id,
            'patient_id': patient_id,
            'ts_id': ts_id,
            # 'roi_coors': roi_coors,
            'ts_cid': ts_cid,
            # 'neighbors': neighbors,
            # 'stn': stn,  # stationery
        }
        self.data, self.label, self.location, self.inv, self.core_id, self.patient_id, self.inv_pred, \
        self.ts_id, self.roi_coors, self.stn, = [None, ] * 10
        self.ts_cid_to_id, self.ts_cid_to_index, self.index = None, None, None
        # self.stn_alpha = stn_alpha
        self.label_corrected = False
        self.last_updated_label = None
        self.transformer = None
        self.core_len = core_len
        self.ts_len = ts_len
        self.n_views, self.aug_type, self.aug_lib, self.n_duplicates = n_views, aug_type, aug_lib, n_duplicates
        self.initial_min_inv = initial_min_inv
        self.n_neighbor = n_neighbor
        self.input_channels = input_channels
        self.make_dataset()

        # Initialize the data augmentation
        if aug_lib == 'self_time':
            self.transform, self.to_tensor_transform = get_transform(
                aug_type if isinstance(aug_type, list) else [aug_type, ],
                prob=transform_prob, degree=degree,
            )
        elif aug_lib == 'tsaug':
            self.transform = (
                    TimeWarp(max_speed_ratio=(2, 3), prob=transform_prob, seed=0) * n_duplicates
                    + Quantize(n_levels=(10, 60), prob=transform_prob, seed=0)
                    + Drift(max_drift=(0.01, 0.4), seed=0) @ transform_prob
                    + Convolve(window=['flattop', 'hann', 'triang'], size=(3, 11), prob=transform_prob, seed=0)
                    + AddNoise(0, (0.01, 0.05), prob=transform_prob, seed=0)
                # + Reverse() @ transform_prob  # with 50% probability, reverse the sequence
            )

    def __getitem__(self, index):
        # x, x_neighbors, y, z, loss_weight = self.getitem_by_index_with_neighbors(index)
        x, y, z, loss_weight = self.getitem_by_index(index)

        index = np.array(index)
        if self.aug_lib == 'tsaug':
            x = self.transform.augment(x)
            if self.n_duplicates > 1:
                rp = lambda _: np.repeat(_, self.n_duplicates, axis=0)
                y, z, loss_weight = rp(y), rp(z), rp(loss_weight)
            return x, y, z, index, loss_weight

        if self.aug_type == 'none':
            return x, y, z, index, loss_weight

        img_list, label_list, loc_list, idx_list, loss_weight_list = [], [], [], [], []

        x = x.T if x.ndim > 1 else x[..., np.newaxis]

        if self.n_views <= 1:
            x = self.to_tensor_transform(self.transform(x).T)
            return x, y, z, index, loss_weight

        for _ in range(self.n_views):
            img_list.append(self.to_tensor_transform(self.transform(x).T))
            loc_list.append(z)
            label_list.append(y)
            idx_list.append(index)
            loss_weight_list.append(loss_weight)
        return img_list, label_list, loc_list, idx_list, loss_weight_list

    @property
    def inv_pred(self):
        return self._inv_pred

    @inv_pred.setter
    def inv_pred(self, inv_pred):
        if inv_pred is not None:
            self._inv_pred = inv_pred
        else:
            self._inv_pred = None

    def get_neighbors(self, index):
        core_loc = self.core_id == self.core_id[index]
        dist = np.abs(self.roi_coors[core_loc] - self.roi_coors[index]).sum(axis=1)
        min_dist_loc = np.argsort(dist)[:self.n_neighbor]
        neighbor_index = self.index[core_loc][min_dist_loc]
        return np.append(index, neighbor_index)

    def cid2id(self, index):
        return self.ts_cid_to_id[self.core_id[index]][self.neighbors[index]]

    def cid2index(self, index):
        return self.ts_cid_to_index[self.core_id[index]][self.neighbors[index]]

    def getitem_by_index(self, index):
        x = self.data[index]  # if self.n_neighbor == 0 else self.data[self.get_neighbors(index)]
        y = self.label[index]
        z = self.location[index]

        # if (x.shape[0] > self.ts_len) or (self.ts_len < 0):
        #     start_idx = random.randint(0, x.shape[0] - self.ts_len)
        #     x = x[..., start_idx: start_idx + self.ts_len]

        # if self.inv_pred is not None:
        #     if isinstance(index, int):
        #         inv_pred = self.inv_pred[self.ts_id[index]]
        #     else:
        #         inv_pred = torch.tensor([self.inv_pred[_] for _ in self.ts_id[index]])
        #     loss_weight = 1 + torch.abs(inv_pred - self.inv[index])
        # else:
        #     loss_weight = np.repeat(1, x.shape[0]).astype('float32') if not isinstance(index, int) else 1
        loss_weight = np.repeat(1, x.shape[0]).astype('float32') if not isinstance(index, int) else 1
        return x, y, z, loss_weight

    def getitem_by_index_with_neighbors(self, index):
        x = self.data[index]
        x_neighbors = self.data[self.cid2index(index)]
        y = self.label[index]
        z = self.location[index]

        loss_weight = 1
        return x, x_neighbors, y, z, loss_weight

    def _filter(self, key, condition):
        return self.data_dict[key][condition]

    def make_dataset(self, condition=None):
        if condition is None:
            condition = (self.data_dict['inv'] >= self.initial_min_inv) + (self.data_dict['inv'] == 0.)
            # condition *= (self.data_dict['stn'] < self.stn_alpha)

            # Range outlier
            # clf = IsolationForest(n_estimators=100, warm_start=True)
            # X = np.vstack((self.data_dict['data'].max(1),
            #                self.data_dict['data'].min(1),
            #                self.data_dict['data'].max(1) - self.data_dict['data'].min(1))).T
            # condition *= clf.fit_predict(X).astype(condition.dtype)

        for k in self.data_dict.keys():
            setattr(self, k, self._filter(k, condition))

        # self.data = torch.tensor(self.data, dtype=torch.float32) if self.aug_type == 'none' else self.data
        self.label = torch.tensor(self.label, dtype=torch.long)
        self.inv = torch.tensor(self.inv, dtype=torch.float32)
        self.location = torch.tensor(self.location, dtype=torch.uint8)
        if self.transformer is not None:
            self.data, _ = robust_norm(self.data, self.transformer)
        self.index = np.arange(len(self.data))

        self.ts_cid_to_id = {cid: self.ts_id[self.core_id == cid] for cid in np.unique(self.core_id)}
        self.ts_cid_to_index = {cid: self.index[self.core_id == cid] for cid in np.unique(self.core_id)}
        # [self.ts_cid.extend(list(range(len(cid))) for cid in )]

    def correct_labels(self, ts_id, core_len, predictions, true_involvement, predicted_involvement, correcting_params):
        """
        Correcting labels if the true and predicted involvements are similar
        :param ts_id: same as 'ts_id' item, except grouped by core
        :param core_len:
        :param predictions:
        :param predicted_involvement:
        :param true_involvement: same as 'true_involvement' item, except grouped by core
        :param correcting_params:
        :return:
        """
        inv_dif_thr, prob_thr = correcting_params.inv_dif_thr, correcting_params.prob_thr
        for _, cl in zip(ts_id, core_len):
            assert len(_) == cl
        inv_dif = np.abs(np.subtract(predicted_involvement, true_involvement))
        inv_dif[np.array(true_involvement) == 0] = 1  # no correction for benign labels
        correcting_mask = np.array(inv_dif <= inv_dif_thr)
        correcting_mask = np.concatenate([np.repeat(_, cl) for (_, cl) in zip(correcting_mask, core_len)])
        print(correcting_mask.sum(), predictions.max(), predictions[correcting_mask].max())
        correcting_mask[predictions.max(1) < prob_thr] = False
        n_correct = correcting_mask.sum()

        if n_correct > 0:
            # add new time-series if available
            print(f'Cls_ratio: Old = {self.label.argmax(1).sum() / len(self.label):.3f}')

            # ts_id_corrected = np.concatenate(ts_id)[correcting_mask]
            # ts_id_updated = np.unique(np.concatenate([self.ts_id, ts_id_corrected]))
            # if not np.all(np.isin(ts_id_updated, np.sort(self.ts_id))):
            #     condition = np.isin(self.data_dict['ts_id'], ts_id_updated)
            #     # condition *= (self.data_dict['stn'] < self.stn_alpha)
            #
            #     print(f'{np.sum(condition) - len(self.ts_id)} new time-series added')
            #     self.make_dataset(condition)
            # else:
            #     condition = np.isin(self.data_dict['ts_id'], self.ts_id)  # use the current time-series IDs
            condition = np.isin(self.data_dict['ts_id'], self.ts_id)  # use the current time-series IDs
            # Assign new labels
            if self.last_updated_label is None:
                new_label = torch.tensor(self.data_dict['label'].argmax(1).copy()).long()
            else:
                new_label = self.last_updated_label.clone()
            new_label[correcting_mask] = torch.tensor(predictions.argmax(1)[correcting_mask]).long()

            self.label = F.one_hot(new_label[condition])
            self.last_updated_label = new_label.clone()
            self.label_corrected = True

            print(f'Cls_ratio: New = {self.label.argmax(1).sum() / len(self.label):.3f}')
            print(f'Correcting amount: {100 * n_correct / len(correcting_mask):.1f}%')
            print(np.unique(self.inv))
        else:
            self.label_corrected = False

    def __len__(self):
        return self.label.size(0)

    @staticmethod
    def estimate_inv(label, core_len):
        new_inv = []
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if np.ndim(label) == 2:
            label = label.argmax(1)
        cur_idx = 0
        for cl in core_len:
            new_inv.append(np.round(label[cur_idx: cur_idx + cl].sum() / cl, 2))
            cur_idx += cl
        return np.array(new_inv).T

    def __updateitem__(self, newones):
        'update labels'
        # Select sample
        # to_categorical(newones)
        newones.shape
        self.label = torch.tensor(newones, dtype=torch.uint8)

    def __updateinv__(self, newones):
        'update labels'
        # Select sample
        self.siginv = torch.tensor(newones, dtype=torch.uint8)


def show_signals(raw, strong, weak):
    import pylab as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plt.plot(raw.T)
    plt.plot(strong.T)
    plt.plot(weak.T)
    plt.show()
    plt.close()


class DatasetUnsupervised(Dataset):
    def __init__(self, data, cfg):
        """

        :param data:
        :param cfg:
        """
        self.data = data.astype('float32')
        # Parse configuration
        self.cfg = cfg
        ssl_alg_cfg = self.cfg[cfg.ssl_name]
        if cfg.ssl_name == 'pl':
            transforms = [ssl_alg_cfg.aug_type_strong, ssl_alg_cfg.aug_type_weak]
        else:
            transforms = [ssl_alg_cfg.aug_type, ssl_alg_cfg.aug_type]

        transform_a, self.to_tensor_transform = get_transform(
            transforms[0] if isinstance(transforms[0], list) else [transforms[0], ],
            prob=ssl_alg_cfg.transform_prob[0], degree=3,
        )
        transform_b, _ = get_transform(
            transforms[1] if isinstance(transforms[1], list) else [transforms[1], ],
            prob=ssl_alg_cfg.transform_prob[1]
        )
        self.unsup_transform = [transform_a, transform_b]

    def __getitem__(self, index):
        x_unsup = self.data[index]
        aug_a = self.to_tensor_transform(self.unsup_transform[0](x_unsup[..., np.newaxis]).squeeze())
        aug_b = self.to_tensor_transform(self.unsup_transform[1](x_unsup[..., np.newaxis]).squeeze())
        return aug_a, aug_b

    def __len__(self):
        return self.data.shape[0]


def remove_empty_data(input_data, set_name, p_thr=.2):
    """

    :param input_data:
    :param set_name:
    :param p_thr: threshold of zero-percentage
    :return:
    """
    data = input_data[f"data_{set_name}"]
    s = [(data[i] == 0).sum() for i in range(len(data))]
    zero_percentage = [s[i] / np.prod(data[i].shape) for i in range(len(data))]
    include_idx = np.array([i for i, p in enumerate(zero_percentage) if p < p_thr])
    if len(include_idx) == 0:
        return input_data

    for k in input_data.keys():
        if set_name in k:
            if isinstance(input_data[k], list):
                input_data[k] = [_ for i, _ in enumerate(input_data[k]) if i in include_idx]
            else:
                input_data[k] = input_data[k][include_idx]
    return input_data


def norm_01(x):
    return (x - x.min(axis=1, keepdims=True)) / (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))


def stratify_groups(groups, num_time_series, marked_array, mark_low_threshold=.2):
    """Get a number of time-series within each group"""
    row_idx = []
    group_unique = np.unique(groups)
    for g in group_unique:
        # Find the time series in group g & mark off those already selected in previous iterations
        is_group = groups == g
        is_group_marked = is_group * marked_array

        # Reset marked array if 80% of the time series in the current group have been selected
        if (is_group_marked.sum() / is_group.sum()) < mark_low_threshold:
            is_group_marked = is_group
            marked_array[is_group] = True

        # Randomly selected time-series within those of the current group
        replace = True if np.sum(is_group_marked) < num_time_series else False
        row_idx.append(np.random.choice(np.where(is_group_marked)[0], num_time_series, replace=replace))
        # print(g, (sum(is_group_marked) / sum(is_group)))

        marked_array[row_idx[-1]] = False
    return np.concatenate(row_idx), marked_array


def normalize(input_data, set_name):
    for i, x in enumerate(input_data[f'data_{set_name}']):
        input_data[f'data_{set_name}'][i] = norm_01(x.astype('float32'))
    return input_data


def encode_unique_id(input_data):
    """Add time-series - wise unique ID"""
    current_idx = 0
    set_len = [0, ]
    for set_name in ['train', 'val', 'test']:
        input_data[f'ts_id_{set_name}'] = []
        input_data[f'core_id_{set_name}'] = []
        set_len.append(len(input_data[f'data_{set_name}']))
        for i, d in enumerate(input_data[f'data_{set_name}']):
            l = len(d)
            input_data[f'ts_id_{set_name}'].append(np.arange(current_idx, current_idx + l))
            input_data[f'core_id_{set_name}'].append(i + np.sum(set_len[:-1]))
            current_idx += l
    return input_data


def encode_location(input_data):
    switcher = {
        "RB": [1, 0, 0, 0, 0, 0, 0, 0],
        "RBL": [1, 0, 0, 0, 0, 0, 0, 0],
        "RBM": [1, 0, 0, 0, 0, 0, 0, 0],
        "RML": [0, 1, 0, 0, 0, 0, 0, 0],
        "RMM": [0, 0, 1, 0, 0, 0, 0, 0],
        "RA": [0, 0, 0, 1, 0, 0, 0, 0],
        "RAL": [0, 0, 0, 1, 0, 0, 0, 0],
        "RAM": [0, 0, 0, 1, 0, 0, 0, 0],
        "LB": [0, 0, 0, 0, 1, 0, 0, 0],
        "LBL": [0, 0, 0, 0, 1, 0, 0, 0],
        "LBM": [0, 0, 0, 0, 1, 0, 0, 0],
        "LML": [0, 0, 0, 0, 0, 1, 0, 0],
        "LMM": [0, 0, 0, 0, 0, 0, 1, 0],
        "LA": [0, 0, 0, 0, 0, 0, 0, 1],
        "LAL": [0, 0, 0, 0, 0, 0, 0, 1],
        "LAM": [0, 0, 0, 0, 0, 0, 0, 1]
    }
    for set_name in ['train', 'val', 'test']:
        loc = [switcher[_] for _ in input_data[f'loc_{set_name}']]
        input_data[f'corename_{set_name}'] = np.array(loc)
    return input_data


def preprocess(input_data, p_thr=.2, to_norm=False):
    """
    Remove data points which have percentage of zeros greater than p_thr
    :param input_data:
    :param p_thr:
    :param to_norm:
    :return:
    """
    for set_name in ['train', 'val', 'test']:
        input_data = remove_empty_data(input_data, set_name, p_thr)
        if to_norm:
            input_data = normalize(input_data, set_name)
    return encode_unique_id(encode_location(input_data))
    # return encode_unique_id(input_data)


def concat_data(included_idx, data, label, core_name, inv, core_id, patient_id, ts_id, roi_coors,
                # neighbors
                ):  # stn
    """ Concatenate data from different cores specified by 'included_idx' """
    core_counter = 0
    data_c, label_c, core_name_c, inv_c, core_len, core_id_c, patient_id_c, ts_id_c, roi_coors_c, = \
        [[] for _ in range(9)]
    neighbors_c = []
    for i in range(len(data)):
        if included_idx[i]:
            data_c.append(data[i])
            label_c.append(np.repeat(label[i], data[i].shape[0]))
            temp = np.tile(core_name[i], data[i].shape[0])
            core_name_c.append(temp.reshape((data[i].shape[0], 8)))
            inv_c.append(np.repeat(inv[i], data[i].shape[0]))
            core_id_c.append(np.repeat(core_id[i], data[i].shape[0]))
            patient_id_c.append(np.repeat(patient_id[i], data[i].shape[0]))
            core_len.append(data[i].shape[0])
            ts_id_c.append(ts_id[i])
            # roi_coors_c.append(roi_coors[i])
            # stn_c.append(stn[i])
            # neighbors_c.append(neighbors[i])
            core_counter += 1
    # roi_coors_c = np.concatenate(roi_coors_c)
    roi_coors_c = None
    data_c = np.concatenate(data_c).astype('float32')
    label_c = to_categorical(np.concatenate(label_c))
    core_name_c = np.concatenate(core_name_c)
    inv_c = np.concatenate(inv_c)
    core_id_c = np.concatenate(core_id_c)
    patient_id_c = np.concatenate(patient_id_c)
    ts_id_c = np.concatenate(ts_id_c)
    # stn_c = np.concatenate(stn_c)
    # neighbors_c = np.concatenate(neighbors_c)

    return data_c, label_c, core_name_c, inv_c, core_id_c, patient_id_c, ts_id_c, core_len, roi_coors_c,  # neighbors_c  # stn_c


def split_train_val(input_data, random_state=0, val_size=.4, verbose=False):
    gs = list(input_data["GS_train"])
    pid = list(input_data["PatientId_train"])
    inv = list(input_data["inv_train"])

    df1 = pd.DataFrame({'pid': pid, 'gs': gs, 'inv': inv})
    df1 = df1.assign(gs_merge=df1.gs.replace({'-': 'Benign',
                                              '3+3': 'G3', '3+4': 'G3', '4+3': 'G4',
                                              '4+4': 'G4', '4+5': 'G4', '5+4': 'G4'}))
    df1 = df1.assign(condition=df1.gs_merge.replace({'-': 'Benign', 'G3': 'Cancer', 'G4': 'Cancer'}))
    df1.gs.replace({'-': 'Benign'}, inplace=True)

    train_inds, test_inds = next(GroupShuffleSplit(test_size=val_size, n_splits=2,
                                                   random_state=random_state).split(df1, groups=df1['pid']))
    df1 = df1.assign(group='train')
    df1.loc[test_inds, 'group'] = 'val'
    df1 = df1.sort_values(by='pid')

    pid_tv = {
        'train': df1[df1.group == 'train'].pid.unique(),
        'val': df1[df1.group == 'val'].pid.unique(),
    }

    keys = [f[:-4] for f in input_data.keys() if 'val' in f]
    merge = {}
    for k in keys:
        merge[k] = deepcopy(input_data[f'{k}_train'])

    # Initialize the new input_data
    target = {}
    for set_name in ['train', 'val']:
        for k in keys:
            target[f'{k}_{set_name}'] = []
    # Re-split data into two sets based on randomized patient ID
    for i, pid in enumerate(merge['PatientId']):
        for set_name in ['train', 'val']:
            for k in keys:
                k_target = f'{k}_{set_name}'
                if pid in pid_tv[set_name]:
                    target[k_target].append(merge[k][i])
    # Assign to original input data after finishing creating a new one
    for set_name in ['train', 'val']:
        for k in keys:
            k_target = f'{k}_{set_name}'
            input_data[k_target] = target[k_target]
            if isinstance(merge[k], np.ndarray):
                input_data[k_target] = np.array(target[k_target]).astype(merge[k].dtype)

    if verbose:
        for set_name in ['train', 'val']:
            for k in keys:
                print(k, len(input_data[f'{k}_{set_name}']))

    return input_data


def add_suf(name, suffix):
    return f'{name}_{suffix}'


def filter_input_data(input_data, set_name, min_inv):
    inv = input_data[add_suf('inv', set_name)]
    included_idx = [True for _ in inv]
    included_idx = [False if (inv < min_inv) and (inv > 0) else tr_idx for inv, tr_idx in zip(inv, included_idx)]
    included_idx = np.argwhere(included_idx).T[0]
    keys = [k for k in input_data if set_name in k]
    for k in keys:
        tmp = [input_data[k][i] for i in included_idx]
        input_data[k] = tmp if not isinstance(input_data[k], np.ndarray) else np.array(tmp).astype(input_data[k])
    return input_data


def extract_subset(input_data, set_name, min_included_inv=.4, to_concat=True, core_list=None):
    """
    Extract subset 'set_name' from input_data using 'min_inv'
    :param input_data: loaded from pkl or matlab file
    :param set_name: 'train', 'val', or 'test'
    :param min_included_inv: minimum involvement for keeping data in the training set
    :param to_concat: concat selected data
    :param core_list: ID of included cores
    :return:
    """
    add_suf = lambda x: f'{x}_{set_name}'
    data = input_data[add_suf('data')]
    # neighbors = input_data[add_suf('neighbors')]
    inv = input_data[add_suf('inv')]
    label = (inv > 0).astype('uint8')
    core_name = input_data[add_suf('corename')].astype(np.float)
    core_id = input_data[add_suf('core_id')]
    patient_id = input_data[add_suf('PatientId')]
    ts_id = input_data[add_suf('ts_id')]
    roi_coors = [np.array(_).T for _ in input_data[add_suf('roi_coors')]]
    # roi_coors = ts_id.copy()
    # stn = deepcopy(ts_id)
    # stn = input_data[add_suf('stn')]

    unused = np.zeros(len(input_data[f'data_{set_name}']))
    if f'unused_{set_name}' in input_data.keys():
        unused = np.array(input_data[f'unused_{set_name}'])

    included_idx = [True for _ in label]
    included_idx = [False if (inv < min_included_inv) and (inv > 0) else tr_idx for inv, tr_idx in
                    zip(inv, included_idx)]

    included_idx = [False if _unused == 1 else _included_idx for (_unused, _included_idx) in zip(unused, included_idx)]

    if core_list is not None:  # filter by core-id
        included_idx = np.bitwise_and(included_idx, np.isin(core_id, core_list))
    if to_concat:
        return concat_data(included_idx, data, label, core_name, inv, core_id, patient_id, ts_id, roi_coors,
                           # neighbors
                           )  # stn
    else:
        core_len = [len(_) for _ in data]
        included_idx = np.argwhere(included_idx).T[0]
        get_included = lambda x: [x[i] for i in included_idx]
        for v in ['data', 'label', 'core_name', 'inv', 'core_id', 'patient_id', 'ts_id', 'core_len',
                  'roi_coors']:  # 'stn'
            eval(f'{v} = get_included({v})')
        return data, label, core_name, inv, core_id, patient_id, ts_id, core_len, roi_coors,  # stn


def create_datasets_v1(data_file, norm=True, min_inv=0.4, aug_type='none', n_views=2,
                       ssl_cfg=None, strategy='none', split_random_state=-1, val_size=.4, transformer=None,
                       get_transformer_only=False, aug_lib='self_time', initial_min_inv=.7, sweep=False,
                       n_neighbor=0, core_list='none', ts_len=200, input_channels=-1, channel_len=25,
                       return_train_ds=False, to_merge_train_val=False):
    """Create training, validation and test sets"""
    if split_random_state >= 0:
        assert 0 < val_size < 1
    if '.mat' in data_file:
        input_data = encode_unique_id(mat73.loadmat(data_file))
    else:
        input_data = load_pickle(data_file)
        # stn = load_pickle('/'.join((os.path.dirname(data_file), 'stationary.pkl')))
        # for set_name in ['train', 'val', 'test']:
        #     input_data[f'stn_{set_name}'] = stn[set_name]
        # print(len(input_data['data_train']))
        input_data = preprocess(input_data)

    if split_random_state >= 0:
        if to_merge_train_val:
            input_data = merge_train_val(input_data)[0]
        else:
            input_data = merge_split_train_val(input_data, random_state=split_random_state, val_size=val_size)
        # if sweep:
        #     input_data = split_train_val(input_data, random_state=split_random_state, val_size=val_size)
    core_list = np.loadtxt(core_list).astype('int').tolist() if core_list != 'none' else None

    # for set_name in ['train', 'val', 'test']:
    #     patches = coor2patches_whole_input(input_data, set_name)
    # exit()

    # if norm or get_transformer_only:
    # for set_name in ['train', 'val', 'test']:
    #     input_data[f'data_{set_name}'] = robust_norm_per_core(input_data[f'data_{set_name}'])

    trn_ds = DatasetV1(*extract_subset(input_data, 'train', min_inv, core_list=core_list),
                       initial_min_inv=initial_min_inv, transform_prob=.2, degree=1, aug_type=aug_type,
                       n_views=n_views, aug_lib=aug_lib, n_neighbor=n_neighbor, ts_len=ts_len,
                       input_channels=input_channels, channel_len=channel_len)

    if norm or get_transformer_only:
        transformer = robust_norm(np.concatenate(trn_ds.data), transformer)[1]
        if get_transformer_only:
            return transformer
        trn_ds.data, transformer = robust_norm(trn_ds.data, transformer)
        trn_ds.transformer = transformer

    if return_train_ds:
        return trn_ds

    if strategy == 'none':
        trn_unsup_ds = None
    else:
        unsup_data = np.concatenate(
            load_pickle(data_file.replace('.pkl', f'{ssl_cfg.unsup_set_suffix}.pkl'))["data_train"])
        unsup_data = robust_norm(unsup_data, transformer)[0] if transformer else unsup_data
        trn_unsup_ds = DatasetUnsupervised(unsup_data, ssl_cfg)

    test_sets = []
    for set_name in ['train', 'val', 'test']:
        test_sets.append(create_datasets_test(
            None, min_inv=min_inv, set_name=set_name, norm=norm, input_data=input_data,
            transformer=transformer, ts_len=ts_len, input_channels=input_channels)
        )
    train_set, val_set, test_set = test_sets

    return trn_ds, trn_unsup_ds, train_set, val_set, test_set


def create_datasets_v2(data_file, norm=True, min_inv=0.4, aug_type='none', n_views=2,
                       ssl_cfg=None, strategy='none', split_random_state=-1, val_size=.4, transformer=None,
                       get_transformer_only=False, aug_lib='self_time', initial_min_inv=.7, sweep=False,
                       n_neighbor=0, core_list='none', ts_len=200, input_channels=-1, channel_len=25,
                       return_train_ds=False):
    """Create training and test sets"""
    if split_random_state >= 0:
        assert 0 < val_size < 1
    if '.mat' in data_file:
        input_data = encode_unique_id(mat73.loadmat(data_file))
    else:
        input_data = load_pickle(data_file)
        input_data = preprocess(input_data)
        input_data = add_neighbors_to_input_data(input_data, data_file)

    if split_random_state >= 0:
        input_data = merge_split(input_data, random_state=split_random_state, val_size=val_size)
    core_list = np.loadtxt(core_list).astype('int').tolist() if core_list != 'none' else None

    # from scipy.signal import correlate, correlation_lags
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import pylab as plt
    # plt.close()
    # coef = np.corrcoef(input_data['data_train'][0])
    # m = (coef.ravel() ** 2) > (np.percentile(coef.ravel() ** 2, 99))
    # plt.imshow(coef ** 2 * (m.reshape(coef.shape)), cmap='jet', vmin=0, vmax=1)
    # plt.show()

    trn_ds = DatasetV1(*extract_subset(input_data, 'train', min_inv, core_list=core_list),
                       initial_min_inv=initial_min_inv, transform_prob=.2, degree=1, aug_type=aug_type,
                       n_views=n_views, aug_lib=aug_lib, n_neighbor=n_neighbor, ts_len=ts_len,
                       input_channels=input_channels, channel_len=channel_len)

    if norm or get_transformer_only:
        transformer = robust_norm(np.concatenate(trn_ds.data), transformer)[1]
        if get_transformer_only:
            return transformer
        trn_ds.data, transformer = robust_norm(trn_ds.data, transformer)
        trn_ds.transformer = transformer

    if return_train_ds:
        return trn_ds

    if strategy == 'none':
        trn_unsup_ds = None
    else:
        unsup_data = np.concatenate(
            load_pickle(data_file.replace('.pkl', f'{ssl_cfg.unsup_set_suffix}.pkl'))["data_train"])
        unsup_data = robust_norm(unsup_data, transformer)[0] if transformer else unsup_data
        trn_unsup_ds = DatasetUnsupervised(unsup_data, ssl_cfg)

    test_sets = []
    for set_name in ['train', 'test']:
        test_sets.append(create_datasets_test(
            None, min_inv=min_inv, state=set_name, norm=norm, input_data=input_data,
            transformer=transformer, ts_len=ts_len, input_channels=input_channels)
        )
    train_set, test_set = test_sets

    return trn_ds, trn_unsup_ds, train_set, test_set


class DatasetV2(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    from typing import Union

    def __init__(self, data, label, location, inv, groups,
                 aug_type: Union[list, str] = ('G2',), n_views=1, transform_prob=.2,
                 time_series_per_group=16):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        # from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, Convolve

        self.data = torch.tensor(data, dtype=torch.float32) if aug_type == 'none' else data
        self.label = torch.tensor(label, dtype=torch.uint8)
        self.location = torch.tensor(location, dtype=torch.uint8)
        self.n_views = n_views
        self.aug_type = aug_type
        self.inv_pred = None
        self.transform, self.to_tensor_transform = get_transform(
            aug_type if isinstance(aug_type, list) else [aug_type, ],
            prob=transform_prob,
        )
        # self.transform = (
        #         TimeWarp() * 1  # random time warping 5 times in parallel
        #         # + Crop(size=170)  # random crop subsequences with length 300
        #         + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
        #         + Reverse() @ 0.5  # with 50% probability, reverse the sequence
        #         + Drift(max_drift=(0.1, 0.5)) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
        #         + Convolve(window='flattop', size=11, prob=.25)
        # )
        self.inv = torch.tensor(inv, dtype=torch.float32)
        self.time_series_per_group = time_series_per_group
        self.groups = groups
        self.marked_array = [np.ones((d.shape[0]), dtype='bool') for d in self.data]

    @property
    def inv_pred(self):
        return self._inv_pred

    @inv_pred.setter
    def inv_pred(self, inv_pred):
        if inv_pred is not None:
            self._inv_pred = torch.tensor(inv_pred, dtype=torch.float32)
        else:
            self._inv_pred = None

    def __getitem__(self, index):
        row_idx, self.marked_array[index] = stratify_groups(self.groups[index], self.time_series_per_group,
                                                            self.marked_array[index])
        x = self.data[index][row_idx]
        y = self.label[index]
        z = self.location[index]
        if self.inv_pred is None:
            loss_weight = 1
        else:
            loss_weight = torch.exp(torch.abs(self.inv_pred[index] - self.inv[index])).item()
            # loss_weight *= 2 if y == 0 else 1

        if self.aug_type == 'none':
            return x, y, z, index

        img_list, label_list, loc_list, idx_list, loss_weight_list = [], [], [], [], []
        if self.transform is not None:
            # img_list = list(self.to_tensor_transform(self.transform.augment(x)))  # .unsqueeze(0)
            # for _x in x:
            img_list = list(x)
            for i in range(x.shape[0]):
                for _ in range(self.n_views):
                    # img_transformed = self.transform(_x[..., np.newaxis]).squeeze()
                    # img_list.append(self.to_tensor_transform(img_transformed))  # .unsqueeze(0)
                    # img_list.append(_x)
                    loc_list.append(z)
                    label_list.append(y)
                    idx_list.append(index)
                    loss_weight_list.append(loss_weight)
        return img_list, label_list, loc_list, idx_list, loss_weight_list

    def __len__(self):
        return self.label.size(0)

    def __updateitem__(self, newones):
        'update labels'
        # Select sample
        # to_categorical(newones)
        newones.shape
        self.label = torch.tensor(newones, dtype=torch.uint8)

    def __updateinv__(self, newones):
        'update labels'
        # Select sample
        self.siginv = torch.tensor(newones, dtype=torch.uint8)


def create_datasets_v2_unused(data_file, norm=None, min_inv=0.4, aug_type='none', n_views=4, to_norm=False):
    input_data = load_pickle(data_file)
    # unsup_data = load_pickle(data_file.replace('mimic', 'unsup'))
    input_data = preprocess(input_data, to_norm=to_norm)

    data_train = input_data["data_train"]
    inv_train = input_data["inv_train"]
    label_train = to_categorical((inv_train > 0).astype('uint8'))
    CoreN_train = input_data["corename_train"].astype(np.float)
    groups_train = load_pickle(data_file.replace('.pkl', '_groups.pkl'))['nc30']

    included_idx = [False if ((inv < min_inv) and (inv > 0)) else True for inv in inv_train]

    # Filter unwanted cores
    label_train = label_train[included_idx]
    inv_train = inv_train[included_idx]
    CoreN_train = CoreN_train[included_idx]
    data_train = [data_train[i].astype('float32') for i, included in enumerate(included_idx) if included]
    groups_train = [groups_train[i] for i, included in enumerate(included_idx) if included]

    # Create training dataset
    trn_ds = DatasetV2(data_train, label_train, CoreN_train, inv_train, groups=groups_train,
                       aug_type=aug_type, n_views=n_views)  # ['magnitude_warp', 'time_warp'])

    train_set = create_datasets_test(None, min_inv=0.4, state='train', norm=norm, input_data=input_data,
                                     train_stats=None)
    val_set = create_datasets_test(None, min_inv=0.4, state='val', norm=norm, input_data=input_data,
                                   train_stats=None)
    test_set = create_datasets_test(None, min_inv=0.4, state='test', norm=norm, input_data=input_data,
                                    train_stats=None)

    return trn_ds, train_set, val_set, test_set


def create_datasets_test(data_file, set_name, norm=True, min_inv=0.4, input_data=None,
                         return_array=False, transformer=None, split_random_state=-1, val_size=.3,
                         ts_len=200, input_channels=1, channel_len=25,
                         ):
    """"""
    if input_data is None:
        input_data = load_pickle(data_file)
        input_data = preprocess(input_data, to_norm=False)

    if split_random_state >= 0:
        input_data = merge_split(input_data, random_state=split_random_state, val_size=val_size)

    data_test = input_data["data_" + set_name]  # [0]
    label_inv = input_data["inv_" + set_name]  # [0]
    roi_coors_test = label_inv.copy()  # input_data['roi_coors_' + state]  # temporary
    label_test = (label_inv > 0).astype('uint8')
    CoreN_test = input_data["corename_" + set_name]
    patient_id_bk = input_data["PatientId_" + set_name]  # [0]
    involvement_bk = input_data["inv_" + set_name]  # [0]
    gs_bk = np.array(input_data["GS_" + set_name])  # [0]
    ts_id = input_data["ts_id_" + set_name]
    c_id = input_data["core_id_" + set_name]
    # neighbors = input_data["neighbors_" + state]

    assert data_test[0].shape[1] >= ts_len  # Check if the input data contain sufficient required time-series points

    unused = np.zeros(len(input_data[f'data_{set_name}']))
    if f'unused_{set_name}' in input_data.keys():
        unused = np.array(input_data[f'unused_{set_name}'])

    included_idx = [False if ((inv < min_inv) and (inv > 0)) else True for inv in label_inv]
    included_idx = [False if _unused == 1 else _included_idx for (_unused, _included_idx) in zip(unused, included_idx)]

    corelen, target_test, name_tst, bags_test, sig_inv_test, bags_ts_id, bags_c_id = [], [], [], [], [], [], []
    # neighbors_test = []
    for i in range(len(data_test)):
        if included_idx[i]:
            bags_test.append(data_test[i])
            target_test.append(np.repeat(label_test[i], data_test[i].shape[0]))
            # temp=np.repeat(onehot_corename(CoreN_test[i]),data_test[i].shape[0])
            temp = np.tile(CoreN_test[i], data_test[i].shape[0])
            name_tst.append(temp.reshape((data_test[i].shape[0], 8)))
            corelen.append(data_test[i].shape[0])
            sig_inv_test.append(np.repeat(label_inv[i], data_test[i].shape[0]))
            bags_ts_id.append(ts_id[i])
            bags_c_id.append(c_id[i])
            # neighbors_test.append(neighbors[i])

    signal_test = np.concatenate(bags_test)
    target_test = np.concatenate(target_test)
    sig_inv_test = np.concatenate(sig_inv_test)
    name_tst = np.concatenate(name_tst)
    # neighbors_test = np.concatenate(neighbors_test)
    ts_cid = []
    [ts_cid.extend(list(range(cl))) for cl in corelen]
    index = np.arange(len(ts_cid))

    roi_coors_test = [roi_coors_test[i] for i in range(len(roi_coors_test)) if included_idx[i] == 1]
    patient_id_bk = patient_id_bk[included_idx]
    gs_bk = gs_bk[included_idx]
    label_inv = label_inv[included_idx].astype('float')

    if norm:
        if transformer is None:
            transformer = robust_norm(np.concatenate(input_data['data_train']), transformer)[1]
        signal_test = robust_norm(signal_test, transformer)[0]

    if return_array:
        return signal_test, corelen, label_inv, patient_id_bk, gs_bk

    label_test = to_categorical(target_test)

    assert (not (signal_test.shape[1] % channel_len))
    # The length of the time-series must be divisible by the number of input channels
    if input_channels > 1:
        signal_test = signal_test.reshape((signal_test.shape[0], -1, channel_len))  # [..., :ts_len]
        signal_test = signal_test[:, :input_channels]
    else:
        signal_test = np.expand_dims(signal_test, 1)  # [..., :ts_len]

    tst_ds = TensorDataset(torch.tensor(signal_test, dtype=torch.float32),
                           torch.tensor(label_test, dtype=torch.uint8),
                           torch.tensor(name_tst, dtype=torch.uint8),
                           torch.tensor(sig_inv_test, dtype=torch.float32).unsqueeze(1))
    return list((tst_ds, corelen, label_inv, patient_id_bk, gs_bk, roi_coors_test, bags_ts_id, bags_c_id))


def _preprocess(x_train):
    # Normalize
    x_train_max = np.max(x_train)
    x_train_min = np.min(x_train)
    x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
    # Test is secret
    # x_val = 2. * (x_val - x_train_min) / (x_train_max - x_train_min) - 1.
    # x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
    return x_train, (x_train_min, x_train_max)


def add_neighbors_to_input_data(input_data, data_file):
    """

    :param input_data: input_data after preprocessed
    :param data_file:
    :return:
    """
    # load "neighbors" (created from preprocessed input_data)
    neighbors = load_pickle(data_file.replace('.pkl', '_neighbors.pkl'))

    if neighbors is not None:
        for set_name in ['train', 'val', 'test']:
            input_data[f'neighbors_{set_name}'] = neighbors[set_name]
    return input_data
