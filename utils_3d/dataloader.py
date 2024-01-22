import os
import numpy as np
# from tsai.data.transforms import *
from torch.utils.data import DataLoader

from .dataset import NumpyDataset, NumpyDatasetExtracted
# from utils import to_categorical, create_loader  # get_transform
from utils import to_categorical, create_loader


def gen_data_loader(opt, input_set, self_train=False, drop_last=False, return_info=False, get_train_only=False,
                    get_eval_only=False, stats=(None, None), re_extract=False, balance=True):
    """

    :param opt: from argparse in misc.py and yaml files
    :param input_set:
    :param self_train:
    :param drop_last:
    :param return_info:
    :param get_train_only:
    :param get_eval_only:
    :param re_extract:
    :param stats: [train_mean, train_std]
    :return:
    """
    assert not (get_eval_only and get_train_only)

    def filter_by_inv(_list, inv, inv_thr):
        return np.array([_ for (_, _inv) in zip(_list, inv) if ((_inv >= inv_thr) or (_inv == 0))])

    def filter_by_gs(_list, gs, remove):
        return np.array([_ for (_, _gs) in zip(_list, gs) if ((str(_gs) != remove) or (str(_gs) == '-'))])

    def verbose():
        print(len(files),
              f'{len(trn_idx)} ({label[trn_idx].argmax(1).sum()})',
              f'{len(val_idx)} ({label[val_idx].argmax(1).sum()})',
              f'{len(test_idx)} ({label[test_idx].argmax(1).sum()})')

    def stratify_patient(patient_id, num_consecutive=(3, 1, 1)):
        set_idx = {'train': [], 'val': [], 'test': []}
        patient_id = list(np.unique(patient_id))
        i = 0
        while i <= len(patient_id):
            set_idx['train'].extend(patient_id[i:i + num_consecutive[0]])
            set_idx['val'].extend(patient_id[i + sum(num_consecutive[:1]):i + sum(num_consecutive[:2])])
            set_idx['test'].extend(patient_id[i + sum(num_consecutive[:2]):i + sum(num_consecutive[:3])])
            i += sum(num_consecutive)
        return set_idx

    def show_set_dist():
        import pylab as plt
        inv = target
        _, ax = plt.subplots(1, 3)
        ax[0].hist(inv[trn_idx])
        ax[1].hist(inv[val_idx])
        ax[2].hist(inv[test_idx])
        plt.show()
        exit()

    def ds2dl(idx):
        ds = NumpyDataset(files[idx], label[idx], ts_len=opt.ts_len, patch_info=(num_patches, patch_size),
                          input_dim=opt.input_dim, norm=opt.normalize_input, metadata=metadata[idx], stats=stats,
                          re_extract=re_extract, ts_start=opt.ts_start)
        dl = DataLoader(ds, opt.test_batch_size, num_workers=opt.num_workers, pin_memory=False)
        return dl

    num_patches, patch_size = input_set[opt.input_set_idx]
    opt.input_channels = num_patches
    dir_dataset = opt.data_source.data_root
    files = np.load(os.path.join(dir_dataset, f'input_{num_patches}_{patch_size}.npy'), allow_pickle=True)
    metadata = np.load(os.path.join(dir_dataset, f'metadata_{num_patches}_{patch_size}.npy'), allow_pickle=True)
    files, metadata = [filter_by_inv(_, [md['Involvement'] for md in metadata], opt.initial_min_inv) for _ in
                       (files, metadata)]
    files, metadata = [filter_by_gs(_, [md['GleasonScore'] for md in metadata], '6') for _ in
                       (files, metadata)]

    pid = np.array([md['PatientId'] for md in metadata])
    label = to_categorical(np.array([md['TrueLabel'] for md in metadata]))
    # inv = np.array([md['Involvement'] for md in metadata])

    if isinstance(opt.split_strategy, list):
        set_idx = stratify_patient(pid, num_consecutive=opt.split_strategy)
        trn_idx = np.where(np.isin(pid, set_idx['train']))[0]
        val_idx = np.where(np.isin(pid, set_idx['val']))[0]
        test_idx = np.where(np.isin(pid, set_idx['test']))[0]
    elif opt.split_strategy in ['by_inv', 'by_gs']:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=opt.seed)
        if 'gs' in opt.split_strategy:
            target = np.array([md['PrimarySecondary'] for md in metadata])
        else:
            target = np.array([md['Involvement'] for md in metadata])
        trn_idx, test_idx = next(gss.split(files, target, pid))
        # gss = GroupShuffleSplit(n_splits=1, train_size=.9, random_state=opt.seed)
        trn_idx, val_idx = next(gss.split(files[trn_idx], target[trn_idx], pid[trn_idx]))

    elif opt.split_strategy == 'by_id':
        trn_idx = np.where(pid <= 100)[0]
        val_idx = np.where((pid > 100) & (pid <= 130))[0]
        test_idx = np.where(pid > 130)[0]
    else:
        raise NotImplementedError

    if opt.paths.extract_root is None:
        trn_ds = NumpyDataset(files[trn_idx], label[trn_idx], random_crop_ts=True, ts_len=opt.ts_len,
                              input_dim=opt.input_dim,
                              augment=True,
                              # augment=False,
                              drop_range=(0.0, 0.3),  # 0.3),
                              norm=opt.normalize_input,
                              transform=None,
                              patch_info=(num_patches, patch_size),
                              is_train=True, metadata=metadata[trn_idx],
                              min_inv=opt.min_inv,
                              max_inv=1,
                              shift_range=opt.shift_range,
                              n_views=opt.n_views,
                              use_anchor=opt.use_anchor,
                              # transform=get_transform(['window_warp']),
                              grid=opt.grid,
                              self_train=self_train,
                              return_info=return_info,
                              exclude_benign=opt.exclude_benign,
                              stats=stats,
                              )
    elif not get_eval_only:
        from functools import partial
        # TS_tfms = [
        #     TSIdentity,
        #     TSMagAddNoise,
        #     (TSMagScale, .02, .2),
        #     (partial(TSMagWarp, ex=0), .02, .2),
        #     (partial(TSTimeWarp, ex=[0, 1, 2]), .02, .2),
        # ]
        # transform = RandAugment(TS_tfms, N=3, M=5)
        # transform = RandAugment([TSSmooth, TSHorizontalFlip, TSVerticalFlip, TSTranslateX, TSRandomShift, ],
        #                         N=2, M=3)
        transform = None
        trn_ds = NumpyDatasetExtracted(opt.paths.extract_root,
                                       opt.extract_dirs,
                                       opt.ts_len,
                                       transform=transform,
                                       # transform=get_transform(['window_warp']),
                                       patch_info=(num_patches, patch_size),
                                       min_inv=opt.min_inv, max_inv=1.,
                                       dif_learning=opt.dif_learning,
                                       cut_mix_1d=opt.cut_mix_1d,
                                       rand_erase=opt.random_erase,
                                       exclude_benign=opt.exclude_benign,
                                       ts_start=opt.ts_start,
                                       stats=stats,
                                       data_suffix='_miv.4_v2',  # '_shuffled'
                                       norm=True,
                                       )
        stats = trn_ds.stats

    # if not self_train:
    #     sampler = create_loader(trn_ds, opt.train_batch_size, jobs=opt.num_workers,
    #                             add_sampler=True,
    #                             get_sampler_only=True, weight_by_inv=False)
    # else:
    #     sampler = None

    trn_dl = make_train_loader(trn_ds, opt, drop_last=drop_last, balance=balance) if not get_eval_only \
        else None

    if self_train or get_train_only:
        return trn_dl

    trn_dl_eval = ds2dl(trn_idx)
    val_dl = ds2dl(val_idx)
    test_dl = ds2dl(test_idx)

    return trn_dl, trn_dl_eval, val_dl, test_dl


def make_train_loader(trn_ds, opt, drop_last=True, balance=True):
    if trn_ds.exclude_benign or not balance:
        sampler = None
    else:
        sampler = create_loader(trn_ds, opt.train_batch_size, jobs=opt.num_workers, add_sampler=True,
                                get_sampler_only=True, weight_by_inv=False)
    # sampler = None
    shuffle = True if not sampler else None
    trn_dl = DataLoader(trn_ds, opt.train_batch_size, num_workers=opt.num_workers, sampler=sampler, drop_last=drop_last,
                        pin_memory=False, shuffle=shuffle)
    return trn_dl
