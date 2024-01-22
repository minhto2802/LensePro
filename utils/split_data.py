import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def positive_percentage(label):
    return np.sum(label > 0) / len(label)


def compare_positive_percentage(label, train_ind, test_ind):
    train_pos_perc = positive_percentage(label[train_ind])
    test_pos_perc = positive_percentage(label[test_ind])
    return train_pos_perc, test_pos_perc


def is_valid_split(split, label, max_dif=.05, min_test_len=50):
    train_ind, test_ind = split
    train_pos_perc, test_pos_perc = compare_positive_percentage(label, train_ind, test_ind)
    if (np.abs(train_pos_perc - test_pos_perc) <= max_dif) and (len(test_ind) >= min_test_len):
        return True
    else:
        return False


def get_valid_split(split_iterator, label, max_dif=.05, min_test_len=50):
    for split in split_iterator:
        if is_valid_split(split, label, max_dif, min_test_len):
            return split
    raise "Cannot find valid split. Try again with a different random state."


def merge_split(input_data, random_state=0, val_size=.4, verbose=False):
    """
    :param input_data:
    :param random_state:
    :param val_size:
    :param verbose:
    :return:
    """
    # merge train-val then randomize patient ID to split train-val again
    gs = list(input_data["GS_train"]) + list(input_data['GS_val'])
    pid = list(input_data["PatientId_train"]) + list(input_data['PatientId_val'])
    inv = list(input_data["inv_train"]) + list(input_data['inv_val'])

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
    # Merge train - val
    keys = [f[:-4] for f in input_data.keys() if 'val' in f]
    merge = {}
    for k in keys:
        if isinstance(input_data[f'{k}_train'], list):
            merge[k] = input_data[f'{k}_train'] + input_data[f'{k}_val']
        else:
            merge[k] = np.concatenate([input_data[f'{k}_train'], input_data[f'{k}_val']], axis=0)

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


def merge_split_train_val(input_data, random_state=0, val_size=.4, verbose=False):
    """
    :param input_data:
    :param random_state:
    :param val_size:
    :param verbose:
    :return:
    """
    # merge train-val then randomize patient ID to split train-val again
    gs = list(input_data["GS_train"]) + list(input_data['GS_val'])
    pid = list(input_data["PatientId_train"]) + list(input_data['PatientId_val'])
    inv = list(input_data["inv_train"]) + list(input_data['inv_val'])

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

    # for _ in df1.pid.unique():
    #     tmp1 = list(np.unique(df1[df1.pid <= _].gs_merge, return_counts=True))
    #     tmp1[1] = np.round(tmp1[1] / tmp1[1].sum(), 2)
    #     tmp2 = list(np.unique(df1[df1.pid > _].gs_merge, return_counts=True))
    #     tmp2[1] = np.round(tmp2[1] / tmp2[1].sum(), 2)
    #     print(_, dict(zip(*tmp1)), dict(zip(*tmp2)))

    # import seaborn as sns
    # import pylab as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # # tr, v = df1[df1.group == 'train'], df1[df1.group == 'val']
    # plt.close('all')
    # sns.countplot(x='gs_merge', data=df1[(df1.inv >= .4) | (df1.inv == 0)], hue='group')
    # plt.show()

    pid_tv = {
        'train': df1[df1.group == 'train'].pid.unique(),
        'val': df1[df1.group == 'val'].pid.unique(),
    }
    # Merge train - val
    keys = [f[:-4] for f in input_data.keys() if 'val' in f]
    merge = {}
    for k in keys:
        if isinstance(input_data[f'{k}_train'], list):
            merge[k] = input_data[f'{k}_train'] + input_data[f'{k}_val']
        else:
            merge[k] = np.concatenate([input_data[f'{k}_train'], input_data[f'{k}_val']], axis=0)

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
