import io
import os
import sys
import pickle
import random

sys.path.insert(0, '../..')

import cv2
import yaml
import time
import string
import argparse
import functools

import torch
import numpy as np
from munch import munchify
from yaml import CLoader as Loader
from tensorboardX import SummaryWriter
from sklearn.preprocessing import RobustScaler

import bz2
import _pickle as cPickle

from datetime import datetime

import numpy as np


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if 'skip_timer' in kwargs.keys():
            skip_timer = kwargs['skip_timer']
            kwargs.pop('skip_timer')
            if skip_timer:
                return func(*args, **kwargs)

        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


class Logger(object):
    def __init__(self, filename, directory='./../logs'):
        self.terminal = sys.stdout
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log = open(f"{directory}/{filename}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def print_date_time():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n\n{("* " * 20)}{dt_string}{(" *" * 20)}')


def norm_01(x: np.ndarray):
    return (x - x.min()) / x.max()


def crop_bbox(x, bbox):
    """

    :param x: size = [..., H, W]
    :param bbox: Bounding box (min_row, min_col, max_row, max_col)
    :return:
    """
    if x is None:
        return x
    return x[..., bbox[0]:bbox[2], bbox[1]:bbox[3]]


def paste_bbox(x, original_shape, bbox):
    """

    :param x: the cropped image
    :param original_shape: shape of the original rf image
    :param bbox: output of regionprops
    :return:
    """
    if x is None:
        return x
    x_large = np.zeros(original_shape, dtype=x.dtype)
    x_large[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = x
    return x_large


def load_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def save_pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def compressed_pickle(title, data):
    """Takes much longer time to save file, but much more storage-friendly"""
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)  # Load any compressed pickle file


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def list_str(values):
    v = values.split(',')
    return [int(_) for _ in v]


def parse_args() -> dict:
    """Read commandline arguments
    Argument list includes mostly tunable hyper-parameters (learning rate, number of epochs, etc).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", default='yamls/coteaching.yml',
                        help="Path for the config file")
    parser.add_argument("--backbone", type=str,
                        help="Backbone name")
    parser.add_argument("--exp-name", type=str,
                        help="Experiment name")
    parser.add_argument("-l", "--loss-name", type=str,
                        help="Name of the loss function use with coteaching")
    parser.add_argument("--exp-suffix", type=str,
                        help="Suffix in the experiment name")
    parser.add_argument("--num-workers", type=int,
                        help="Training seed")
    parser.add_argument("--seed", type=int,
                        help="Training seed")
    parser.add_argument("-trbs", "--train-batch-size", type=int,
                        help="Batch size during training")
    parser.add_argument("-tbs", "--test-batch-size", type=int,
                        help="Batch size during test or validation")
    parser.add_argument("-ne", "--n-epochs", type=int,
                        help="Batch size during training")
    parser.add_argument("--total-iters", type=int,
                        help="Total number of iterations (currently not used)")
    parser.add_argument("--lr", type=float,
                        help="Learning rate")
    parser.add_argument("--shift-range", type=list_str,
                        help="")
    parser.add_argument("--loss-coef", type=str, default='',
                        help="")
    parser.add_argument("--optimizer", type=str,
                        help="Optimizer", choices=['adam', 'novo', 'sgd'])
    parser.add_argument("--initial-min-inv", type=float,
                        help="Minimum involvement of cancer cores used for initial training")
    parser.add_argument("--input-channels", type=int,
                        help="Number of segments used")
    parser.add_argument("--min-inv", type=float,
                        help="Minimum involvement of cancer cores in training set")
    parser.add_argument("--epoch-start-correct", type=int,
                        help="Epoch that label correction takes place")
    parser.add_argument("--elr_alpha", type=float,
                        help="lambda in early-learning regularization")
    parser.add_argument("--split_random_state", type=int,
                        help="Random state for splitting training/validation set, "
                             "set to smaller than 0 for using the current split")
    parser.add_argument("--val_size", type=float,
                        help="Percentage of training data used for validation")
    parser.add_argument("--gpus-id", type=int, nargs='+', default=[0],
                        help="Path for the config file")
    parser.add_argument("--eval", action='store_true', default=False,
                        help='Perform evaluation; Training is performed if not set')
    parser.add_argument("--use-wandb", action='store_true', default=False,
                        help='Perform evaluation; Training is performed if not set')

    subparsers = parser.add_subparsers()
    parser_hg = subparsers.add_parser('hgpsl_p', help='HGP-SL-Pooling architecture')
    parser_hg.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser_hg.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser_hg.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser_hg.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
    parser_hg.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser_hg.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser_hg.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')

    # VICREG #
    parser_ssl = subparsers.add_parser('ssl', help="Pretrain a model with VICReg")
    # Checkpoints
    parser_ssl.add_argument("--exp-dir", default="./exp",
                            help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser_ssl.add_argument("--log-freq-time", type=int, default=60,
                            help='Print logs to the stats.txt file every [log-freq-time] seconds')
    # Model
    parser_ssl.add_argument("--mlp",
                            # default="8192-8192-8192",
                            default="2048-2048-2048",
                            help='Size and number of layers of the MLP expander head')
    # Optim
    parser_ssl.add_argument("--batch-size", type=int, default=2048)
    parser_ssl.add_argument("--base-lr", type=float, default=0.2,
                            help='Base learning rate, effective learning after warmup is [base-lr] * '
                                 '[batch-size] / 256')
    parser_ssl.add_argument("--wd", type=float, default=1e-6,
                            help='Weight decay')
    # Loss
    parser_ssl.add_argument("--sim-coeff", type=float, default=25.0,
                            help='Invariance regularization loss coefficient')
    parser_ssl.add_argument("--std-coeff", type=float, default=25.0,
                            help='Variance regularization loss coefficient')
    parser_ssl.add_argument("--cov-coeff", type=float, default=1.0,
                            help='Covariance regularization loss coefficient')

    # VICREG Evaluation

    args = parser.parse_args()

    # Remove arguments that were not set and do not have default values
    args = {k: v for k, v in args.__dict__.items() if v is not None}
    return args


def read_yaml(verbose=False, setup_dir=False, args=None, parser=None) -> yaml:
    """Read config files stored in yml"""

    # Read commandline arguments
    parser = parse_args if parser is None else parser
    args = parser() if args is None else args
    if verbose:
        print_separator('READ YAML')

    # Read in yaml
    with open(args['config']) as f:
        opt = yaml.load(f, Loader)
    # Update option with commandline argument
    opt.update(args)
    # Convert dictionary to class-like object (just for usage convenience)
    opt = munchify(opt)

    if setup_dir:
        opt = setup_directories(opt)

    if verbose:
        # print yaml on the screen
        lines = print_yaml(opt)
        for line in lines:
            print(line)

    return opt


def fix_random_seed(seed, benchmark=False, deterministic=True):
    """Ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def print_separator(text, total_len=50):
    print('#' * total_len)
    left_width = (total_len - len(text)) // 2
    right_width = total_len - len(text) - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)


def print_yaml(opt):
    lines = []
    if isinstance(opt, dict):
        for key in opt.keys():
            tmp_lines = print_yaml(opt[key])
            tmp_lines = ["%s.%s" % (key, line) for line in tmp_lines]
            lines += tmp_lines
    else:
        lines = [": " + str(opt)]
    return lines


def create_path(opt):
    for k, v in opt['paths'].items():
        if (k == 'folder') or (v is None):
            continue
        if not os.path.isfile(v):
            os.makedirs(v, exist_ok=True)


def update_directories(opt, reset_idx=0):
    current_folder = '' + opt.paths.folder
    if reset_idx > 0:
        if f'_reset{reset_idx - 1}' in current_folder:
            new_folder = current_folder.replace(f'_reset{reset_idx - 1}', f'_reset{reset_idx}')
        else:
            new_folder = current_folder + f'_reset{reset_idx}'
        #
        for k, v in opt.paths.__dict__.items():
            setattr(opt.paths, k, v.replace(current_folder, new_folder))

        # make directories
        create_path(opt)
    return opt


def setup_directories(opt, to_create_paths=True, is_graph=False):
    opt.exp_suffix = '' if opt.exp_suffix == 'none' else opt.exp_suffix

    if opt.paths is None:
        raise TypeError('log_dir, result_dir, and checkpoint_dir need to be specified!')
    prefix = '_'.join(opt.backbone) if isinstance(opt.backbone, list) else opt.backbone
    opt.paths.folder = prefix + opt.exp_suffix
    for k, v in opt.paths.__dict__.items():
        if k == 'extract_root':
            continue
        opt.paths[k] = v.replace('exp_name',
                                 '/'.join([opt.exp_name, opt.paths.folder]))
    project_root = opt.project_root if os.path.exists(opt.project_root) else opt.project_root_server
    opt.data_source.data_root = '/'.join((project_root, opt.data_source.data_root))
    if hasattr(opt.paths, 'self_train_checkpoint'):
        opt.paths.self_train_checkpoint = '/'.join((project_root, opt.paths.self_train_checkpoint))

    if opt.data_source.train_set:  # not NONE
        if not is_graph:
            # Dataset
            if opt.data_source.use_mat:
                # Try using the local data or data in the project first
                data_file = '/'.join(([opt.data_source.data_root, opt.data_source.train_set_mat]))
                if not os.path.exists(data_file):  # use data on the server instead
                    data_file = '/'.join(
                        (opt.data_source.data_root_server, opt.data_source.train_set_mat)
                    )
            else:
                data_file = '/'.join([opt.data_source.data_root, opt.data_source.train_set])
                try:
                    assert os.path.exists(data_file)
                except AssertionError:
                    raise FileExistsError(f'"{data_file}" does not exist.')
            opt.data_file = data_file
        else:
            opt.graph_root = os.path.join(opt.data_source.data_root, opt.data_source.graph_name)

    # make directories
    if to_create_paths:
        create_path(opt)
    return opt


def get_time_suffix() -> str:
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("_%Y%m%d_%H%M%S")
    return dt_string


def print_date_time():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n\n{("* " * 20)}{dt_string}{(" *" * 20)}')


def main():
    get_time_suffix()


class Logger(object):
    def __init__(self, filename, directory='./../logs'):
        self.terminal = sys.stdout
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log = open(f"{directory}/{filename}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def setup_tensorboard(opt):
    """

    :param opt: return from read_yaml
    :return:
    """
    writer = SummaryWriter(
        logdir=opt.paths.log_dir,
        flush_secs=opt.tensorboard.flush_secs,
        filename_suffix=opt.tensorboard.filename_suffix
    )
    return writer


def eval_mode(func):
    """wrapper for torch evaluation"""

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def plot_to_image(fig, dpi=200):
    """Convert figure to image"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def init_weights(net, init_fn):
    from torch import nn

    # out_counter = 0
    for child in net.children():
        if isinstance(child, nn.Conv1d):
            init_fn(child.weights)


def robust_norm(x, transformer=None, is_timeseries=True, quantile_range=(25, 75)):
    """

    :param x:
    :param transformer:
    :param is_timeseries:
    :param quantile_range:
    :return:
    """
    shape_org = None
    if is_timeseries:
        shape_org = x.shape
        x = x.flatten()[..., np.newaxis]
    if transformer is None:
        transformer = RobustScaler(quantile_range=quantile_range).fit(x)

    x = transformer.transform(x)
    if shape_org is not None:
        x = x[..., 0].reshape(shape_org)

    return x, transformer


def robust_norm_per_core(x, core_id=None):
    if isinstance(x, list):
        for _x in x:
            shape_org = _x.shape
            _x[...] = RobustScaler().fit_transform(_x.flatten()[..., np.newaxis])[..., 0].reshape(shape_org)
    else:
        core_id_unique = np.unique(core_id)
        for cid in core_id_unique:
            _x = x[core_id == cid]
            shape_org = _x.shape
            _x = _x.flatten()[..., np.newaxis]
            x[core_id == cid] = RobustScaler().fit_transform(_x)[..., 0].reshape(shape_org)
    return x


def patch_opt_sweep(opt, config):
    """

    :param opt:
    :param config: wandb config
    :return:
    """
    opt.arch.mid_channels = config.mid_channels
    opt.arch.num_blocks = config.num_blocks
    opt.lr = config.lr
    opt.train_batch_size = config.train_batch_size
    opt.train.coteaching.forget_rate = config.forget_rate
    opt.train.coteaching.num_gradual = config.num_gradual
    return opt


def gen_random_str(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def print_graph_dataset_info(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')

    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def get_attr_from_graph_dataset(dataset, attr):
    if len(getattr(dataset[0], attr).shape) != 0:
        return np.array([getattr(_, attr)[0].item() for _ in dataset])
    else:
        return [getattr(_, attr) for _ in dataset]


def accumulate_dict(accumulator, single_dict, idx=None):
    if single_dict:
        if not accumulator:
            accumulator = {}
            for k in single_dict.keys():
                accumulator[k] = []
        for k, v in single_dict.items():
            if hasattr(v, '__len__'):
                if idx is not None:
                    v = idx[v]
                accumulator[k].extend(list(v))
            else:
                accumulator[k].append(v)
        return accumulator
    return None


get_attr = get_attr_from_graph_dataset
