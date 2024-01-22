import gc
import os
import math
import json
import time

import warnings

warnings.filterwarnings("ignore")
import yaml
import wandb
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from tqdm import tqdm
from munch import munchify
from yaml import CLoader as Loader
from cleanlab.experimental.coteaching import forget_rate_scheduler

from sklearn.mixture import GaussianMixture

import torch
from torch import nn, optim
import torch.nn.functional as F

from training_strategy import *
from loss_functions.protoc_loss import protoc_loss
from loss_functions.coteaching_loss import loss_coteaching
from loss_functions.isomaxplus import IsoMaxPlusLossSecondPart as IsoMaxLossSecondPart

from utils import compute_metrics_core
from utils import fix_random_seed, net_interpretation, accumulate_dict

from utils_3d.dataloader import create_loader
from utils_3d.optimizers import get_scheduler
from utils_3d.dataset import PatchLabeledDataset, CropFixSize

from tensorboardX import SummaryWriter


def parse_args() -> dict:
    import argparse
    """Read commandline arguments
    Argument list includes mostly tunable hyper-parameters (learning rate, number of epochs, etc).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", default='vicreg_patch_evaluate.yml',
                        help="Path for the config file")
    parser.add_argument("--exp-suffix", default='', type=str,
                        help="Suffix in the experiment name")
    parser.add_argument("--pretrained", default=None,
                        type=str, help="path to the pretrained model")
    parser.add_argument("--random-crop", action='store_true', default=False,
                        help="random crop instead of center crop")
    parser.add_argument("--time-series", action='store_true', default=False,
                        help="time-series classification")
    parser.add_argument("--seed", type=int, default=0,
                        help='Seed')
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--in-channels", type=int, default=1,
                        help='Number of time frames')
    parser.add_argument("--workers", type=int, default=10,
                        help='Number of CPUs')

    # Co-teaching
    parser.add_argument("--forget-rate", type=float, default=0.1,
                        help='percentage of training data that will be filtered')
    parser.add_argument("--exponent", type=int, default=1,
                        help='')
    parser.add_argument("--num-gradual", type=int, default=20,
                        help='Number of epochs to reach the forget-rate')

    parser.add_argument("--min-inv", type=float, default=0.7,
                        help='')

    # IsoMax
    parser.add_argument("--ood-thr", type=float, default=0.0,
                        help='Number of epochs to reach the forget-rate')
    parser.add_argument("-es", "--entropic-scale", type=float, default=0.0,
                        help='Number of epochs to reach the forget-rate')

    arg = parser.parse_args()

    # Remove arguments that were not set and do not have default values
    arg = {k: v for k, v in arg.__dict__.items() if v is not None}
    return arg


def match_noisy_percent(noisy_idx_b, labels):
    """Keep percentage of noisy labels of all classes to be the same"""
    min_noisy_perc = min([noisy_idx_b[labels == l].sum() / sum(labels == l) for l in np.unique(labels)])
    noisy_idx_b_prime = np.zeros_like(noisy_idx_b)
    noisy_l_list, choices_list = [], []
    for l in np.unique(labels):
        noisy_l = np.where(noisy_idx_b * (labels == l))[0]
        n_choice = int(sum(labels == l) * min_noisy_perc)
        choices = np.random.choice(noisy_l, n_choice, replace=False)
        noisy_idx_b_prime[choices] = True
        noisy_l_list.append(noisy_l)
        choices_list.append(choices)
    assert not len(np.intersect1d(*noisy_l_list))
    assert not len(np.intersect1d(*choices_list))
    return noisy_idx_b_prime


def on_batch_end(print_freq, lr_backbone, lr_head, losses, protoc_losses, t_epoch, epoch, current_step, start_time,
                 batch_extra_outputs, extra_outputs=None, idx=None, stats_file=None, writer=None, net_idx=1):
    if isinstance(losses, list):
        t_epoch.set_postfix(loss0=losses[0].item(), loss1=losses[1].item(), lr=lr_head)
    else:
        t_epoch.set_postfix(loss=losses, lr=lr_head)

    if writer:
        writer.add_scalar('LEARNING_RATE', lr_head, current_step)
        if isinstance(losses, list):
            writer.add_scalar(f'loss/cls_net{1}', losses[0].item(), current_step)
            writer.add_scalar(f'loss/cls_net{2}', losses[1].item(), current_step)
        else:
            writer.add_scalar(f'loss/cls_net{net_idx}', losses.item(), current_step)
        if len(protoc_losses):
            writer.add_scalar(f'loss/protoc_dst_net1', protoc_losses[0].item(), current_step)
            writer.add_scalar(f'loss/protoc_dst_net2', protoc_losses[1].item(), current_step)
        if batch_extra_outputs:
            writer.add_scalar(f'extra/perc_common_update', batch_extra_outputs['perc_common_update'],
                              current_step)
        writer.flush()
    if extra_outputs is not None:
        extra_outputs = accumulate_dict(extra_outputs, batch_extra_outputs, idx.numpy())
    return extra_outputs


def get_model(arg, checkpoint_name="checkpoint.pth", len_loader=None, const_init: float = 0):
    arch, root_dir, pretrained, weights, lr_head, lr_backbone, weight_decay, epochs, exp_dir = \
        arg.arch, arg.root_dir, arg.pretrained, arg.weights, arg.lr_head, arg.lr_backbone, \
            arg.weight_decay, arg.epochs, arg.exp_dir

    if arg.time_series:
        arg.arch = 'inception'
        from tsai.models.InceptionTimePlus import InceptionTimePlus
        backbone = InceptionTimePlus(16, 2, fc_dropout=0)
        backbone[1][0] = backbone[1][0][0]
        if 'pretrained' in arg.arch:
            backbone.load_state_dict(torch.load('experiments/tsai/mvp_200.pth'), strict=False)  #
        embedding = 128

    if "convnext" in arg.arch:
        backbone, embedding = convnext.__dict__[arg.arch](
            drop_path_rate=arg.drop_path_rate,
            layer_scale_init_value=arg.layer_scale_init_value,
            in_chans=arg.in_channels,
        )
    elif "resnet" in arg.arch:
        backbone, embedding = resnet.__dict__[arg.arch](
            zero_init_residual=True,
            num_channels=arg.in_channels)

    if pretrained:
        print('/'.join([root_dir, pretrained]))
        state_dict = torch.load('/'.join([root_dir, pretrained]), map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.", ""): value
            for (key, value) in state_dict.items()
        }
        backbone.load_state_dict(state_dict, strict=False)

    if arg.single_model:
        head = nn.Linear(embedding, 2)
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
    else:
        from loss_functions.isomaxplus import IsoMaxPlusLossFirstPart as IsoMaxLossFirstPart
        head = IsoMaxLossFirstPart(embedding, arg.num_class, const_init=const_init)

    model = nn.Sequential(backbone, head)
    model.cuda()

    if weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)

    optims = [
        optim.SGD(head.parameters(), lr_head, momentum=0.9, weight_decay=weight_decay, nesterov=True),
        optim.SGD(backbone.parameters(), lr_backbone, momentum=0.9, weight_decay=weight_decay, nesterov=True)]
    schedulers = [
        get_scheduler('one_cycle', len_loader, arg.epochs, optims[0], arg.lr_head),
        get_scheduler('one_cycle', len_loader, arg.epochs, optims[1], arg.lr_backbone)
    ]

    # automatically resume from checkpoint if it exists
    if os.path.exists('/'.join([exp_dir, checkpoint_name])):
        ckpt = torch.load('/'.join([exp_dir, checkpoint_name]), map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc_b = ckpt["best_acc_b"]
        best_auc = ckpt["best_auc"]
        best_tst_acc_b = ckpt["best_tst_acc_b"]
        model.load_state_dict(ckpt[f"model"])
        optims[0].load_state_dict(ckpt["optimizer0"])
        optims[1].load_state_dict(ckpt["optimizer1"])
        schedulers[0].load_state_dict(ckpt["scheduler0"])
        schedulers[1].load_state_dict(ckpt["scheduler1"])
    else:
        start_epoch = 0
        best_acc_b = 0
        best_auc = 0
        best_tst_acc_b = 0

    return model, optims, schedulers, start_epoch, best_acc_b, best_auc, best_tst_acc_b


def save_results(arg, models, epoch, scores, scores_tst, optims, schedulers,
                 best_acc_b, best_tst_acc_b, best_auc, stats_file=None):
    if scores_tst['acc_b'] > best_tst_acc_b:
        torch.save(models[0].state_dict(), '/'.join([arg.exp_dir, "best_test.pth"]))

    if scores['acc_b'] > best_acc_b:
        torch.save(models[0].state_dict(), '/'.join([arg.exp_dir, "best_val.pth"]))

    if epoch == arg.epochs - 1:
        torch.save(models[0].state_dict(), '/'.join([arg.exp_dir, "last.pth"]))

    best_acc_b = max(best_acc_b, scores['acc_b'])
    best_auc = max(best_auc, scores['auc'])
    best_tst_acc_b = max(best_tst_acc_b, scores_tst['acc_b'])
    stats = dict(
        epoch=epoch,
        acc_b=scores['acc_b'],
        auc_b=scores['auc'],
        test_acc_b=scores_tst['acc_b'],
        test_auc=scores_tst['acc_b'],
        best_acc_b=best_acc_b,
        best_auc=best_auc,
    )
    # print(json.dumps(stats))
    if stats_file:
        print(json.dumps(stats), file=stats_file)

    for i in range(len(models)):
        state = dict(
            epoch=epoch + 1, best_acc_b=best_acc_b, best_auc=best_auc, best_tst_acc_b=best_tst_acc_b,
            model=models[i].state_dict(),
            optimizer0=optims[i * len(models)].state_dict(), optimizer1=optims[i * len(models) + 1].state_dict(),
            scheduler0=schedulers[i * len(models)].state_dict(),
            scheduler1=schedulers[i * len(models) + 1].state_dict(),
        )

    return best_acc_b, best_tst_acc_b, best_auc


def correct_labels_with_ilr(arg, train_loader, train_eval_loader, epoch, correcting_info, kwargs_loader):
    # trn_eval_ds.correct_labels(*correcting_info)
    trn_ds = train_loader.dataset
    corrected = trn_ds.correct_labels(*correcting_info, verbose=True)
    if corrected:
        print('Creating a new training dataloader...')
        # n_training = [len(train_loader.dataset.label), sum(train_loader.dataset.label==1)]
        train_sampler = create_loader(trn_ds, arg.batch_size, jobs=arg.workers, add_sampler=True,
                                      get_sampler_only=True, weight_by_inv=False)
        train_loader = torch.utils.data.DataLoader(trn_ds, sampler=train_sampler, drop_last=True,
                                                   **kwargs_loader)
    return train_loader


def get_coteaching_models(arg, len_loader):
    model1, optimizers1, schedulers1, *_ = get_model(arg, checkpoint_name="checkpoint0.pth",
                                                     len_loader=len_loader, const_init=-1e-1)
    model2, optimizers2, schedulers2, *last_run = get_model(arg, checkpoint_name="checkpoint1.pth",
                                                            len_loader=len_loader, const_init=-1e-1)
    return [model1, model2], optimizers1 + optimizers2, schedulers1 + schedulers2, last_run


def get_single_model(arg, len_loader):
    model, optimizers, schedulers, *last_run = get_model(arg, checkpoint_name="checkpoint1.pth",
                                                            len_loader=len_loader, const_init=-1e-1)
    return [model], optimizers, schedulers, last_run


def eval_forward(model, dataloader, description='Extracting features'):
    model.eval()
    inputs, feats, outputs = [], [], []
    with torch.no_grad():
        inputs = []
        with tqdm(dataloader, unit="batch") as t_epoch:
            t_epoch.set_description(f"{description}")
            for batch in t_epoch:
                images, labels = batch[:2]
                inputs.append(images)
                feat = model[0](images.to('cuda', non_blocking=True))  # extract features
                feats.append(feat.cpu().detach())
                output = model[1](feat)  # extract logits (distances)
                outputs.append(output.cpu().detach())
    inputs = torch.concat(inputs)
    feats = torch.concat(feats)
    outputs = torch.concat(outputs)
    return inputs, feats, outputs


def evaluate_ood_viz(model, dataloader, ood_test_loader, ood_control_loader,
                     set_name='train_tsne', epoch=0, writer=None, declare_thr=0.35, result_dir=None,
                     correcting_params: tuple = None, emb_2d=None, n_classes=2,
                     val_loader=None, test_loader=None, out_val=None, out_tst=None,
                     criterion=nn.CrossEntropyLoss(),
                     ):
    """
    ood_masks = evaluate_ood_viz(models, train_raw_loader, ood_test_loader, ood_control_loader,
                                 set_name='train_tsne', epoch=epoch, writer=writer,
                                 val_loader=val_loader, test_loader=test_loader, out_val=out_val, out_tst=out_tst)
    """
    import pandas as pd

    def get_median_filtered(signal, threshold=3):
        signal = signal.copy()
        difference = np.abs(signal - np.median(signal))
        median_difference = np.median(difference)
        if median_difference == 0:
            s = 0
        else:
            s = difference / float(median_difference)
        mask = s > threshold
        return mask, signal[mask].min()

    outputs = eval_forward(model, dataloader)[-1]
    outputs_ood_test = eval_forward(model, ood_test_loader)[-1]
    outputs_ood_control = eval_forward(model, ood_control_loader)[-1]

    ds = dataloader.dataset
    ds_ood_test = ood_test_loader.dataset
    ds_ood_control = ood_control_loader.dataset
    ds_val = val_loader.dataset
    ds_tst = test_loader.dataset

    # Make assumptions about the labels of OOD data
    ds_ood_test.label = np.zeros_like(ds_ood_test.label)
    ds_ood_control.label = np.zeros_like(ds_ood_control.label)

    outputs = torch.concat([outputs, outputs_ood_test, outputs_ood_control,
                            torch.tensor(out_val), torch.tensor(out_tst)])
    labels = np.concatenate([ds.label, ds_ood_test.label, ds_ood_control.label,
                             ds_val.label, ds_tst.label])
    set_names = np.concatenate([
        np.tile('train', len(ds.label)),
        np.tile('ood_test', len(ds_ood_test.label)),
        np.tile('ood_control', len(ds_ood_control.label)),
        np.tile('val', len(ds_val.label)),
        np.tile('test', len(ds_tst.label))]
    )

    loss = criterion(outputs, labels, reduction='none').cpu().numpy()
    # min_dist = torch.tensor(-outputs).min(dim=1)[0]
    scores = torch.tensor(outputs).max(dim=1)[0]

    d = {'loss': loss, 'scores': scores, 'labels': labels, 'set_names': set_names}
    df = pd.DataFrame(data=d)
    fig = sns.histplot(df, x='scores', hue='set_names', common_norm=False, element="step", fill=False)
    ood_scores_trn = df.scores[df.set_names == 'train'].to_numpy()
    thr_train = -get_median_filtered(-ood_scores_trn)[1]
    thr_ood_test = np.median(df.scores[df.set_names == 'ood_test'].to_numpy())
    plt.axvline(thr_train, color='r')
    plt.axvline(thr_ood_test, color='b')
    writer.add_figure(f'ood/scores', fig.get_figure(), global_step=epoch)
    plt.close()

    fig2 = sns.histplot(df, x='loss', hue='set_names', common_norm=False, fill=False)
    writer.add_figure(f'ood/IsoMax2Lss', fig2.get_figure(), global_step=epoch)
    plt.close('all')

    return ood_scores_trn < thr_ood_test  # .astype('uint8')


def eval_ood(models, dataloader, ood_test_loader, epoch=0, writer=None, start_filter_epoch=9,
             ood_thr=50, ind_common_update=None, ind_non_update=None, net_idx=None):
    if (epoch < start_filter_epoch - 1) or (ood_thr == 100):
        return None

    if net_idx is None or len(net_idx) < len(models):
        net_idx = range(len(models))
    outputs_ood_test = [eval_forward(model, ood_test_loader, f'Net{net_idx} Eval OOD Test')[-1]
                        for net_idx, model in zip(net_idx, models)]

    outputs = outputs_ood_test  # for debugging

    set_names = np.tile('ood_test', len(ood_test_loader.dataset))

    scores = [torch.tensor(out).max(dim=1)[0] for out in outputs]
    thr_ood_test = None

    for i, _scores in enumerate(scores):
        d = {'scores': _scores, 'scores_normed': scores[i], 'set_names': set_names}
        df = pd.DataFrame(data=d)
        thr_ood_test = {_ood_perc: np.percentile(df.scores[df.set_names == 'ood_test'].to_numpy(), 100 - _ood_perc)
                        for _ood_perc in [10, 20, 30, 40, 50, 0]}

    return thr_ood_test


def eval_ood_v0(models, dataloader, ood_test_loader, epoch=0, writer=None, start_filter_epoch=9,
                ood_thr=50,
                ind_common_update=None, ind_non_update=None, net_idx=None):
    if (epoch < start_filter_epoch - 1) or (ood_thr == 100):
        return None

    if net_idx is None or len(net_idx) < len(models):
        net_idx = range(len(models))
    outputs_train = [eval_forward(model, dataloader, f'Net{net_idx} Eval Train')[-1]
                     for net_idx, model in zip(net_idx, models)]
    outputs_ood_test = [eval_forward(model, ood_test_loader, f'Net{net_idx} Eval OOD Test')[-1]
                        for net_idx, model in zip(net_idx, models)]

    def min_max_norm(x):
        return (x - x.min()) / (x.max() - x.min())

    outputs = [torch.concat([out, out_ood_test]) for out, out_ood_test in zip(outputs_train, outputs_ood_test)]
    set_names = np.concatenate([np.tile('train', len(dataloader.dataset)),
                                np.tile('ood_test', len(ood_test_loader.dataset))])

    scores = [torch.tensor(out).max(dim=1)[0] for out in outputs]
    scores_normed = [min_max_norm(score) for score in scores]
    ood_masks = []

    for i, _scores in enumerate(scores):
        d = {'scores': _scores, 'scores_normed': scores_normed[i], 'set_names': set_names}
        df = pd.DataFrame(data=d)
        thr_ood_test = np.percentile(df.scores[df.set_names == 'ood_test'].to_numpy(), ood_thr)
        ood_masks.append(np.array([_scores > thr_ood_test for _scores in df.scores[df.set_names == 'train']],
                                  dtype='uint8'))

        if writer is not None:
            fig = sns.histplot(df, x='scores_normed', hue='set_names', common_norm=False, element="step", fill=False)
            plt.axvline(np.percentile(df.scores_normed[df.set_names == 'ood_test'].to_numpy(), ood_thr), color='r')
            writer.add_figure(f'ood/scores_{net_idx}', fig.get_figure(), global_step=epoch)
            plt.close()

    return ood_masks, outputs_train, thr_ood_test


def eval_train(model, dataloader, epoch, net_idx, writer=None, stats_file=None,
               ood_masks: np.ndarray = None, outputs: np.ndarray = None, multi_gmm=False,
               criterion=nn.CrossEntropyLoss()):
    if ood_masks is None:
        ood_masks = np.zeros(len(dataloader.dataset), dtype='uint8')
    if outputs is None:
        outputs = eval_forward(model, dataloader, f'Net{net_idx} Eval Train')[-1]
    labels = dataloader.dataset.label
    losses = criterion(outputs, labels, reduction='none').cpu().numpy()
    losses = (losses - losses.min()) / (losses.max() - losses.min())

    def cluster_by_gmm(loss, verbose=False):
        gmm = GaussianMixture(n_components=2, max_iter=200, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss.reshape(-1, 1))
        clean_idx, noisy_idx = gmm.means_.argmin(), gmm.means_.argmax()
        _prob = gmm.predict_proba(loss.reshape(-1, 1))
        _prob = _prob[:, clean_idx]
        return _prob

    # GMM
    """Single GMM"""
    if not multi_gmm:
        prob = cluster_by_gmm(losses)
    else:
        prob = np.zeros(len(losses))
        for cls in np.unique(labels):
            _prob = cluster_by_gmm(losses[labels == cls])
            prob[labels == cls] = _prob

    p_thr = np.clip(0.5, prob.min() + 1e-5, prob.max() - 1e-5)
    noisy = prob <= p_thr  # Note that noisy_idx_b comes from a filtered dataset, not original one

    noisy_rate_per_class = np.array([noisy[labels == l].sum() / sum(labels == l) for l in np.unique(labels)])
    if np.any(noisy_rate_per_class > 0.3):
        noisy = np.ones_like(noisy)
        clean_idx = np.argsort(losses)[:int(len(noisy) * 0.9)]
        noisy[clean_idx] = 0

    # match the percentage of noisy labels of each class
    prob[np.invert(noisy)] = 1

    def plot(separator: str = 'Clean/Noisy', suptitle='GMM clustering', hue='noisy', fig_tag='gmm',
             target='Noisy', hue_order=None):
        sns.set_context("paper")

        classes = ['Benign', 'Cancer']
        graphs = []
        fig, axes = plt.subplots(1, 4 if hue == 'noisy' else 2, figsize=(8, 4))
        plt.suptitle(suptitle)

        ax_idx = 0
        if hue == 'noisy':
            graphs.append(sns.histplot(df, x='losses', hue='labels', ax=axes[0]))
            axes[0].set_xlabel('Losses')
            axes[0].set_title(f'Losses by classes')
            ax_idx += 1

        for label in ['Benign', 'Cancer']:
            graphs.append(
                sns.histplot(df[df.labels == label], x='losses', hue=hue, element="step", fill=False,
                             ax=axes[ax_idx], hue_order=hue_order))
            axes[ax_idx].set_xlabel('Losses')
            axes[ax_idx].set_title(f'{label} losses')
            ax_idx += 1

        graphs.append(
            sns.histplot(df, x='labels', hue=hue, hue_order=hue_order, multiple="stack", binwidth=0.05, bins=2,
                         fill=True, ax=axes[ax_idx]))
        axes[ax_idx].set_xlabel('Classes')
        axes[ax_idx].set(ylabel=None)
        axes[ax_idx].set_xticks([0, 1])
        axes[ax_idx].set_xticklabels(classes)
        axes[ax_idx].set_title(f'Labels by {separator}')

        def autolabel(graph, texts):
            """Attach a text label above each bar in *rects*, displaying its height."""
            height = 0
            h = []
            for rectangle in graph.patches:
                h.append(rectangle.get_height())
            h = np.array(h).reshape(2, int(len(h) / 2))
            h = h[0] + h[1]

            for i, text in enumerate(texts):
                # height += rect.get_height()
                graph.annotate(f'{text:.0f}% {target}',
                               xy=(graph.patches[i].get_x() + graph.patches[i].get_width() / 2., h[i] + 1),
                               ha='center', va='center',
                               xytext=(0, 5),
                               textcoords='offset points'
                               )

        clean_perc = [len(df[(df.labels == c) & (df[hue] == target)]) * 100 / len(df[df.labels == c]) for c in classes]

        autolabel(graphs[-1], clean_perc)
        [g.legend_.set_title(None) for g in graphs]

        writer.add_figure(f'{fig_tag}/net{net_idx}', fig.get_figure(), global_step=epoch)
        plt.close('all')

    if writer:
        # noisy = (prob <= 0.5).astype('uint8')
        noisy = noisy.astype('uint8')

        d = {'losses': losses, 'noisy': noisy, 'labels': labels, 'ood': ood_masks,
             'removed': np.clip(noisy + ood_masks, 0, 1, ), 'overlapped': noisy * ood_masks}
        df = pd.DataFrame(data=d)
        df.labels.replace({0: 'Benign', 1: 'Cancer'}, inplace=True)
        df.noisy.replace({0: 'Clean', 1: 'Noisy'}, inplace=True)
        df.removed.replace({0: 'Kept', 1: 'Removed'}, inplace=True)
        df.overlapped.replace({0: 'Not Both', 1: 'OOD & Noisy'}, inplace=True)
        df.ood.replace({0: 'ID', 1: 'OOD'}, inplace=True)
        plot(hue_order=['Clean', 'Noisy'])
        if np.any(ood_masks):
            plot('OOD', suptitle='Isomax OOD', hue='ood', fig_tag='ood', target='OOD')
            plot('OOD+Noisy', suptitle='Removed by IsoMax & GMM', hue='removed', fig_tag='removed', target='Removed')
            plot('OODxNoisy', suptitle='Overlapped by IsoMax & GMM', hue='overlapped', fig_tag='overlapped',
                 target='OOD & Noisy')
        # exit()
    return noisy, losses


def compute_core_metrics(
        outputs, ds, all_logits, declare_thr, epoch, set_name, writer, result_dir,
        _ood_scores=None, _ood_perc: int = None, _ood_thr: float = None):
    if _ood_scores is None:
        id_masks = np.ones(len(outputs), dtype='bool')
        suffix = ''
    else:
        if _ood_perc == 0:
            id_masks = np.ones(len(outputs), dtype='bool')
        else:
            id_masks = _ood_scores <= _ood_thr
        suffix = f'_{_ood_perc}'

    outputs_c = []
    u_id_idx_org = sorted(np.unique(ds.id, return_index=True)[1])
    u_id_org = [ds.id[idx] for idx in u_id_idx_org]
    u_id_idx, u_id = [], []
    discarded_id, discarded_id_pred_acc = [], []
    patches_removed_class = []

    # for _id in u_id:
    for i, _id in enumerate(u_id_org):
        idx = ds.id == _id
        tmp = outputs[idx]
        pred_cls = tmp.argmax(1)
        label = ds.label[idx]
        id_mask = id_masks[idx]
        kept_rate = id_mask.sum() / len(id_mask)
        pred_inv = sum(pred_cls[id_mask]) / sum(id_mask)
        patches_removed_class.extend(label[np.invert(id_mask)])
        if np.isnan(pred_inv):
            discarded_id.append(_id)
            discarded_id_pred_acc.append(((sum(pred_cls) / len(tmp)) >= declare_thr[-1]) == label[0])
            continue
        outputs_c.append(pred_inv)
        u_id_idx.append(u_id_idx_org[i])
        u_id.append(u_id_org[i])
        # outputs_c.append(sum(pred_cls) / tmp.shape[0])

    if len(patches_removed_class) and len(all_logits):
        print(f"Discard {len(patches_removed_class)} / {len(all_logits)} patches,"
              f" ({sum(patches_removed_class) * 100 / len(patches_removed_class):.0f}% are from cancer cores)")
    if len(discarded_id):
        print(f"Discard {len(discarded_id)} cores. Prediction accuracy of these cores are "
              f"{sum(discarded_id_pred_acc) / len(discarded_id)}%")
    outputs_c = np.array(outputs_c)
    # inv_c = ds.inv[u_id_idx]
    # core_len = [np.sum(ds.id == _id) for _id in u_id]
    inv_c = ds.inv[u_id_idx]
    core_len = [np.sum(ds.id == _id) for _id in u_id]

    major_thr = declare_thr[-1]
    for i, _declare_thr in enumerate(declare_thr[:-1]):
        verbose = True if _declare_thr == major_thr else False
        scores = compute_metrics_core(
            outputs_c,
            inv_c,
            predicted_involvement=outputs_c,
            declare_thr=_declare_thr,
            current_epoch=epoch, verbose=verbose, scores={}, set_name=set_name,
        )

        # Tensorboard logging
        if writer:
            if _declare_thr == major_thr:
                for score_name, score in zip(scores.keys(), scores.values()):
                    writer.add_scalar(f'{set_name}/{score_name.upper()}{suffix}', score, epoch)
                gs_c = ['-' if _ == 0 else _ for _ in ds.gs[u_id_idx]]

                net_interpretation(
                    outputs_c,
                    ds.pid[u_id_idx],
                    inv_c, gs_c, result_dir, ood=None,
                    latents=None, declare_thr=_declare_thr, current_epoch=epoch, set_name=set_name, writer=writer,
                    scores=scores, core_id=ds.id[u_id_idx], core_list=None,
                )
                plt.close('all')
            else:
                writer.add_scalar(f'{set_name}/ACC_B_{_declare_thr}{suffix}', scores['acc_b'], epoch)
    return core_len, inv_c, outputs_c, scores


def evaluate(models, dataloader, set_name='val', epoch=0, writer=None,
             declare_thr=None, result_dir=None,
             correcting_params: tuple = None, entropic_scale=1, ood_thr_dict=None):
    """
    params:
        declare_thr: [thr1, thr2, thr3,..., major_thr]
    """
    # evaluate
    if declare_thr is None:
        declare_thr = [0.5, 0.5]
    ds = dataloader.dataset
    [model.eval() for model in models]
    outputs, all_logits = [], []
    features = []
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[:2]
            images = torch.split(images, 1, dim=1) if ds.tta else [images]
            logits = [model(_images.to('cuda', non_blocking=True)) for model in models for _images in images]
            if ood_thr_dict is not None:
                all_logits.append(logits[0].cpu())
            # Merge predictions
            outputs.append(sum([F.softmax(logit, dim=1).cpu() for logit in logits]) / len(logits))
        # outputs = np.concatenate([_.numpy() for _ in outputs])
        outputs = torch.concat(outputs).numpy()

    if ood_thr_dict is not None:
        assert isinstance(ood_thr_dict, dict)
        all_logits = torch.concat(all_logits).numpy()
        ood_scores = all_logits.max(axis=1)
        for ood_perc, ood_thr in ood_thr_dict.items():
            core_len, inv_c, outputs_c, scores = compute_core_metrics(
                outputs, ds, all_logits, declare_thr, epoch, set_name, writer, result_dir,
                ood_scores, ood_perc, ood_thr)
    else:
        core_len, inv_c, outputs_c, scores = compute_core_metrics(
            outputs, ds, all_logits, declare_thr, epoch, set_name, writer, result_dir,
        )

    if (set_name == 'train') and (correcting_params is not None):
        return ds.get_correcting_mask(core_len, outputs, inv_c, outputs_c, correcting_params)

    return scores, all_logits


def warmup(arg, models, optims, schedulers, train_loader, epoch, forget_rate_schedule, ood_masks,
           start_time, writer=None, stats_file=None):
    extra_outputs = None  # reset this at the beginning of each epoch    protoc_losses = []
    first_batch = True
    protoc_losses = []
    if arg.weights == "finetune":
        [model.train() for model in models]
    elif arg.weights == "freeze":
        [model.eval() for model in models]
    else:
        assert False

    with tqdm(train_loader, unit="batch") as t_epoch:
        t_epoch.set_description(f"Epoch {epoch}")
        for step, (images, labels, idx) in enumerate(t_epoch, start=epoch * len(t_epoch)):
            # break
            images = images.to('cuda', non_blocking=True)
            labels = F.one_hot(labels, 2).float().to('cuda', non_blocking=True)

            outputs = [model(images) for model in models]
            *losses, batch_extra_outputs = loss_coteaching(
                *outputs, labels, forget_rate_schedule[epoch],
                num_classes=2, extra_outputs=True, mask=[ood_mask[idx] for ood_mask in ood_masks]
            )
            _writer = writer if first_batch else None

            [optimizer.zero_grad() for optimizer in optims]
            [loss.backward() for loss in losses]
            # [torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) for model in models]
            [optimizer.step() for optimizer in optims]
            [scheduler.step() for scheduler in schedulers]

            extra_outputs = on_batch_end(arg.print_freq, optims[0].param_groups[0]["lr"],
                                         optims[0].param_groups[0]["lr"],
                                         losses, protoc_losses, t_epoch, epoch, schedulers[0].last_epoch,
                                         start_time, batch_extra_outputs, extra_outputs, idx,
                                         stats_file, writer)
            first_batch = False

    return extra_outputs


def run_train_epoch(arg, model, optims, schedulers, train_loader, epoch, forget_rate_schedule, ood_masks,
                    start_time, writer=None, stats_file=None, net_idx=1, criterion=nn.CrossEntropyLoss()):
    protoc_losses = []
    if arg.weights == "finetune":
        model.train()
    elif arg.weights == "freeze":
        model.eval()
    else:
        assert False

    with tqdm(train_loader, unit="batch") as t_epoch:
        t_epoch.set_description(f"Epoch {epoch} Net{net_idx}")
        for step, (images, labels, idx) in enumerate(t_epoch, start=epoch * len(t_epoch)):
            # break
            images = images.to('cuda', non_blocking=True)
            if arg.criterion in ['ce', 'isomax']:
                labels = F.one_hot(labels, 2).float().to('cuda', non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            [optimizer.zero_grad() for optimizer in optims]
            loss.backward()
            [optimizer.step() for optimizer in optims]
            [scheduler.step() for scheduler in schedulers]
            on_batch_end(arg.print_freq, optims[0].param_groups[0]["lr"],
                         optims[0].param_groups[0]["lr"],
                         loss, protoc_losses, t_epoch, epoch, schedulers[0].last_epoch,
                         start_time, False, None, idx,
                         stats_file, writer, net_idx=net_idx)
    return


def run_train_loop(arg, train_loader, val_loader, test_loader, train_eval_loader,
                   train_raw_loader, ood_test_loader, ood_control_loader,
                   kwargs_loader, writer=None, stats_file=None, criterion=None):
    if arg.single_model:
        _get_model = get_single_model
        arg.warmup = np.inf
    else:
        _get_model = get_coteaching_models
    models, optims, schedulers, last_run = _get_model(arg, len_loader=len(train_loader))
    start_epoch, best_acc_b, best_auc, best_tst_acc_b = last_run

    forget_rate_schedule = forget_rate_scheduler(arg.epochs, arg.forget_rate, arg.num_gradual, arg.exponent)
    ood_thr_schedule = forget_rate_scheduler(arg.epochs, arg.ood_thr, arg.num_gradual, arg.exponent)

    ood_masks = [np.zeros(len(train_loader.dataset), dtype='uint8') for _ in range(2)]
    # models = models[:1]
    ood_thr_dict = None

    start_time = time.time()
    for epoch in range(start_epoch, arg.epochs):
        # train
        outputs_train_raw = [None, None]
        if epoch < arg.warmup:

            for i in range(len(models)):
                if arg.single_model:
                    _optims, _schedulers = optims, schedulers
                else:
                    _optims, _schedulers = optims[i * 2:i * 2 + 2], schedulers[i * 2:i * 2 + 2]
                run_train_epoch(arg, models[i], _optims, _schedulers,
                                train_loader, epoch, None, None,
                                start_time, writer=writer, net_idx=i + 1, criterion=criterion)
        else:
            def train_net(model_a, model_b, _optims, _schedulers, net_idx):
                nonlocal train_raw_loader, outputs_train_raw

                """Filter label noise from the actual training set """
                noisy_idx_b, losses_b = eval_train(model_b, train_raw_loader, epoch, 2 if net_idx == 1 else 1,
                                                   writer, stats_file, None, outputs_train_raw[0], criterion=criterion)
                _train_loader = make_train_loader_wrapper(
                    arg,
                    removed_idx=[noisy_idx_b],  # order matters
                    kwargs_loader=kwargs_loader,
                )
                run_train_epoch(
                    arg, model_a, _optims, _schedulers, _train_loader, epoch, None, None,
                    start_time, writer=writer, net_idx=net_idx, criterion=criterion,
                )
                _ood_thr = eval_ood(
                    [model_b], train_raw_loader, ood_test_loader, epoch=epoch, writer=writer,
                    start_filter_epoch=-1, ood_thr=100 - ood_thr_schedule[epoch],
                    net_idx=[2 if net_idx == 1 else 1]
                    # net_idx=[net_idx]
                )
                return _ood_thr  # _ood_thr

            ood_thr_dict1 = train_net(*models, optims[:2], schedulers[:2], net_idx=1)  # Net 1
            ood_thr_dict2 = train_net(*models[::-1], optims[2:], schedulers[2:], net_idx=2)  # Net 2
            if ood_thr_dict1 is not None:
                ood_thr_dict = {k: max(ood_thr_dict1[k], ood_thr_dict2[k]) for k in ood_thr_dict1.keys()}

        kwargs_eval = dict(epoch=epoch, writer=writer, declare_thr=[0.4, 0.4],
                           ood_thr_dict=ood_thr_dict)
        correcting_params = (arg.inv_dif_thr, arg.prob_thr) if epoch >= arg.epoch_start_correct else None
        correcting_info = evaluate(models, train_eval_loader, set_name='train',
                                   correcting_params=correcting_params, **kwargs_eval)
        scores, out_val = evaluate(models, val_loader, set_name='val', **kwargs_eval)
        scores_tst, out_tst = evaluate(models, test_loader, set_name='test', **kwargs_eval)
        # best_acc_b, best_tst_acc_b, best_auc = save_results(arg, models, epoch, scores, scores_tst,
        #                                                     optims, schedulers,
        #                                                     best_acc_b, best_tst_acc_b, best_auc, stats_file=None)

        if epoch >= arg.epoch_start_correct:
            correct_labels_with_ilr(arg, train_loader, train_eval_loader, epoch, correcting_info, kwargs_loader)
            # evaluate_ood_viz(models[0], train_raw_loader, ood_test_loader, ood_control_loader,
            #                  set_name='train_tsne', epoch=epoch, writer=writer,
            #                  val_loader=val_loader, test_loader=test_loader, out_val=out_val, out_tst=out_tst)


def make_train_loader_wrapper(arg, noisy_idx=None, ood_idx=None, kwargs_loader=None, is_raw=False, trn_ds=None,
                              removed_idx: list = None, oversampling=False, transform=None):
    def filter_ds(ds):
        file_idx = np.arange(len(ds))
        if removed_idx is not None:  # remove files in order of list in removed_idx
            assert noisy_idx is None and ood_idx is None
            for rm_idx in removed_idx:
                if rm_idx is not None:
                    file_idx = file_idx[np.invert(rm_idx.astype('bool'))]
        else:
            if noisy_idx is not None:
                file_idx = np.setdiff1d(file_idx, np.argwhere(noisy_idx))
            if ood_idx is not None:
                file_idx = np.setdiff1d(file_idx, np.argwhere(ood_idx))
        if len(file_idx) < len(ds):
            kept_idx = np.zeros(len(ds), dtype='bool')
            kept_idx[file_idx] = True
            ds.filter_by_idx(kept_idx)

    if transform is None:
        if is_raw:
            transform = CropFixSize(
                sz=arg.crop_size,
                in_channels=arg.in_channels
            )
        else:
            aug_func = OneCropTransform if not arg.time_series else JustCropTransform  # OneCropTransform OneCropTransformSupervise
            transform = aug_func(not arg.random_crop,
                                 size=(arg.crop_size, arg.crop_size),
                                 in_channels=arg.in_channels)

    if trn_ds is None:
        trn_ds = PatchLabeledDataset(
            arg.data_root,
            inv_range=(arg.min_inv, 1), pid_range=(0, 100), gs_range=(7, 10),
            # inv_range=(arg.min_inv, arg.min_inv+0.3), pid_range=(0, 100), gs_range=(7, 10),
            transform=transform,
            slide_idx=None, stats=None, norm=arg.norm, time_series=arg.time_series,
            return_idx=not is_raw, file_idx=None,
            oversampling_cancer=False if arg.single_model else True,
            # oversampling_cancer=False,
        )
    filter_ds(trn_ds)

    if kwargs_loader is None:
        kwargs_loader = dict(batch_size=arg.batch_size, num_workers=arg.workers, pin_memory=True)

    train_sampler = None
    drop_last = False
    if (not is_raw) or oversampling:
        train_sampler = create_loader(trn_ds, arg.batch_size, jobs=arg.workers, add_sampler=True,
                                      get_sampler_only=True, weight_by_inv=False)
        drop_last = True
    train_loader = torch.utils.data.DataLoader(trn_ds, sampler=train_sampler, drop_last=drop_last,
                                               # shuffle=True,
                                               **kwargs_loader)
    return train_loader


def train(arg, writer=None):
    stats_file = open('/'.join([arg.exp_dir, "stats.txt"]), "a", buffering=1)
    if arg.pretrained is None:
        arg.weights = 'finetune'  # must train the entire network
    print(arg, file=stats_file)

    data_root = arg.data_root
    data_root_test = arg.data_root_test
    stats = None

    kwargs_ds = dict(gs_range=(7, 10),
                     transform=CropFixSize(
                         sz=arg.crop_size,
                         in_channels=arg.in_channels
                     ),
                     stats=stats, norm=arg.norm,
                     time_series=arg.time_series)
    trn_eval_ds = PatchLabeledDataset(data_root_test, inv_range=(0.4, 1), pid_range=(0, 100), **kwargs_ds)
    val_ds = PatchLabeledDataset(data_root_test, inv_range=(0.4, 1), pid_range=(101, 130), **kwargs_ds)
    tst_ds = PatchLabeledDataset(data_root_test, inv_range=(0.4, 1), pid_range=(131, 200), **kwargs_ds)

    kwargs_raw_ds = dict(inv_range=(arg.min_inv, 1), pid_range=(0, 100), gs_range=(7, 10),
                         transform=CropFixSize(
                             sz=arg.crop_size,
                             in_channels=arg.in_channels
                         ),
                         slide_idx=None,
                         stats=stats, norm=arg.norm, time_series=arg.time_series)
    trn_raw_ds = PatchLabeledDataset(data_root, **kwargs_raw_ds)
    ood_test_ds = PatchLabeledDataset(arg.data_root_ood_test, **kwargs_raw_ds)
    ood_control_ds = PatchLabeledDataset(arg.data_root_ood_control, **kwargs_raw_ds)

    kwargs_loader = dict(batch_size=arg.batch_size, num_workers=arg.workers, pin_memory=True)
    train_eval_loader = torch.utils.data.DataLoader(trn_eval_ds, **kwargs_loader)
    val_loader = torch.utils.data.DataLoader(val_ds, **kwargs_loader)
    test_loader = torch.utils.data.DataLoader(tst_ds, **kwargs_loader)
    ood_test_loader = torch.utils.data.DataLoader(ood_test_ds, **kwargs_loader)
    ood_control_loader = torch.utils.data.DataLoader(ood_control_ds, **kwargs_loader)

    train_loader = make_train_loader_wrapper(arg)
    train_raw_loader = make_train_loader_wrapper(arg, is_raw=True)

    if arg.criterion.lower() != 'isomax':
        from utils import get_loss_function
        criterion = get_loss_function(arg.criterion, reduction='mean')
        # criterion = nn.CrossEntropyLoss()
    else:
        criterion = IsoMaxLossSecondPart(arg.entropic_scale)

    # Training loop
    run_train_loop(arg, train_loader,
                   val_loader, test_loader, train_eval_loader,
                   train_raw_loader, ood_test_loader, ood_control_loader,
                   kwargs_loader, writer, stats_file, criterion=criterion)


def main():
    gc.collect()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args_cmd = parse_args()
    with open(f'{root_dir}/yamls/{args_cmd["config"]}') as f:
        arg = yaml.load(f, Loader)
    arg.update(args_cmd)
    arg = munchify(arg)
    print(arg)

    arg.exp_dir = '/'.join([arg.root_dir, arg.exp_dir]) + args_cmd['exp_suffix']
    os.makedirs(arg.exp_dir, exist_ok=True)

    use_wandb = False

    # read the yaml
    fix_random_seed(arg.seed, benchmark=True, deterministic=True)

    # Log in to your W&B account
    if use_wandb:
        wandb.login()
        # init wandb using config and experiment name
        wandb.init(config=vars(arg),
                   project='prostate_cancer_classification_space_time',
                   group=arg.exp_name,
                   sync_tensorboard=True,  # enable tensorboard sync
                   name=arg.exp_dir)

    writer = SummaryWriter(logdir=arg.exp_dir, flush_secs=10, filename_suffix='')
    train(arg, writer)

    # Mark the run as finished
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
