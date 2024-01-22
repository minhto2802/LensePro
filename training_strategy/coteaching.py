import copy
import random

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from utils import scheduler
from torch.nn import functional as F
from utils.analyze_non_update import Analyzer
try:
    from cleanlab.coteaching import forget_rate_scheduler
except:
    from cleanlab.experimental.coteaching import forget_rate_scheduler

from loss_functions import (
    avu_utils as util,
    avu_loss,
)
from .vanilla import Model
from utils.novograd import NovoGrad
from utils import nested_dropout as utils
from utils.get_ssl_alg import gen_ssl_alg
from utils.semi_supervise import TSA, STSA
from loss_functions.simclr import info_nce_loss
from utils.scheduler import WARMUP_SCHEDULER
from utils.get_loss_function import uncertainty_driven
from loss_functions.coteaching_loss import get_loss_coteaching


class CoTeaching(Model):
    def __init__(self, network: list, device, num_class, mode, aug_type='none',
                 loss_name='gce', use_plus=False, num_positions=8, *args, **kwargs):
        super(CoTeaching, self).__init__(device, num_class, mode, aug_type, *args, **kwargs)
        self.model_name = 'CoTeaching'
        self.net1 = network[0]()
        self.net2 = network[1]()
        self.optimizer1, self.optimizer2, self.scheduler1, self.scheduler2 = [None, ] * 4
        self.forget_rate_schedule = None
        self.loss_func = get_loss_coteaching(loss_name, self.num_class, use_plus=use_plus, **kwargs)
        self.loss_func_pos = get_loss_coteaching(loss_name, num_classes=num_positions, use_plus=use_plus, **kwargs)
        self.network_list = [self.net1, self.net2]
        self.params_list = [list(self.net1.parameters()), list(self.net2.parameters())]
        self.class_names = ['benign', 'cancer']
        self.extra = {'loss_thr': [np.Inf for _ in range(self.num_class)]}

    class _Decorators:
        @classmethod
        def pre_forward(cls, decorated):
            """Check and return correct inputs before doing supervised forward
            :param decorated: forward_backward method
            """

            def wrapper(model: Model, *args, **kwargs):
                if 'semi_supervised' in kwargs.keys():
                    if kwargs['semi_supervised']:
                        return model.forward_backward_semi_supervised(*args, **kwargs)
                return decorated(model, *args, **kwargs)

            return wrapper

    def init_optimizers(self, lr: float = 1e-3, n_epochs=None, n_batches=None, coteaching_cfg=None, *args, **kwargs):
        """
        Create optimizers for networks listed in self.network_list
        :param lr:
        :param n_epochs:
        :param n_batches:
        :param coteaching_cfg
        :return:
        """
        # Set-up learning rate scheduler alpha and betas for Adam Optimizer
        net1, net2 = self.network_list
        # self.optimizer1 = optim.AdamW(self.params_list[0], lr=float(lr))  #, weight_decay=1e-3)
        # self.optimizer2 = optim.AdamW(self.params_list[1], lr=float(lr))  #, weight_decay=1e-3)
        self.optimizer1 = NovoGrad(self.params_list[0], lr=float(lr),
                                   weight_decay=1e-3)  # , grad_averaging=True, weight_decay=0.001)
        self.optimizer2 = NovoGrad(self.params_list[1], lr=float(lr),
                                   weight_decay=1e-3)  # , grad_averaging=True, weight_decay=0.001)

        self.scheduler1 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer1, lr, epochs=n_epochs,
                                                              steps_per_epoch=n_batches,
                                                              pct_start=0.2, anneal_strategy='cos', cycle_momentum=True,
                                                              base_momentum=0.85,
                                                              max_momentum=0.95, div_factor=10.0,
                                                              final_div_factor=10000.0, three_phase=False,
                                                              last_epoch=-1, verbose=False)
        self.scheduler2 = torch.optim.lr_scheduler.OneCycleLR(self.optimizer2, lr, epochs=n_epochs,
                                                              steps_per_epoch=n_batches,
                                                              pct_start=0.2, anneal_strategy='cos', cycle_momentum=True,
                                                              base_momentum=0.85,
                                                              max_momentum=0.95, div_factor=10.0,
                                                              final_div_factor=10000.0, three_phase=False,
                                                              last_epoch=-1, verbose=False)
        # self.scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer1, 100, T_mult=1, eta_min=lr / 10, last_epoch=-1)  # lr/10
        # self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer2, 100, T_mult=1, eta_min=lr / 10, last_epoch=-1)  # lr/10

        if coteaching_cfg is None:
            ValueError('CoTeaching configuration must be specified when coteaching training is used')
        if coteaching_cfg.num_gradual_iter > 0:  # the number of iterations is given
            num_gradual = coteaching_cfg.num_gradual_iter
        elif coteaching_cfg.num_gradual > 0:  # the number of epochs is given
            num_gradual = coteaching_cfg.num_gradual * n_batches
        else:  # forget_rate = 0
            coteaching_cfg.forget_rate = 0
            num_gradual = 0
        self.forget_rate_schedule = WARMUP_SCHEDULER[coteaching_cfg.schedule](
            coteaching_cfg.forget_rate, num_gradual)
        # self.forget_rate_schedule = forget_rate_scheduler(n_epochs, forget_rate, num_gradual, exponent)

    def optimize(self, loss: tuple):
        """

        :param loss:
        :return:
        """
        loss1, loss2 = loss
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        if self.scheduler1:
            self.scheduler1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        if self.scheduler2:
            self.scheduler2.step()

    def save(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        [torch.save(net.state_dict(), f'{checkpoint_dir}/{prefix}{sep}{name}_{i + 1}_state_dict.pth')
         for (i, net) in enumerate(self.network_list)]
        [torch.save(net, f'{checkpoint_dir}/{prefix}{sep}{name}_{i + 1}.pth')
         for (i, net) in enumerate(self.network_list)]

    def load(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        [net.load_state_dict(torch.load(f'{checkpoint_dir}/{prefix}{sep}{name}_{i + 1}.pth'))
         for (i, net) in enumerate(self.network_list)]

    def infer(self, x_raw, positions=None, mode='train', **kwargs):
        if mode == 'train':
            # return self.net1(x_raw, positions), self.net2(x_raw, positions)
            return self.net1(x_raw), self.net2(x_raw)
        # mode == test, torch.no_grad() was called outside
        # return self.net1(x_raw, positions, **kwargs)
        return self.net1(x_raw)

    def forward_backward(self, x_raw, n_batch, labels, *args, **kwargs):
        x_raw = x_raw.unsqueeze(1) if x_raw.ndim < 3 else x_raw

        out1, out2 = self.infer(x_raw, n_batch)
        loss1, loss2, extra = self.compute_loss(out1, out2, labels, self.loss_func, **kwargs)
        self.optimize((loss1, loss2))
        return out1, out2, loss1, loss2, extra

    @staticmethod
    def compute_loss(out1, out2, labels, loss_funcs, loss_coefficients=None, **kwargs):
        return loss_funcs(out1, out2, torch.argmax(labels, dim=1), **kwargs)

    @staticmethod
    def on_batch_end(t_epoch, **kwargs):
        t_epoch.set_postfix(**kwargs)

    @staticmethod
    def add_histogram_v0(writer, all_ind, epoch):
        for k in range(2):
            writer.add_histogram(f'Benign_{k + 1}',
                                 np.concatenate([_ for (i, _) in enumerate(all_ind['benign']) if i % 2 == k]),
                                 epoch
                                 )
            writer.add_histogram(f'Cancer_{k + 1}',
                                 np.concatenate([_ for (i, _) in enumerate(all_ind['cancer']) if i % 2 == k]),
                                 epoch
                                 )
            writer.add_histogram(f'cancer_percentage_{k + 1}',
                                 np.array([_ for (i, _) in enumerate(all_ind['cancer_ratio']) if i % 2 == k]),
                                 epoch)

    def add_histogram(self, writer, extra_outputs, epoch):
        for k in extra_outputs.keys():
            if k == 'loss':
                continue
            extra_outputs[k] = np.array(extra_outputs[k], dtype='float32')
        # min_loss = np.array(extra_outputs['min_loss']).min(axis=0)

        for n in range(2):  # 2 networks
            for i, c in enumerate(self.class_names):
                # writer.add_histogram(f'{c}_min_max_dist_{n+1}', extra_outputs['cutoff_thr'][:, i, n] - min_loss[i, n], epoch)
                writer.add_histogram(f'{c}_loss_{n + 1}', np.concatenate([l[i][n] for l in extra_outputs['loss']]),
                                     epoch)
            writer.add_histogram(f'num_kept_{n + 1}', extra_outputs['num_kept'][:, n], epoch)

    def set_train(self):
        [_.train() for _ in self.network_list]

    def train(self, epoch, trn_dl, writer=None, *args, **kwargs):
        """

        :param writer:
        :param epoch:
        :param trn_dl:
        :return:
        """
        self.set_train()
        correct, total = 0, 0
        end_training = False
        # total_steps = self.scheduler1.total_steps
        # all_ind = {'benign': [], 'cancer': [], 'cancer_ratio': []}
        # extra_outputs = {'cutoff_thr': [], 'min_loss': [], 'num_kept': [], 'loss': [],
        #                  'ind_non_update': []}  # for two networks
        # unc_list = []
        x_nu, y_nu, n_nu, loss_weights_nu = None, None, None, None
        index_nu = []  # non-update
        # analyzer = Analyzer(trn_dl.dataset)

        with tqdm(trn_dl, unit="batch") as t_epoch:
            global_step = self.scheduler1.last_epoch
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):
                batch = self.reshape_batch(batch, trn_dl)

                # Parse batch
                x_raw, y_batch, n_batch, index, loss_weights = batch
                # x_raw = self.intra_cut_mix.before_batch(x_raw, y_batch.argmax(1))

                forget_rate = self.forget_rate_schedule(global_step)

                if x_nu is not None:  # and (epoch > 4)
                    x_raw = torch.cat([x_raw, x_nu])
                    y_batch = torch.cat([y_batch, y_nu])
                    n_batch = torch.cat([n_batch, n_nu])
                    index = torch.cat([index, index_nu])
                    loss_weights = torch.cat([loss_weights, loss_weights_nu])
                    forget_rate += x_nu.size(0) / trn_dl.batch_size

                # Forward & Backward
                # try:
                out1, out2, loss1, loss2, extra_outputs = self.forward_backward(
                    x_raw, n_batch, y_batch,
                    loss_weights=loss_weights,
                    forget_rate=forget_rate,
                    step=epoch * i,
                    index=index,
                    epoch=epoch,
                    batch_size=trn_dl.batch_size,
                    n_views=trn_dl.dataset.n_views,
                    global_step=global_step,
                    # extra_outputs=extra_outputs,
                    loss_thr=self.extra['loss_thr'],
                )
                # except:
                #     end_training = True
                #     break

                total += y_batch.size(0)
                correct += (self.to_prob(out1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
                self.on_batch_end(t_epoch, loss=loss1.item() + loss2.item(), acc=correct / total,
                                  opt1_lr=self.optimizer1.param_groups[0]['lr'],
                                  opt2_lr=self.optimizer2.param_groups[0]['lr'])
                if writer:
                    writer.add_scalar(f'LEARNING_RATE', self.optimizer1.param_groups[0]['lr'], global_step)

                # if 'ind' in extra.keys():
                #     ind = extra['ind']
                #     [extra_outputs[k].extend(ind[k]) for k in ind.keys()]
                # if 'unc' in extra.keys():
                #     unc_list.append(extra['unc'])

                # if (global_step % 10) == 0:
                #     pass
                # break

                # if len(extra_outputs['ind_non_update']) > 0:
                #     ind_nu = extra_outputs['ind_non_update']
                #     index_nu.append(index[ind_nu].cpu().detach())

                # x_nu, y_nu, n_nu, loss_weights_nu, index_nu = \
                #     x_raw[ind_nu], y_batch[ind_nu], n_batch[ind_nu], loss_weights[ind_nu], index[ind_nu]

        # if epoch > 14:
        #     self.estimate_loss_thr(extra_outputs, global_step, writer)

        # self.estimate_loss_thr(extra_outputs, global_step)
        # self.add_histogram(writer, extra_outputs, epoch)
        # analyzer.update(index_nu, epoch, forget_rate, writer)

        return loss1.item() + loss2.item(), correct / total, end_training

    def eval(self, tst_dl, device=None, net_index=1, **kwargs):
        """

        :param net_index: 1 or 2
        :param tst_dl:
        :param device:
        :return: outputs and signal-wise accuracy
        """
        [_.eval() for _ in self.network_list]
        return super(CoTeaching, self).eval(tst_dl, device, **kwargs)

    def estimate_loss_thr(self, extra_outputs, global_step, writer=None):
        """

        :param extra_outputs:
        :param global_step:
        :param writer:
        :return:
        """
        if 'loss' not in extra_outputs.keys():
            return

        for n in range(2):  # 2 networks
            loss_c = []
            self.extra['loss_thr'][n] = []
            for i, c in enumerate(self.class_names):
                loss_c.append(np.concatenate([l[i][n] for l in extra_outputs['loss']]))
                self.extra['loss_thr'][n].append(np.quantile(loss_c[-1], 1 - self.forget_rate_schedule(global_step)))
                if writer is not None:
                    writer.add_scalar(f'loss_thr/{c}_{n}', self.extra['loss_thr'][n][-1], global_step)


class CoTeachingMultiTask(CoTeaching):
    def __init__(self, *args, **kwargs):
        super(CoTeachingMultiTask, self).__init__(*args, **kwargs)
        self.loss_coefficients = kwargs['opt'].train.loss_coefficients

    def forward_backward(self, x_raw, n_batch, labels, *args, **kwargs):
        out1, out2 = self.infer(x_raw, n_batch)

        loss1, loss2, ind = self.compute_loss(out1, out2,
                                              [labels, n_batch],
                                              [self.loss_func, self.loss_func_pos],
                                              loss_coefficients=self.loss_coefficients,
                                              **kwargs)
        self.optimize((loss1, loss2))
        return out1[0], out2[0], loss1, loss2, {'ind': ind}

    def infer(self, x_raw, positions, mode='train'):
        if mode == 'train':
            return self.net1(x_raw, positions), self.net2(x_raw, positions)
        # mode == test, torch.no_grad() was called outside
        return self.net1(x_raw, positions)[0]  # only collect the first output

    @staticmethod
    def compute_loss(out1, out2, labels, loss_funcs, loss_coefficients=None, **kwargs):
        if loss_coefficients is None:
            loss_coefficients = [1. for _ in range(len(out1))]
        loss0, loss1, ind = [], [], []
        for _out1, _out2, _labels, _loss_func in zip(out1, out2, labels, loss_funcs):
            _loss0, _loss1, _ind = _loss_func(_out1, _out2, torch.argmax(_labels, dim=1), **kwargs)
            loss0.append(_loss0), loss1.append(_loss1), ind.append(_ind)

        loss0 = torch.stack([l * coef for (l, coef) in zip(loss0, loss_coefficients)]).sum()
        loss1 = torch.stack([l * coef for (l, coef) in zip(loss1, loss_coefficients)]).sum()
        ind = ind[0]

        return loss0, loss1, ind


class CoTeachingSelfTrain(CoTeaching):
    def __init__(self, *args, **kwargs):
        super(CoTeachingSelfTrain, self).__init__(*args, **kwargs)
        device = args[1]
        ckpt = kwargs['ckpt']

        self.net1.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
        self.net1.eval()
        out = self.net1(torch.rand((2, 1, 200)).cuda(device))
        self.clf1 = kwargs['classifier'][0](in_channels=out.shape[1])
        self.clf2 = kwargs['classifier'][1](in_channels=out.shape[1])

        # Override the network list
        # self.network_list = [self.clf1, self.clf2]
        self.params_list = [self.params_list[0] + list(self.clf1.parameters()),
                            self.params_list[1] + list(self.clf2.parameters())]
        # self.network_list += [self.clf1, self.clf2]

    def infer(self, x_raw, positions, mode='train'):
        if mode == 'train':
            # with torch.no_grad():
            #     out = self.net1(x_raw)
            # return self.clf1(out, positions), self.clf2(out, positions)
            return self.clf1(self.net1(x_raw), positions), self.clf2(self.net2(x_raw), positions)
        # mode == test, torch.no_grad() was called outside
        return self.clf1(self.net1(x_raw), positions)


class CoTeachingUncertaintyAvU(CoTeaching):
    """Co-teaching with uncertainty AvU"""
    opt_h = 1.  # optimal threshold
    beta = 3.
    avu_criterion = avu_loss.AvULoss().cuda()
    unc = 0

    def on_batch_end(self, t_epoch, **kwargs):
        kwargs['unc'] = self.unc
        super(CoTeachingUncertaintyAvU, self).on_batch_end(t_epoch, **kwargs)

    def add_unc_loss(self, loss, out, labels, kl=None):
        labels = labels.argmax(dim=1)

        probs_ = self.to_prob(out)
        probs = probs_.data.cpu().numpy()
        pred_entropy = util.entropy(probs)
        unc = np.mean(pred_entropy, axis=0)
        preds = np.argmax(probs, axis=-1)
        scaled_kl = kl.data / 1e6 if kl is not None else 0.

        # avu = util.accuracy_vs_uncertainty(np.array(preds),
        #                                    np.array(labels.cpu().data.numpy()),
        #                                    np.array(pred_entropy), self.opt_h)
        #
        # # cross_entropy_loss = criterion(output, target_var)
        #
        elbo_loss = loss + scaled_kl
        avu_loss = self.beta * self.avu_criterion(out, labels, self.opt_h, type=0)
        loss = loss + scaled_kl + avu_loss
        return loss, None, elbo_loss, unc

    def forward_backward(self, x_raw, n_batch, labels, **kwargs):
        # (out1, kl1), (out2, kl2) = self.infer(x_raw, n_batch)
        out1, out2 = self.infer(x_raw, n_batch)

        loss1, loss2, ind = self.loss_func(out1, out2, torch.argmax(labels, dim=1), **kwargs)
        # loss1 = loss1 + kl1.data / 1e6
        # loss2 = loss2 + kl2.data / 1e6

        loss1, avu1, elbo_loss1, unc1 = self.add_unc_loss(loss1, out1, labels, None)
        loss2, avu2, elbo_loss2, unc2 = self.add_unc_loss(loss2, out2, labels, None)

        extra = {'ind': ind}  # 'unc': unc1 + unc2, 'elbo_loss': elbo_loss1 + elbo_loss2}
        self.unc = unc1.item() + unc2.item()

        self.optimize((loss1, loss2))
        return out1, out2, loss1, loss2, extra


class CoTeachingSemiSupervise(CoTeaching):
    def __init__(self, *args, **kwargs):
        super(CoTeachingSemiSupervise, self).__init__(*args, **kwargs)

        self.coef_sup = torch.nn.Parameter(torch.Tensor([.6, .6]))
        self.coef_unsup = torch.nn.Parameter(torch.Tensor([.6, .6]))
        self.params_list += [list(self.coef_sup), list(self.coef_sup)]

        opt = None if 'opt' not in kwargs.keys() else kwargs['opt']
        if opt.is_eval:
            return
        if opt is not None:
            cfg = opt.ssl
            self.ssl_alg = gen_ssl_alg(cfg[cfg.ssl_name]) if cfg.ssl_name != 'none' else None
            self.coef = scheduler.ExpWarmup(float(opt.ssl.coef), opt.ssl.warmup_iter)

            total_steps = int(np.ceil(opt.num_samples['train'] / opt.train_batch_size)) * opt.n_epochs
            if cfg.tsa > cfg.stsa:
                self.anneal = TSA(cfg.tsa, total_steps, opt.tasks_num_class[0], cfg.schedule)
            else:
                self.anneal = STSA(cfg.stsa, total_steps, opt.tasks_num_class[0], cfg.schedule)
        else:
            raise ValueError('Option must be specified if semi-supervised training is used!')

    def forward_backward(self, x_raw, n_batch, labels, *args, **kwargs):
        """

        :param x_raw:
        :param n_batch:
        :param labels:
        :param args:
        :param kwargs:
        :return:
        """
        x_raw = x_raw.unsqueeze(1) if x_raw.ndim < 3 else x_raw
        out1, out2 = self.infer(x_raw, n_batch)
        loss1, loss2, ind = self.loss_func(out1, out2, torch.argmax(labels, dim=1),
                                           anneal=self.anneal,
                                           **kwargs)

        x_unsup = kwargs['x_unsup'].unsqueeze(1)
        out_unsup_1, out_unsup_2 = self.infer(x_unsup)

        loss_unsup_1 = self.ssl_alg.consistency(*self.ssl_alg(*torch.chunk(out_unsup_1, 2)))
        loss_unsup_2 = self.ssl_alg.consistency(*self.ssl_alg(*torch.chunk(out_unsup_2, 2)))

        if self.coef.base_value < 0:
            loss1, loss2 = uncertainty_driven(loss1, self.coef_sup[0]), uncertainty_driven(loss2, self.coef_sup[1])
            loss_unsup_1 = uncertainty_driven(loss_unsup_1, self.coef_unsup[0])
            loss_unsup_2 = uncertainty_driven(loss_unsup_2, self.coef_unsup[1])
        else:
            coef = self.coef(kwargs['global_step'])
            loss_unsup_1 = loss_unsup_1 * self.coef(kwargs['global_step'])
            loss_unsup_2 = loss_unsup_2 * self.coef(kwargs['global_step'])

            if 'writer' in kwargs.keys():
                kwargs['writer'].add_scalar('UNSUP_COEF', coef, kwargs['global_step'])

        loss1 += loss_unsup_1
        loss2 += loss_unsup_2
        self.optimize((loss1, loss2))
        return (out1, out2,
                loss1, loss2,
                loss_unsup_1.cpu().item(), loss_unsup_2.cpu().item(), {'ind': ind})

    def train(self, epoch, trn_dl, trn_unsup_dl=None, writer=None):
        """

        :param writer:
        :param epoch:
        :param trn_dl:
        :param trn_unsup_dl:
        :return:
        """
        [_.train() for _ in self.network_list]
        correct, total = 0, 0
        all_ind = {'benign': [], 'cancer': [], 'cancer_ratio': []}
        unc_list = []
        global_step = len(trn_dl) * epoch
        # total_loss_unsup_1, total_loss_unsup_2 = [], []

        t_epoch = tqdm(unit="batch", total=min(len(trn_dl), len(trn_unsup_dl)))
        t_epoch.set_description(f"Epoch {epoch}")

        for i, batch in enumerate(zip(trn_dl, trn_unsup_dl)):
            # Parse batch
            batch_sup, batch_unsup = batch
            if isinstance(batch_sup[0], list):
                batch_sup = [torch.cat(_, 0).to(self.device) for _ in batch_sup]
            else:
                if batch_sup[0].ndim > 2:
                    batch_sup = [_.view(-1, _.shape[-1]).to(self.device)
                                 if _.ndim > 2 else _.view(-1).to(self.device) for _ in batch_sup]

            x_raw, y_batch, n_batch, index, loss_weights = batch_sup
            x_unsup = torch.cat(batch_unsup, 0)

            # Forward & Backward
            out1, out2, loss1, loss2, loss_unsup_1, loss_unsup_2, extra = self.forward_backward(
                x_raw, n_batch, y_batch,
                loss_weights=loss_weights,
                forget_rate=self.forget_rate_schedule(global_step),
                step=epoch * i,
                index=index,
                epoch=epoch,
                batch_size=trn_dl.batch_size,
                x_unsup=x_unsup,
                global_step=global_step,
                writer=writer,
            )
            total += y_batch.size(0)
            correct += (self.to_prob(out1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
            self.on_batch_end(t_epoch, loss=loss1.item() + loss2.item(), acc=correct / total,
                              g_step=global_step,
                              opt1_lr=self.optimizer1.param_groups[0]['lr'],
                              opt2_lr=self.optimizer2.param_groups[0]['lr'])
            # total_loss_unsup_1.append(loss_unsup_1)
            # total_loss_unsup_2.append(loss_unsup_2)
            t_epoch.update()

            writer.add_scalar(f'loss/TRAIN_UNSUP_1', loss_unsup_1, global_step)
            writer.add_scalar(f'loss/TRAIN_UNSUP_2', loss_unsup_2, global_step)
            writer.add_scalar(f'LEARNING_RATE', self.optimizer1.param_groups[0]['lr'], global_step)

            if 'unc' in extra.keys():
                unc_list.append(extra['unc'])

            total += y_batch.size(0)
            correct += (self.to_prob(out1).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
            global_step += 1

        t_epoch.close()
        # writer.add_scalar(f'loss/TRAIN_UNSUP_1', np.mean(total_loss_unsup_1), epoch)
        # writer.add_scalar(f'loss/TRAIN_UNSUP_2', np.mean(total_loss_unsup_2), epoch)
        return 0, 0
        # return loss1.item() + loss2.item(), correct / total


class CoTeachingNested(CoTeaching):

    def __init__(self, *args, **kwargs):
        super(CoTeachingNested, self).__init__(*args, **kwargs)

        opt = None if 'opt' not in kwargs.keys() else kwargs['opt']
        if opt is None:
            raise ValueError('Option must be specified if nested-dropout training is used!')
        self.cfg = opt.nested

        # Optimizer
        _optim = self.cfg.optim
        self.warmup_optimizer1 = torch.optim.Adam(self.net1.parameters(), _optim.lr, weight_decay=_optim.weight_decay)
        self.warmup_optimizer2 = torch.optim.Adam(self.net2.parameters(), _optim.lr, weight_decay=_optim.weight_decay)
        self.warmup_optimizers = [self.warmup_optimizer1, self.warmup_optimizer2]

        # generate mask
        feat_dim = self.net1.feature_size
        self.mask_feat_dim = []
        for i in range(feat_dim):
            tmp = torch.cuda.FloatTensor(1, feat_dim).fill_(0)
            tmp[:, : (i + 1)] = 1
            self.mask_feat_dim.append(tmp)

        # distribution and test function
        self.dist = self.gaussian_dist(self.cfg.mu, self.cfg.nested, feat_dim) if self.cfg.nested > 0 else None

    # def forward_backward(self, x, n_batch, labels, *args, **kwargs):
    #     with torch.no_grad():
    #         self.set_eval()
    #         ind_1, ind_2 = self.compute_loss(*self.infer(x, n_batch), labels, self.loss_func, get_index_only=True,
    #                                          **kwargs)
    #     self.set_train()
    #     out1 = self.infer_nested(self.net1, x, n_batch)
    #     out2 = self.infer_nested(self.net2, x, n_batch)
    #     loss1, loss2, ind = self.compute_loss(out1, out2, labels, self.loss_func,
    #                                           ind_1_update=ind_1, ind_2_update=ind_2, **kwargs)
    #
    #     self.optimize((loss1, loss2))
    #     return out1, out2, loss1, loss2, {'ind': ind}

    @staticmethod
    def gaussian_dist(mu, std, n):
        dist = np.array([np.exp(-((i - mu) / std) ** 2) for i in range(1, n + 1)])
        return dist / np.sum(dist)

    def infer_nested(self, net, x, n_batch):
        """

        :param net:
        :param x:
        :param n_batch:
        :return:
        """
        feature = net.module.feat_extractor(x, n_batch)

        if self.dist is not None:
            k = np.random.choice(range(len(self.mask_feat_dim)), p=self.dist)
            mask_k = self.mask_feat_dim[k]
            feature_masked = feature * mask_k
        else:
            feature_masked = F.dropout(feature, p=self.cfg.dropout, training=True)

        logits = net.module.classifier(feature_masked, n_batch)
        return logits

    def warmup_forward_backward(self, net, x, y, n_batch, optimizer, loss_cls, acc):
        """

        :param net:
        :param x:
        :param y:
        :param n_batch:
        :param optimizer:
        :param loss_cls:
        :param acc:
        :return:
        """
        logits = self.infer_nested(net, x, n_batch)
        loss = F.cross_entropy(logits, y.argmax(1))

        # update optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metrics
        loss_cls.update(loss.item(), y.size()[0])
        acc_batch = utils.accuracy(logits, y.argmax(1), topk=(1,))
        acc.update(acc_batch[0].item(), y.size()[0])
        return loss_cls, acc

    def warmup_lr(self, train_loader):
        warmup_iter = self.cfg.warmup_iter
        t_iter = tqdm(unit="iter", total=warmup_iter, desc='Warmup')
        loss_cls1, loss_cls2, acc1, acc2 = [utils.AverageMeter() for _ in range(4)]
        while t_iter.n < warmup_iter:
            [net.train(freeze_bn=False) for net in self.network_list]

            for batchIdx, batch in enumerate(train_loader):
                if self.aug_type != 'none':
                    batch_data = [torch.cat(_, 0).to(self.device) for _ in batch]
                else:
                    batch_data = [_.to(self.device) for _ in batch]
                x_raw, y_batch, n_batch, *_ = batch_data

                lr_update = t_iter.n / float(warmup_iter) * self.cfg.optim.lr
                for _optim in self.warmup_optimizers:
                    for g in _optim.param_groups:
                        g['lr'] = lr_update

                loss_cls1, acc1 = self.warmup_forward_backward(self.net1, x_raw.unsqueeze(1), y_batch, n_batch,
                                                               self.warmup_optimizer1, loss_cls1, acc1)
                loss_cls2, acc2 = self.warmup_forward_backward(self.net2, x_raw.unsqueeze(1), y_batch, n_batch,
                                                               self.warmup_optimizer2, loss_cls2, acc2)
                t_iter.set_postfix(loss1=loss_cls1.avg, loss2=loss_cls2.avg,
                                   acc1=acc1.avg, acc2=acc2.avg,
                                   lr=lr_update)
                t_iter.update()
                if t_iter.n == warmup_iter:
                    t_iter.close()
                    return

    def set_train(self):
        [net.train(self.cfg.freeze_bn) for net in self.network_list]

    def set_eval(self):
        [net.eval() for net in self.network_list]

    # def eval(self, tst_dl, device=None, **kwargs):
    #     [_.eval() for _ in self.network_list]
    #     if device:
    #         self.device = device
    #     outputs = []
    #     entropic_scores = []
    #     total = correct = 0
    #
    #     # apply model on test signals
    #     for batch in tst_dl:
    #         x_raw, y_batch, n_batch, _ = [t.to(self.device) for t in batch]
    #
    #         feature1 = self.net1.module.feat_extractor(x_raw, n_batch)
    #         feature2 = self.net2.module.feat_extractor(x_raw, n_batch)
    #         logits_bag = []
    #
    #         if self.dist is not None:
    #             for i in range(len(self.mask_feat_dim)):
    #                 logits1 = self.net1.module.classifier(feature1 * self.mask_feat_dim[i])
    #                 logits2 = self.net2.module.classifier(feature2 * self.mask_feat_dim[i])
    #                 logits = (logits1 + logits2) * 0.5
    #                 logits_bag.append(logits.unsqueeze(0))
    #         else:
    #             logits1 = self.net1.module.classifier(F.dropout(feature1, p=self.cfg.dropout, training=False))
    #             logits2 = self.net2.module.classifier(F.dropout(feature2, p=self.cfg.dropout, training=False))
    #             logits = (logits1 + logits2) * 0.5
    #             logits_bag.append(logits.unsqueeze(0))
    #
    #         pred = (torch.cat(logits_bag, dim=0)).sum(0)
    #         pred = F.softmax(pred, dim=1)
    #
    #         probabilities = pred  # torch.nn.Softmax(dim=1)(pred)
    #         entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
    #         entropic_scores.append((-entropies).cpu().numpy())
    #
    #         outputs.append(pred.cpu().numpy())
    #         total += y_batch.size(0)
    #         correct += (pred.argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
    #
    #     outputs = np.concatenate(outputs)
    #     entropic_scores = np.concatenate(entropic_scores)
    #     return outputs, entropic_scores, correct / total
