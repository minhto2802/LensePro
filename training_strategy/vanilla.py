import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F

from utils.novograd import NovoGrad
from utils.get_loss_function import *


class Model:
    def __init__(self, device, num_class, mode, aug_type='none', network=None,
                 loss_name=None, *args, **kwargs):
        self.num_class = num_class
        self.device = device
        self.aug_type = aug_type
        self.mode = mode
        self.optimizer, self.scheduler, self.net = [None, ] * 3
        self.model_name = 'Vanilla'
        # self.intra_cut_mix = IntraClassCutMix1d()

        if network:
            self.net = network()
        if loss_name:
            self.loss_func = get_loss_function(loss_name, self.num_class, **kwargs)

    def init_optimizers(self, lr=1e-3, n_epochs=None, n_batches=None, scheduler=None, optimizer='adam', *args,
                        **kwargs):
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=float(lr), amsgrad=True)  # , weight_decay=1e-4)
        elif optimizer.lower() == 'novo':
            self.optimizer = NovoGrad(self.net.parameters(), lr=float(lr),
                                      weight_decay=1e-3)  # , grad_averaging=True, weight_decay=0.001)
        elif optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=float(lr), weight_decay=1e-3)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, steps_per_epoch=554,
        #                                                       epochs=n_epochs)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=lr * 10,
        #                                                    step_size_up=100, cycle_momentum=False,
        #                                                    mode="triangular")
        if scheduler == 'warm_restart':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 10, T_mult=1, eta_min=lr / 10, last_epoch=-1)
        elif scheduler == 'one_cycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, lr, epochs=n_epochs,
                                                                 steps_per_epoch=n_batches,
                                                                 pct_start=0.3, anneal_strategy='cos',
                                                                 cycle_momentum=True,
                                                                 base_momentum=0.85,
                                                                 max_momentum=0.95, div_factor=10.0,
                                                                 final_div_factor=10000.0, three_phase=False,
                                                                 last_epoch=-1, verbose=False)
        else:
            self.scheduler = None

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def save(self, checkpoint_dir, prefix='', name='coreN'):
        sep = '' if prefix == '' else '_'
        torch.save(self.net.state_dict(), f'{checkpoint_dir}/{prefix}{sep}{name}_state_dict.pth')
        torch.save(self.net, f'{checkpoint_dir}/{prefix}{sep}{name}.pth')

    def load(self, checkpoint_dir, prefix='', name='coreN', load_dict=False):
        sep = '' if prefix == '' else '_'
        if load_dict:
            self.net.load_state_dict(torch.load(f'{"/".join([checkpoint_dir, prefix, ])}{sep}{name}_state_dict.pth'))
        else:
            self.net = torch.load(f'{"/".join([checkpoint_dir, prefix, ])}{sep}{name}.pth')

    def infer(self, x, positions, **kwargs):
        # return self.net(x_raw, positions, **kwargs)
        return self.net(x)

    def reshape_batch(self, batch, trn_dl):
        if isinstance(batch[0], list):
            if (self.aug_type != 'none') and (trn_dl.batch_sampler is not None):
                batch = [torch.cat(_, 0).to(self.device) for _ in batch]
            elif (self.aug_type != 'none') and (batch[0].ndim > 2):
                batch = [_.view(-1, _.shape[-1]).to(self.device)
                         if _.ndim > 2 else _.view(-1).to(self.device) for _ in batch]
        else:
            batch = [_.to(self.device) for _ in batch]
            batch[0] = batch[0].unsqueeze(1) if batch[0].ndim == 2 else batch[0]
        return batch

    def forward_backward(self, x_raw, n_batch, y_batch, *args, **kwargs):
        out = self.infer(x_raw, n_batch)
        loss = self.loss_func(out, torch.argmax(y_batch, dim=1), *args, **kwargs).mean()  # index=index
        self.loss_func.update_hist(out.data.detach(),
                                   kwargs['index'].cpu().detach().numpy().tolist())
        self.optimize(loss)
        return out, loss

    @staticmethod
    def to_prob(out):
        return F.softmax(out, dim=1)

    def train(self, epoch, trn_dl, writer=None, *args, **kwargs):
        self.net.train()
        correct, total = 0, 0
        end_training = False
        # total_steps = self.scheduler.total_steps

        with tqdm(trn_dl, unit="batch") as t_epoch:
            global_step = self.scheduler.last_epoch
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):
                batch = self.reshape_batch(batch, trn_dl)

                # Parse batch
                x_raw, y_batch, n_batch, index, loss_weights = batch
                # x_raw = self.intra_cut_mix.before_batch(x_raw, y_batch.argmax(1))

                # Forward &
                # try:
                #     out, loss = self.forward_backward(x_raw, n_batch, y_batch, current_epoch=epoch, iteration=global_step, index=index)
                # except:
                #     end_training = True
                #     break
                out, loss = self.forward_backward(x_raw, n_batch, y_batch, current_epoch=epoch, iteration=global_step,
                                                  index=index)

                total += y_batch.size(0)
                correct += (self.to_prob(out).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
                t_epoch.set_postfix(loss=loss.item(), acc=correct / total,
                                    lr=self.optimizer.param_groups[0]['lr'])
                # break

                if writer:
                    writer.add_scalar(f'LEARNING_RATE', self.optimizer.param_groups[0]['lr'], global_step)

        return loss, correct / total, end_training

    def eval(self, tst_dl, device=None, **kwargs):
        if device:
            self.device = device
        outputs = []
        entropic_scores = []
        features = []
        total = correct = 0
        inputs = []
        labels = []
        if 'coteaching' not in self.model_name.lower():
            self.net.eval()

        # apply model on test signals
        for batch in tst_dl:
            x_raw, y_batch, n_batch, _ = [t.to(self.device) for t in batch]

            # x_raw = x_raw[:, :2]

            pred = self.infer(x_raw, n_batch, mode='test', **kwargs)
            # if 'get_feat' in kwargs.keys():
            #     pred, feature = pred
                # features.append(feature.cpu().numpy())
            pred = self.to_prob(pred)

            probabilities = pred
            entropies = -(probabilities * torch.log(probabilities)).sum(dim=1)
            entropic_scores.append((-entropies).cpu().numpy())

            inputs.append(x_raw.cpu().numpy())
            outputs.append(pred.cpu().numpy())
            labels.append(y_batch.cpu().numpy())
            total += y_batch.size(0)
            correct += (pred.argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()

        inputs = np.concatenate(inputs)
        outputs = np.concatenate(outputs)
        labels = np.concatenate(labels)

        entropic_scores = np.concatenate(entropic_scores)
        # features = np.concatenate(features) if 'get_feat' in kwargs.keys() else features

        return inputs, outputs, labels, entropic_scores, features, correct / total

    def forward_backward_semi_supervised(self, *args, **kwargs):
        pass

    @staticmethod
    def get_activation(name: str, activation: dict):
        """For getting intermediate layer outputs"""

        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    def fade_signal(self, m, stsa_rat=.9):
        """Remove training signal with respect to the current training iteration"""
        num_signal = int(
            min(np.floor(self.current_it / (self.total_iter * stsa_rat / m.shape[0])) + 1, m.shape[0]))
        if num_signal == self.batch_size:
            return m
        else:
            signal = np.zeros(self.batch_size)[:, np.newaxis, np.newaxis, np.newaxis]
            signal[np.random.permutation(self.batch_size)[:num_signal]] = 1
            return m * np.array(signal)


class ModelSam(Model):
    """A adaptive version of Model to train with SAM optimizer"""

    def init_optimizers(self, lr=1e-3, n_epochs=None, rho=5e-2, weight_decay=0, *args, **kwargs):
        from utils.get_optimizer import SAMSGD
        self.optimizer = SAMSGD(self.net.parameters(), lr=float(lr), rho=rho, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 10, T_mult=1, eta_min=0, last_epoch=-1)

    def closure(self, x_raw, y_batch, n_batch, index):
        pass

    def train(self, epoch, trn_dl, **kwargs):
        self.net.train()
        correct, total = 0, 0

        with tqdm(trn_dl, unit="batch") as t_epoch:
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):

                if self.aug_type != 'none':
                    x_raw, y_batch, n_batch, index = [torch.cat(_, 0).to(self.device) for _ in batch]
                else:
                    x_raw, y_batch, n_batch, index = [_.to(self.device) for _ in batch]

                out = None

                def closure():
                    nonlocal out
                    self.optimizer.zero_grad()
                    out = self.infer(x_raw, n_batch)
                    loss = self.loss_func(out, torch.argmax(y_batch, dim=1), index=index).mean()
                    # loss = smooth_crossentropy(out, torch.argmax(y_batch, dim=1)).mean()
                    loss.backward()
                    return loss

                loss = self.optimizer.step(closure)
                self.scheduler.step()

                total += y_batch.size(0)
                correct += (self.to_prob(out).argmax(dim=1) == torch.argmax(y_batch, dim=1)).sum().item()
                t_epoch.set_postfix(loss=loss.item(), acc=correct / total,
                                    lr=self.optimizer.param_groups[0]['lr'])

        return loss, correct / total
