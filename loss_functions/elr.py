from copy import copy
import torch
from torch.nn import functional as F

import numpy as np
from loss_functions.isomax import IsoMaxLossSecondPart


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def sigmoid_rampdown(current, rampdown_length):
    """Exponential rampdown"""
    if rampdown_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampdown_length)
        phase = 1.0 - (rampdown_length - current) / rampdown_length
        return float(np.exp(-12.5 * phase * phase))


class ELR(torch.nn.Module):
    def __init__(self, num_examp, num_classes=2, lmbda=0.5, beta=0.7):
        r"""Early Learning Regularization.
         Parameters
         * `num_examp` Total number of training examples.
         * `num_classes` Number of classes in the classification problem.
         * `lmbda` Regularization strength; must be a positive float, controling the strength of the ELR.
         * `beta` Temporal ensembling momentum for target estimation.
         """
        super(ELR, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.beta = beta
        self.lmbda = lmbda

    def forward(self, output, label, index):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lmbda * elr_reg
        return final_loss


def cross_entropy(output, target, M=3):
    return F.cross_entropy(output, target)


class ELRPlus(torch.nn.Module):
    def __init__(self, num_examp, device, num_classes=10, _lambda=3, beta=0.3, coef_step=0):
        super(ELRPlus, self).__init__()
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self._lambda = _lambda
        self.coef_step = coef_step
        self.num_classes = num_classes

    def forward(self, output, y_labeled, **kwargs):
        iteration = kwargs['current_iter']
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_labeled = F.one_hot(y_labeled, self.num_classes)

        if self.num_classes == 100:
            y_labeled = y_labeled * self.q
            y_labeled = y_labeled / (y_labeled).sum(dim=1, keepdim=True)

        ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim=-1))
        reg = ((1 - (self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + sigmoid_rampup(iteration, self.coef_step) * (
                self._lambda * reg)

        if 'reduction' in kwargs.keys():
            if kwargs['reduction'] == 'none':
                return final_loss
        return final_loss.mean()

        # return final_loss, y_pred.cpu().detach()

    def update_hist(self, out, index=None, mix_index=..., mixup_l=1):
        y_pred_ = F.softmax(out, dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] + (1 - self.beta) * y_pred_ / (y_pred_).sum(dim=1,
                                                                                                              keepdim=True)
        self.q = mixup_l * self.pred_hist[index] + (1 - mixup_l) * self.pred_hist[index][mix_index]
