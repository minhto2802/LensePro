import torch
import torch.nn as nn
import torch.nn.functional as F


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
          torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl


def log_likelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    log_likelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    log_likelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    log_likelihood = log_likelihood_err + log_likelihood_var
    return log_likelihood


class EvidentialLoss(nn.Module):
    """Abstract class for Evidential loss functions"""

    def __init__(self, num_classes, annealing_steps):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_steps = annealing_steps

    def loss_func(self, *args, **kwargs) -> torch.Tensor:
        pass

    def compute_kl_div(self, alpha, y, current_epoch, device):
        annealing_coef = torch.min(torch.tensor(
            1.0, dtype=torch.float32), torch.tensor(current_epoch / self.annealing_steps, dtype=torch.float32))

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * kl_divergence(kl_alpha, self.num_classes, device)
        return kl_div

    def forward(self, output, target, current_epoch):
        evidence = relu_evidence(output)
        alpha = evidence + 1
        target = F.one_hot(target) if target.ndim == 1 else target
        return torch.mean(self.loss_func(target, alpha, current_epoch))


class EdLMSELoss(EvidentialLoss):

    def loss_func(self, y, alpha, current_epoch):
        device = y.device
        y = y.to(device)
        alpha = alpha.to(device)
        log_likelihood = log_likelihood_loss(y, alpha, device)
        kl_div = self.compute_kl_div(alpha, y, current_epoch, device)
        return log_likelihood + kl_div


class EdLLogLoss(EvidentialLoss):
    def __init__(self, *args, **kwargs):
        super(EdLLogLoss, self).__init__(*args, **kwargs)
        self.func = torch.log

    def loss_func(self, y, alpha, current_epoch):
        device = y.device
        y = y.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y * (self.func(S) - self.func(alpha)), dim=1, keepdim=True)
        kl_div = self.compute_kl_div(alpha, y, current_epoch, device)
        return A + kl_div


class EdLDigammaLoss(EdLLogLoss):
    def __init__(self, *args, **kwargs):
        super(EdLDigammaLoss, self).__init__(*args, **kwargs)
        self.func = torch.digamma
