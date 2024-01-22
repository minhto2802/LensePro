import numpy as np
import torch


def make_pseudo_label(logits, threshold):
    max_value, hard_label = logits.softmax(1).max(1)
    mask = (max_value >= threshold)
    return hard_label, mask


def sharpening(soft_labels, temp=5):
    soft_labels = soft_labels.pow(temp)
    return soft_labels / soft_labels.abs().sum(1, keepdim=True)


def tempereture_softmax(logits, tau):
    return (logits / tau).softmax(1)


def mixup(x, y, alpha):
    device = x.device
    b = x.shape[0]
    permute = torch.randperm(b)
    perm_x = x[permute]
    perm_y = y[permute]
    factor = torch.distributions.beta.Beta(alpha, alpha).sample((b, 1)).to(device)
    if x.ndim == 4:
        x_factor = factor[..., None, None]
    else:
        x_factor = factor
    mixed_x = x_factor * x + (1 - x_factor) * perm_x
    mixed_y = factor * y + (1 - factor) * perm_y
    return mixed_x, mixed_y


class TSA:
    def __init__(self, percentage, total_steps, num_classes, schedule='linear', end=1, scale=5):
        self.max_iter = percentage * total_steps
        self.schedule = eval(f'self.{schedule}')
        self.start = 1 / num_classes
        self.end = end
        self.scale = scale

    @staticmethod
    def linear(step_ratio):
        return step_ratio

    def exp(self, step_ratio):
        return ((step_ratio - 1) * self.scale).exp()

    def log(self, step_ratio):
        return 1 - (-step_ratio * self.scale).exp()

    def anneal_loss(self, logits, labels, loss, global_step, reduction='none'):
        if self.max_iter == 0:
            return loss

        threshold = self.get_tsa_threshold(global_step)
        with torch.no_grad():
            probs = logits.softmax(1)
            correct_label_probs = probs.gather(1, labels[:, None]).squeeze()
            mask = correct_label_probs < threshold
        return self.mask_loss(loss, mask, reduction)

    @staticmethod
    def mask_loss(loss, mask, reduction='none'):
        loss = loss * mask
        if reduction == 'none':
            return loss
        return loss.mean()

    def get_tsa_threshold(self, global_step):
        coef = self.schedule(global_step / self.max_iter)
        return coef * (self.end - self.start) + self.start


class STSA(TSA):
    def __init__(self, *args, **kwargs):
        super(STSA, self).__init__(*args, **kwargs)
        self.start = 0.05
        self.num_classes = args[2]

    def anneal_loss(self, logits, labels, loss, global_step, reduction='none'):
        if self.max_iter == 0:
            return loss

        percent_visible = self.get_tsa_threshold(global_step)
        mask = torch.zeros_like(labels)
        for c in range(self.num_classes):
            idx = np.where(labels.detach().cpu() == c)[0]
            num_visible = min(round(len(idx) * percent_visible), len(idx))
            mask[np.random.choice(idx, num_visible, replace=False)] = 1
        return self.mask_loss(loss, mask, reduction)
