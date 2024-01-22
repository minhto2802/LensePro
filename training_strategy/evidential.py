import torch.nn.functional as F
from .vanilla import Model
from utils.get_loss_function import *


class Evidential(Model):
    def __init__(self, *args, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)
        self.class_names = ['benign', 'cancer']
        annealing_steps = kwargs['opt'].n_epochs if 'opt' in kwargs.keys() else 100  # for EDL loss
        assert kwargs['loss_name'] in ['edl_mse', 'edl_digamma', 'edl_log']
        self.loss_func = get_loss_function(annealing_steps=annealing_steps, **kwargs)

    def to_prob(self, out):
        alpha = F.relu(out) + 1
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        return prob

    def forward_backward(self, x_raw, n_batch, y_batch, *args, **kwargs):
        out = self.infer(x_raw, n_batch)
        loss = self.loss_func(out, y_batch, current_epoch=kwargs['epoch']).mean()
        self.optimize(loss)
        return out, loss
