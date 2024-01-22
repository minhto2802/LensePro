import torch
from torch.nn import functional as F

from loss_functions import *


def get_loss_function(loss_name, num_classes=2, train_loader=None, annealing_steps=100, reduction='none', **kwargs):
    loss_functions = {
        'mae': L1Loss(),
        'ce': CrossEntropy(),
        'nll': F.nll_loss,
        'fl': FocalLoss(gamma=2.),
        'isomax': IsoMaxLossSecondPart(),
        'isomaxplus': IsoMaxPlusLossSecondPart(),
        'kl': kl_div,
        'c_ce': C_CrossEntropy(),
        'nce_agce': NCEandAGCE(num_classes=num_classes, alpha=0, beta=1, a=4, q=0.2, reduction=reduction),
        'nce_rce': NCEandRCE(alpha=1., beta=100., num_classes=num_classes),
        'nfl_rce': NFLandRCE(alpha=1., beta=1., gamma=1, num_classes=num_classes),
        'gce': GeneralizedCrossEntropy(num_classes),
        'agce': AGCELoss(num_classes=num_classes, reduction=reduction),
        'edl_digamma': EdLDigammaLoss(num_classes, annealing_steps),
        'edl_log': EdLLogLoss(num_classes, annealing_steps),
        'edl_mse': EdLMSELoss(num_classes, annealing_steps),
        'poly': PolyLoss(eps=1),
        # 'nlnl': NLNL(train_loader, num_classes)
    }
    opt = kwargs['opt'] if 'opt' in kwargs else None

    if loss_name == 'elr':  # https://github.com/shengliu66/ELR
        assert opt is not None
        loss_functions['elr'] = ELR(
            opt.num_samples['train'], num_classes, alpha=opt.elr.alpha, beta=opt.elr.beta)
    elif loss_name == 'elr+':
        assert opt is not None
        loss_functions['elr+'] = ELRPlus(
            opt.num_samples['train'], num_classes=num_classes,
            device=opt.device, beta=opt.elr.beta,
            coef_step=opt.elr.coef_step, _lambda=opt.elr._lambda
        )
    if loss_name == 'abc':  # https://github.com/thulas/dac-label-noise
        assert opt is not None
        for field in ['abstention', 'n_epochs']:
            assert hasattr(opt, field)
        args = opt.abstention
        loss_functions['abc'] = DacLossPid(learn_epochs=args.learn_epochs,
                                           total_epochs=opt.n_epochs,
                                           abst_rate=args.abst_rate,
                                           alpha_final=args.alpha_final,
                                           alpha_init_factor=args.alpha_init_factor,
                                           pid_tunings=list(args.pid_tunings)
                                           )
    return loss_functions[loss_name]


def uncertainty_driven(loss, coef):
    """

    :param loss:
    :param coef:
    :return:
    """
    var = torch.square(coef)
    return (1 / var) * loss + torch.log(var)
