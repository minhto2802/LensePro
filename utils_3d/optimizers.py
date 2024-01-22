# A convenient module for getting some pre-defined optimizers and schedulers
from torch import optim
from utils import NovoGrad


def get_optimizer(parameters, optim_name, lr, scheduler_name=None, n_epochs=None, steps_per_epoch=None):
    scheduler, optimizer = None, None

    optim_name = optim_name.lower()
    if optim_name == 'novograd':
        optimizer = NovoGrad(parameters, lr=float(lr), weight_decay=5e-2)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=float(lr), weight_decay=7e-2, nesterov=True, momentum=0.9)  # 5e-2
    elif optim_name == 'adam':
        optimizer = optim.Adam(parameters, lr=float(lr), amsgrad=True, weight_decay=1e-4)
    elif optim_name == 'adamw':
        optimizer = optim.AdamW(parameters, lr=float(lr), amsgrad=True, weight_decay=1e-2)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} is not available.")
    scheduler = get_scheduler(scheduler_name, steps_per_epoch, n_epochs, optimizer, lr)
    return optimizer, scheduler


def get_scheduler(scheduler_name, steps_per_epoch, n_epochs, optimizer, lr):
    if scheduler_name is not None:
        assert n_epochs is not None and steps_per_epoch is not None
    if scheduler_name == 'one_cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, lr, epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.01, anneal_strategy='cos',
            # pct_start=0.2, anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=100000,  # 2.0,
            final_div_factor=100000,  # 10000.0,
            three_phase=False,
            last_epoch=-1, verbose=False)
    elif scheduler_name == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    return scheduler

