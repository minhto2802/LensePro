from torch import nn
from training_strategy import *


STRATEGY_DICT = {
    'none': CoTeaching,
    'ssl': CoTeachingSemiSupervise,
    'multitask': CoTeachingMultiTask,
    'self_train': CoTeachingSelfTrain,
    'variational': CoTeachingUncertaintyAvU,
    'nested': CoTeachingNested
}

MODELS = {
    'vanilla': Model,
    'sam': ModelSam,
    'evidential': Evidential,
    'gnn': GNN,
}


def get_model(opt, network, device, mode, classifier=None):
    """

    :param classifier:
    :param mode: 'train' or 'eval'
    :param opt: output of 'read_yaml' (utils.misc) or munchify(dict)
    :param network: output of 'construct_network' (utils.misc)
    :param device:
    :return:
    """
    model = None
    opt.device = device
    if opt.model_name.lower() == 'coteaching':
        cot = STRATEGY_DICT[opt.strategy]
        if mode.lower() == 'train':
            model = cot(
                network, device, opt.tasks_num_class[0], mode, opt.aug_type,
                loss_name=opt.loss_name,
                use_plus=opt.train.coteaching.use_plus,
                relax=opt.train.coteaching.relax,
                classifier=classifier,
                ckpt=opt.paths.self_train_checkpoint,
                opt=opt,  # optional
            )
            model.init_optimizers(lr=opt.lr, n_epochs=opt.n_epochs, n_batches=opt.num_batches['train'],
                                  coteaching_cfg=opt.train.coteaching, optimizer=opt.optimizer)
        else:
            opt.is_eval = True
            model = cot(network, device, opt.tasks_num_class[0], mode='test', classifier=classifier,
                        opt=opt, epochs=opt.n_epochs,  # optional
                        )
        model.net1 = nn.DataParallel(model.net1).to(device)
        model.net2 = nn.DataParallel(model.net2).to(device)

        return model
    elif opt.model_name.lower() in ['vanilla', 'sam', 'evidential', 'gnn']:
        _model = MODELS[opt.model_name.lower()]
        model = _model(device, num_class=opt.tasks_num_class[0], mode=mode, aug_type=opt.aug_type,
                       network=network, loss_name=opt.loss_name,
                       opt=opt,  # optional
                       )
        model.init_optimizers(opt.lr, n_epochs=opt.n_epochs, n_batches=opt.num_batches['train'],
                              scheduler=opt.scheduler, optimizer=opt.optimizer)
    return model
