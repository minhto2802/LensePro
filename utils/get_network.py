from torch import nn
# from torchinfo import summary
from torchsummary import summary
# from tsai.models.all import InceptionTime, XceptionTime, transfer_weights, ResNetPlus, xresnet1d18, xresnet1d34

from networks import *


def _construct_network(device, opt, backbone, num_class=None, num_positions=None, is_graph=False, *args, **kwargs):
    """
    Create networks
    :param device:
    :param opt:
    :param backbone:
    :param num_positions:
    :return:
    """

    def _get_network():
        get = get_gnn if is_graph else get_network

        return get(backbone, device, opt.input_channels,
                   num_class if num_class is not None else opt.tasks_num_class[0],
                   num_blocks=opt.arch.num_blocks,
                   out_channels=opt.arch.out_channels,
                   mid_channels=opt.arch.mid_channels,
                   num_positions=opt.arch.num_positions if num_positions is None else num_positions,
                   self_train=opt.strategy == 'self_train',
                   variational=opt.strategy == 'variational',
                   verbose=True,
                   weights_path=opt.weights_path,
                   fc_dropout=opt.fc_dropout,
                   input_length=opt.ts_len,
                   opt=opt,
                   *args, **kwargs)

    return _get_network


def _construct_classifier(device, opt, num_positions=None):
    """
    Create classifiers
    :param device:
    :param opt:
    :param num_positions:
    :return:
    """

    def _get_classifier(in_channels):
        """in_channels is needed, can be obtained by running the backbone once"""
        return get_classifier(device, in_channels,
                              # in_channels // 2,
                              1024,
                              opt.tasks_num_class[0],
                              num_positions=opt.arch.num_positions if num_positions is None else num_positions)

    return _get_classifier


def construct_network(device, opt, is_graph=False, *args, **kwargs):
    """
    Create one network for vanilla training and two networks for coteaching
    :param device:
    :param opt:
    :param is_graph:
    :return:
    """
    if opt.is_eval:
        return _construct_network(device, opt, opt.backbone, opt.tasks_num_class[0], is_graph=is_graph, *args, **kwargs)

    num_class = 1 if opt.strategy == 'self_train' else opt.tasks_num_class[0]
    num_class += 1 if opt.loss_name == 'abc' else 0

    if 'coteaching' in opt.model_name:
        backbones = opt.backbone
        if isinstance(opt.backbone, str):
            backbones = [opt.backbone, ] * 2
        elif isinstance(opt.backbone, list) and (len(opt.backbone) == 1):
            backbones = opt.backbone * 2
        # Create two networks, in which the first one does not have location encoder
        networks = []
        # for (num_positions, backbone) in zip([opt.arch.num_positions, opt.arch.num_positions], backbones):
        for (num_positions, backbone) in zip([0, 0], backbones):
            networks.append(_construct_network(device, opt, backbone, num_class, num_positions, is_graph=is_graph))
        return networks
    return _construct_network(device, opt, opt.backbone, num_class, is_graph=is_graph, *args, **kwargs)


def construct_classifier(device, opt):
    """
    Create one classifier for vanilla training and two classifiers for coteaching when self-training is used
    :param device:
    :param opt:
    :return:
    """
    if 'coteaching' in opt.model_name:
        clf = [_construct_classifier(device, opt, num_positions) for num_positions in
               [opt.arch.num_positions, opt.arch.num_positions]]
        return clf
    return _construct_classifier(device, opt, opt.arch.num_positions)


def get_network(backbone, device, in_channels, nb_class, num_positions=0,
                verbose=False, self_train=False, num_blocks=3,
                out_channels=30, mid_channels=32, variational=False,
                input_length=200,
                weights_path=None,
                *args, **kwargs):
    backbone = backbone.lower()
    # in_channels += 32
    # in_channels = 2
    if backbone == 'simconv4':
        from self_time.model.model_backbone import SimConv4
        net = SimConv4(in_channels, is_classifier=True, nb_class=nb_class, num_positions=num_positions,
                       self_train=self_train)
    elif backbone == 'inception':
        if variational:
            raise NotImplemented('Variational version of inception is not implemented yet')

        _net = InceptionModel if not variational else InceptionModelVariational
        net = _net(num_blocks, in_channels, out_channels=out_channels, stride=1,
                   bottleneck_channels=12, kernel_sizes=15, input_length=input_length,
                   # bottleneck_channels=12, kernel_sizes=5, input_length=input_length,
                   use_residuals='default',
                   num_pred_classes=nb_class, self_train=self_train, num_positions=num_positions)
    elif 'inception_' in backbone:
        in_channels = 24
        get_embedding = True if 'feat' in backbone or 'pretrained' in backbone else False
        net = Inception(in_channels, 64, num_classes=nb_class, bottleneck_channels=64, eca='eca' in backbone,
                        from_spatio='spatio' in backbone, use_residual=False, add_identity=False,
                        get_embedding=get_embedding, *args, **kwargs)
    # elif backbone == 'inception_time':
    #     net = InceptionTime(in_channels, nb_class, nf=8)
    #     if weights_path:
    #         weights_path = weights_path.replace('net_name', backbone)
    #         transfer_weights(net,
    #                          weights_path, exclude_head=True)
    # elif backbone == 'inception_gm':
    #     net = InceptionTimeGM(in_channels, nb_class, base_channels=8, bottleneck_channels=8)
    elif backbone == 'xception_time':
        net = XceptionTime(in_channels, nb_class, nf=4)
    elif backbone == 'resnet_ucr':
        net = ResNet(in_channels, mid_channels=mid_channels,
                     num_pred_classes=nb_class, num_positions=num_positions)
    elif backbone == 'resnet':
        # _net = resnet20 if not variational else resnet20_variational
        # net = _net(num_classes=nb_class, in_channels=in_channels)
        net = ResNetPlus(in_channels, nb_class)
    elif backbone == 'resnet1d':
        net = xresnet1d18(in_channels, nb_class)
        # net = xresnet1d34(in_channels, nb_class, nf=8)
    elif backbone == 'fnet':
        net = FNet(dim=200, depth=5, mlp_dim=32, dropout=.5, num_pred_classes=nb_class, num_positions=num_positions)
    elif backbone == 'transformer':
        net = TransformerClassifier(num_classes=nb_class, n_hid=in_channels, n_inp=input_length // in_channels)
    else:
        # This network already has a positional encoder
        net = Classifier_3L(in_channels, nb_class, drop=.5)
        if verbose:
            summarize_net(net, in_channels, num_positions)
        return net.to(device)

    # Wrap the feature extractor with the position encoding network
    # net = PosEncoder(net, num_positions=num_positions, variational=variational)
    if verbose:
        summarize_net(net, in_channels, num_positions, input_length)
    return net.to(device)
    # return nn.DataParallel(net.to(device))


def get_gnn(backbone, device, in_channels, nb_class, mid_channels=32, *args, **kwargs):
    net_dict = {
        'basic': GCN,
        'basic2': GINC,
        'hgpsl_p': HGPSL_Pool,
    }
    if backbone == 'hgpsl_p':
        gnn = net_dict[backbone](kwargs['opt'], in_channels, nb_class)
    else:
        gnn = net_dict[backbone](in_channels, mid_channels, nb_class, *args, **kwargs)
    return nn.DataParallel(gnn.to(device))


def summarize_net(net, in_channels, num_positions, input_length=200):
    input_size = [(in_channels, input_length), ]
    if num_positions > 0:
        input_size = [input_size, (num_positions,)]
    summary(net, input_size=input_size, device='cpu')


def get_classifier(device, in_channels, out_channels, nb_class, num_positions):
    """

    :param device:
    :param in_channels:
    :param out_channels:
    :param nb_class:
    :param num_positions:
    :return:
    """
    clf = Classifier(in_channels, out_channels, nb_class, num_positions).cuda(device)
    return clf


def main():
    from networks.r2plus1d_18 import r2plus1d_18
    net = r2plus1d_18(num_classes=2, in_channels=24, strides=(1, 1, 2, 2))
    import torch
    x = torch.randn((2, 24, 100, 32, 32))
    print(net(x).shape)


if __name__ == '__main__':
    main()
