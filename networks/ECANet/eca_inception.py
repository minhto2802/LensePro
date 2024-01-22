import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ECANet.eca_module import EcaLayer
# from networks.ECANet.eca_module import EIcaLayer as EcaLayer
from networks.ECANet.spatial_net import SpatialNet
from loss_functions.isomaxplus import IsoMaxPlusLossFirstPart

DROP_RATE = 0.2


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(x):
    return x


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(),
                 eca=True, groups=1, add_identity=False, first_block=False):
        """
        : param in_channels				Number of input channels (input features)
        : param n_filters				Number of filters per convolution layer => out_channels = 4*n_filters
        : param kernel_sizes			List of kernel sizes for each convolution.
                                        Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
                                        This is nessesery because of padding size.
                                        For correction of kernel_sizes use function "correct_sizes".
        : param bottleneck_channels		Number of output channels in bottleneck.
                                        Bottleneck wont be used if nuber of in_channels is equal to 1.
        : param activation				Activation function for output tensor (nn.ReLU()).
        """
        super(Inception, self).__init__()
        self.add_identity = add_identity

        # if in_channels > 1:
        #     self.bottleneck = nn.Conv1d(
        #         in_channels=in_channels,
        #         out_channels=bottleneck_channels,
        #         kernel_size=1,
        #         stride=1,
        #         bias=False,
        #         groups=groups,
        #     )
        # else:
        #     self.bottleneck = pass_through
        #     bottleneck_channels = 1
        # self.eca_bottle_neck = EcaLayer(bottleneck_channels, 3)
        self.bottleneck = nn.Conv1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=groups,
        )

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False,
            groups=groups,
        )
        # self.eca1 = EcaLayer(n_filters, 3)  # kernel_sizes[0])
        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False,
            groups=groups,
        )
        # self.eca2 = EcaLayer(n_filters, 3)  # kernel_sizes[1])

        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False,
            groups=groups,
        )
        # self.eca3 = EcaLayer(n_filters, 3)  # kernel_sizes[2])

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        # self.conv_from_maxpool = nn.Conv1d(
        #     in_channels=in_channels,
        #     out_channels=n_filters,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False
        # )
        # self.eca_max_pool = EcaLayer(n_filters, 3)  # 1)
        self.eca_layer = EcaLayer() if eca else None

        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)   # if not first_block else nn.Identity()

        # self.batch_norm = nn.GroupNorm(num_groups=n_filters, num_channels=4 * n_filters)
        # self.batch_norm = nn.GroupNorm(num_groups=1, num_channels=4 * n_filters)
        self.activation = activation  # TODO: original

    def forward(self, x, *args):
        # step 1
        z_bottleneck = self.bottleneck(x)
        # z_bottleneck = self.eca_bottle_neck(self.bottleneck(x))

        z_maxpool = self.max_pool(x)
        # step 2
        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        # z4 = self.conv_from_maxpool(z_maxpool)
        z4 = self.bottleneck(z_maxpool)
        # z1 = self.eca1(self.conv_from_bottleneck_1(z_bottleneck))
        # z2 = self.eca2(self.conv_from_bottleneck_2(z_bottleneck))
        # z3 = self.eca3(self.conv_from_bottleneck_3(z_bottleneck))
        # z4 = self.eca_max_pool(self.conv_from_maxpool(z_maxpool))
        # step 3
        z = torch.cat([z1, z2, z3, z4], axis=1)
        # z = self.activation(self.batch_norm(z))  # TODO: original

        z = self.batch_norm(z)
        if self.eca_layer is not None:
            z = self.eca_layer(z)
        if self.activation is not None:
            z = self.activation(z)

        # z = self.classifier(z, *args)
        return z

    def forward_v1(self, x, *args):
        # step 1
        z = self.batch_norm(x)
        z = self.activation(z)

        z_bottleneck = self.bottleneck(z)
        z_maxpool = self.max_pool(x)

        # step 2
        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.bottleneck(z_maxpool)

        # step 3
        z = torch.cat([z1, z2, z3, z4], dim=1)
        if self.eca_layer is not None:
            z = self.eca_layer(z)
        if self.add_identity:
            z += x

        return z


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), num_classes=2, eca=True, input_length=200, from_spatio=False,
                 get_embedding=False, zero_init_residual=True, groups=1, add_identity=False,
                 fc_dropout=.3, *args, **kwargs):
        super(InceptionBlock, self).__init__()
        # self.spatio2temporal = SpatialNet(input_length) if from_spatio else nn.Identity()
        self.use_residual = use_residual
        # self.activation = activation
        self.get_embedding = get_embedding

        # self.eca_input = EcaLayer(in_channels, 3)

        def make_inception():
            def _make_inception():
                inception = Inception(
                    in_channels=n_filters * 4,
                    n_filters=n_filters,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    activation=activation,
                    eca=eca,
                    groups=groups,
                    add_identity=add_identity,
                )
                return inception

            return _make_inception()

        inception_1 = Inception(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,  # nn.Identity(),  # activation
            first_block=True,
            eca=eca,
            groups=groups,
        )
        # self.inception_2 = Inception(
        #     in_channels=4 * n_filters,
        #     n_filters=n_filters,
        #     kernel_sizes=kernel_sizes,
        #     bottleneck_channels=bottleneck_channels,
        #     activation=activation,
        #     eca=eca,
        #     groups=groups,
        # )
        # self.inception_3 = Inception(
        #     # in_channels=4 * n_filters,
        #     in_channels=2 * n_filters,
        #     n_filters=n_filters,
        #     kernel_sizes=kernel_sizes,
        #     bottleneck_channels=bottleneck_channels,
        #     activation=None if self.use_residual else activation,
        #     eca=eca,
        #     groups=groups,
        # )

        self.inception_blocks = nn.Sequential(inception_1, *[make_inception() for _ in range(2)])
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=groups,
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                ),
            )
            if eca:
                self.residual.add_module('eca', EcaLayer())
        # if self.use_residual:
        #     self.residual = nn.Sequential(
        #         nn.BatchNorm1d(
        #             num_features=in_channels,
        #         ),
        #         nn.ReLU(),
        #         nn.Conv1d(
        #             in_channels=in_channels,
        #             out_channels=in_channels,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1
        #         ),
        #         nn.BatchNorm1d(
        #             num_features=in_channels
        #         ),
        #         nn.ReLU(),
        #         nn.Conv1d(
        #             in_channels=in_channels,
        #             out_channels=4 * n_filters,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1
        #         ),
        #         EIcaLayer(4 * n_filters),
        #     )

        self.gap = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        if self.get_embedding:
            self.head = nn.Identity()
        else:
            # self.classifier = Classifier(4 * n_filters, 4 * n_filters, num_classes)
            classifier = []
            if fc_dropout:
                classifier += [nn.Dropout(fc_dropout)]
            classifier += [nn.Linear(4 * n_filters, num_classes)]
            self.head = nn.Sequential(*classifier)
            self.head[-1].weight.data.normal_(mean=0.0, std=0.01)
            self.head[-1].bias.data.zero_()

        # self.distance = IsoMaxPlusLossFirstPart(4 * n_filters, num_classes)

        # self._initialize_weights()
        # self.regressor = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1),
        #                                Regressor(4 * n_filters, 4 * n_filters))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         # nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        #         # nn.init.xavier_normal_(m.weight)
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        #         import scipy.stats as stats
        #         stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #         x = stats.truncnorm(-2, 2, scale=stddev)
        #         values = torch.as_tensor(x.rvs(m.weight.numel()), dtype=m.weight.dtype)
        #         values = values.view(m.weight.size())
        #         with torch.no_grad():
        #             m.weight.copy_(values)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        # self._initialize_weights()

    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, get_feat=False, *args):
        # x = self.spatio2temporal(x)
        # x = self.eca_input(x)
        # z = self.inception_1(x)
        # # z = F.dropout1d(z, DROP_RATE)
        # z = self.inception_2(z)
        # # z = F.dropout1d(z, DROP_RATE)
        # z = self.inception_3(z)
        z = self.inception_blocks(x)
        # if self.activation:
        #     z = self.activation(z)

        # if self.use_residual:
        #     z = z + self.residual(x)
        #     z = self.activation(z)
        if self.use_residual:
            z = z + self.residual(x)
            z = self.activation(z)

        z = self.gap(z)
        if get_feat or self.get_embedding:
            return z

        # if self.get_embedding:
        #     # distance = self.distance(z)
        #     # return logit, distance  # , inv
        #     return z

        # logit = self.classifier(z, *args)
        logit = self.head(z, *args)
        return logit


class Classifier(nn.Module):
    def __init__(self, feat_dim, out_dim, num_pred_classes):
        super().__init__()
        # self.linear01 = IsoMaxLossFirstPart(out_dim, num_pred_classes)
        self.head = nn.Sequential(
            nn.Dropout(DROP_RATE),
            nn.Linear(feat_dim, out_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Dropout(DROP_RATE),
            nn.Linear(out_dim, num_pred_classes, bias=False)
        )

        # self.head = nn.Sequential(
        #     nn.Linear(feat_dim, out_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(out_dim, num_pred_classes)
        # )

    def forward(self, x, *args):
        # return self.linear01(self.linear00(x))
        return self.head(x)


class Regressor(nn.Module):
    def __init__(self, feat_dim, out_dim):
        super().__init__()
        # self.linear01 = IsoMaxLossFirstPart(out_dim, num_pred_classes)
        self.linear01 = nn.Linear(out_dim, 1)

    def forward(self, x, *args):
        # return self.linear01(self.linear00(x))
        return self.linear01(x)


def summarize_net(net, in_channels, input_length=200, spatial_dim=None):
    if spatial_dim is None:
        spatial_dim = []
    else:
        in_channels, input_length = input_length, in_channels
    from torchsummary import summary
    input_size = (in_channels, input_length, *spatial_dim)
    summary(net, input_size=input_size)


def main():
    in_channels = 24
    num_positions = -1
    input_length = 200
    spatial_dim = None  # (32, 32)
    net = InceptionBlock(24, 32, from_spatio=True if spatial_dim else False)
    device = 'cuda'
    summarize_net(net.to(device), in_channels, input_length, spatial_dim)
    # summarize_net(net.to(device), in_channels, num_positions, input_length)


if __name__ == '__main__':
    main()
