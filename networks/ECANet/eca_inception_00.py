import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ECANet.eca_module import EcaLayer

DROP_RATE = 0.2


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(X):
    return X


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(),
                 return_indices=False):
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
        : param return_indices			Indices are needed only if we want to create decoder with InceptionTranspose with MaxUnpool1d.
        """
        super(Inception, self).__init__()

        self.return_indices = return_indices
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                stride=1,
                bias=False
            )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1
        self.eca_bottle_neck = EcaLayer(5)

        self.conv_from_bottleneck_1 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=kernel_sizes[0] // 2,
            bias=False
        )
        self.eca1 = EcaLayer(kernel_sizes[0])

        self.conv_from_bottleneck_2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding=kernel_sizes[1] // 2,
            bias=False
        )
        self.eca2 = EcaLayer(kernel_sizes[1])

        self.conv_from_bottleneck_3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=n_filters,
            kernel_size=kernel_sizes[2],
            stride=1,
            padding=kernel_sizes[2] // 2,
            bias=False
        )
        self.eca3 = EcaLayer(kernel_sizes[2])

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.eca_max_pool = EcaLayer(1)

        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        # self.batch_norm = nn.GroupNorm(num_groups=n_filters, num_channels=4 * n_filters)
        # self.batch_norm = nn.GroupNorm(num_groups=1, num_channels=4 * n_filters)
        self.activation = activation

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        #         import scipy.stats as stats
        #         stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #         X = stats.truncnorm(-2, 2, scale=stddev)
        #         values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
        #         values = values.view(m.weight.size())
        #         with torch.no_grad():
        #             m.weight.copy_(values)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, X, *args):
        # step 1
        # Z_bottleneck = self.bottleneck(X)
        Z_bottleneck = self.eca_bottle_neck(self.bottleneck(X))

        if self.return_indices:
            Z_maxpool, indices = self.max_pool(X)
        else:
            # Z_maxpool = self.max_pool(X)
            Z_maxpool = self.max_pool(X)
        # step 1b
        # Z_bottleneck = F.dropout1d(Z_bottleneck, DROP_RATE)
        # step 2
        # Z1 = self.conv_from_bottleneck_1(Z_bottleneck)
        # Z2 = self.conv_from_bottleneck_2(Z_bottleneck)
        # Z3 = self.conv_from_bottleneck_3(Z_bottleneck)
        # Z4 = self.conv_from_maxpool(Z_maxpool)
        Z1 = self.eca1(self.conv_from_bottleneck_1(Z_bottleneck))
        Z2 = self.eca2(self.conv_from_bottleneck_2(Z_bottleneck))
        Z3 = self.eca3(self.conv_from_bottleneck_3(Z_bottleneck))
        Z4 = self.eca_max_pool(self.conv_from_maxpool(Z_maxpool))
        # step 2b
        # Z1 = F.dropout1d(Z1, DROP_RATE)
        # Z2 = F.dropout1d(Z2, DROP_RATE)
        # Z3 = F.dropout1d(Z3, DROP_RATE)
        # Z4 = F.dropout1d(Z4, DROP_RATE)
        # step 3
        Z = torch.cat([Z1, Z2, Z3, Z4], axis=1)
        Z = self.activation(self.batch_norm(Z))

        # Z = self.classifier(Z, *args)
        if self.return_indices:
            return Z, indices
        else:
            return Z


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False, num_classes=2):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.eca_input = EcaLayer(1)

        self.inception_1 = Inception(
            in_channels=in_channels,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_2 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        self.inception_3 = Inception(
            in_channels=4 * n_filters,
            n_filters=n_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            activation=activation,
            return_indices=return_indices
        )
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=4 * n_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm1d(
                    num_features=4 * n_filters
                ),
                # nn.GroupNorm(
                #     num_groups=n_filters,
                #     num_channels=4 * n_filters
                # ),
            )

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1),
                                        Classifier(4 * n_filters, 4 * n_filters, num_classes))
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

    def forward(self, X, *args):
        X = self.eca_input(X)
        if self.return_indices:
            Z, i1 = self.inception_1(X)
            Z, i2 = self.inception_2(Z)
            Z, i3 = self.inception_3(Z)
        else:
            Z = self.inception_1(X)
            Z = self.inception_2(Z)
            Z = self.inception_3(Z)
        if self.use_residual:
            Z = Z + self.residual(X)
            Z = self.activation(Z)
        logit = self.classifier(Z, *args)
        # inv = self.regressor(Z, *args)
        if self.return_indices:
            return logit, [i1, i2, i3]
        else:
            return logit  # , inv


class Classifier(nn.Module):
    def __init__(self, feat_dim, out_dim, num_pred_classes):
        super().__init__()
        # self.linear01 = IsoMaxLossFirstPart(out_dim, num_pred_classes)
        self.linear01 = nn.Sequential(nn.Linear(out_dim, num_pred_classes))

    def forward(self, x, *args):
        # return self.linear01(self.linear00(x))
        return self.linear01(x)


class Regressor(nn.Module):
    def __init__(self, feat_dim, out_dim):
        super().__init__()
        # self.linear01 = IsoMaxLossFirstPart(out_dim, num_pred_classes)
        self.linear01 = nn.Linear(out_dim, 1)

    def forward(self, x, *args):
        # return self.linear01(self.linear00(x))
        return self.linear01(x)


def summarize_net(net, in_channels, num_positions, input_length=200):
    from torchsummary import summary
    input_size = [(in_channels, input_length), ]
    if num_positions > 0:
        input_size = [input_size, (num_positions,)]
    summary(net, input_size=input_size)


def main():
    in_channels = 24
    num_positions = -1
    input_length = 200
    net = InceptionBlock(24, 32)
    device = 'cuda'
    summarize_net(net.to(device), in_channels, num_positions, input_length)


if __name__ == '__main__':
    main()
