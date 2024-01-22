import torch
from torch import nn
from torch.nn.parameter import Parameter


class EcaLayer(nn.Module):
    """Constructs a ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=11):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class EIcaLayer0(nn.Module):
    """Constructs a EICA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=12):
        super(EIcaLayer0, self).__init__()
        assert (k_size >= 9) and (k_size % 2 == 0)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.conv1_a = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 1, kernel_size=(k_size // 2 - 1), padding=(k_size // 2 - 2) // 2, bias=False),
        )
        self.conv1_b = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 1, kernel_size=k_size - 1, padding=(k_size - 2) // 2, bias=False),
        )
        self.max_conv1 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = y.transpose(-1, -2)
        y = self.conv1(y) + self.conv1_a(y) + self.conv1_b(y) + self.max_conv1(y)
        y = y.transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class EIcaLayer(EcaLayer):
    """Constructs a EIcaLayer module.
    Args:
        kernel_sizes: Adaptive selection of kernel size
    """

    def __init__(self, kernel_sizes=(9, 19, 39)):
        super(EIcaLayer, self).__init__()
        self.conv = InceptionModule(kernel_sizes=kernel_sizes)


def pass_through(x):
    return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels=1, n_filters=1, kernel_sizes=(9, 19, 39), bottleneck_channels=1,
                 activation=nn.ReLU()):
        """
        Inception module for channel attention
        :param in_channels: Number of input channels (input features)
        :param n_filters: Number of filters per convolution layer => out_channels = 4*n_filters
        :param kernel_sizes: List of kernel sizes for each convolution.
         Each kernel size must be odd number that meets -> "kernel_size % 2 !=0".
         This is necessary because of padding size.
         For correction of kernel_sizes use function "correct_sizes".
        :param bottleneck_channels: Number of output channels in bottleneck.
        Bottleneck won't be used if nuber of in_channels is equal to 1.
        :param activation: Activation function for output tensor (nn.ReLU())
        """
        super(InceptionModule, self).__init__()

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1,
                                        stride=1, bias=False)
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=bottleneck_channels, out_channels=n_filters,
                                             kernel_size=k, stride=1, padding=k // 2,
                                             bias=False) for k in kernel_sizes])
        self.conv_from_maxpool = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                               nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=1,
                                                         stride=1, padding=0, bias=False))

        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation
        self.merge_channels = nn.Conv1d(n_filters * 4, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        z_bottleneck = self.bottleneck(x)
        outs = [conv(z_bottleneck) for conv in self.conv]
        outs.append(self.conv_from_maxpool(x))
        outs = self.activation(self.batch_norm(torch.cat(outs, dim=1)))
        return self.merge_channels(outs)

    # z_bottleneck = F.dropout1d(z_bottleneck, 0.2)
    # outs = [F.dropout1d(out, 0.2) for out in outs]
