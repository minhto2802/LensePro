import math
import torch
from torch import nn

from networks.utils import Conv1dSamePadding

from typing import cast, Union, List
import torch.nn.functional as F
import torch.nn.init as init
from loss_functions.isomax import IsoMaxLossFirstPart  # , IsoMaxLossFirstPartV1
# from loss_functions.isomaxplus import IsoMaxPlusLossFirstPart as IsoMaxLossFirstPart


ACT = nn.ReLU
# ACT = nn.SELU


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939
    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 input_length: int, use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1, self_train=False, stride=1,
                 num_positions=0,
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes,
            'stride': stride,
            'input_length': input_length,
        }
        self.self_train = self_train
        # channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
        #                                                                   num_blocks))
        # bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels, num_blocks))
        channels = [in_channels] + [out_channels for i in range(num_blocks)]
        bottleneck_channels = [bottleneck_channels] * num_blocks
        # bottleneck_channels = [c//2 for c in channels[1:]]
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        strides = cast(List[int], self._expand_to_blocks(stride, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
            # use_residuals = [False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks))

        self.blocks = nn.Sequential(*[
            nn.Sequential(InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                                         residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                         stride=strides[i],
                                         kernel_size=kernel_sizes[i],
                                         groups=1),
                          # nn.BatchNorm1d(channels[i + 1]),
                          # ACT(),
                          ) for i in range(num_blocks)
        ])

        # self.blocks = nn.Sequential(*[
        #     InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
        #                    residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
        #                    stride=strides[i],
        #                    kernel_size=kernel_sizes[i]) for i in range(num_blocks)])

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.feature_size = channels[-1]
        self.num_positions = num_positions
        # self.pos_encoder = nn.Linear(in_features=channels[-1] + num_positions, out_features=channels[-1])

        # linear_in = channels[-1] + num_positions if num_positions > 0 else channels[-1]
        # self.pos_encoder1 = nn.Sequential(
        #     nn.Dropout(.2),
        #     nn.Linear(linear_in, channels[-1]), nn.PReLU(), nn.BatchNorm1d(channels[-1]))
        # self.pos_encoder2 = nn.Sequential(
        #     nn.Dropout(.2),
        #     nn.Linear(linear_in, channels[-1]), nn.PReLU(), nn.BatchNorm1d(channels[-1]))
        # self.out = nn.Sequential(
        #     nn.Linear(linear_in, channels[-1]), nn.ReLU(inplace=True), nn.Linear(channels[-1], channels[-1]))

        self.feat_extractor = FeatureExtractor(self.blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.classifier = Classifier(channels[-1], channels[-1], num_pred_classes)

        # self.fc = IsoMaxLossFirstPart(channels[-1] * int(output_length), num_pred_classes)

        # self.linear10 = nn.Linear(channels[-1], channels[-1])
        # self.linear11 = IsoMaxLossFirstPart(channels[-1], num_positions)

        # self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.feat_extractor(x, *args)

        if self.num_positions > 0:
            x = self.pos_encoder1(torch.cat((x, args[0].float()), 1))
            x = self.pos_encoder2(torch.cat((x, args[0].float()), 1))
            x = self.out(torch.cat((x, args[0].float()), 1))
        if self.self_train:
            return F.normalize(x, dim=1)
        out1 = self.classifier(x, *args)
        # out1 = self.fc(x)

        # if self.num_positions > 0:
        #     x = self.pos_encoder1(torch.cat((x, args[0].float()), 1))
        #     x = self.pos_encoder2(torch.cat((x, args[0].float()), 1))
        #     x = self.pos_encoder(torch.cat((x, args[0].float()), 1))
        #     if self.num_positions > 0:
        #         x = self.pos_encoder1(torch.cat((x, args[0].float()), 1))
        #         x = self.pos_encoder2(torch.cat((x, args[0].float()), 1))
        #     else:
        #         x = self.pos_encoder2(self.pos_encoder1(x))
        #     out2 = self.linear11(F.relu(self.linear10(x)))
        #     return out1, out2

        if 'get_feat' in kwargs.keys():
            return out1, x
        return out1

    def train(self, mode=True, freeze_bn=False):
        """
        Override the default train() to freeze the BN parameters
        """
        super(InceptionModel, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41, drop: float = None, groups=1) -> None:
        super().__init__()
        self.drop = drop

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, bias=False, groups=1)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        strides = [1, 1, stride]
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=strides[i], bias=False, groups=groups)
            for i in range(len(kernel_size_s))
        ])

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm1d(out_channels),
                # nn.GroupNorm(1, out_channels),
                ACT()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        # print(x.shape, self.residual(org_x).shape)
        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.net = nn.Sequential(*args)

    def forward(self, x, *args):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, feat_dim, out_dim, num_pred_classes):
        super().__init__()
        # self.linear01 = IsoMaxLossFirstPart(out_dim, num_pred_classes)
        self.linear01 = nn.Sequential(nn.Linear(out_dim, num_pred_classes))

    def forward(self, x, *args):
        # return self.linear01(self.linear00(x))
        return self.linear01(x)


def main():
    from torchinfo import summary
    num_blocks, in_channels, pred_classes = 3, 1, 2
    net = InceptionModel(num_blocks, in_channels, out_channels=16, input_length=200,
                         bottleneck_channels=12, kernel_sizes=15, use_residuals='default',  # 'default',
                         stride=1, num_pred_classes=pred_classes, num_positions=0)
    summary(net, input_size=[(2, 1, 200), (2, 8)])


if __name__ == '__main__':
    main()
