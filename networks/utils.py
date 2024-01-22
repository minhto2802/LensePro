import torch
from torch import nn
import torch.nn.functional as F

# from .layers.conv_variational import Conv1dVariational

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -2.0


class Conv1dSamePaddingV0(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    # padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    padding = (((l_out - 1)) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)


def transfer_weights(model, weights_path: str, device: torch.device = 'cuda', exclude_head: bool = True):
    """Utility function that allows to easily transfer weights between models.
    Taken from the great self-supervised repository created by Kerem Turgutlu.
    https://github.com/KeremTurgutlu/self_supervised/blob/d87ebd9b4961c7da0efd6073c42782bbc61aaa2e/self_supervised/utils.py"""

    state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    matched_layers = 0
    unmatched_layers = []
    for name, param in state_dict.items():
        if exclude_head and 'head' in name: continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
