import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, hidden_dim):
    # dw
    block = nn.Sequential(
        nn.Conv3d(in_channels, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm3d(hidden_dim),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv3d(hidden_dim, hidden_dim, 1, (1, 2, 2), bias=False),
        nn.BatchNorm3d(hidden_dim),
        nn.ReLU6(inplace=True),
    )
    return block


class SpatialNet(nn.Module):
    def __init__(self, in_channels):
        super(SpatialNet, self).__init__()
        self.conv = nn.Sequential()
        for i in range(5):
            self.conv.add_module(f'block{i}', conv_block(in_channels, in_channels))

    def forward(self, x):
        x = self.conv(x)
        # x = F.adaptive_max_pool3d(x, output_size=(x.shape[0], 1, 1))
        return x.squeeze().transpose(-1, -2)

    def _initialize_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    from torch.autograd import Variable

    net = SpatialNet(200)
    input_var = Variable(torch.randn(2, 200, 24, 32, 32))
    output = net(input_var)
    print(output.shape)
