import torch
from torch import nn


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout
    layers right after a separable convolution layer.
    """

    def __init__(self, ni, no, kernel, stride, pad,
                 drop=None, bn=True,
                 activ=lambda: nn.PReLU()):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Classifier3LV1(nn.Module):
    def __init__(self, raw_ni, no, drop=.5, num_positions=8):
        super().__init__()
        self.num_positions = num_positions
        self.raw = nn.Sequential(
            SepConv1d(raw_ni, 32, 5, 2, 3, drop=drop),
            SepConv1d(32, 32, 3, 1, 1, drop=drop),
            SepConv1d(32, 64, 5, 4, 3, drop=drop),
            SepConv1d(64, 64, 3, 1, 1, drop=drop),
            SepConv1d(64, 128, 5, 4, 1, drop=drop),
            SepConv1d(128, 128, 3, 1, 1, drop=drop),
            SepConv1d(128, 256, 1, 4, 2),
            Flatten())
        self.feat1 = nn.Sequential(
            nn.Dropout(drop if drop else 0),
            nn.Linear(768 + self.num_positions, 256), nn.PReLU(), nn.BatchNorm1d(256))
        self.feat2 = nn.Sequential(
            nn.Dropout(drop if drop else 0),
            nn.Linear(256 + self.num_positions, 128), nn.PReLU(), nn.BatchNorm1d(128), nn.PReLU())

        self.out = nn.Sequential(
            nn.Linear(128 + self.num_positions, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            out_counter = 0
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
                if isinstance(child, nn.Linear):
                    out_counter += 1
                    nn.init.uniform_(child.weights, a=0.0, b=1.0)
                    if out_counter > 4:
                        nn.init.normal_(child.weights, mean=0.0, std=1.0)

        init(self)

    def forward(self, t_raw, n_raw, **kwargs):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        # feat = torch.cat((raw_out, n_raw.float()), 1)
        # f1 = self.feat1(feat)
        # feat = torch.cat((f1, n_raw.float()), 1)
        # f2 = self.feat2(feat)
        # feat = torch.cat((f2, n_raw.float()), 1)
        # out = self.out(feat)
        if 'get_feat' in kwargs.keys():
            return out, raw_out
        return out


class Classifier3L(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()

        self.raw = nn.Sequential(
            SepConv1d(raw_ni, 32, 5, 2, 3, drop=drop),
            SepConv1d(32, 32, 3, 1, 1, drop=drop),
            SepConv1d(32, 64, 5, 4, 3, drop=drop),
            SepConv1d(64, 64, 3, 1, 1, drop=drop),
            SepConv1d(64, 128, 5, 4, 1, drop=drop),
            SepConv1d(128, 128, 3, 1, 1, drop=drop),
            SepConv1d(128, 256, 1, 4, 2),
            Flatten())

        self.feat1 = nn.Sequential(
            nn.Dropout(drop), nn.Linear(768, 256), nn.PReLU(), nn.BatchNorm1d(256))
        self.feat2 = nn.Sequential(
            nn.Dropout(drop), nn.Linear(256, 128), nn.PReLU(), nn.BatchNorm1d(128), nn.PReLU())

        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            out_counter = 0;
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
                if isinstance(child, nn.Linear):
                    out_counter += 1
                    nn.init.uniform_(child.weights, a=0.0, b=1.0)
                    if out_counter > 4:
                        nn.init.normal_(child.weights, mean=0.0, std=1.0)

        init(self)

    def forward(self, t_raw, *args, **kwargs):
        raw_out = self.raw(t_raw)
        f1 = self.feat1(raw_out)
        f2 = self.feat2(f1)
        out = self.out(f2)
        if 'get_feat' in kwargs.keys():
            return out, f2
        return out
