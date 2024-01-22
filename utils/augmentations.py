import math

import torch
# from tsai.utils import random_shuffle
import torch.nn as nn
from torch.distributions import Beta

import numpy as np
from PIL import Image


class IntraClassCutMix1d():
    "Implementation of CutMix applied to examples of the same class"
    run_valid = False

    def __init__(self, alpha=1.):
        self.distrib = Beta(alpha, alpha)

    def before_batch(self, x, y):
        bs, *_, seq_len = x.size()
        idxs = torch.arange(bs, device=x.device)
        # y = torch.tensor(y)
        unique_c = torch.unique(y).tolist()
        idxs_by_class = torch.cat([idxs[torch.eq(y, c)] for c in unique_c])
        idxs_shuffled_by_class = torch.cat([random_shuffle(idxs[torch.eq(y, c)]) for c in unique_c])

        lam = self.distrib.sample((1,))
        x1, x2 = self.rand_bbox(seq_len, lam)
        xb1 = x[idxs_shuffled_by_class]
        x[idxs_by_class, :, x1:x2] = xb1[..., x1:x2]
        return x

    def rand_bbox(self, seq_len, lam):
        cut_seq_len = torch.round(seq_len * (1. - lam)).type(torch.long)
        half_cut_seq_len = torch.div(cut_seq_len, 2, rounding_mode='floor')

        # uniform
        cx = torch.randint(0, seq_len, (1,))
        x1 = torch.clamp(cx - half_cut_seq_len, 0, seq_len)
        x2 = torch.clamp(cx + half_cut_seq_len, 0, seq_len)
        return x1, x2

class Grid(object):
    """Modified version of https://github.com/dvlab-research/GridMask/blob/master/imagenet_grid/utils/grid.py"""
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        # h = img.size(1)
        # w = img.size(2)
        h, w = img.shape[:2]

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        # mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask[..., np.newaxis]  # .expand_as(img)
        img = img * mask

        return img


class GridMask(nn.Module):
    """Currently incompatible with the modified version of Grid"""
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n, c, h, w)
        return y


def main():
    intra_cut_mix = IntraClassCutMix1d()
    x = torch.rand([12, 1, 200])
    y = torch.randint(0, 2, [12, ])
    x_out = intra_cut_mix.before_batch(x, y)
    print(x_out.size())


if __name__ == '__main__':
    main()
