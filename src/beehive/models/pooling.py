import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d


# From: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
def gem_1d(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1),)).pow(1./p)


def gem_2d(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def gem_3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./p)


_GEM_FN = {
    1: gem_1d, 2: gem_2d, 3: gem_3d
}


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return _GEM_FN[self.dim](x, p=self.p, eps=self.eps)


class AdaptiveConcatPool1d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool1d(x, 1), F.adaptive_max_pool1d(x, 1)), dim=1)


class AdaptiveConcatPool2d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)), dim=1)


class AdaptiveConcatPool3d(nn.Module):

    def forward(self, x):
        return torch.cat((F.adaptive_avg_pool3d(x, 1), F.adaptive_max_pool3d(x, 1)), dim=1)