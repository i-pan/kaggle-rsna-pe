import torch
import numpy as np

from torch import nn
from torch.nn import functional as F


class WSO(nn.Module):
    #
    def __init__(self, nch, wl, ww, input_ch=1, act='sigmoid', upper=255.0, dim=3, smooth=1., trainable=True):
        super(WSO, self).__init__()
        self.nch = nch
        self.wl  = wl 
        self.ww  = ww
        self.act = act 
        self.upper = upper
        self.smooth = smooth
        #
        _conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.conv = _conv(input_ch, nch, kernel_size=1, stride=1, groups=input_ch)
        self.init_weights(wl, ww)
        if not trainable:
            self.conv.weight.requires_grad = False
            self.conv.bias.requires_grad = False
    #
    def forward(self, x):
        x = self.conv(x)
        if self.act == 'relu':
            x = F.relu(x)
            x[x > self.upper] = self.upper
        elif self.act == 'sigmoid': 
            x = torch.sigmoid(x) * self.upper
        return x
    #
    def init_weights(self, wl, ww):
        # if type(wl) != list: wl = [wl]
        # if type(ww) != list: ww = [ww]
        assert len(ww) == len(wl) == self.nch
        params = self.get_params(wl, ww)
        self.conv.state_dict()['weight'].copy_(params[0].reshape(self.conv.weight.shape))
        self.conv.state_dict()['bias'].copy_(params[1].reshape(self.conv.bias.shape))
    #
    def get_params(self, wl, ww):
        ws = []
        bs = []
        for i in range(len(wl)):
            if self.act == 'relu':
                w = self.upper / ww[i]
                b = -1. * self.upper * (wl[i] - ww[i] / 2.) / ww[i]
            elif self.act == 'sigmoid':
                w = 2./ww[i] * np.log(self.upper/self.smooth - 1.)
                b = -2.*wl[i]/ww[i] * np.log(self.upper/self.smooth - 1.)
            ws.append(w)
            bs.append(b)
        return torch.tensor(ws, requires_grad=True), \
               torch.tensor(bs, requires_grad=True)


class WSO2d(WSO):

    def __init__(self, *args, **kwargs):
        kwargs['dim'] = 2
        super().__init__(*args, **kwargs)


class WSO3d(WSO):

    def __init__(self, *args, **kwargs):
        kwargs['dim'] = 3
        super().__init__(*args, **kwargs)


