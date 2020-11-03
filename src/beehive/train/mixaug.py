import torch
import numpy as np


def apply_mixup(X, alpha):
    lam = np.random.beta(alpha, alpha, X.size(0))
    lam = np.max((lam, 1-lam), axis=0)
    index = torch.randperm(X.size(0))
    lam = torch.Tensor(lam).to(X.device)
    for dim in range(X.ndim - 1):
        lam = lam.unsqueeze(-1)
    X = lam * X + (1 - lam) * X[index]
    return X, index, lam


def rand_bbox(size, lam, margin=0):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = (H * cut_rat).astype(np.int)
    cut_w = (W * cut_rat).astype(np.int)
    # uniform
    if margin < 1 and margin > 0:
        h_margin = margin*H
        w_margin = margin*W
    else:
        h_margin = margin
        w_margin = margin
    cx = np.random.randint(0+h_margin, H-h_margin, B)
    cy = np.random.randint(0+w_margin, W-w_margin, B)
    #
    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)
    return bbx1, bby1, bbx2, bby2


def apply_cutmix(X, alpha):
    batch_size = X.size(0)
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.max((lam, 1-lam), axis=0)
    x1, y1, x2, y2 = rand_bbox(X.size(), lam)
    index = torch.randperm(batch_size)
    for b in range(X.size(0)):
        X[b, ..., x1[b]:x2[b], y1[b]:y2[b]] = X[index[b], ..., x1[b]:x2[b], y1[b]:y2[b]]
    lam = 1. - ((x2 - x1) * (y2 - y1) / float((X.size(-1) * X.size(-2))))
    return X, index, lam


MIX_FN = {'cutmix': apply_cutmix, 'mixup': apply_mixup}
def apply_mixaug(X, y, mix):
    mixer = np.random.choice([*mix])
    X, index, lam = MIX_FN[mixer](X, mix[mixer])
    return X, {
        'y1':  y,
        'y2':  y[index],
        'lam': lam
    }





