import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable


WEIGHTS = [
    0.0736196319,
    0.09202453988,
    0.1042944785,
    0.1042944785,
    0.1877300613,
    0.06257668712,
    0.06257668712,
    0.2346625767,
    0.0782208589
]


class MulticlassLogLoss(nn.Module):

    def __init__(self, weights=WEIGHTS):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS).float()

    def forward(self, p, t):
        return torch.sum(self.weights.to(t.device)*F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none').mean(0))


class PEStudyLogLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS[:-2]).float()
        self.weights = self.weights / self.weights.sum()

    def forward(self, p, t):
        return torch.sum(self.weights.to(t.device)*F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none').mean(0))


class PELocalLogLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([WEIGHTS[0]]+WEIGHTS[4:7]).float()
        self.weights = self.weights / self.weights.sum()

    def forward(self, p, t):
        return torch.sum(self.weights.to(t.device)*F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none').mean(0))


class PETypeLogLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS[:4]).float()
        self.weights = self.weights / self.weights.sum()

    def forward(self, p, t):
        return torch.sum(self.weights.to(t.device)*F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none').mean(0))


class RVLVLogLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([[WEIGHTS[-2],WEIGHTS[-1]]]).float()
        self.weights = self.weights / self.weights.sum()

    def forward(self, p, t):
        return torch.sum(self.weights.to(t.device)*F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none').mean(0))


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        return F.binary_cross_entropy_with_logits(p.float(), t.float())


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self, *args, **kwargs):
        self.weighted = kwargs.pop('weighted', False)
        super().__init__(*args, **kwargs)

    def forward(self, p, t):
        t,mask = t
        if self.weighted:
            loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none')
            # loss.shape = (B, D)
            # Compute proportion of positives for each batch element
            w = t.float().sum(dim=1) / mask.float().sum(dim=1)
            # w.shape = (B,)
            loss = (w.unsqueeze(1)*loss).flatten()
            mask = mask.flatten()
            return loss[mask==1].mean()
        else:
            loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none')
            return (loss[mask==1]).mean()


class PerSliceWeightedBCE(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        t,mask,wt = t
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none')
        loss = (wt.unsqueeze(1)*loss)
        loss = loss.flatten()
        mask = mask.flatten()
        return loss[mask==1].mean()


class PEProbRefineLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        t,wt = t
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction='none')
        if loss.ndim == 2:
            wt = wt.unsqueeze(1)
        assert loss.shape == wt.shape
        loss = wt*loss
        return loss.mean()


class MulticlassPerSliceLoss(nn.Module):

    def __init__(self, weights=[0.2,1]):
        super().__init__()
        self.weights = torch.tensor(WEIGHTS).float()
        self.multiclass_loss = MulticlassLogLoss()

    def forward(self, p, t):
        slice_loss = F.binary_cross_entropy_with_logits(p[0].float(), t['pe_present_on_image'].float())
        exam_loss  = self.multiclass_loss(p[1], t['exam_level_labels'])
        return self.weights[0]*slice_loss + self.weights[1]*exam_loss


class MultiSliceBCE(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        loss = 0.
        for i in range(p.size(1)):
            loss += F.binary_cross_entropy_with_logits(p[:,i].float(), t[:,i].float())
        return loss / p.size(1)


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, p, t):
        t = t.view(-1)
        if self.weight:
            return F.cross_entropy(p.float(), t.long(), weight=self.weight.float().to(t.device))
        else:
            return F.cross_entropy(p.float(), t.long())


class OneHotCrossEntropy(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class SmoothCrossEntropy(nn.Module):
    
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return F.cross_entropy(x, target.long())


class MixCrossEntropy(nn.Module):

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.cross_entropy(p.float(), t['y_true1'].long(), reduction='none')
        loss2 = F.cross_entropy(p.float(), t['y_true2'].long(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.cross_entropy(p.float(), t.long())


class DenseCrossEntropy(nn.Module):

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.Module):

    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        labels = F.one_hot(labels.long(), logits.size(1)).float().to(labels.device)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss
