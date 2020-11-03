import numpy as np

from sklearn import metrics


def auc(t, p, **kwargs):
    # y_pred.shape = (N,)
    if p.ndim == 2 and t.ndim == 2:
        p = p[:,0]
        t = t[:,0]
    return {'auc': metrics.roc_auc_score(t, p)}


def multi_auc(t, p, **kwargs):
    # y_pred.shape = (N,C)
    metrics_dict = {}
    for i in range(p.shape[-1]):
        try:
            metrics_dict[f'auc{i}'] = metrics.roc_auc_score(t[:,i], p[:,i])
        except ValueError:
            metrics_dict[f'auc{i}'] = 0
    return metrics_dict


def multi_loss(t, p, **kwargs):
    # y_pred.shape = (N,C)
    metrics_dict = {}
    for i in range(p.shape[-1]):
        metrics_dict[f'loss{i}'] = metrics.log_loss(t[:,i], p[:,i], labels=[0,1], eps=1e-6)
    return metrics_dict


def accuracy(t, p, **kwargs):
    return {'accuracy': np.mean(np.argmax(p, axis=1) == t)}


def pos_accuracy(t, p, **kwargs):
    # p.shape (N, 4)
    # t is 0-3 (0=no PE, 1=right, 2=left, 3=central)
    positives = (t > 0)
    p = p[positives]
    t = t[positives]
    return {'pos_accuracy': np.mean(np.argmax(p[:,1:], axis=1)+1 == t)}