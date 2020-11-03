import torch
import os


def is_tensor(x): 
    return isinstance(x, torch.Tensor)


def _cudaify(x, device):
    dev = f'cuda:{device}'
    if isinstance(x, dict):
        return {k:v.to(dev) if is_tensor(v) else v for k,v in x.items()}

    if isinstance(x, (tuple,list)):
        return type(x)([_.to(dev) if is_tensor(_) else _cudaify(_, device) for _ in x])

    return x.to(dev)


def cudaify(batch, labels, device): 
    return _cudaify(batch, device), _cudaify(labels, device)


def create_dir(d):
    if not os.path.exists(d):
        print(f'Creating directory : {d} ...')
        os.makedirs(d)