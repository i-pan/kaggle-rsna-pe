import random, torch, os, numpy as np
import logging
# From: https://github.com/liaopeiyuan/ml-arsenal-public/blob/master/reproducibility.py

def set_reproducibility(SEED, rank):
    logger = logging.getLogger('root')
    print = logger.info
    if rank == 0:
        print('Fixing random seed for reproducibility ...')
    if rank > 0:
        SEED = int(f'{SEED}{rank}')
    os.environ['PYTHONHASHSEED'] = f'{SEED}'
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f'  Setting random seed to {SEED} !')
    print('')
    #
    torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = True
    if rank == 0:
        print('PyTorch environment ...')
        print(f'  torch.__version__              = {torch.__version__}')
        print(f'  torch.version.cuda             = {torch.version.cuda}')
        print(f'  torch.backends.cudnn.version() = {torch.backends.cudnn.version()}')
        print('\n')