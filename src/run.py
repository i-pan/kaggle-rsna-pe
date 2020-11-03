import argparse
import logging
import os, os.path as osp
import re
import torch

from omegaconf import OmegaConf
from beehive.controls import datamaker, training, inference
from beehive.reproducibility import set_reproducibility
from beehive.utils import create_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', type=str) 
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--dist-val', action='store_true')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--kfold', type=int, default=-1)
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--save-features-dir', type=str)
    parser.add_argument('--save-probas-dir', type=str)
    parser.add_argument('--save-pe-slice', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--annotations', type=str)
    parser.add_argument('--load-backbone', type=str)
    parser.add_argument('--load-transformer', type=str)
    return parser.parse_args()


def setup_dist(rank):
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', init_method='env://')
    return torch.distributed.get_world_size()


def create_logger(save_dir, mode):
    create_dir(save_dir)
    logfile = osp.join(save_dir, '{}_log.txt'.format(mode))
    if osp.exists(logfile): 
        print(f'Removing existing log file {logfile} ...')
        os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))
    logger.info(f'Saving logs to {save_dir} ...\n')


def setup(args, cfg):
    if args.dist:
        # Distributed training
        cfg.world_size = setup_dist(args.local_rank)
        cfg.train.params.dist_val = args.dist_val
    elif args.device_id >= 0:
        # Single GPU
        torch.cuda.set_device(args.device_id)
        cfg.world_size = 1

    cfg.experiment.distributed = args.dist
    cfg.experiment.sync_bn = args.sync_bn
    if args.name:
        cfg.experiment.name = args.name
    
    if args.seed:
        cfg.experiment.seed = args.seed

    if args.annotations:
        cfg.data.annotations = args.annotations

    cfg.local_rank = args.local_rank
    
    cfg.train.params.cuda = cfg.evaluate.params.cuda = cfg.experiment.cuda

    if not cfg.experiment.name:
        cfg.experiment.name = osp.basename(args.config).replace('.yaml','')

    if args.load_backbone:
        cfg.model.params.load_backbone = args.load_backbone

    if args.load_transformer:
        cfg.model.params.load_transformer = args.load_transformer

    # Make all necessary changes to config file here
    if args.num_workers > 0:
        cfg.data.num_workers = args.num_workers

    save_log_dir = osp.join(cfg.experiment.save_logs, cfg.experiment.name)
    save_checkpoints_dir = osp.join(cfg.experiment.save_checkpoints,
                                    cfg.experiment.name)

    if args.kfold >= 0:
        if isinstance(cfg.data.inner_fold, (int,float)): 
            cfg.data.inner_fold = None
        cfg.data.outer_fold = args.kfold
        save_log_dir = osp.join(save_log_dir, f'fold{args.kfold}')
        save_checkpoints_dir = osp.join(save_checkpoints_dir, f'fold{args.kfold}')
        cfg.experiment.seed = int(f'{cfg.experiment.seed}{args.kfold}')
        if re.search(r'fold[0-9]+', cfg.data.data_dir):
            cfg.data.data_dir = re.sub(r'fold[0-9]+', f'fold{args.kfold}', cfg.data.data_dir)
        if args.local_rank == 0:
            print(f'Loading data from {cfg.data.data_dir} ...')

    if not cfg.evaluate.batch_size:
        cfg.evaluate.batch_size = cfg.train.batch_size        

    cfg.evaluate.params.save_checkpoint_dir = save_checkpoints_dir

    if args.local_rank == 0:
        create_logger(save_log_dir, args.mode)
        logger = logging.getLogger('root')

    set_reproducibility(cfg.experiment.seed, args.local_rank)

    if args.local_rank == 0 and args.mode == 'train':
        logger.info(f'Running experiment {cfg.experiment.name} ...')
        logger.info(f'CONFIG : {args.config}')
        if args.dist:
            logger.info('==DISTRIBUTED TRAINING==')
        elif args.device_id >= 0:
            logger.info(f'==SINGLE GPU, ID#{args.device_id}==')
        create_dir(save_checkpoints_dir)
        logger.info(f'Saving checkpoints to {save_checkpoints_dir} ...')
        # We will set all the seeds we can, in vain ...

    if cfg.transform:
        if cfg.transform.augment == 'grid_mask':
            assert 'grid_mask' in cfg.train.params

    return cfg


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    cfg = setup(args, cfg)

    if args.mode == 'train':
        training.train(cfg)
    elif args.mode == 'predict_full':
        inference.predict_full(cfg)
    else:
        cfg.save_features_dir = args.save_features_dir
        cfg.save_probas_dir = args.save_probas_dir
        cfg.checkpoint = args.checkpoint
        cfg.save_pe_slice = args.save_pe_slice
        getattr(inference, args.mode)(cfg)


if __name__ == '__main__':
    main()












