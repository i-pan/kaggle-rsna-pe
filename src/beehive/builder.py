import logging

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from . import data
from . import evaluate
from . import losses
from . import models 
from . import optim
from . import train 


def get_name_and_params(base):
    return getattr(base, 'name'), getattr(base, 'params')


def build_transforms(cfg, mode):
    # 1-Resize
    name, params = get_name_and_params(cfg.transform.resize)
    resizer = getattr(data.transforms, name)(**params)
    # 2-(Optional) Data augmentation
    if mode == 'train' and cfg.transform.augment:
        name, params = get_name_and_params(cfg.transform.augment)
        augmenter = getattr(data.transforms, name)(**params)
    else:
        augmenter = None
    # 3-(Optional) Crop
    if cfg.transform.crop:
        name, params = get_name_and_params(cfg.transform.crop)
        cropper = getattr(data.transforms, name)(mode=mode, **params)
    else:
        cropper = None
    # 4-Preprocess
    if cfg.transform.preprocess:
        name, params = get_name_and_params(cfg.transform.preprocess)
        preprocessor = getattr(data.transforms, name)(**params)
    else:
        preprocessor = None
    return {
        'resize': resizer,
        'augment': augmenter,
        'crop': cropper,
        'preprocess': preprocessor
    }


def build_dataset(cfg, data_info, mode):
    dataset_class = getattr(data.datasets, cfg.data.dataset.name)
    dataset_params = cfg.data.dataset.params
    if not dataset_params:
        dataset_params = OmegaConf.create()
    dataset_params.test_mode = mode != 'train'
    dataset_params = dict(dataset_params)
    if cfg.transform:
        transforms = build_transforms(cfg, mode)
        dataset_params.update(transforms)
    dataset_params.update(data_info)
    return dataset_class(**dataset_params)


def build_dataloader(cfg, dataset, mode):
    dataloader_params = {}
    dataloader_params['num_workers'] = cfg.data.num_workers
    dataloader_params['drop_last'] = mode == 'train'
    dataloader_params['shuffle'] = mode == 'train'
    if mode == 'train':
        dataloader_params['batch_size'] = cfg.train.batch_size
        sampler = None
        if cfg.data.sampler:
            name, params = get_name_and_params(cfg.data.sampler)
            sampler = getattr(data.samplers, name)(dataset, **params0)
        if cfg.experiment.distributed:
            sampler = DistributedSampler(dataset,
                                         shuffle=True)
        if cfg.local_rank == 0 and sampler:
            logger = logging.getLogger('root')
            logger.info(f'Using sampler {sampler} for training ...')
        if sampler:
            dataloader_params['shuffle'] = False
            dataloader_params['sampler'] = sampler
    else:
        dataloader_params['batch_size'] = cfg.evaluate.batch_size
        if mode != 'predict_full':
            sampler = None
            if cfg.experiment.distributed and cfg.train.params.dist_val:
                sampler = DistributedSampler(dataset,
                                             shuffle=False)
            if cfg.local_rank == 0 and sampler:
                logger = logging.getLogger('root')
                logger.info(f'Using sampler {sampler} for validation ...')
            if sampler:
                dataloader_params['shuffle'] = False
                dataloader_params['sampler'] = sampler

    loader = DataLoader(dataset,
        **dataloader_params)
    return loader


def build_model(cfg):
    name, params = get_name_and_params(cfg.model)
    model = getattr(models.engine, name)(**params)
    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'Creating model <{name}> ...')
        if 'backbone' in cfg.model.params:
            logger.info(f'  Using backbone <{cfg.model.params.backbone}> ...')
        if 'pretrained' in cfg.model.params:
            logger.info(f'  Pretrained : {cfg.model.params.pretrained}')
    return model 


def build_loss(cfg):
    name, params = get_name_and_params(cfg.loss)
    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'Using loss function <{name}> ...')
    if not params:
        params = {} 
    criterion = getattr(losses, name)(**params)
    return criterion


def build_scheduler(cfg, optimizer):
    # Some schedulers will require manipulation of config params
    # My specifications were to make it more intuitive for me
    name, params = get_name_and_params(cfg.scheduler)
    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'Using learning rate schedule <{name}> ...')

    if name == 'CosineAnnealingWarmRestarts':
        # eta_min <-> final_lr
        # T_0 calculated from num_epochs and num_snapshots
        params = {
            # Use num_epochs from training parameters
            'T_0': int(cfg.train.params.num_epochs / params.num_snapshots),
            'eta_min': params.final_lr
        }

    if name == 'CosineAnnealingLR':
        # eta_min <-> final_lr
        # T_max calculated from num_epochs and steps_per_epoch
        params = {
            'T_max': int(cfg.train.params.num_epochs * cfg.train.params.steps_per_epoch),
            'eta_min': params.final_lr
        }

    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        # Use learning rate from optimizer parameters as initial learning rate
        INIT_LR  = cfg.optimizer.params.lr
        MAX_LR = params.max_lr
        FINAL_LR = params.final_lr
        params = {}
        params['max_lr'] = MAX_LR
        params['steps_per_epoch'] = cfg.train.params.steps_per_epoch
        params['epochs'] = cfg.train.params.num_epochs
        params['div_factor'] = MAX_LR / INIT_LR
        params['final_div_factor'] = INIT_LR / FINAL_LR

    scheduler = getattr(optim, name)(optimizer=optimizer, **params)
    
    # Some schedulers might need more manipulation after instantiation
    if name == 'CosineAnnealingWarmRestarts':
        scheduler.T_cur = 0
    # Set update frequency
    if name in ('CosineAnnealingWarmRestarts', 'OneCycleLR', 'CustomOneCycleLR', 'CosineAnnealingLR'):
        scheduler.update = 'on_batch'
    elif name in ('ReduceLROnPlateau'):
        scheduler.update = 'on_valid'
    else:
        scheduler.update = 'on_epoch'

    return scheduler


def build_optimizer(cfg, parameters):
    name, params = get_name_and_params(cfg.optimizer)
    if cfg.local_rank == 0:
        logger = logging.getLogger('root')
        logger.info(f'Using optimizer <{name}> ...')
    optimizer = getattr(optim, name)(parameters, **params)
    return optimizer


def build_evaluator(cfg, loader):
    name, params = get_name_and_params(cfg.evaluate)
    params.prefix = cfg.model.name
    evaluator = getattr(evaluate, name)(loader, **params)
    return evaluator




