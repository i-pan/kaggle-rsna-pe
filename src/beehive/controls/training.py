import logging

from torch.nn.parallel import DistributedDataParallel
from torch.nn import SyncBatchNorm as sbn
from .. import train as beehive_train
from .. import builder
from .datamaker import get_train_val_dataloaders


def build_trainer(cfg):
    # Create data loaders
    train_loader, valid_loader = get_train_val_dataloaders(cfg)
    if cfg.train.params.steps_per_epoch == 0:
        cfg.train.params.steps_per_epoch = len(train_loader)
    # Create model
    model = builder.build_model(cfg)
    if cfg.experiment.distributed:
        if cfg.experiment.sync_bn:
            model = sbn.convert_sync_batchnorm(model)
        if cfg.experiment.cuda: 
            model.to(f'cuda:{cfg.local_rank}')
        model = DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
    else:
        if cfg.experiment.cuda: 
            model.to(f'cuda:{cfg.local_rank}')
    model.train()
    # Create loss
    criterion = builder.build_loss(cfg)
    # Create optimizer
    optimizer = builder.build_optimizer(cfg, model.parameters())
    # Create learning rate scheduler
    scheduler = builder.build_scheduler(cfg, optimizer)
    # Create evaluator 
    evaluator = builder.build_evaluator(cfg, valid_loader)
    trainer = getattr(beehive_train, cfg.train.name)
    trainer = trainer(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        logger=logging.getLogger('root'),
        cuda=cfg.train.params.pop('cuda'),
        dist=cfg.experiment.distributed
    )
    return trainer


def train(cfg):
    trainer = build_trainer(cfg)
    trainer.set_local_rank(cfg.local_rank)
    trainer.set_world_size(cfg.world_size)
    trainer.train(**cfg.train.params)


