import datetime
import glob
import numpy as np
import os.path as osp
import pickle
import time

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    print('PyTorch automatic mixed precision unavailable !')
    print('Set train.params.amp to False')

from collections import defaultdict
from ..utils import cudaify
from .mixaug import apply_mixaug


class TimeMeter:

    def __init__(self, mavg=100):
        self.mavg = mavg
        self.load_time = []
        self.step_time = []

    def set_time(self, t):
        self.load_time.append(t[0])
        self.step_time.append(t[1])

    def get_time(self):
        return (np.mean(self.load_time[-int(self.mavg):]),
                np.mean(self.step_time[-int(self.mavg):]))


class LossMeter: 

    def __init__(self, mavg=100): 
        self.mavg = mavg
        self.losses  = defaultdict(list)
        self.history = defaultdict(list)

    def set_loss(self, minibatch_loss):
        for k,v in minibatch_loss.items():
            self.losses[k].append(v) 

    def get_loss(self): 
        for k,v in self.losses.items():
            self.history[k].append(np.mean(v[-self.mavg:]))
        return {k:v[-1] for k,v in self.history.items()}

    def reset(self): 
        self.losses = defaultdict(list)

    def get_history(self): 
        return self.history


class Step:
    
    def __init__(self, loader, cuda=True, dist=False):
        self.loader = loader
        self.cuda = cuda
        self.dist = dist
        self.generator = self._generator()
        self.loss_meter = LossMeter(mavg=100)
        self.time_meter = TimeMeter(mavg=100)
        self.success = False 
        self._sampler_epoch = 0

    # Wrap data loader in generator
    def _generator(self):
        loader_len = len(self.loader)
        while 1:
            for i, data in enumerate(self.loader, 1):
                if self.dist and i % loader_len == 0 and i > 0:
                    self._sampler_epoch += 1
                    self.loader.sampler.set_epoch(self._sampler_epoch)
                yield data 

    # Get data
    def _data(self):
        batch, labels = next(self.generator)
        if self.cuda:
            batch, labels = cudaify(batch, labels, self.local_rank)

        if hasattr(self, 'mix') and self.mix:
            assert isinstance(self.mix, dict)
            assert all(i in ['cutmix', 'mixup'] for i in self.mix)
            batch, labels = apply_mixaug(self.mix)

        return batch, labels

    def _loss(self, output, labels):
        loss = self.criterion(output, labels)
        metered = {'loss': loss.item()}
        return loss, metered

    def _step(self):
        data_tic = time.time()
        batch, labels = self._data()
        data_toc = time.time() - data_tic

        def closure(): 
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss, metered = self._loss(output, labels)
            loss.backward()
            self.loss_meter.set_loss(metered)
            return loss

        step_tic = time.time()
        if self.amp:
            self.optimizer.zero_grad()
            with autocast():
                output = self.model(batch)
                loss, metered = self._loss(output, labels)
                self.scaler.scale(loss).backward()
                self.loss_meter.set_loss(metered)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss = self.optimizer.step(closure=closure)
        step_toc = time.time() - step_tic
        self.time_meter.set_time((data_toc, step_toc))

    def _accumulate_step(self):
        data_tic = time.time()
        batch, labels = self._data()
        data_toc = time.time() - data_tic
        batch_size = batch.size(0)
        assert batch_size % self.gradient_accumulation == 0, \
            f'Batch size <{batch_size}> must be multiple of gradient accumulation <{self.gradient_accumulation}>'

        splits = torch.split(torch.arange(batch_size), int(batch_size / self.gradient_accumulation))

        def closure(): 
            metered = defaultdict(list)
            self.optimizer.zero_grad()
            for i in range(self.gradient_accumulation):
                output = self.model(batch[splits[i]])
                if isinstance(labels, dict):
                    loss, _metered = self._loss(output, 
                        {k:v[splits[i]] for k,v in labels.items()})
                else:
                    loss, _metered = self._loss(output, labels[splits[[i]]])
                (loss / self.gradient_accumulation).backward()
                for k,v in _metered: metered[k] += [v]
            self.loss_meter.set_loss({k:np.mean(v) for k,v in metered})
            return loss

        step_tic = time.time()
        if self.amp:
            self.optimizer.zero_grad()
            with autocast():
                metered = defaultdict(list)
                self.optimizer.zero_grad()
                for i in range(self.gradient_accumulation):
                    output = self.model(batch[splits[i]])
                    if isinstance(labels, dict):
                        loss, _metered = self._loss(output, 
                            {k:v[splits[i]] for k,v in labels.items()})
                    else:
                        loss, _metered = self._loss(output, labels[splits[[i]]])
                    self.scaler.scale(loss / self.gradient_accumulation).backward()
                    for k,v in _metered: metered[k] += [v]
            self.loss_meter.set_loss({k:np.mean(v) for k,v in metered})
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.optimizer.step(closure=closure)
        step_toc = time.time() - step_start
        self.time_meter.set_time((data_toc, step_toc))

    def train_step(self):
        self._accumulate_step() if self.gradient_accumulation > 1 else self._step()

        if not self.success:
            self.print('\n// first batch successful ^^')
            self.success = True

        if self.grid_mask:
            # grid_mask is a float in (0, 1)
            # anneal from p_start to p_end in grid_mask*total_steps
            self.loader.dataset.transform.set_p(int(self.grid_mask * self.total_steps))


class Trainer(Step):

    def __init__(self,
                 loader,
                 model,
                 optimizer,
                 scheduler,
                 criterion,
                 evaluator,
                 logger,
                 cuda=True,
                 dist=False):
        super().__init__(loader=loader, cuda=cuda, dist=dist)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.evaluator = evaluator
        self.logger = logger
        self.print  = self.logger.info
        self.evaluator.set_logger(self.logger)
        self.steps = 0
        self.current_epoch = 0

    def check_end_train(self): 
        return self.current_epoch >= self.num_epochs

    def check_end_epoch(self):
        return (self.steps % self.steps_per_epoch) == 0 and (self.steps > 0)

    def check_validation(self):
        # We add 1 to current_epoch when checking whether to validate
        # because epochs are 0-indexed. E.g., if validate_interval is 2,
        # we should validate after epoch 1. We need to add 1 so the modulo
        # returns 0
        return self.check_end_epoch() and self.steps > 0 and ((self.current_epoch + 1) % self.validate_interval) == 0

    def scheduler_step(self):
        if 'warmrestarts' in str(self.scheduler).lower():
            self.scheduler.step(self.current_epoch + self.steps * 1./self.steps_per_epoch)
        else:
            self.scheduler.step()

    def print_progress(self):
        train_loss = self.loss_meter.get_loss()
        loss_statement = ''
        for k,v in train_loss.items():
            loss_statement += '{}={:.4f} '.format(k,v)
        learning_rates = np.unique([_['lr'] for _ in self.optimizer.param_groups])
        lr_statement = 'lr='
        for lr in learning_rates:
            lr_statement += '{:.1e}/'.format(lr)
        lr_statement = lr_statement[:-1]
        if self.local_rank == 0:
            self.print('epoch {epoch}, batch {batch}/{steps_per_epoch}: {loss_statement}(data: {load_time:.3f}s/batch, step: {step_time:.3f}s/batch, {lr_statement})'
                    .format(epoch=str(self.current_epoch).zfill(len(str(self.num_epochs))), \
                            batch=str(self.steps).zfill(len(str(self.steps_per_epoch))), \
                            steps_per_epoch=self.steps_per_epoch, \
                            loss_statement=loss_statement, \
                            load_time=self.time_meter.get_time()[0],
                            step_time=self.time_meter.get_time()[1],
                            lr_statement=lr_statement))

    def set_local_rank(self, local_rank=0):
        self.local_rank = local_rank
        self.evaluator.set_local_rank(local_rank)

    def set_world_size(self, world_size=1):
        self.world_size = world_size

    @staticmethod
    def load_pickle(fp):
        with open(fp, 'rb') as f:
            return pickle.load(f)

    def train(self,
              num_epochs,
              steps_per_epoch,
              validate_interval,
              gradient_accumulation=1,
              amp=True,
              verbosity=None,
              grid_mask=None,
              mix=None,
              dist_val=False):
        # Epochs are 0-indexed
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.validate_interval = validate_interval
        self.gradient_accumulation = gradient_accumulation
        self.amp = amp
        if self.amp: self.scaler = GradScaler()
        if verbosity:
            verbosity = verbosity
        else:
            verbosity = self.steps_per_epoch // 10
        self.grid_mask = grid_mask
        self.mix = mix
        self.dist_val = dist_val
        tic = datetime.datetime.now() 
        while 1: 
            self.train_step()
            self.steps += 1
            if self.scheduler.update == 'on_batch':
                 self.scheduler_step()
            # Check- print training progress
            if self.steps % verbosity == 0 and self.steps > 0:
                self.print_progress()
            # Check- run validation
            if self.check_validation():
                # if self.local_rank == 0:
                    self.print('VALIDATING ...')
                    validation_start_time = datetime.datetime.now()
                    # Start validation
                    self.model.eval()
                    if self.dist_val:
                        self.evaluator.validate(self.model,
                            self.criterion,
                            str(self.current_epoch).zfill(len(str(self.num_epochs))),
                            save_pickle=True)
                        if self.local_rank == 0:
                            while 1:
                                predictions = glob.glob(osp.join(self.evaluator.save_checkpoint_dir, '.tmp_preds_rank*.pkl'))
                                if len(predictions) == self.world_size:
                                    break
                            time.sleep(5)
                            # Combine and calculate validation stats
                            predictions = [self.load_pickle(p) for p in predictions]
                            y_true = np.concatenate([p['y_true'] for p in predictions])
                            y_pred = np.concatenate([p['y_pred'] for p in predictions])
                            losses = np.concatenate([p['losses'] for p in predictions])
                            del predictions
                            valid_metric = self.evaluator.calculate_metrics(y_true, y_pred, losses)
                            self.evaluator.save_checkpoint(self.model, valid_metric, y_true, y_pred)
                            self.print('Validation took {} !'.format(datetime.datetime.now() - validation_start_time))
                    elif self.local_rank == 0:
                        y_true, y_pred, losses = self.evaluator.validate(self.model,
                            self.criterion,
                            str(self.current_epoch).zfill(len(str(self.num_epochs))),
                            save_pickle=False)
                        valid_metric = self.evaluator.calculate_metrics(y_true, y_pred, losses)
                        self.evaluator.save_checkpoint(self.model, valid_metric, y_true, y_pred)
                        self.print('Validation took {} !'.format(datetime.datetime.now() - validation_start_time))

                    if self.scheduler.update == 'on_valid':
                        self.scheduler.step(valid_metric)
                    # End validation
                    self.model.train()
            # Check- end of epoch
            if self.check_end_epoch():
                if self.scheduler.update == 'on_epoch':
                    self.scheduler.step()
                self.current_epoch += 1
                self.steps = 0
                # RESET BEST MODEL IF USING COSINEANNEALINGWARMRESTARTS
                if 'warmrestarts' in str(self.scheduler).lower():
                    if self.current_epoch % self.scheduler.T_0 == 0:
                        self.evaluator.reset_best()
            #
            if self.evaluator.check_stopping(): 
                # Make sure to set number of epochs to max epochs
                # Remember, epochs are 0-indexed and we added 1 already
                # So, this should work (e.g., epoch 99 would now be epoch 100,
                # thus training would stop after epoch 99 if num_epochs = 100)
                self.current_epoch = num_epochs
            if self.check_end_train():
                # Break the while loop
                break
        self.print('TRAINING : END') 
        self.print('Training took {}\n'.format(datetime.datetime.now() - tic))
