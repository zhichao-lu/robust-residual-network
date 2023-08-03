import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_core.attacks import create_attack
from adv_core.attacks import CWLoss
from adv_core.utils.metrics import accuracy

from adv_core.utils import ctx_noparamgrad_and_eval
from adv_core.utils import set_bn_momentum  # Good! After we have this, we do not need to set by ourself in the model file
from adv_core.utils import seed
from adv_core.dataaugs.cutmix import cutmix
from adv_core.losses.trades import trades_loss
from adv_core.utils import misc

from .train import Trainer

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

from contextlib import suppress

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainer(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """

    def __init__(self, info, args, config):
        super(WATrainer, self).__init__(info, args, config=config)
        self.with_cutmix = False
        if self.params.apex_amp and has_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.amp_autocast = suppress
        elif self.params.native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
        else:
            self.amp_autocast = torch.cuda.amp.autocast
        self.wa_model = copy.deepcopy(self.model)
        if args.resume and args.weight_path is not None:
            # Load Torch State Dict
            checkpoints = torch.load(args.weight_path, map_location=device)
            self.model.load_state_dict(checkpoints['unaveraged_model_state_dict'], strict=False)
            self.wa_model.load_state_dict(checkpoints['model_state_dict'], strict=False)
            if self.optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
                self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            if self.scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
            self.start_epoch = checkpoints['epoch'] + 1
            self.best_val_acc = checkpoints['best_val_acc']
            self.best_test_acc = checkpoints['best_test_acc']
            self.best_test_clean = checkpoints['best_test_clean']
            if misc.is_main_process():
                print("   ### Loading checkpoint from {} and resume training from {} ###  ".format(args.weight_path,
                                                                                                   self.start_epoch))
                print("   ### With Best-test-acc@{} and best-val-acc@{}% ### ".format(self.best_test_acc,
                                                                                      self.best_val_acc))
        if self.params.cutmix_beta > 0 or self.params.cut_window > 0:
            self.with_cutmix = True
        if misc.is_main_process():
            print(self.model)
            param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
            print("   ### Number of parameters for full capacity network is : {:.2f} ###".format(param_count))
            if self.with_cutmix:
                print("   ### Using CutMix during Training ###    ")
                if self.params.cut_window > 0:
                    print("   ### Fixed the cutout window size to {} ###   ".format(self.params.cut_window))
                else:
                    print("   ### Randomly sample the cutout window size ###   ")
            if self.params.deepmind_impl or self.params.attack_step >= 0.1:
                print("    ### Update the step size following Deepmind ###  ")
        if self.params.apex_amp and has_apex:
            # Apex DDP preferred unless native amp is activated
            if misc.is_main_process():
                print("Using NVIDIA APEX DistributedDataParallel.")
            self.model = ApexDDP(self.model, delay_allreduce=True)
        else:
            if misc.is_main_process():
                print("Using native Torch DistributedDataParallel.")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.params.local_rank],
                                                                   find_unused_parameters=True)
#         self.wa_model = copy.deepcopy(self.model)

        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4 * args.attack_iter,
                                         2 / 255)  # distribute evaluation
        num_samples = self.params.len_train # 50000 if 'cifar' in self.params.data else 73257
        # num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        self.update_steps = int(np.floor(num_samples / self.params.batch_size) + 1)
        self.warmup_steps = self.params.warmup_epochs * self.update_steps  # warmup 10 epoch
        self.lr_decay_steps = int((self.update_steps * self.params.num_adv_epochs) / 3) * 2
        self.last_lr = 0

        # self.params.lr = self.params.lr * self.params.batch_size / 256  # linear scaling the lr
        if misc.is_main_process():
            if self.params.advnorm:
                print("   #### Only update BN with final adversarial samples and without clean samples ### ")
            if self.params.adjust_bn:
                print("   #### Putting BN parameters into group without weight decay ### ")
            print(self.params)

    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """

        def group_weight(model):
            group_decay = []
            group_no_decay = []
            i = 0
            for n, p in model.named_parameters():
                # To adapt other norm implementation, pay attention to this
                if ('batchnorm' in n or "bn" in n) and self.params.adjust_bn:
                    if misc.is_main_process() and i < 4:
                        print("   ### Without using L2 regularization {} ### ".format(n))
                        i += 1
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups

        if self.params.batch_size > 256 and self.params.lr <= 0.1:
            self.params.lr = self.params.lr * self.params.batch_size / 256  # linear scaling the lr
        print("   #### Initial learning rate is {} ### ".format(self.params.lr))
        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr,
                                         weight_decay=self.params.weight_decay,
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        update_iter = 0
        if self.params.scheduler in ['converge', 'cosine', 'cosinew', 'cyclic']:
            self.last_lr = self.scheduler.get_last_lr()[0]
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1

            x, y = data
            # cutmix attention
            if self.with_cutmix:
                x, y = cutmix(x, y, beta=self.params.cutmix_beta, cut_window=self.params.cut_window)
            x, y = x.to(device), y.to(device)
            if (self.params.apex_amp and has_apex) or self.params.native_amp:  # if using apex to accelerate the speed
                with self.amp_autocast():
                    if adversarial:
                        if self.params.beta is not None and self.params.mart:
                            loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                        elif self.params.beta is not None:
                            loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta, )
                        else:
                            loss, batch_metrics = self.adversarial_loss(x, y)
                    else:
                        loss, batch_metrics = self.standard_loss(x, y)
            else:
                if adversarial:
                    if self.params.beta is not None and self.params.mart:
                        loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None:
                        loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)
            if (self.params.apex_amp and has_apex) or self.params.native_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            global_step = (epoch - 1) * self.update_steps + update_iter
            if self.params.scheduler == 'step':
                # change the lr each step instead of each epoch
                if global_step <= self.warmup_steps:  # linearly increasing lr from 0 - max_lr
                    lr = (self.params.lr / self.warmup_steps) * global_step
                    self.update_lr(lr)
                    if global_step == 100 or global_step == 2000:
                        if misc.is_main_process():
                            print("   ## Warmup lr at step@{} is {} ##  ".format(global_step, lr))
                elif global_step > self.lr_decay_steps:
                    lr = self.params.lr * 0.1
                    self.update_lr(lr)
                    if global_step == self.lr_decay_steps + 10:
                        if misc.is_main_process():
                            print("   ## lr at step@{} is {} ##  ".format(global_step, lr))
                else:
                    lr = self.params.lr
                    self.update_lr(lr)
                    if global_step == self.lr_decay_steps - 200:
                        if misc.is_main_process():
                            print("   ## lr at step@{} is {} ##  ".format(global_step, lr))
                self.last_lr = lr
            else:
                self.last_lr = self.scheduler.get_last_lr()[0]

            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            torch.cuda.synchronize()
            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
        if self.params.scheduler in ['converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        update_bn(self.wa_model, self.model)
        return dict(metrics.mean())

    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step,
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter,
                                          beta=beta, attack=self.params.attack, with_cutmix=self.with_cutmix,
                                          advnorm=self.params.advnorm)
        return loss, batch_metrics

    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the EMA model
        """
        acc = 0.0
        self.wa_model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        total, correct = 0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                out = self.wa_model(x_adv)
            else:
                out = self.wa_model(x)
            acc = accuracy(y, out)
            metric_logger.meters['acc'].update(acc, n=batch_size)
            total += batch_size
        # distribute evaluation
        metric_logger.synchronize_between_processes()
        results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if misc.is_main_process():
            print(metric_logger.meters['acc'].num_total, metric_logger.meters['acc'].num_count, results['acc'] * 100)
        return results['acc']

    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(),
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)

    def save_training_statu(self, path, epoch=0, best_val_acc=[], best_test_acc=[], best_test_clean=[]):
        # Torch Save State Dict
        state = {
            'epoch': epoch,
            'model_state_dict': self.wa_model.state_dict(),
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_test_clean": best_test_clean,
        }
        torch.save(state, path)

    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    ema_has_module = hasattr(wa_model, 'module')
    needs_module = hasattr(model, 'module') and not ema_has_module
    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in wa_model.state_dict().items():
            if needs_module:
                k = 'module.' + k
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.module.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
