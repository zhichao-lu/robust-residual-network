import time
from . import util, misc

import torch
import torch.optim as optim
from torch.autograd import Variable

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer:
    def __init__(self, criterion, data_loader, logger, config, amp_scaler=False, amp_autocast=torch.cuda.amp.autocast,
                 global_step=0, args=None):
        self.criterion = criterion
        self.args = args
        self.data_loader = data_loader
        self.amp_scaler = amp_scaler
        self.amp_autocast = amp_autocast
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.warmup_steps = args.warmup_steps
        self.static_decay = args.static_decay
        self.decay_tau = args.tau

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer, teacher_model=None, model_ema=None):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader["train_dataset"]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer, teacher_model=teacher_model)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                if misc.is_main_process():
                    self.logger.info(display)
            self.global_step += 1
            if self.args.ema:
                """
                Exponential model weight averaging update.
                """
                factor = int(self.global_step >= self.warmup_steps)
                if not self.static_decay:
                    delta = self.global_step - self.warmup_steps
                    decay = min(self.decay_tau, (1. + delta) / (10. + delta)) if 10. + delta != 0 else self.decay_tau
                else:
                    decay = self.decay_tau
                decay *= factor
                ema_has_module = hasattr(model_ema, 'module')
                needs_module = hasattr(model, 'module') and not ema_has_module
                with torch.no_grad():
                    msd = model.state_dict()
                    for k, ema_v in model_ema.state_dict().items():
                        if needs_module:
                            k = 'module.' + k
                        model_v = msd[k].detach()
                        ema_v.copy_(ema_v * decay + (1. - decay) * model_v)

        return self.global_step

    def train_batch(self, images, labels, model, optimizer, teacher_model=None):
        model.zero_grad()
        optimizer.zero_grad()
        if self.amp_autocast is not None:
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                with self.amp_autocast():
                    logits = model(images)
                    loss = self.criterion(logits, labels)
            else:
                with self.amp_autocast():
                    logits, loss, _ = self.criterion(model, images, labels, optimizer)
        else:
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                logits = model(images)
                loss = self.criterion(logits, labels)
            else:
                logits, loss, _ = self.criterion(model, images, labels, optimizer)

        if self.amp_scaler:   # amp scale loss, only for apex
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if self.config.grad_clip != -1:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        else:
            if isinstance(model.parameters(), torch.Tensor):
                parameters = [model.parameters()]
            else:
                parameters = model.parameters()
            parameters = [p for p in parameters if p.grad is not None]
            grad_norm = 0.0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** (1. / 2)
        optimizer.step()
        if len(labels.shape) > 1:
            labels = labels.argmax(dim=1)
        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc.item(), labels.shape[0])
        self.acc5_meters.update(acc5.item(), labels.shape[0])

        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload


class Evaluator:
    def __init__(self, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        return

    def eval(self, epoch, model, amp_autocast=torch.cuda.amp.autocast, amp_scaler=False):
        model.eval()
        metric_logger = misc.MetricLogger(delimiter="  ")
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            batch_size = images.size(0)
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model,
                                          amp_autocast=amp_autocast, amp_scaler=amp_scaler)
            end = time.time()
            time_used = end - start
            metric_logger.meters['clean_acc'].update(log_payload['val_acc'], n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_payload['val_acc_avg'] = results['clean_acc'] * 100
        display = util.log_display(epoch=epoch, global_step=i, time_elapse=time_used, **log_payload)
        self.acc_meters.reset()
        self.acc_meters.update(results['clean_acc'], n=1)
        if self.logger is not None and misc.is_main_process():
            print(metric_logger.meters['clean_acc'].num_total, metric_logger.meters['clean_acc'].num_count, results['clean_acc'] * 100)
            self.logger.info(display)
        return

    def eval_batch(self, images, labels, model, amp_autocast=torch.cuda.amp.autocast, amp_scaler=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if amp_autocast is not None:
            with amp_autocast():
                pred = model(images)
                loss = self.criterion(pred, labels)
        else:
            pred = model(images)
            loss = self.criterion(pred, labels)
        acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        payload = {"val_acc": acc.item(),
                   "val_acc_avg": self.acc_meters.avg,
                   "val_acc5": acc5.item(),
                   "val_acc5_avg": self.acc5_meters.avg,
                   "val_loss": loss.item(),
                   "val_loss_avg": self.loss_meters.avg}
        return payload

    def _pgd_whitebox(self, model, X, y, random_start=True, epsilon=0.031, num_steps=20, step_size=0.003,
                      amp_autocast=torch.cuda.amp.autocast, amp_scaler=False):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            if amp_autocast is not None:
                with amp_autocast():
                    output = model(X_pgd)
                    loss = torch.nn.CrossEntropyLoss()(output, y)
            else:
                output = model(X_pgd)
                loss = torch.nn.CrossEntropyLoss()(output, y)
            if amp_scaler:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()
        return acc.item(), acc_pgd.item(), loss.item(), stable.item(), X_pgd

    def _pgd_cw_whitebox(self, model, X, y, random_start=True, epsilon=0.031, num_steps=20, step_size=0.003,
                         amp_autocast=torch.cuda.amp.autocast, amp_scaler=False):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)

        def CWLoss(output, target, confidence=0):
            """
            CW loss (Marging loss).
            """
            num_classes = output.shape[-1]
            target = target.data
            target_onehot = torch.zeros(target.size() + (num_classes,))
            target_onehot = target_onehot.cuda()
            target_onehot.scatter_(1, target.unsqueeze(1), 1.)
            target_var = Variable(target_onehot, requires_grad=False)
            real = (target_var * output).sum(1)
            other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
            loss = - torch.clamp(real - other + confidence, min=0.)
            loss = torch.sum(loss)
            return loss

        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            if amp_autocast is not None:
                with amp_autocast():
                    output = model(X_pgd)
                    loss = CWLoss(output, y)
            else:
                output = model(X_pgd)
                loss = CWLoss(output, y)
            if amp_scaler:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        X_pgd = Variable(X_pgd.data, requires_grad=False)
        predict_pgd = model(X_pgd).data.max(1)[1].detach()
        predict_clean = model(X).data.max(1)[1].detach()
        acc_pgd = (predict_pgd == y.data).float().sum()
        stable = (predict_pgd.data == predict_clean.data).float().sum()
        return acc.item(), acc_pgd.item(), loss.item(), stable.item(), X_pgd
