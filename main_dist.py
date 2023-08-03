import mlconfig
import argparse

import time
import os
import torch
import shutil
import copy
import numpy as np
from torch.autograd import Variable
from torchprofile import profile_macs
import random

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

from contextlib import suppress

from core import dataset
from core.dist_engine import Evaluator, Trainer
from auto_attack.autoattack import AutoAttack
from core import util, misc

parser = argparse.ArgumentParser(description='RobustArc')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--version', type=str, default="arch_001")
parser.add_argument('--fix_seed', action='store_true', default=False)
parser.add_argument("--stop_epoch", type=int, default=None)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument('--exp_name', type=str, default="new_ablation/more_depths/")
parser.add_argument('--config_path', type=str, default='configs')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_best_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--attack_choice', default='PGD', choices=['PGD', 'AA', 'GAMA', 'CW', 'none', 'MI-FGSM', "TI-FGSM"])
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=20, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--train_eval_epoch', default=0.5, type=float, help='PGD Eval in training after this epoch')

# for distribute learning
parser.add_argument('--apex-amp', action='store_true', default=False, help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False, help='Use Native Torch AMP mixed precision')

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

args = parser.parse_args()
if args.epsilon > 1:
    args.epsilon = args.epsilon / 255
    args.step_size = args.step_size / 255

device = torch.device('cuda')
misc.init_distributed_mode(args)
if not args.fix_seed:
    args.seed = random.randint(0, 1000)
# fix the seed for reproducibility
args.seed = args.seed + misc.get_rank()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

exp_path = args.exp_name
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = "{}/checkpoints".format(exp_path)
print(log_file_path, checkpoint_path)
search_results_checkpoint_file_name = None

checkpoint_path_file = os.path.join(checkpoint_path, args.version)
if misc.is_main_process():
    if not os.path.isdir(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

if not args.train:
    logger = util.setup_logger(name=args.version,
                               log_file=log_file_path + '_eval_at-{}-{}steps'.format(args.attack_choice, args.num_steps) + ".log")
else:
    logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
if misc.is_main_process():
    shutil.copyfile(config_file, os.path.join(exp_path, args.version + '.yaml'))

# resolve AMP arguments based on PyTorch / Apex availability
use_amp = None
if args.apex_amp and has_apex:
    use_amp = 'apex'
elif args.native_amp:
    use_amp = 'native'
elif args.apex_amp or args.native_amp:
    print("Neither APEX or native Torch AMP is available, using float32. Install NVIDA apex or upgrade to PyTorch 1.6")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
if misc.is_main_process():
    logger.info(args, args.seed)
    logger.info("GPU List: %s" % (device_list))
if args.stop_epoch == None:
    args.stop_epoch = config.epochs
if config.epochs < 100:  # at least training for 100 epochs
    config.epochs = 100
if hasattr(config, 'ema'):
    args.ema, args.tau, args.static_decay = config.ema, config.tau, config.static_decay
    logger.info("   ### EMA is using for improving the performance ### ")
else:
    args.ema, args.tau, args.static_decay = False, 0, False
if misc.is_main_process():
    logger.info(" ### Start to evaluate from {} Epoch ### ".format(config.epochs * args.train_eval_epoch))
    logger.info(" ### Start to evaluate from {} Epoch ### ".format(config.epochs * args.train_eval_epoch))
    logger.info(" ### Start to evaluate from {} Epoch ### ".format(config.epochs * args.train_eval_epoch))

num_tasks = misc.get_world_size()
global_rank = misc.get_rank()

datasets = config.dataset().getDataLoader()
dataset_train, dataset_test = datasets['train_set'], datasets['test_set']
train_bs, test_bs, num_workers = datasets['train_batch_size'], datasets['test_batch_size'], datasets['num_workers']
train_bs_per, test_bs_per = train_bs // misc.get_world_size(), test_bs // misc.get_world_size()
sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
data_loader = {}
data_loader['train_dataset'] = torch.utils.data.DataLoader(dataset=dataset_train,
                                                           batch_size=train_bs_per, sampler=sampler_train,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           num_workers=num_workers)

data_loader['test_dataset'] = torch.utils.data.DataLoader(dataset=dataset_test,
                                                          batch_size=test_bs_per, sampler=sampler_test,
                                                          pin_memory=True,
                                                          drop_last=False,
                                                          num_workers=num_workers)


def whitebox_eval(data_loader, model, evaluator, log=True, amp_autocast=torch.cuda.amp.autocast, amp_scaler=False):
    natural_count, pgd_count, total, stable_count = 0, 0, 0, 0
    loss_meters = util.AverageMeter()
    lip_meters = util.AverageMeter()
    metric_logger = misc.MetricLogger(delimiter="  ")

    model.eval()
    for i, (images, labels) in enumerate(data_loader["test_dataset"]):
        images, labels = images.to(device), labels.to(device)
        # pgd attack
        images, labels = Variable(images, requires_grad=True), Variable(labels)
        if args.attack_choice == 'PGD':
            rs = evaluator._pgd_whitebox(model, images, labels, random_start=True, epsilon=args.epsilon,
                                         num_steps=args.num_steps, step_size=args.step_size,
                                         amp_autocast=amp_autocast, amp_scaler=amp_scaler)
        elif args.attack_choice == 'CW':
            print("    ###    Using PGD-CW ####    ")
            rs = evaluator._pgd_cw_whitebox(model, images, labels, random_start=True, epsilon=args.epsilon,
                                            num_steps=args.num_steps, step_size=args.step_size,
                                            amp_autocast=amp_autocast, amp_scaler=amp_scaler)
        else:
            raise ('Not implemented')
        acc, acc_pgd, loss, stable, X_pgd = rs
        batch_size = images.size(0)
        local_lip = util.local_lip(model, images, X_pgd).item()
        lip_meters.update(local_lip)
        loss_meters.update(loss)

        metric_logger.meters['clean_acc'].update(acc / batch_size, n=batch_size)
        metric_logger.meters['robust_acc'].update(acc_pgd / batch_size, n=batch_size)
        metric_logger.meters['stable_acc'].update(stable / batch_size, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    natural_count, clean_total = metric_logger.meters['clean_acc'].num_total, metric_logger.meters['clean_acc'].num_count
    robust_count, robust_total = metric_logger.meters['robust_acc'].num_total, metric_logger.meters['robust_acc'].num_count
    stable_count, stabel_total = metric_logger.meters['stable_acc'].num_total, metric_logger.meters['stable_acc'].num_count

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    natural_acc, pgd_acc, stable_acc = results['clean_acc'] * 100, results['robust_acc'] * 100, results['stable_acc'] * 100

    if misc.is_main_process():
        logger.info('Natural Correct Count: %d/%d Acc: %.2f ' % (natural_count, clean_total, natural_acc))
        logger.info('%s Correct Count: %d/%d Acc: %.2f ' % (args.attack_choice, robust_count, robust_total, pgd_acc))
        logger.info('Natural Acc: %.2f ' % (natural_acc))
        logger.info('%s Acc: %.2f ' % (args.attack_choice, pgd_acc))
        logger.info('%s with %.1f steps Loss Avg: %.2f ' % (args.attack_choice, args.num_steps, loss_meters.avg))
        logger.info('LIP Avg: %.4f ' % (lip_meters.avg))
        logger.info('Stable Count: %d/%d StableAcc: %.2f ' % (stable_count, stabel_total, stable_acc))
    return natural_acc, pgd_acc, stable_acc, lip_meters.avg


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = config.optimizer.lr
    schedule = config.lr_schedule if hasattr(config, 'lr_schedule') else 'fixed'
    if schedule == 'fixed':
        if epoch >= 0.75 * config.epochs:
            lr = config.optimizer.lr * 0.1
        if epoch >= 0.9 * config.epochs:
            lr = config.optimizer.lr * 0.01
        if epoch >= config.epochs:
            lr = config.optimizer.lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = config.optimizer.lr * 0.5 * (1 + np.cos((epoch - 1) / config.epochs * np.pi))
    elif schedule == 'search':
        if epoch >= 75:
            lr = 0.01
        if epoch >= 90:
            lr = 0.001
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_eps(epoch, config):
    eps_min = 2 / 255
    eps_max = 8 / 255
    ratio = epoch / (config.epochs * 0.2)
    eps = (eps_min + 0.5 * (eps_max - eps_min) * (1 - np.cos(ratio * np.pi)))
    return eps


def adjust_weight_decay(model, l2_value):
    conv, fc = [], []
    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            # frozen weights
            continue
        if 'fc' in name:
            fc.append(param)
        else:
            conv.append(param)
    params = [{'params': conv, 'weight_decay': l2_value}, {'params': fc, 'weight_decay': 0.01}]
    print(fc)
    return params


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


def train(starting_epoch, model, genotype, optimizer, scheduler, criterion,
          trainer, evaluator, ENV, data_loader, model_ema=None,
          amp_autocast=torch.cuda.amp.autocast, amp_scaler=False):
    print(model)
    for epoch in range(starting_epoch, config.epochs):
        data_loader['train_dataset'].sampler.set_epoch(epoch)
        if misc.is_main_process():
            logger.info("=" * 20 + "Training Epoch %d" % (epoch) + "=" * 20)
        adjust_learning_rate(optimizer, epoch)
        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer, model_ema=model_ema)
        if args.ema:    # update BN
            update_bn(model_ema, model)
        # Eval
        evaluator.eval(epoch, model, amp_autocast=amp_autocast, amp_scaler=amp_scaler)
        if misc.is_main_process():
            logger.info("=" * 20 + "Eval Epoch %d" % (epoch) + "=" * 20)
            logger.info(('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg * 100)))
        ENV['eval_history'].append(evaluator.acc_meters.avg * 100)
        ENV['curren_acc'] = evaluator.acc_meters.avg * 100

        is_best = False
        if epoch >= config.epochs * args.train_eval_epoch and args.train_eval_epoch >= 0:
            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()
            for param in model.parameters():
                param.requires_grad = False
            natural_acc, pgd_acc, stable_acc, lip = whitebox_eval(data_loader, model, evaluator, log=False,
                                                                  amp_autocast=amp_autocast, amp_scaler=amp_scaler)
            for param in model.parameters():
                param.requires_grad = True
            is_best = True if pgd_acc > ENV['best_pgd_acc'] else False
            ENV['best_pgd_acc'] = max(ENV['best_pgd_acc'], pgd_acc)
            ENV['pgd_eval_history'].append((epoch, pgd_acc))
            ENV['stable_acc_history'].append(stable_acc)
            ENV['lip_history'].append(lip)
            if misc.is_main_process():
                logger.info('Best PGD accuracy: %.2f' % (ENV['best_pgd_acc']))
        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model, re-shape the procedure for saving model
        if misc.is_main_process():
            target_model = model.module if args.data_parallel else model
            filename = checkpoint_path_file + '.pth'
            util.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            alpha_optimizer=None,
                            scheduler=None,
                            genotype=genotype,
                            save_best=is_best,
                            filename=filename)
            logger.info('Model Saved at %s\n', filename)
            if args.ema:        # update the Model_EMA at every epoch for resume training
                filename = checkpoint_path_file + '_ema.pth'
                target_model = model_ema.module if hasattr(model_ema, 'module') else model_ema
                util.save_model(ENV=ENV,
                                epoch=epoch,
                                model=target_model,
                                optimizer=None,
                                alpha_optimizer=None,
                                scheduler=None,
                                genotype=genotype,
                                save_best=False,
                                filename=filename)
                logger.info('Latest Model-EMA Saved at %s\n', filename)
            if config.epochs == 400:
                save_epochs = [300, 325, 350, 370]
            else:
                save_epochs = [int(config.epochs * 0.7)]
            if epoch in save_epochs:
                filename = checkpoint_path_file + '_{}.pth'.format(epoch)
                util.save_model(ENV=ENV,
                                epoch=epoch,
                                model=target_model,
                                optimizer=optimizer,
                                alpha_optimizer=None,
                                scheduler=None,
                                genotype=genotype,
                                save_best=False,
                                filename=filename)
                logger.info('Model Saved at %s\n', filename)
                if args.ema:        # update the Model_EMA at every epoch for resume training
                    filename = checkpoint_path_file + '_{}_ema.pth'.format(epoch)
                    target_model = model_ema.module if hasattr(model_ema, 'module') else model_ema
                    util.save_model(ENV=ENV,
                                    epoch=epoch,
                                    model=target_model,
                                    optimizer=None,
                                    alpha_optimizer=None,
                                    scheduler=None,
                                    genotype=genotype,
                                    save_best=False,
                                    filename=filename)
                    logger.info('Latest Model-EMA Saved at %s\n', filename)

        if (epoch + 1) == args.stop_epoch:  # setting this
            break   # finish the training
    return


def main():
    # Load Search Version Genotype
    model = config.model().to(device)
    if args.local_rank == 0:
        logger.info(model)
    genotype = None

    # Setup ENV
    if hasattr(config, 'adjust_weight_decay') and config.adjust_weight_decay:
        params = adjust_weight_decay(model, config.optimizer.weight_decay)
    else:
        params = model.parameters()

    optimizer = config.optimizer(params)
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        amp_autocast = suppress
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
    else:
        amp_autocast = None
    amp_scaler = (args.apex_amp and has_apex)
    criterion = config.criterion()

    trainer = Trainer(criterion, data_loader, logger, config, amp_scaler=amp_scaler, amp_autocast=amp_autocast, args=args)
    evaluator = Evaluator(data_loader, logger, config)
    if hasattr(config.dataset, "input_size"):
        print("   ## FLOPs with input shape {} ##  ".format([1, 3, config.dataset.input_size, config.dataset.input_size]))
        profile_inputs = (torch.randn([1, 3, config.dataset.input_size, config.dataset.input_size]).to(device),)
    elif config.dataset.dataset_type == "TINYIMAGENET":
        print("   ## FLOPs with input shape {} ##  ".format([1, 3, 64, 64]))
        profile_inputs = (torch.randn([1, 3, 64, 64]).to(device),)
    else:
        profile_inputs = (torch.randn([1, 3, 32, 32]).to(device),)
    flops = profile_macs(model, profile_inputs) / 1e6
    starting_epoch = 0

    config.set_immutable()
    if args.local_rank == 0:
        for key in config:
            logger.info("%s: %s" % (key, config[key]))
        logger.info("param size = %fMB", util.count_parameters_in_MB(model))
        logger.info("flops: %.4fM" % flops)
        logger.info("PyTorch Version: %s" % (torch.__version__))

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'flops': flops,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'stable_acc_history': [],
           'lip_history': [],
           'genotype_list': []}

    if args.load_model or args.load_best_model:
        filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
        checkpoint = util.load_model(filename=filename,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=None)
        starting_epoch = checkpoint['epoch'] + 1
        ENV = checkpoint['ENV']
        if 'stable_acc_history' not in ENV:
            ENV['stable_acc_history'] = []
        if 'lip_history' not in ENV:
            ENV['lip_history'] = []
        trainer.global_step = ENV['global_step']
        if args.local_rank == 0:
            logger.info("File %s loaded!" % (filename))
    # adding EMA before DDP
    if args.ema:
        model_ema = copy.deepcopy(model)
    else:
        model_ema = None
    if use_amp == 'apex' and has_apex:
        # Apex DDP preferred unless native amp is activated
        if args.local_rank == 0:
            logger.info("Using NVIDIA APEX DistributedDataParallel.")
        model = ApexDDP(model, delay_allreduce=True)
    else:
        if args.local_rank == 0:
            logger.info("Using native Torch DistributedDataParallel.")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          find_unused_parameters=True, broadcast_buffers=False)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    if args.local_rank == 0:
        logger.info("Starting Epoch: %d" % (starting_epoch))

    if args.train:
        train(starting_epoch, model, genotype, optimizer, None, criterion, trainer, evaluator,
              ENV, data_loader, model_ema=model_ema)
    elif args.attack_choice in ['PGD', 'GAMA', 'CW', "MI-FGSM", "TI-FGSM"]:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        natural_acc, adv_acc, stable_acc, lip = whitebox_eval(data_loader, model, evaluator)

    elif args.attack_choice == 'AA':
        for param in model.parameters():
            param.requires_grad = False
        x_test = [x for (x, y) in data_loader['test_dataset']]
        x_test = torch.cat(x_test, dim=0)
        y_test = [y for (x, y) in data_loader['test_dataset']]
        y_test = torch.cat(y_test, dim=0)
        model.eval()

        adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, logger=logger, verbose=True)
        adversary.plus = False

        logger.info('=' * 20 + 'AA Attack Eval' + '=' * 20)
        x_adv, robust_accuracy = adversary.run_standard_evaluation(x_test, y_test, bs=config.dataset.eval_batch_size)
        robust_accuracy = robust_accuracy * 100
        logger.info('AA Accuracy: %.2f' % (robust_accuracy))

    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)
