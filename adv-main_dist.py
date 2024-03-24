"""
Adversarial Training (with improvements from Gowal et al., 2020).
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd
import mlconfig

import torch
import torch.nn as nn

from adv_core.datas import get_data_info
from adv_core.datas import load_data
from adv_core.datas import SEMISUP_DATASETS, DATASETS

from adv_core.utils import format_time
from adv_core.utils import Logger
from adv_core.utils import parser_train
from adv_core.utils import seed
from adv_core.furnace.watrain_dist import WATrainer

from core import misc

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

# Setup

parse = parser_train()
parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
# distributed training parameters
parse.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parse.add_argument('--local_rank', default=-1, type=int)
parse.add_argument('--dist_on_itp', action='store_true')
parse.add_argument("--start_eval", default=0, type=int, help='when start to eval the test adv acc')
parse.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
args = parse.parse_args()
assert args.data in DATASETS  # SEMISUP_DATASETS, f'Only data in {SEMISUP_DATASETS} is supported!'

os.environ["NCCL_NET_GDR_LEVEL"] = "1"

misc.init_distributed_mode(args)

DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
WEIGHTS_test = os.path.join(LOG_DIR, 'weights-best_test.pt')
WEIGHTS_test_clean = os.path.join(LOG_DIR, 'weights-best_test_clean.pt')
if misc.is_main_process():
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

num_tasks = misc.get_world_size()
global_rank = misc.get_rank()

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size
NUM_ADV_EPOCHS = args.num_adv_epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if misc.is_main_process():
    logger.log('Using device: {}'.format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1

# To speed up training
torch.backends.cudnn.benchmark = True

# Load data
train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader = load_data(
    DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, shuffle_train=True,
    aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=True
)
# Only for no Semi-supervised dataset
if args.data not in SEMISUP_DATASETS:
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

if misc.is_main_process():
    print("   ### Num of Train@{}; Num of Val@{}; Num of Test@{} ###  ".format(
        len(train_dataset), len(val_dataset), len(test_dataset)))
args.len_train, args.len_val, args.len_test = len(train_dataset), len(val_dataset), len(test_dataset)
train_bs_per, test_bs_per = BATCH_SIZE // misc.get_world_size(), BATCH_SIZE_VALIDATION // misc.get_world_size()
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                           batch_size=train_bs_per, sampler=sampler_train,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           num_workers=16)
# Please verify this: Do not need to put the weight-averaged model distributely
if misc.is_main_process():
    print("   ### Using distribute evaluation instead ###  ")
    print("   ### Using distribute evaluation instead ###  ")
sampler_test = torch.utils.data.DistributedSampler(test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
sampler_eval = torch.utils.data.DistributedSampler(val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=test_bs_per, sampler=sampler_test, pin_memory=True, drop_last=False, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=test_bs_per, sampler=sampler_eval, pin_memory=True, drop_last=False, num_workers=8)

del train_dataset, test_dataset, val_dataset

# Adversarial Training
seed(args.seed)
if args.resume:
    args.weight_path = WEIGHTS.replace("weights-best.pt", "weights-last.pt")
config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
trainer = WATrainer(info, args, config=config)
last_lr = args.lr

# fix the seed for reproducibility
args.seed = args.seed + misc.get_rank()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if NUM_ADV_EPOCHS > 0:
    metrics = pd.DataFrame()
    acc = trainer.eval(test_dataloader)

    val_scores = [0.0, 0.0]
    test_scores = [0.0, 0.0]
    best_test_clean = 0.0
    if misc.is_main_process():
        logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(acc * 100))
        logger.log('RST Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))
    test_adv_acc, eval_adv_acc = 0.0, 0.0

if args.resume and args.weight_path is not None:
    start_epoch = trainer.start_epoch
    val_scores = trainer.best_val_acc
    test_scores = trainer.best_test_acc       # Pay attention to this !!!
    best_test_clean = trainer.best_test_clean
else:
    start_epoch = 1

for epoch in range(start_epoch, NUM_ADV_EPOCHS + 1):
    sampler_train.set_epoch(epoch)
    start = time.time()
    if misc.is_main_process():
        logger.log('======= Epoch {} ======='.format(epoch))

    res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
    test_clean_acc = trainer.eval(test_dataloader)
    # We also can save one model for best test clean accuracy

    if test_clean_acc > best_test_clean:
        best_test_clean = test_clean_acc
        if misc.is_main_process():
            trainer.save_training_statu(WEIGHTS_test_clean, epoch, best_val_acc=val_scores,
                                        best_test_acc=test_scores, best_test_clean=best_test_clean)
    # adding warmup for 10 epochs
    last_lr = trainer.last_lr
    if misc.is_main_process():
        logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
        if 'clean_acc' in res:
            logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc'] * 100,
                                                                                    test_clean_acc * 100))
        else:
            logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_clean_acc * 100))

    epoch_metrics = {'train_' + k: v for k, v in res.items()}
    epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_clean_acc, 'test_adversarial_acc': ''})

    if epoch == 250:    # save for resume training
        if misc.is_main_process():
            trainer.save_training_statu(WEIGHTS_test.replace("-best", "-250e"), epoch, best_val_acc=val_scores,
                                        best_test_acc=test_scores, best_test_clean=best_test_clean)
    if (epoch % args.adv_eval_freq == 0 and epoch > args.start_eval) or epoch == NUM_ADV_EPOCHS:
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        if misc.is_main_process():
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc'] * 100,
                                                                                   test_adv_acc * 100))
        epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        if test_adv_acc > test_scores[1]:
            test_scores[0], test_scores[1] = test_clean_acc, test_adv_acc
            if misc.is_main_process():
                # trainer.save_model(WEIGHTS_test)
                trainer.save_training_statu(WEIGHTS_test, epoch, best_val_acc=val_scores,
                                            best_test_acc=test_scores, best_test_clean=best_test_clean)
    else:
        if misc.is_main_process():
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc'] * 100))
    if epoch > 0:    # After we use 1024 samples from test set instead, we can use this as best model selection
        eval_adv_acc = trainer.eval(val_dataloader, adversarial=True)
    else:
        eval_adv_acc = 0
    if misc.is_main_process():
        logger.log('Adversarial Accuracy-\tEval: {:.2f}%.'.format(eval_adv_acc * 100))
    epoch_metrics['eval_adversarial_acc'] = eval_adv_acc

    if eval_adv_acc >= val_scores[1]:
        val_scores[0], val_scores[1] = test_clean_acc, eval_adv_acc
        if misc.is_main_process():
            # trainer.save_model(WEIGHTS)
            trainer.save_training_statu(WEIGHTS, epoch, best_val_acc=val_scores,
                                        best_test_acc=test_scores, best_test_clean=best_test_clean)
    metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    if misc.is_main_process():
        # trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))
        trainer.save_training_statu(os.path.join(LOG_DIR, 'weights-last.pt'), epoch, best_val_acc=val_scores,
                                    best_test_acc=test_scores, best_test_clean=best_test_clean)
        logger.log('Time taken: {}'.format(format_time(time.time() - start)))
        metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)

# Record metrics
train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
if misc.is_main_process():
    logger.log('\nTraining completed.')
    logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc * 100, val_scores[0] * 100))
    if NUM_ADV_EPOCHS > 0:
        logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.'.format(res['adversarial_acc'] * 100,
                                                                                   val_scores[1] * 100))

    logger.log('Script Completed.')
