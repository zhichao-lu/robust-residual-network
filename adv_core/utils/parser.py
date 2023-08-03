import argparse

from adv_core.attacks import ATTACKS
from adv_core.data import DATASETS
from adv_core.furnace.train import SCHEDULERS

from .utils import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')

    parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=512, help='Batch size for testing.')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--config-path', type=str, default='configs')
    parser.add_argument('--version', type=str, default="arch_001")
    parser.add_argument('--start-eval', type=int, default=310, help='evaluation until the specific epoch')

    parser.add_argument('--deepmind-impl', action="store_true", default=False, help="Whether follow deepmind implementation "
                                                                                    "change step size of inner optimization or not ")
    # Resume training
    parser.add_argument('--resume', default=False, action='store_true')
    # Only update BN with final adversarial samples when True
    parser.add_argument('--advnorm', default=False, action='store_true')
    # Put BN into group_no_decay if it is True
    parser.add_argument('--adjust-bn', type=str2bool, default=True)

    # For Cutmix data augmentation
    parser.add_argument('--cutmix-beta', type=float, default=0)
    parser.add_argument('--cut-window', type=int, default=0)

    parser.add_argument('--data-parallel', action='store_true', default=False)
    parser.add_argument('--apex-amp', action='store_true', default=False, help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False, help='Use Native Torch AMP mixed precision')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')

    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, required=True, 
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('-m', '--model', default='wrn-28-10-swish', help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.') # if norm, the input is in [-1, 1] range
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-na', '--num-adv-epochs', type=int, default=400, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=10, help='Adversarial evaluation frequency (in epochs).')
    
    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')
    
    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')

    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Debug code. Run 1 epoch of training and evaluation.')
    parser.add_argument('--mart', action='store_true', default=False, help='MART training.')
    
    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional Tiny Images data.', 
                        default='/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser


def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')
        
    parser.add_argument('--desc', type=str, required=True, help='Description of model to be evaluated.')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')
    
    # eval-aa.py
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'], 
                        help='Version of AA.')

    # eval-adv.py
    parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
    parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')
    
    # eval-rb.py
    parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
                        help='Threat model for RobustBench evaluation.')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    return parser

