from .base import Attack

from .fgsm import FGMAttack
from .fgsm import FGSMAttack
from .fgsm import L2FastGradientAttack
from .fgsm import LinfFastGradientAttack

from .pgd import PGDAttack
from .pgd import L2PGDAttack
from .pgd import LinfPGDAttack

from .utils import CWLoss


ATTACKS = ['fgsm', 'linf-pgd', 'fgm', 'l2-pgd']


def create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform',
                  clip_min=0., clip_max=1.):
    """
    Initialize adversary.
    Arguments:
        model (nn.Module): forward pass function.
        criterion (nn.Module): loss function.
        attack_type (str): name of the attack.
        attack_eps (float): attack radius.
        attack_iter (int): number of attack iterations.
        attack_step (float): step size for the attack.
        rand_init_type (str): random initialization type for PGD (default: uniform).
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
   Returns:
       Attack
   """
#     print(" ### Attacker Eval settings loss@{}, type@{}, eps@{}, iter@{}, step@{} ### ".format(criterion, attack_type, attack_eps, attack_iter, attack_step))
    if attack_type == 'fgsm':
        attack = FGSMAttack(model, criterion, eps=attack_eps, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'fgm':
        attack = FGMAttack(model, criterion, eps=attack_eps, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'linf-pgd':
        attack = LinfPGDAttack(model, criterion, eps=attack_eps, nb_iter=attack_iter, eps_iter=attack_step,
                               rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max)
    elif attack_type == 'l2-pgd':
        attack = L2PGDAttack(model, criterion, eps=attack_eps, nb_iter=attack_iter, eps_iter=attack_step,
                             rand_init_type=rand_init_type, clip_min=clip_min, clip_max=clip_max)
    else:
        raise NotImplementedError('{} is not yet implemented!'.format(attack_type))
    return attack
