import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from adv_core.utils.metrics import accuracy
from adv_core.utils import SmoothCrossEntropyLoss
from adv_core.utils import track_bn_stats


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0,
                attack='linf-pgd', label_smoothing=0.1, with_cutmix=False, advnorm=False):
    """
    TRADES training (Zhang et al, 2019).
    """
    # TO-DO List: 1. change the step size: they have different step size before step < 5 and step > 5
    # if we use extra data, we should add the label_smoothing, pay attention to this!!!
    # 1. Update batch normalization statistics from adversarial examples only
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    if with_cutmix:
        criterion_ce = SoftTargetCrossEntropy()
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    # please make sure, the range is in [0, 1] instead of [-1, 1]
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    if not with_cutmix:
        p_natural = F.softmax(model(x_natural), dim=1)

    for i in range(perturb_steps):
        if i == 5 and step_size >= 0.1:  # Follow DeepMind implementation
            step_size *= 0.1
        #             print("   ### Step_size@{:.4f} ###  ".format(step_size))
        x_adv.requires_grad_()
        with torch.enable_grad():
            if with_cutmix:
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), y)
            else:
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    if not advnorm:
        print("   ## Update BN with Adv and Clean samples ## ")
        track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # Calculate robust loss, pay attention to the following question !!!! Urgent
    logits_natural = model(x_natural)
    if advnorm:  # The put the update of BN after clean inputs instead, i.e., don't use clean input to update BN
        track_bn_stats(model, True)

    logits_adv = model(x_adv)
    loss_natural = criterion_ce(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    if with_cutmix:
        batch_metrics = {'loss': loss.item(), 'clean_acc': 0, 'adversarial_acc': 0}
    else:
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()),
                         'adversarial_acc': accuracy(y, logits_adv.detach())}

    return loss, batch_metrics
