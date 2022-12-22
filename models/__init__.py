import mlconfig
import torch
from models.resnet import PreActResNet
from models.robnet import RobNet
from models.advrush import AdvRush

# Setup mlconfig
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.Adamax)

mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

mlconfig.register(torch.nn.CrossEntropyLoss)

mlconfig.register(PreActResNet)
mlconfig.register(RobNet)
mlconfig.register(AdvRush)

