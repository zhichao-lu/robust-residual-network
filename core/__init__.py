import mlconfig

from .losses.trades import TradesLoss
from .losses.madrys import MadrysLoss
from .losses.mart import MartLoss
from .dataset import DatasetGenerator
from .dist_engine import Evaluator, Trainer

# Setup mlconfig
mlconfig.register(TradesLoss)
mlconfig.register(MadrysLoss)
mlconfig.register(MartLoss)
mlconfig.register(DatasetGenerator)

