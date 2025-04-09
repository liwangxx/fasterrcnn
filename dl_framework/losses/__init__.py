from .registry import LossRegistry
from .base_loss import BaseLoss

# 导入内置损失函数以便自动注册
from .cross_entropy import CrossEntropyLoss
from .mse_loss import MSELoss
from .l1_loss import L1Loss
from .combined_loss import CombinedLoss

__all__ = [
    'LossRegistry',
    'BaseLoss',
    'CrossEntropyLoss',
    'MSELoss',
    'L1Loss',
    'CombinedLoss',
] 