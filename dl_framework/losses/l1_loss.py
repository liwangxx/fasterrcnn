import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Optional

from .base_loss import BaseLoss
from .registry import LossRegistry

@LossRegistry.register('l1')
class L1Loss(BaseLoss):
    """L1损失函数（绝对误差损失）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 损失函数配置
        """
        super().__init__(config)
        
        # 提取配置参数
        self.reduction = self.config.get('reduction', 'mean')
        
        # 创建L1损失函数
        self.loss_fn = nn.L1Loss(reduction=self.reduction)
    
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算L1损失
        
        Args:
            outputs: 模型输出，形状为 [N, *] 或包含预测值的字典
            targets: 目标值，形状为 [N, *] 或包含目标值的字典
            
        Returns:
            损失值
        """
        # 处理字典输入
        if isinstance(outputs, dict):
            outputs = outputs.get('predictions', next(iter(outputs.values())))
        
        if isinstance(targets, dict):
            targets = targets.get('targets', next(iter(targets.values())))
        
        return self.loss_fn(outputs, targets) * self.weight 