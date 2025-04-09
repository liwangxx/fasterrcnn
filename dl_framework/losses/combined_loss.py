import torch
import torch.nn as nn
from typing import Dict, Any, Union, List

from .base_loss import BaseLoss
from .registry import LossRegistry

@LossRegistry.register('combined')
class CombinedLoss(BaseLoss):
    """组合多个损失函数"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 损失函数配置，必须包含losses列表
        """
        super().__init__(config)
        
        # 提取子损失函数配置
        loss_configs = self.config.get('losses', [])
        if not loss_configs:
            raise ValueError("组合损失函数必须指定子损失函数列表")
        
        # 创建子损失函数
        self.losses = []
        for loss_config in loss_configs:
            loss = LossRegistry.create(loss_config)
            self.losses.append(loss)
    
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算组合损失
        
        Args:
            outputs: 模型输出
            targets: 目标值
            
        Returns:
            组合损失值
        """
        total_loss = 0.0
        for loss in self.losses:
            loss_value = loss(outputs, targets)
            total_loss = total_loss + loss_value
        
        return total_loss * self.weight 