import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Optional, List

from .base_loss import BaseLoss
from .registry import LossRegistry

@LossRegistry.register('cross_entropy')
class CrossEntropyLoss(BaseLoss):
    """交叉熵损失函数"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 损失函数配置
        """
        super().__init__(config)
        
        # 提取配置参数
        self.reduction = self.config.get('reduction', 'mean')
        self.ignore_index = self.config.get('ignore_index', -100)
        self.label_smoothing = self.config.get('label_smoothing', 0.0)
        
        # 创建交叉熵损失函数
        self.loss_fn = nn.CrossEntropyLoss(
            reduction=self.reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
    
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算交叉熵损失
        
        Args:
            outputs: 模型输出，形状为 [N, C] 或包含 'logits' 键的字典
            targets: 目标标签，形状为 [N] 或包含 'labels' 键的字典
            
        Returns:
            损失值
        """
        # 处理字典输入
        if isinstance(outputs, dict):
            outputs = outputs.get('logits', next(iter(outputs.values())))
        
        if isinstance(targets, dict):
            targets = targets.get('labels', next(iter(targets.values())))
        
        return self.loss_fn(outputs, targets) * self.weight 