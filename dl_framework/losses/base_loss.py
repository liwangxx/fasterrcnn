import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, List, Tuple

class BaseLoss(nn.Module):
    """损失函数基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 损失函数配置
        """
        super().__init__()
        self.config = config or {}
        self._weight = self.config.get('weight', 1.0)
        
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
            
        Returns:
            损失值
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    @property
    def weight(self) -> float:
        """获取损失权重
        
        Returns:
            损失权重
        """
        return self._weight 