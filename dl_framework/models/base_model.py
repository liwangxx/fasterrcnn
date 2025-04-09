import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Union, List, Optional

from ..losses.registry import LossRegistry

class BaseModel(nn.Module):
    """所有模型的基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config or {}
        
        # 如果配置中包含损失函数配置，则创建损失函数
        self.loss = None
        if 'loss' in self.config:
            self.loss = LossRegistry.create(self.config['loss'])
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            模型输出
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def get_loss(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算损失
        
        如果模型配置中指定了损失函数，则使用该损失函数；
        否则，子类应该实现自己的损失计算逻辑。
        
        Args:
            outputs: 模型输出
            targets: 目标值
            
        Returns:
            损失值
        """
        if self.loss is not None:
            return self.loss(outputs, targets)
        
        # 对于没有指定损失函数的旧模型，子类需要实现此方法
        raise NotImplementedError("模型未指定损失函数，子类必须实现get_loss方法")
    
    def predict(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """预测（用于推理）
        
        Args:
            x: 输入张量
            
        Returns:
            预测结果
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def load_weights(self, checkpoint_path: str) -> None:
        """加载模型权重
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            self.load_state_dict(checkpoint['model'])
        else:
            self.load_state_dict(checkpoint)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数
        
        Returns:
            可训练参数列表
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量
        
        Returns:
            包含总参数和可训练参数数量的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params
        } 