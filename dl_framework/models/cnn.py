import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Union

from .base_model import BaseModel
from .registry import ModelRegistry

@ModelRegistry.register('cnn')
class CNN(BaseModel):
    """简单的CNN模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        
        # 从配置中获取参数
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 10)
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层的输入特征数量
        # 假设输入图像大小为 32x32
        fc_input_features = 128 * (32 // 8) * (32 // 8)  # 三次下采样后
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_features, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        
        # dropout
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，形状为 [N, C, H, W]
            
        Returns:
            输出张量，形状为 [N, num_classes]
        """
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 第三个卷积块
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失
        
        Args:
            outputs: 模型输出，形状为 [N, num_classes]
            targets: 目标张量，形状为 [N]
            
        Returns:
            损失值
        """
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, targets) 