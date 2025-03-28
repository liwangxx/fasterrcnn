import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Union, Optional

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
        
        # 用于存储中间特征的字典
        self.features = {}
    
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
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """执行前向传播并保存中间特征
        
        Args:
            x: 输入张量，形状为 [N, C, H, W]
            
        Returns:
            元组，包含输出张量和特征字典
        """
        features = {}
        
        # 第一个卷积块
        conv1_output = self.conv1(x)
        features['conv1'] = conv1_output
        
        relu1_output = F.relu(conv1_output)
        features['relu1'] = relu1_output
        
        pool1_output = self.pool(relu1_output)
        features['pool1'] = pool1_output
        
        # 第二个卷积块
        conv2_output = self.conv2(pool1_output)
        features['conv2'] = conv2_output
        
        relu2_output = F.relu(conv2_output)
        features['relu2'] = relu2_output
        
        pool2_output = self.pool(relu2_output)
        features['pool2'] = pool2_output
        
        # 第三个卷积块
        conv3_output = self.conv3(pool2_output)
        features['conv3'] = conv3_output
        
        relu3_output = F.relu(conv3_output)
        features['relu3'] = relu3_output
        
        pool3_output = self.pool(relu3_output)
        features['pool3'] = pool3_output
        
        # 展平
        flattened = pool3_output.view(pool3_output.size(0), -1)
        features['flattened'] = flattened
        
        # 全连接层
        fc1_output = self.fc1(flattened)
        features['fc1'] = fc1_output
        
        relu_fc1_output = F.relu(fc1_output)
        features['relu_fc1'] = relu_fc1_output
        
        dropout_output = self.dropout(relu_fc1_output)
        features['dropout'] = dropout_output
        
        fc2_output = self.fc2(dropout_output)
        features['fc2'] = fc2_output
        
        # 保存特征字典
        self.features = features
        
        return fc2_output, features
    
    def visualize_features(self, visualizer, input_tensor: torch.Tensor, global_step: int) -> None:
        """使用指定的可视化器可视化模型的特征图
        
        Args:
            visualizer: 可视化器实例，需要实现add_feature_maps方法
            input_tensor: 输入张量，形状为 [N, C, H, W]
            global_step: 全局步数
        """
        # 确保输入是批次形式
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # 执行前向传播并获取特征
        with torch.no_grad():
            _, features = self.forward_with_features(input_tensor)
            
        # 针对每个卷积层特征图进行可视化
        for layer_name, feature_map in features.items():
            # 只可视化卷积层输出
            if layer_name.startswith('conv') or layer_name.startswith('pool'):
                if len(feature_map.shape) == 4:  # 确保是特征图 [N, C, H, W]
                    # 将特征图移动到CPU进行可视化
                    feature_map = feature_map.cpu()
                    
                    # 使用可视化器的方法
                    for i in range(min(feature_map.size(1), 16)):  # 最多显示16个通道
                        channel_map = feature_map[0, i].unsqueeze(0)  # [1, H, W]
                        tag = f"features/{layer_name}/channel_{i}"
                        visualizer.add_image(tag, channel_map, global_step)
                        
                    # 添加特征图统计信息
                    visualizer.add_histogram(f"features/{layer_name}/histogram", feature_map, global_step)
    
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