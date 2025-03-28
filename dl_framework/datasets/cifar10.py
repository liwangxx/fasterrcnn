import os
import torch
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Any, Optional, Tuple

from .base_dataset import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register('cifar10')
class CIFAR10Dataset(BaseDataset):
    """CIFAR-10数据集"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__(config, is_training)
        self.transform = self._get_transforms()
        self._load_data()
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换
        
        Returns:
            数据变换组合
        """
        transform_config = self.config.get('transforms', {})
        
        transform_list = []
        
        # 调整大小
        if 'resize' in transform_config:
            size = transform_config['resize']
            transform_list.append(transforms.Resize(size))
        
        # 如果是训练集，添加数据增强
        if self.is_training:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
            ])
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 标准化
        if 'normalize' in transform_config:
            mean = transform_config['normalize'].get('mean', [0.5, 0.5, 0.5])
            std = transform_config['normalize'].get('std', [0.5, 0.5, 0.5])
            transform_list.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(transform_list)
    
    def _load_data(self) -> None:
        """加载数据"""
        # 下载CIFAR-10数据集（如果需要）
        self.dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=self.is_training,
            download=True,
            transform=self.transform
        )
    
    def __len__(self) -> int:
        """获取数据集长度
        
        Returns:
            数据集长度
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            图像张量和标签
        """
        image, label = self.dataset[idx]
        return image, label 