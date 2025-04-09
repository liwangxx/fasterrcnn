from typing import Dict, Type, Any, Optional, Union
import torch.nn as nn

class LossRegistry:
    """损失函数注册器，用于注册和获取损失函数类"""
    _losses = {}
    
    @classmethod
    def register(cls, name):
        """注册损失函数类
        
        Args:
            name: 损失函数名称
            
        Returns:
            装饰器函数
        """
        def wrapper(loss_class):
            cls._losses[name] = loss_class
            return loss_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取损失函数类
        
        Args:
            name: 损失函数名称
            
        Returns:
            损失函数类
        
        Raises:
            ValueError: 如果损失函数未注册
        """
        if name not in cls._losses:
            raise ValueError(f"未注册的损失函数: {name}")
        return cls._losses[name]
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> nn.Module:
        """根据配置创建损失函数实例
        
        Args:
            config: 损失函数配置，必须包含'type'字段
            
        Returns:
            损失函数实例
            
        Raises:
            ValueError: 如果配置中没有type字段或者类型未注册
        """
        if not isinstance(config, dict):
            raise ValueError(f"损失函数配置必须是字典类型，得到了: {type(config)}")
            
        if 'type' not in config:
            raise ValueError("损失函数配置必须包含'type'字段")
            
        loss_type = config.pop('type')
        loss_class = cls.get(loss_type)
        return loss_class(config)
    
    @classmethod
    def list_losses(cls):
        """列出所有已注册的损失函数
        
        Returns:
            已注册损失函数名称列表
        """
        return list(cls._losses.keys()) 