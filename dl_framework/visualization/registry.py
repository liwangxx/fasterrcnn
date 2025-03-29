from typing import Dict, Type, Any
from .base_visualizer import BaseVisualizer

class VisualizerRegistry:
    """可视化器注册表，用于动态注册和获取可视化器类"""
    
    _registry: Dict[str, Type[BaseVisualizer]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """注册可视化器类
        
        Args:
            name: 可视化器名称，如果为None则使用类名
            
        Returns:
            装饰器函数
        """
        def decorator(visualizer_cls):
            visualizer_name = name or visualizer_cls.__name__
            cls._registry[visualizer_name] = visualizer_cls
            return visualizer_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseVisualizer]:
        """获取可视化器类
        
        Args:
            name: 可视化器名称
            
        Returns:
            可视化器类
            
        Raises:
            KeyError: 如果可视化器类不存在
        """
        if name not in cls._registry:
            raise KeyError(f"可视化器类型 '{name}' 未注册")
        return cls._registry[name]
    
    @classmethod
    def list(cls) -> Dict[str, Type[BaseVisualizer]]:
        """获取所有注册的可视化器类
        
        Returns:
            可视化器名称到类的映射
        """
        return cls._registry.copy() 