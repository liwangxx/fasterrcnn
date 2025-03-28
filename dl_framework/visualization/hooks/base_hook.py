import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from ..base_visualizer import BaseVisualizer

class BaseHook:
    """可视化钩子基类"""
    
    def __init__(self, config: Dict[str, Any], visualizer: BaseVisualizer):
        """初始化
        
        Args:
            config: 钩子配置
            visualizer: 可视化器
        """
        self.config = config
        self.visualizer = visualizer
        self.name = config.get('name', self.__class__.__name__)
        self.frequency = config.get('frequency', 100)  # 多少步触发一次
        self.targets = config.get('targets', [])  # 目标模块
        
        # 存储注册的钩子句柄
        self.hooks = {}
        
        # 设置钩子
        self._setup()
    
    def _setup(self) -> None:
        """设置钩子环境"""
        # 子类可以重写此方法进行特定设置
        pass
    
    def should_trigger(self, step: int) -> bool:
        """检查是否应该触发
        
        Args:
            step: 当前步数
            
        Returns:
            是否应该触发
        """
        return step % self.frequency == 0
    
    def register_hooks(self, model: nn.Module) -> None:
        """向模型注册钩子
        
        Args:
            model: 模型
        """
        for target in self.targets:
            if '.' in target:
                # 处理嵌套模块，例如 "layer1.0"
                module = model
                for part in target.split('.'):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        raise ValueError(f"模块 {module.__class__.__name__} 没有子模块 {part}")
                self._register_to_module(module, target)
            else:
                # 直接获取模块
                if hasattr(model, target):
                    module = getattr(model, target)
                    self._register_to_module(module, target)
                else:
                    raise ValueError(f"模型 {model.__class__.__name__} 没有模块 {target}")
    
    def _register_to_module(self, module: nn.Module, name: str) -> None:
        """向模块注册钩子
        
        Args:
            module: 模块
            name: 模块名称
        """
        # 子类需要实现此方法
        pass
    
    def before_training(self, model: nn.Module) -> None:
        """训练开始前调用
        
        Args:
            model: 模型
        """
        # 注册钩子
        self.register_hooks(model)
    
    def before_epoch(self, epoch: int, model: nn.Module) -> None:
        """每个epoch开始前调用
        
        Args:
            epoch: 当前epoch
            model: 模型
        """
        pass
    
    def before_step(self, step: int, batch: Any, model: nn.Module) -> None:
        """每步开始前调用
        
        Args:
            step: 当前步数
            batch: 数据批次
            model: 模型
        """
        pass
    
    def after_step(self, step: int, batch: Any, outputs: Any, loss: torch.Tensor, model: nn.Module) -> None:
        """每步结束后调用
        
        Args:
            step: 当前步数
            batch: 数据批次
            outputs: 模型输出
            loss: 损失
            model: 模型
        """
        pass
    
    def after_epoch(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> None:
        """每个epoch结束后调用
        
        Args:
            epoch: 当前epoch
            model: 模型
            metrics: 验证指标
        """
        pass
    
    def after_training(self, model: nn.Module, metrics: Dict[str, float]) -> None:
        """训练结束后调用
        
        Args:
            model: 模型
            metrics: 验证指标
        """
        pass
    
    def cleanup(self) -> None:
        """清理钩子资源"""
        # 移除所有注册的钩子
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear() 