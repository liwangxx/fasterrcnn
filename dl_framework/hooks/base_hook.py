import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from ..visualization.base_visualizer import BaseVisualizer
from ..utils.logger import get_logger

class BaseHook:
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 钩子配置
        """
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.frequency = self.config.get('frequency', 100)
        self.targets = self.config.get('targets', [])
        self.hooks = {}
        self.services = {}  # 用于存储注入的服务
        # 设置钩子
        self._setup()
    
    def register_service(self, service_name: str, service: Any) -> None:
        """注册外部服务（依赖注入）
        
        Args:
            service_name: 服务名称
            service: 服务实例
        """
        self.services[service_name] = service
    
    def get_service(self, service_name: str) -> Any:
        """获取已注册的服务
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务实例或None
        """
        return self.services.get(service_name)
    
    def _setup(self) -> None:
        """设置钩子环境"""
        # 子类可以重写此方法进行特定设置
        pass
    
    def should_trigger(self, step: int) -> bool:
        """检查是否应该触发钩子
        
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

class HookFactory:
    """Hook工厂类，用于从配置创建Hook实例"""
    
    @staticmethod
    def create_hooks_from_config(hooks_config: List[Dict[str, Any]]) -> List[BaseHook]:
        """根据配置创建Hook实例
        
        Args:
            hooks_config: Hook配置列表
            
        Returns:
            创建的Hook实例列表
        """
        hooks = []
        logger = get_logger('hooks')
        for hook_config in hooks_config:
            hook_type = hook_config.get('type')
            if not hook_type:
                continue
                
            try:
                # 从注册表获取Hook类
                from .registry import HookRegistry
                hook_cls = HookRegistry.get(hook_type)
                # 创建Hook实例
                hook = hook_cls(hook_config)
                hooks.append(hook)
            except KeyError:
                logger.warning(f"警告: 钩子类型 '{hook_type}' 未注册")
            except Exception as e:
                logger.warning(f"创建钩子 '{hook_type}' 时出错: {str(e)}")
                
        return hooks