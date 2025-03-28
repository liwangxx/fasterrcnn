from typing import Dict, Type, Any
from .base_hook import BaseHook

class HookRegistry:
    """钩子注册表，用于动态注册和获取钩子类"""
    
    _registry: Dict[str, Type[BaseHook]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """注册钩子类
        
        Args:
            name: 钩子名称，如果为None则使用类名
            
        Returns:
            装饰器函数
        """
        def decorator(hook_cls):
            hook_name = name or hook_cls.__name__
            cls._registry[hook_name] = hook_cls
            return hook_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[BaseHook]:
        """获取钩子类
        
        Args:
            name: 钩子名称
            
        Returns:
            钩子类
            
        Raises:
            KeyError: 如果钩子类不存在
        """
        if name not in cls._registry:
            raise KeyError(f"钩子类型 '{name}' 未注册")
        return cls._registry[name]
    
    @classmethod
    def list(cls) -> Dict[str, Type[BaseHook]]:
        """获取所有注册的钩子类
        
        Returns:
            钩子名称到类的映射
        """
        return cls._registry.copy() 