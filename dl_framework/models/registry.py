class ModelRegistry:
    """模型注册器，用于注册和获取模型类"""
    _models = {}
    
    @classmethod
    def register(cls, name):
        """注册模型类
        
        Args:
            name: 模型名称
            
        Returns:
            装饰器函数
        """
        def wrapper(model_class):
            cls._models[name] = model_class
            return model_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取模型类
        
        Args:
            name: 模型名称
            
        Returns:
            模型类
        
        Raises:
            ValueError: 如果模型未注册
        """
        if name not in cls._models:
            raise ValueError(f"未注册的模型: {name}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls):
        """列出所有已注册的模型
        
        Returns:
            已注册模型名称列表
        """
        return list(cls._models.keys()) 