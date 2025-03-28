class DatasetRegistry:
    """数据集注册器，用于注册和获取数据集类"""
    _datasets = {}
    
    @classmethod
    def register(cls, name):
        """注册数据集类
        
        Args:
            name: 数据集名称
            
        Returns:
            装饰器函数
        """
        def wrapper(dataset_class):
            cls._datasets[name] = dataset_class
            return dataset_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取数据集类
        
        Args:
            name: 数据集名称
            
        Returns:
            数据集类
        
        Raises:
            ValueError: 如果数据集未注册
        """
        if name not in cls._datasets:
            raise ValueError(f"未注册的数据集: {name}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls):
        """列出所有已注册的数据集
        
        Returns:
            已注册数据集名称列表
        """
        return list(cls._datasets.keys()) 