import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List, Union, Optional
import os

class BaseDataset(Dataset):
    """所有数据集的基类"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__()
        self.config = config
        self.is_training = is_training
        self.data_dir = self._get_data_dir()
        self.transform = self._get_transforms()
        
    def _get_data_dir(self) -> str:
        """获取数据目录
        
        Returns:
            数据目录路径
        """
        data_dir = self.config.get('data_dir', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        return data_dir
    
    def _get_transforms(self) -> Optional[Any]:
        """获取数据变换
        
        Returns:
            数据变换对象
        """
        # 子类应实现此方法来提供适当的变换
        return None
    
    def _load_data(self) -> None:
        """加载数据
        
        子类必须实现此方法来加载数据
        """
        raise NotImplementedError("子类必须实现_load_data方法")
    
    def __len__(self) -> int:
        """获取数据集长度
        
        Returns:
            数据集长度
        """
        raise NotImplementedError("子类必须实现__len__方法")
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项
        """
        raise NotImplementedError("子类必须实现__getitem__方法")
    
    def get_collate_fn(self):
        """获取数据整理函数
        
        Returns:
            数据整理函数
        """
        # 默认返回None，使用PyTorch默认的collate_fn
        return None 