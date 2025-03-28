import os
from typing import Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np
import torch

class BaseVisualizer:
    """可视化器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if self.enabled:
            self.log_dir = self._get_log_dir()
    
    def _get_log_dir(self) -> str:
        """获取日志目录
        
        Returns:
            日志目录路径
        """
        log_dir = self.config.get('log_dir', 'logs/visualization')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """添加标量
        
        Args:
            tag: 标量标签
            scalar_value: 标量值
            global_step: 全局步数
        """
        if not self.enabled:
            return
        self._add_scalar_impl(tag, scalar_value, global_step)
    
    def _add_scalar_impl(self, tag: str, scalar_value: float, global_step: int) -> None:
        """添加标量的具体实现
        
        Args:
            tag: 标量标签
            scalar_value: 标量值
            global_step: 全局步数
        """
        # 子类需要实现此方法
        pass
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int) -> None:
        """添加多个标量
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-标量字典
            global_step: 全局步数
        """
        if not self.enabled:
            return
        self._add_scalars_impl(main_tag, tag_scalar_dict, global_step)
    
    def _add_scalars_impl(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int) -> None:
        """添加多个标量的具体实现
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-标量字典
            global_step: 全局步数
        """
        # 子类需要实现此方法
        pass
    
    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        """添加图像
        
        Args:
            tag: 图像标签
            img_tensor: 图像张量，形状为[C, H, W]或[N, C, H, W]
            global_step: 全局步数
        """
        if not self.enabled:
            return
        self._add_image_impl(tag, img_tensor, global_step)
    
    def _add_image_impl(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        """添加图像的具体实现
        
        Args:
            tag: 图像标签
            img_tensor: 图像张量
            global_step: 全局步数
        """
        # 子类需要实现此方法
        pass
    
    def add_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], global_step: int) -> None:
        """添加直方图
        
        Args:
            tag: 直方图标签
            values: 数值
            global_step: 全局步数
        """
        if not self.enabled:
            return
        self._add_histogram_impl(tag, values, global_step)
    
    def _add_histogram_impl(self, tag: str, values: Union[torch.Tensor, np.ndarray], global_step: int) -> None:
        """添加直方图的具体实现
        
        Args:
            tag: 直方图标签
            values: 数值
            global_step: 全局步数
        """
        # 子类需要实现此方法
        pass
    
    def add_figure(self, tag: str, figure: plt.Figure, global_step: int) -> None:
        """添加matplotlib图表
        
        Args:
            tag: 图表标签
            figure: matplotlib图表
            global_step: 全局步数
        """
        if not self.enabled:
            return
        self._add_figure_impl(tag, figure, global_step)
    
    def _add_figure_impl(self, tag: str, figure: plt.Figure, global_step: int) -> None:
        """添加matplotlib图表的具体实现
        
        Args:
            tag: 图表标签
            figure: matplotlib图表
            global_step: 全局步数
        """
        # 子类需要实现此方法
        pass
    
    def add_text(self, tag: str, text_string: str, global_step: int) -> None:
        """添加文本
        
        Args:
            tag: 文本标签
            text_string: 文本内容
            global_step: 全局步数
        """
        if not self.enabled:
            return
        self._add_text_impl(tag, text_string, global_step)
    
    def _add_text_impl(self, tag: str, text_string: str, global_step: int) -> None:
        """添加文本的具体实现
        
        Args:
            tag: 文本标签
            text_string: 文本内容
            global_step: 全局步数
        """
        # 子类需要实现此方法
        pass
    
    def add_graph(self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None) -> None:
        """添加模型图
        
        Args:
            model: 模型
            input_to_model: 输入张量
        """
        if not self.enabled:
            return
        self._add_graph_impl(model, input_to_model)
    
    def _add_graph_impl(self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None) -> None:
        """添加模型图的具体实现
        
        Args:
            model: 模型
            input_to_model: 输入张量
        """
        # 子类需要实现此方法
        pass
    
    def flush(self) -> None:
        """刷新可视化器"""
        if not self.enabled:
            return
        self._flush_impl()
    
    def _flush_impl(self) -> None:
        """刷新可视化器的具体实现"""
        # 子类需要实现此方法
        pass
    
    def close(self) -> None:
        """关闭可视化器"""
        if not self.enabled:
            return
        self._close_impl()
    
    def _close_impl(self) -> None:
        """关闭可视化器的具体实现"""
        # 子类需要实现此方法
        pass 