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
        self.global_config = config
        self.config = config.get('visualization', {})
        self.experiment_dir = config.get('experiment_dir', 'experiments/default')
        # 查找是否有已经注册了的可视化器
        from .registry import VisualizerRegistry
        for key in self.config.keys():
            if key.lower() in [name.lower() for name in VisualizerRegistry.list().keys()]:
                self.visualizer_config = self.config.get(key, None)
        
        if self.visualizer_config:
            self.vis_dir = self.global_config.get('visualization_dir')
            
            # 确保日志目录存在
            os.makedirs(self.vis_dir, exist_ok=True)
            
            # 创建标准子目录
            self.img_dir = os.path.join(self.vis_dir, 'images')
            self.sample_dir = os.path.join(self.vis_dir, 'samples')
            
            # 确保子目录存在
            os.makedirs(self.img_dir, exist_ok=True)
            os.makedirs(self.sample_dir, exist_ok=True)


    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """添加标量
        
        Args:
            tag: 标量标签
            scalar_value: 标量值
            global_step: 全局步数
        """
        if not self.visualizer_config:
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
        if not self.visualizer_config:
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
        if not self.visualizer_config:
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
        if not self.visualizer_config:
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
        if not self.visualizer_config:
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
        if not self.visualizer_config:
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
        if not self.visualizer_config:
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
        if not self.visualizer_config:
            return
        self._flush_impl()
    
    def _flush_impl(self) -> None:
        """刷新可视化器的具体实现"""
        # 子类需要实现此方法
        pass
    
    def close(self) -> None:
        """关闭可视化器"""
        if not self.visualizer_config:
            return
        self._close_impl()
    
    def _close_impl(self) -> None:
        """关闭可视化器的具体实现"""
        # 子类需要实现此方法
        pass
    
    def add_images_grid(self, tag: str, img_tensor: torch.Tensor, global_step: int, nrow: int = 8) -> None:
        """添加图像网格
        
        Args:
            tag: 图像标签
            img_tensor: 形状为[B, C, H, W]的图像张量批次
            global_step: 全局步数
            nrow: 网格中每行的图像数量
        """
        if not self.visualizer_config:
            return
        
        self._add_images_grid_impl(tag, img_tensor, global_step, nrow)
    
    def _add_images_grid_impl(self, tag: str, img_tensor: torch.Tensor, global_step: int, nrow: int = 8) -> None:
        """添加图像网格的具体实现
        
        Args:
            tag: 图像标签
            img_tensor: 形状为[B, C, H, W]的图像张量批次
            global_step: 全局步数
            nrow: 网格中每行的图像数量
        """
        # 子类需要实现此方法
        pass 