import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Optional
from torch.utils.tensorboard import SummaryWriter

from .base_visualizer import BaseVisualizer
from .registry import VisualizerRegistry

@VisualizerRegistry.register("TensorBoard")
class TensorBoardVisualizer(BaseVisualizer):
    """TensorBoard可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 可视化配置
        """
        super().__init__(config)
        
        if self.enabled:
            tensorboard_config = config.get('tensorboard', {})
            self.log_dir = tensorboard_config.get('log_dir', self.log_dir)
            self.flush_secs = tensorboard_config.get('flush_secs', 30)
            
            # 创建SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.log_dir,
                flush_secs=self.flush_secs
            )
    
    def _add_scalar_impl(self, tag: str, scalar_value: float, global_step: int) -> None:
        """添加标量的具体实现
        
        Args:
            tag: 标量标签
            scalar_value: 标量值
            global_step: 全局步数
        """
        self.writer.add_scalar(tag, scalar_value, global_step)
    
    def _add_scalars_impl(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int) -> None:
        """添加多个标量的具体实现
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-标量字典
            global_step: 全局步数
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
    
    def _add_image_impl(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        """添加图像的具体实现
        
        Args:
            tag: 图像标签
            img_tensor: 图像张量
            global_step: 全局步数
        """
        self.writer.add_image(tag, img_tensor, global_step)
    
    def _add_histogram_impl(self, tag: str, values: Union[torch.Tensor, np.ndarray], global_step: int) -> None:
        """添加直方图的具体实现
        
        Args:
            tag: 直方图标签
            values: 数值
            global_step: 全局步数
        """
        self.writer.add_histogram(tag, values, global_step)
    
    def _add_figure_impl(self, tag: str, figure: plt.Figure, global_step: int) -> None:
        """添加matplotlib图表的具体实现
        
        Args:
            tag: 图表标签
            figure: matplotlib图表
            global_step: 全局步数
        """
        self.writer.add_figure(tag, figure, global_step)
    
    def _add_text_impl(self, tag: str, text_string: str, global_step: int) -> None:
        """添加文本的具体实现
        
        Args:
            tag: 文本标签
            text_string: 文本内容
            global_step: 全局步数
        """
        self.writer.add_text(tag, text_string, global_step)
    
    def _add_graph_impl(self, model: torch.nn.Module, input_to_model: Optional[torch.Tensor] = None) -> None:
        """添加模型图的具体实现
        
        Args:
            model: 模型
            input_to_model: 输入张量
        """
        if input_to_model is not None:
            self.writer.add_graph(model, input_to_model)
        else:
            # 如果没有提供输入张量，创建一个假的输入
            try:
                # 尝试获取模型的输入形状
                if hasattr(model, 'config') and 'in_channels' in model.config:
                    in_channels = model.config.get('in_channels', 3)
                    input_shape = (1, in_channels, 32, 32)  # 假设是图像输入
                else:
                    input_shape = (1, 3, 32, 32)  # 默认假设是RGB图像
                
                dummy_input = torch.rand(input_shape).to(next(model.parameters()).device)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                print(f"添加模型图失败: {e}")
    
    def _flush_impl(self) -> None:
        """刷新可视化器的具体实现"""
        self.writer.flush()
    
    def _close_impl(self) -> None:
        """关闭可视化器的具体实现"""
        self.writer.close() 