import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from .base_hook import BaseHook
from ..base_visualizer import BaseVisualizer

class GradientFlowHook(BaseHook):
    """梯度流可视化钩子，用于可视化模型参数梯度的变化"""
    
    def __init__(self, config: Dict[str, Any], visualizer: BaseVisualizer):
        """初始化
        
        Args:
            config: 钩子配置
            visualizer: 可视化器
        """
        super().__init__(config, visualizer)
    
    def _setup(self) -> None:
        """设置钩子环境"""
        self.save_dir = self.config.get('save_dir', 'visualization/grad_flow')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 存储梯度数据
        self.grad_data = {}
    
    def _register_to_module(self, module: nn.Module, name: str) -> None:
        """向模块注册钩子
        
        Args:
            module: 模块
            name: 模块名称
        """
        # 对于梯度流，我们只需要存储模块中的参数梯度
        # 不需要实际注册钩子，只需要记录需要跟踪的模块
        self.grad_data[name] = None
    
    def after_step(self, step: int, batch: Any, outputs: Any, loss: torch.Tensor, model: nn.Module) -> None:
        """每步结束后调用
        
        Args:
            step: 当前步数
            batch: 数据批次
            outputs: 模型输出
            loss: 损失
            model: 模型
        """
        if not self.should_trigger(step):
            return
        
        # 收集指定模块的梯度
        for target in self.targets:
            if '.' in target:
                # 处理嵌套模块，例如 "layer1.0"
                module = model
                for part in target.split('.'):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        continue
                self.grad_data[target] = self._get_module_grad(module)
            else:
                # 直接获取模块
                if hasattr(model, target):
                    module = getattr(model, target)
                    self.grad_data[target] = self._get_module_grad(module)
        
        # 创建梯度流图表
        self._visualize_grad_flow(step)
    
    def _get_module_grad(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        """获取模块的梯度
        
        Args:
            module: 模块
            
        Returns:
            模块参数名称和梯度的字典
        """
        grad_dict = {}
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_dict[name] = param.grad.abs().mean().item()
        return grad_dict
    
    def _visualize_grad_flow(self, step: int) -> None:
        """可视化梯度流
        
        Args:
            step: 当前步数
        """
        # 创建图表
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        # 准备数据
        module_names = []
        grad_means = []
        
        for module_name, grad_dict in self.grad_data.items():
            if grad_dict is None:
                continue
                
            for param_name, grad_mean in grad_dict.items():
                module_names.append(f"{module_name}.{param_name}")
                grad_means.append(grad_mean)
        
        # 绘制柱状图
        if module_names:
            bars = ax.bar(range(len(module_names)), grad_means, align='center')
            ax.set_xticks(range(len(module_names)))
            ax.set_xticklabels(module_names, rotation=90)
            ax.set_xlabel('Layers')
            ax.set_ylabel('Average Gradient')
            ax.set_title(f'Gradient Flow (Step {step})')
            ax.grid(True)
            
            # 添加颜色映射
            cm = plt.cm.get_cmap('viridis')
            for i, bar in enumerate(bars):
                bar.set_color(cm(i / len(bars)))
            
            # 设置布局
            plt.tight_layout()
            
            # 保存图表
            fig_path = os.path.join(self.save_dir, f"grad_flow_{step}.png")
            plt.savefig(fig_path)
            
            # 添加到可视化器
            self.visualizer.add_figure('grad_flow', fig, step)
        
        plt.close(fig) 