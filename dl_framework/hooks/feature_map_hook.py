import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union

from .base_hook import BaseHook
from .registry import HookRegistry
from ..models.cnn import CNN

@HookRegistry.register("FeatureMapHook")
class FeatureMapHook(BaseHook):
    """特征图可视化钩子"""
    
    def __init__(self, config: Dict[str, Any], visualizer):
        """初始化
        
        Args:
            config: 钩子配置
            visualizer: 可视化器
        """
        super().__init__(config, visualizer)
        
        # 获取配置
        self.max_features = config.get('max_features', 16)  # 每层最多显示的特征数量
        self.sample_batch_idx = config.get('sample_batch_idx', 0)  # 用于可视化的样本在批次中的索引
        self.input_shape = config.get('input_shape', None)  # 输入形状，如果为None，则使用实际输入
    
    def _setup(self) -> None:
        """设置钩子环境"""
        pass  # 不需要特别设置
    
    def before_step(self, step: int, batch: Any, model: nn.Module) -> None:
        """在每步开始前调用
        
        Args:
            step: 当前步数
            batch: 数据批次
            model: 模型
        """
        # 不需要在每步开始前执行操作
        pass
    
    def after_step(self, step: int, batch: Any, outputs: Any, loss: torch.Tensor, model: nn.Module) -> None:
        """在每步结束后调用
        
        Args:
            step: 当前步数
            batch: 数据批次
            outputs: 模型输出
            loss: 损失
            model: 模型
        """
        # 检查是否应该触发可视化
        if not self.should_trigger(step):
            return
            
        # 检查模型是否为CNN模型
        if not isinstance(model, CNN):
            return
            
        # 获取输入数据
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            # 假设批次是(inputs, targets)或(inputs, targets, ...)的形式
            inputs = batch[0]
        else:
            # 如果批次格式不明确，直接使用
            inputs = batch
            
        # 确保输入是张量
        if not isinstance(inputs, torch.Tensor):
            return
            
        # 选择用于可视化的样本
        if inputs.dim() == 4 and inputs.size(0) > self.sample_batch_idx:
            # 对于图像数据，选择一个样本进行可视化
            sample = inputs[self.sample_batch_idx].unsqueeze(0)
        else:
            # 如果批次大小不足或不是图像数据，使用全部数据
            sample = inputs
            
        # 确保样本和模型在同一设备上
        device = next(model.parameters()).device
        sample = sample.to(device)
            
        # 调用模型的visualize_features方法
        model.visualize_features(self.visualizer, sample, step)
    
    def after_epoch(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> None:
        """在每个epoch结束后调用
        
        Args:
            epoch: 当前epoch
            model: 模型
            metrics: 验证指标
        """
        # 不需要在每个epoch结束后执行特别操作
        pass
    
    def cleanup(self) -> None:
        """清理钩子资源"""
        # 不需要特别清理
        super().cleanup()