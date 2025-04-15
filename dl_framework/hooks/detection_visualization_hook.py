import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any, Optional, Tuple
import torchvision.transforms as T
import time

from .base_hook import BaseHook
from .registry import HookRegistry

@HookRegistry.register("DetectionVisualizationHook")
class DetectionVisualizationHook(BaseHook):
    """目标检测可视化钩子，用于可视化模型的检测结果"""
    
    def __init__(self, name: str, frequency: int = 100, num_images: int = 4, 
                 confidence_threshold: float = 0.5, output_dir: str = 'outputs/visualizations',
                 figsize: Tuple[int, int] = (12, 8)):
        """初始化检测可视化钩子
        
        Args:
            name: 钩子名称
            frequency: 触发频率（每N步）
            num_images: 每次显示的图像数量
            confidence_threshold: 置信度阈值，只显示高于此阈值的检测框
            output_dir: 输出目录
            figsize: 图像大小
        """
        super().__init__(name, frequency)
        self.num_images = num_images
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        self.figsize = figsize
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 用于反归一化图像的变换
        self.denormalize = T.Compose([
            T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])
        
        # 颜色映射，用于不同类别
        self.colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()  # 假设最大21个类别
    
    def on_step_end(self, state: Dict[str, Any]) -> None:
        """每步结束时调用
        
        Args:
            state: 训练状态
        """
        # 检查是否应该触发
        if not self._should_trigger(state):
            return
        
        # 获取模型和数据集
        model = state.get('model')
        if model is None:
            return
        
        batch = state.get('batch')
        if batch is None:
            return
        
        # 如果batch是字典，获取images
        if isinstance(batch, dict):
            images = batch.get('images')
            targets = batch.get('targets', None)
        else:
            # 如果batch是列表或元组，假设第一个元素是images
            images = batch[0]
            targets = batch[1] if len(batch) > 1 else None
        
        # 转为评估模式
        model.eval()
        
        # 限制图像数量
        batch_size = min(len(images), self.num_images)
        
        # 创建图像网格
        fig, axs = plt.subplots(1, batch_size, figsize=self.figsize)
        if batch_size == 1:
            axs = [axs]
        
        # 获取类别名
        dataset = state.get('dataset')
        if dataset and hasattr(dataset, 'CLASSES'):
            class_names = dataset.CLASSES
        else:
            # 如果没有类别名，使用索引
            class_names = [str(i) for i in range(21)]  # 默认21个类别（VOC）
        
        # 可视化每张图像
        with torch.no_grad():
            for i in range(batch_size):
                img = images[i].detach().cpu()
                
                # 反归一化图像
                img = self.denormalize(img)
                
                # 将图像转换为numpy数组并调整到0-1范围
                img_np = img.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                
                # 显示图像
                axs[i].imshow(img_np)
                axs[i].axis('off')
                
                # 获取预测结果
                # 为了预测单个图像，需要将其包装在列表中
                predictions = model.predict([images[i].to(model.device)])[0]
                
                # 提取预测框、分数和标签
                pred_boxes = predictions['boxes'].detach().cpu().numpy()
                pred_scores = predictions['scores'].detach().cpu().numpy()
                pred_labels = predictions['labels'].detach().cpu().numpy()
                
                # 应用置信度阈值
                mask = pred_scores >= self.confidence_threshold
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
                
                # 绘制预测框
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    # 获取框的坐标
                    x1, y1, x2, y2 = box
                    
                    # 获取类别名称和颜色
                    class_name = class_names[label] if label < len(class_names) else f"class_{label}"
                    color = self.colors[label % len(self.colors)]
                    
                    # 绘制矩形框
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, 
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    axs[i].add_patch(rect)
                    
                    # 添加标签文本
                    axs[i].text(
                        x1, y1 - 5, 
                        f"{class_name}: {score:.2f}",
                        color=color, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                    )
                
                # 标题
                axs[i].set_title(f"Batch {state['step']}, Image {i}")
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"detection_{state['step']}_{timestamp}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        
        # 恢复训练模式
        model.train()
        
        # 记录结果路径
        self.logger.info(f"检测可视化已保存到: {save_path}")
        
    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        """每个epoch结束时调用
        
        Args:
            state: 训练状态
        """
        pass  # 在epoch结束时不需要特殊处理 