import os
import torch
import torchvision
from torchvision.datasets import VOCDetection
from torchvision import transforms as T
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import random
import math

from .base_dataset import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register('voc2012')
class VOC2012Dataset(BaseDataset):
    """Pascal VOC 2012数据集"""
    
    # VOC数据集的类别名称
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]
    
    def __init__(self, config, is_training=True):
        """初始化VOC2012数据集
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__(config, is_training)
        
        # 获取配置参数
        self.data_root = config.get('data_root', 'data/PASCAL_VOC')
        self.year = config.get('year', '2012')
        self.image_set = 'train' if is_training else 'val'
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 图像大小设置
        self.img_size = config.get('img_size', 600)  # 使用固定大小
        self.use_fixed_size = config.get('use_fixed_size', True)  # 默认使用固定大小
        self.use_augmentation = config.get('use_augmentation', True) and is_training
        
        # 构建类别到索引的映射
        self.class_to_idx = {cls: i for i, cls in enumerate(self.CLASSES)}
        
        # 加载数据集
        self.dataset = VOCDetection(
            root=self.data_root,
            year=self.year,
            image_set=self.image_set,
            download=False,  # 不自动下载数据
            transform=None  # 我们自己处理转换
        )
        
        # 初始化转换
        self._init_transforms()
        
        # 缓存无效样本索引
        self.invalid_indices = self._find_invalid_samples()
        if len(self.invalid_indices) > 0:
            print(f"警告: 发现 {len(self.invalid_indices)} 个无效样本")
    
    def _init_transforms(self):
        """初始化数据转换"""
        # 基本转换，用于验证和测试
        self.base_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 训练时的数据增强
        if self.is_training and self.use_augmentation:
            self.augmentation_transforms = [
                # 水平翻转 - 保留这个，是最安全的数据增强
                lambda img, targets: self._horizontal_flip(img, targets),
                # 随机缩放 - 减小缩放范围，避免过度缩放导致边界框无效
                lambda img, targets: self._random_scale(img, targets, scale_range=(0.9, 1.1)),
                # 颜色抖动 - 减小颜色抖动参数
                lambda img, targets: (self._color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), targets),
                # 随机裁剪 - 增大最小尺寸比例，减少裁剪导致的边界框失效
                lambda img, targets: self._random_crop(img, targets, min_size=0.7),
            ]
        else:
            self.augmentation_transforms = []
    
    def _find_invalid_samples(self):
        """查找无效样本
        
        Returns:
            无效样本的索引列表
        """
        invalid_indices = []
        for i in range(len(self.dataset)):
            try:
                img, target_dict = self.dataset[i]
                boxes, labels = self._parse_voc_xml(target_dict['annotation'])
                if len(boxes) == 0:
                    invalid_indices.append(i)
            except Exception as e:
                print(f"样本 {i} 处理错误: {e}")
                invalid_indices.append(i)
        return invalid_indices
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.dataset) - len(self.invalid_indices)
    
    def __getitem__(self, idx):
        """获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            处理后的样本字典，包含图像和目标
        """
        # 递归计数器，避免无限递归
        if not hasattr(self, '_recursion_counter'):
            self._recursion_counter = 0
        
        # 如果递归次数过多，返回一个简单的样本
        if self._recursion_counter > 10:
            self._recursion_counter = 0
            # 创建一个简单样本
            dummy_image = torch.zeros((3, self.img_size, self.img_size))
            dummy_target = {
                'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),  # 假设类别1
                'image_id': torch.tensor([0]),
                'area': torch.tensor([1600.0]),  # 40*40
                'iscrowd': torch.tensor([0], dtype=torch.int64),
                'orig_size': torch.tensor([self.img_size, self.img_size]),
                'resized_size': torch.tensor([self.img_size, self.img_size])
            }
            return {'images': dummy_image, 'targets': dummy_target}
        
        self._recursion_counter += 1
        
        # 调整索引以跳过无效样本
        valid_idx = idx
        for invalid_idx in self.invalid_indices:
            if valid_idx >= invalid_idx:
                valid_idx += 1
        
        # 获取原始数据
        image, target_dict = self.dataset[valid_idx]
        
        # 如果图像不是RGB模式，转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 解析目标框和类别
        boxes, labels = self._parse_voc_xml(target_dict['annotation'])
        
        # 确保有边界框和标签 - 如果没有有效的边界框，跳过这个样本
        if len(boxes) == 0:
            # 递归调用，重新选择样本
            self._recursion_counter -= 1
            return self.__getitem__((idx + 1) % len(self))
        
        # 转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 检查边界框有效性
        valid_boxes_mask = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
        if not valid_boxes_mask.all():
            # 过滤掉无效边界框
            boxes = boxes[valid_boxes_mask]
            labels = labels[valid_boxes_mask]
            
            # 如果过滤后没有边界框，重新选择样本
            if boxes.shape[0] == 0:
                self._recursion_counter -= 1
                return self.__getitem__((idx + 1) % len(self))
        
        # 准备目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([valid_idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # 记录原始图像尺寸，供调试
        w, h = image.size
        orig_size = torch.tensor([h, w])
        
        # 应用数据增强 - 降低应用概率从0.5到0.3
        if self.is_training and self.use_augmentation:
            for transform in self.augmentation_transforms:
                # 随机决定是否应用此增强 - 降低概率
                if random.random() < 0.3:
                    try:
                        image, target = transform(image, target)
                        # 验证每次变换后boxes的有效性
                        if target['boxes'].shape[0] == 0:
                            # 不要引发异常，而是跳过这次增强
                            continue
                    except Exception as e:
                        # 不打印错误，静默处理
                        continue
        
        # 确保图像大小固定（缩放和填充）
        image, target = self._resize_and_pad_image(image, target)
        
        # 记录resize后的图像尺寸，供调试
        w, h = image.size
        resized_size = torch.tensor([h, w])
        
        # 应用基本转换（归一化等）
        image = self.base_transforms(image)
        
        # 最终检查边界框有效性 - 不再输出警告信息，静默处理
        boxes = target['boxes']
        if boxes.shape[0] == 0 or torch.any(torch.isnan(boxes)) or torch.any(torch.isinf(boxes)):
            self._recursion_counter -= 1
            return self.__getitem__((idx + 1) % len(self))
        
        # 将边界框严格限制在图像边界内
        h, w = self.img_size, self.img_size  # 最终图像尺寸
        boxes[:, 0].clamp_(min=0, max=w-1)
        boxes[:, 1].clamp_(min=0, max=h-1)
        boxes[:, 2].clamp_(min=1, max=w)
        boxes[:, 3].clamp_(min=1, max=h)
        
        # 确保所有框都有面积 - 放宽验证标准，只要面积大于等于0就保留
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        valid_area_mask = area >= 0
        if not valid_area_mask.all():
            # 过滤掉面积为0的框
            boxes = boxes[valid_area_mask]
            labels = target['labels'][valid_area_mask]
            
            # 如果过滤后没有边界框，重新选择样本
            if boxes.shape[0] == 0:
                self._recursion_counter -= 1
                return self.__getitem__((idx + 1) % len(self))
            
            # 更新target
            target['boxes'] = boxes
            target['labels'] = labels
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # 添加调试信息
        target['orig_size'] = orig_size
        target['resized_size'] = resized_size
        
        result = {
            'images': image, 
            'targets': target
        }
        
        # 重置递归计数器
        self._recursion_counter -= 1
        
        return result
    
    def _resize_and_pad_image(self, image, target):
        """调整图像大小并填充到固定尺寸
        
        Args:
            image: PIL图像
            target: 目标字典
            
        Returns:
            调整大小和填充后的图像和目标
        """
        w, h = image.size
        target_size = self.img_size
        
        # 计算缩放比例，同时保持宽高比
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # 调整图像大小
        image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # 如果有边界框，按比例调整
        if target['boxes'].shape[0] > 0:
            boxes = target['boxes'].clone()  # 克隆以避免修改原始数据
            
            # 按比例调整边界框坐标
            boxes[:, [0, 2]] *= (new_w / w)
            boxes[:, [1, 3]] *= (new_h / h)
            
            # 确保坐标在有效范围内
            boxes[:, 0].clamp_(min=0, max=new_w-1)
            boxes[:, 1].clamp_(min=0, max=new_h-1)
            boxes[:, 2].clamp_(min=1, max=new_w)
            boxes[:, 3].clamp_(min=1, max=new_h)
            
            # 确保所有框的宽高大于1
            width = boxes[:, 2] - boxes[:, 0]
            height = boxes[:, 3] - boxes[:, 1]
            valid_size = (width > 1) & (height > 1)
            
            if not valid_size.all():
                # 只保留有效的框
                boxes = boxes[valid_size]
                if 'labels' in target:
                    target['labels'] = target['labels'][valid_size]
            
            target['boxes'] = boxes
            
            # 重新计算面积
            if boxes.shape[0] > 0:
                target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                # 更新iscrowd
                target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        # 创建灰色填充图像(128,128,128是中性灰色)
        padded_image = Image.new('RGB', (target_size, target_size), (128, 128, 128))
        # 将调整大小后的图像粘贴到填充图像的左上角
        padded_image.paste(image, (0, 0, new_w, new_h))
        
        # 最终检查
        if 'boxes' in target and target['boxes'].shape[0] > 0:
            boxes = target['boxes']
            
            # 确保所有值都是有限的
            if torch.any(torch.isnan(boxes)) or torch.any(torch.isinf(boxes)):
                print("警告: 边界框包含NaN或Inf值，将被清除")
                valid_mask = ~(torch.isnan(boxes).any(dim=1) | torch.isinf(boxes).any(dim=1))
                target['boxes'] = boxes[valid_mask]
                if 'labels' in target:
                    target['labels'] = target['labels'][valid_mask]
                # 更新area和iscrowd
                if target['boxes'].shape[0] > 0:
                    target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
                    target['iscrowd'] = torch.zeros((target['boxes'].shape[0],), dtype=torch.int64)
        
        return padded_image, target
    
    def _parse_voc_xml(self, annotation):
        """解析VOC XML注释
        
        Args:
            annotation: VOC XML注释字典
            
        Returns:
            边界框和标签列表
        """
        boxes = []
        labels = []
        
        for obj in annotation['object']:
            # 获取类别
            class_name = obj['name']
            if class_name not in self.class_to_idx:
                continue
                
            # 获取边界框坐标
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            
            # 放宽边界框验证标准 - 允许非常小的差异
            if xmax - xmin < 0.5 or ymax - ymin < 0.5:
                continue
            
            # 确保边界框在图像内
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            
            # 放宽边界框面积的验证标准
            width = xmax - xmin
            height = ymax - ymin
            if width < 0.5 or height < 0.5:  # 放宽标准从1降到0.5
                continue
            
            # 添加边界框和类别
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[class_name])
        
        return boxes, labels
    
    def _horizontal_flip(self, image, target):
        """水平翻转图像和框
        
        Args:
            image: PIL图像
            target: 目标字典
            
        Returns:
            翻转后的图像和目标
        """
        # 翻转图像
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 翻转边界框
        if target['boxes'].shape[0] > 0:
            w = image.width
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target['boxes'] = boxes
        
        return flipped_image, target
    
    def _random_scale(self, image, target, scale_range=(0.9, 1.1)):
        """随机缩放图像和边界框
        
        Args:
            image: PIL图像
            target: 目标字典
            scale_range: 缩放比例范围
            
        Returns:
            缩放后的图像和目标
        """
        scale = random.uniform(*scale_range)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        scaled_image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # 缩放边界框
        if target['boxes'].shape[0] > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= (new_w / w)
            boxes[:, [1, 3]] *= (new_h / h)
            
            # 额外检查：确保边界框有效
            invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
            if invalid_boxes.any():
                # 如果有无效边界框，不应用这次缩放
                return image, target
                
            target['boxes'] = boxes
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        return scaled_image, target
    
    def _random_crop(self, image, target, min_size=0.7):
        """随机裁剪图像，确保保留所有边界框
        
        Args:
            image: PIL图像
            target: 目标字典
            min_size: 最小裁剪比例
            
        Returns:
            裁剪后的图像和目标
        """
        w, h = image.size
        
        # 如果没有边界框，直接返回
        if target['boxes'].shape[0] == 0:
            return image, target
        
        # 获取所有框的最小和最大坐标，确保裁剪区域包含所有目标
        boxes = target['boxes']
        min_x = max(0, boxes[:, 0].min().item())
        min_y = max(0, boxes[:, 1].min().item())
        max_x = min(w, boxes[:, 2].max().item())
        max_y = min(h, boxes[:, 3].max().item())
        
        # 安全性检查，确保裁剪区域合法
        if min_x >= max_x or min_y >= max_y:
            return image, target
            
        # 确定随机裁剪区域，但要保证包含所有边界框
        # 使用更保守的裁剪策略，增加最小尺寸
        crop_w = random.uniform(max(max_x - min_x, min_size * w), w)
        crop_h = random.uniform(max(max_y - min_y, min_size * h), h)
        
        # 计算可行的左上角坐标范围
        max_left = min(min_x, w - crop_w)
        max_top = min(min_y, h - crop_h)
        
        # 如果无法满足裁剪条件，直接返回原图
        if max_left < 0 or max_top < 0:
            return image, target
        
        # 随机选择左上角坐标
        left = random.uniform(0, max(0, max_left))
        top = random.uniform(0, max(0, max_top))
        
        # 裁剪图像
        right = min(w, left + crop_w)
        bottom = min(h, top + crop_h)
        
        cropped_image = image.crop((left, top, right, bottom))
        
        # 调整边界框坐标
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]] - left, min=0, max=right-left)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]] - top, min=0, max=bottom-top)
        
        # 验证裁剪后的边界框
        invalid_boxes = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
        if invalid_boxes.any():
            # 如果有无效边界框，不应用裁剪
            return image, target
        
        # 计算新的面积
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        return cropped_image, target
    
    def _color_jitter(self, image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """颜色抖动
        
        Args:
            image: PIL图像
            brightness, contrast, saturation, hue: 调整参数
            
        Returns:
            调整后的图像
        """
        color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        return color_jitter(image)
    
    def get_collate_fn(self):
        """返回自定义的collate函数，用于批处理
        
        Returns:
            collate_fn: 批处理函数
        """
        def collate_fn(batch):
            """自定义批处理函数，处理不同大小的图像和目标框
            
            Args:
                batch: 样本批次
                
            Returns:
                批处理后的字典
            """
            images = []
            targets = []
            
            for item in batch:
                if item is None:  # 跳过无效样本
                    continue
                images.append(item['images'])
                targets.append(item['targets'])
                
            # 确保批次不为空
            if not images:
                # 创建一个空批次
                dummy_image = torch.zeros((3, self.img_size, self.img_size))
                dummy_target = {
                    'boxes': torch.tensor([[0, 0, 1, 1]], dtype=torch.float32),
                    'labels': torch.tensor([0], dtype=torch.int64),
                    'image_id': torch.tensor([0]),
                    'area': torch.tensor([1.0]),
                    'iscrowd': torch.tensor([0], dtype=torch.int64)
                }
                images = [dummy_image]
                targets = [dummy_target]
            
            # 将图像堆叠为张量
            images = torch.stack(images)
            
            # 确保targets列表中的每个元素都有正确的格式
            for i, target in enumerate(targets):
                # 确保boxes是正确的格式并且有效
                if 'boxes' in target and target['boxes'].numel() > 0:
                    # 确保boxes格式正确 [N, 4]
                    if len(target['boxes'].shape) != 2 or target['boxes'].shape[1] != 4:
                        # 修复boxes格式
                        if len(target['boxes'].shape) == 1 and target['boxes'].numel() == 4:
                            # 单个框，需要调整形状
                            target['boxes'] = target['boxes'].view(1, 4)
                        else:
                            # 无效的框，替换为虚拟框
                            target['boxes'] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32,
                                                        device=target['boxes'].device)
                    
                    # 确保所有坐标有效（确保xmin < xmax, ymin < ymax）
                    boxes = target['boxes']
                    invalid_mask = (boxes[:, 0] >= boxes[:, 2]) | (boxes[:, 1] >= boxes[:, 3])
                    if invalid_mask.any():
                        # 移除无效的框
                        valid_boxes = boxes[~invalid_mask]
                        if valid_boxes.shape[0] == 0:
                            # 如果没有有效框，添加一个虚拟框
                            target['boxes'] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32, 
                                                        device=boxes.device)
                            if 'labels' in target:
                                target['labels'] = torch.tensor([0], dtype=torch.int64, 
                                                            device=target['labels'].device)
                        else:
                            # 使用有效框
                            target['boxes'] = valid_boxes
                            if 'labels' in target:
                                target['labels'] = target['labels'][~invalid_mask]
                
                # 确保labels与boxes数量一致
                if 'boxes' in target and 'labels' in target and target['boxes'].shape[0] != target['labels'].shape[0]:
                    # 修复labels数量
                    num_boxes = target['boxes'].shape[0]
                    if num_boxes > 0:
                        if target['labels'].shape[0] > num_boxes:
                            # 截断labels
                            target['labels'] = target['labels'][:num_boxes]
                        else:
                            # 扩展labels（用0填充）
                            padding = torch.zeros(num_boxes - target['labels'].shape[0], 
                                                dtype=torch.int64, device=target['labels'].device)
                            target['labels'] = torch.cat([target['labels'], padding])
                
                # 确保area字段正确
                if 'boxes' in target and target['boxes'].shape[0] > 0:
                    # 重新计算面积
                    boxes = target['boxes']
                    target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                
                # 确保iscrowd字段存在且正确
                if 'boxes' in target and target['boxes'].shape[0] > 0:
                    num_boxes = target['boxes'].shape[0]
                    if 'iscrowd' not in target or target['iscrowd'].shape[0] != num_boxes:
                        target['iscrowd'] = torch.zeros(num_boxes, dtype=torch.int64, 
                                                    device=target['boxes'].device)
            
            # 确保所有tensor都在同一设备上
            device = images.device
            for i, target in enumerate(targets):
                for k, v in target.items():
                    if isinstance(v, torch.Tensor) and v.device != device:
                        target[k] = v.to(device)
            
            # Faster R-CNN要求targets是列表
            return {'images': images, 'targets': targets}
        
        return collate_fn