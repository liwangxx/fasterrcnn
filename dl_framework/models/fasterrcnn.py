import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, Any, Tuple, List

from .base_model import BaseModel
from .registry import ModelRegistry

@ModelRegistry.register('fasterrcnn')
class FasterRCNNModel(BaseModel):
    """Faster R-CNN目标检测模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Faster R-CNN模型
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        self.config = config
        
        # 初始化模型
        self._build_model()
    
    def _build_model(self):
        """构建Faster R-CNN模型"""
        config = self.config
        
        # 获取类别数量
        num_classes = config.get('num_classes', 21)  # 默认为COCO数据集的91类或VOC的21类
        
        # 获取backbone类型
        backbone_name = config.get('backbone', 'resnet50')
        pretrained = config.get('pretrained', True)
        trainable_backbone_layers = config.get('trainable_backbone_layers', 5)
        
        # 创建backbone
        if backbone_name == 'resnet50':
            backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
                'resnet50', 
                pretrained=pretrained,
                trainable_layers=trainable_backbone_layers
            )
        else:
            raise ValueError(f"不支持的backbone: {backbone_name}")
        
        # 获取RPN配置
        rpn_config = config.get('rpn', {})
        anchor_sizes = rpn_config.get('anchor_sizes', ((32,), (64,), (128,), (256,), (512,)))
        aspect_ratios = rpn_config.get('aspect_ratios', ((0.5, 1.0, 2.0),) * len(anchor_sizes))
        
        # 创建AnchorGenerator
        if isinstance(anchor_sizes[0], int):
            # 如果传入的是简单列表，转换为元组格式
            anchor_sizes = tuple((size,) for size in anchor_sizes)
        
        anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=(aspect_ratios,) * len(anchor_sizes) if isinstance(aspect_ratios, (list, tuple)) and not isinstance(aspect_ratios[0], (list, tuple)) else aspect_ratios
        )
        
        # RPN参数
        rpn_pre_nms_top_n_train = rpn_config.get('pre_nms_top_n_train', 2000)
        rpn_pre_nms_top_n_test = rpn_config.get('pre_nms_top_n_test', 1000)
        rpn_post_nms_top_n_train = rpn_config.get('post_nms_top_n_train', 2000)
        rpn_post_nms_top_n_test = rpn_config.get('post_nms_top_n_test', 1000)
        rpn_nms_thresh = rpn_config.get('nms_thresh', 0.7)
        rpn_fg_iou_thresh = rpn_config.get('fg_iou_thresh', 0.7)
        rpn_bg_iou_thresh = rpn_config.get('bg_iou_thresh', 0.3)
        
        # ROI Heads参数
        roi_config = config.get('roi_heads', {})
        box_fg_iou_thresh = roi_config.get('fg_iou_thresh', 0.5)
        box_bg_iou_thresh = roi_config.get('bg_iou_thresh', 0.5)
        box_batch_size_per_image = roi_config.get('batch_size_per_image', 512)
        box_positive_fraction = roi_config.get('positive_fraction', 0.25)
        box_score_thresh = roi_config.get('score_thresh', 0.05)
        box_nms_thresh = roi_config.get('nms_thresh', 0.5)
        box_detections_per_img = roi_config.get('detections_per_img', 100)
        
        # 创建Faster R-CNN模型
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,  # 先设为None，后面会替换掉预测头
            rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
            rpn_nms_thresh=rpn_nms_thresh,
            rpn_fg_iou_thresh=rpn_fg_iou_thresh,
            rpn_bg_iou_thresh=rpn_bg_iou_thresh,
            box_fg_iou_thresh=box_fg_iou_thresh,
            box_bg_iou_thresh=box_bg_iou_thresh,
            box_batch_size_per_image=box_batch_size_per_image,
            box_positive_fraction=box_positive_fraction,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )
        
       
        
        # 打印模型结构摘要
        print(f"Faster R-CNN模型已创建，backbone: {backbone_name}, 类别数: {num_classes}")
    
    def forward(self, inputs):
        """前向传播
        
        Args:
            inputs: 输入数据
            
        Returns:
            模型输出
        """
        # 训练模式下，inputs是一个字典，包含images和targets
        # 测试模式下，inputs只包含images
        if isinstance(inputs, dict):
            images = inputs.get('images')
            targets = inputs.get('targets', None)
        else:
            # 如果输入不是字典，假设它是图像张量列表
            images = inputs
            targets = None
            
        # 确保图像是列表，如果不是，转换为列表
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            images = [img for img in images]
            
        # 检查device
        device = next(self.model.parameters()).device
        images = [img.to(device) for img in images]
        
        if self.training and targets is not None:
            # 将targets转移到正确的设备
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in t.items()} for t in targets]
            
            losses = self.model(images, targets)
            return losses
        else:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(images)
            return predictions
    
    def predict(self, inputs):
        """进行预测
        
        Args:
            inputs: 输入数据
            
        Returns:
            预测结果
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            return self.forward(inputs)
    
    def get_loss(self, outputs, targets=None):
        """获取损失
        
        Note: Faster R-CNN模型在forward中直接计算损失，因此这个方法不需要额外计算
        
        Args:
            outputs: 模型输出
            targets: 目标值（对于Faster R-CNN，这在forward中已经使用）
            
        Returns:
            损失值字典
        """
        # 在训练模式下，outputs就是损失
        if self.training:
            return outputs
        else:
            # 在评估模式下，不计算损失
            return {'loss': torch.tensor(0.0, device=self.device)}
    
    def load_weights(self, weights_path):
        """加载预训练权重
        
        Args:
            weights_path: 权重文件路径
        """
        print(f"加载权重: {weights_path}")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
    def save_weights(self, weights_path):
        """保存模型权重
        
        Args:
            weights_path: 保存路径
        """
        print(f"保存权重: {weights_path}")
        torch.save(self.model.state_dict(), weights_path)
        
    @property
    def device(self):
        """获取模型设备"""
        return next(self.model.parameters()).device 