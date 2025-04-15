# VOC数据集与Faster R-CNN训练指南

本指南将介绍如何使用我们的深度学习框架来训练Faster R-CNN模型，以实现对VOC数据集的目标检测。

## 数据集准备

1. 下载VOC2012数据集，将其解压到`data/PASCAL_VOC`目录下。目录结构应如下：
   ```
   data/PASCAL_VOC/
   ├── VOCdevkit/
   │   └── VOC2012/
   │       ├── Annotations/
   │       ├── ImageSets/
   │       ├── JPEGImages/
   │       └── ...
   ```

2. 数据集已经通过`VOC2012Dataset`类实现并注册到框架中，无需额外修改代码。该实现支持各种数据增强功能，包括水平翻转、随机缩放、随机裁剪和颜色抖动。

## 模型配置

我们提供了预配置的Faster R-CNN模型，基于ResNet-50骨干网络。配置文件位于：
- `configs/models/fasterrcnn.yaml`：模型配置
- `configs/datasets/voc2012.yaml`：数据集配置
- `configs/training/fasterrcnn_voc.yaml`：训练配置

## 训练模型

使用以下命令开始训练：

```bash
python tools/train.py --config configs/training/fasterrcnn_voc.yaml
```

训练过程将使用以下配置：
- 30个epochs的训练
- SGD优化器，初始学习率为0.005
- 每10个epochs学习率衰减为原来的0.1
- 早停机制：如果5个epoch内验证损失没有改善，则停止训练
- 每个epoch自动保存检查点，并保留最近的3个检查点

## 可视化与监控

训练过程中，系统会自动使用以下钩子进行监控：

1. `TimeTrackingHook`：记录每个epoch的训练时间
2. `DetectionVisualizationHook`：每100步可视化模型的检测结果
3. `SystemMonitorHook`：监控系统资源使用情况

可视化结果将保存在`outputs/visualizations`目录下。

## 评估模型

每个epoch结束后，系统会自动在验证集上评估模型性能。主要指标是mAP（平均精度均值），这是目标检测领域常用的评估指标。

## 模型结构

使用的Faster R-CNN模型具有以下特点：
- 使用ResNet-50作为骨干网络，带有特征金字塔网络(FPN)
- RPN（区域建议网络）配置了多尺度anchor
- 两阶段检测架构：先产生区域建议，再进行分类和边界框回归

## 预测与推理

训练完成后，可以使用以下代码进行预测：

```python
from dl_framework.models import ModelRegistry

# 加载模型
model_config = {...}  # 从配置文件加载
model = ModelRegistry.get('fasterrcnn')(model_config)
model.load_weights('path/to/checkpoint.pth')

# 进行预测
image = ...  # 预处理后的图像
predictions = model.predict(image)

# 处理预测结果
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
labels = predictions[0]['labels']
```

## 自定义参数

如果需要调整模型参数，可以修改相应的配置文件：

1. 修改`configs/models/fasterrcnn.yaml`来调整模型结构
2. 修改`configs/datasets/voc2012.yaml`来调整数据集参数
3. 修改`configs/training/fasterrcnn_voc.yaml`来调整训练参数

## 常见问题

1. **内存不足**：减小batch size或图像尺寸
2. **训练速度慢**：减少可训练的backbone层数或使用更小的模型
3. **过拟合**：增加数据增强强度或调整正则化参数

## 后续改进

可以考虑以下改进方向：
1. 加入更多数据增强方法，如Mosaic、MixUp等
2. 尝试不同的骨干网络，如ResNet-101、EfficientNet等
3. 使用其他目标检测模型，如YOLOv5、RetinaNet等 