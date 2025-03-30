# 深度学习框架 - TensorBoard可视化指南

可视化是深度学习实验中至关重要的一环，它帮助我们理解模型的训练过程、分析模型的行为以及调试潜在问题。本指南将详细介绍如何在我们的框架中使用TensorBoard进行可视化。

## 什么是可视化系统？

可视化系统是通过`BaseVisualizer`类及其子类实现的，它提供了一套统一的接口，允许用户在训练过程中记录和查看各种数据，如损失、准确率、特征图、参数分布等。TensorBoard是目前实现的主要可视化工具。

```python
class BaseVisualizer:
    """可视化器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化"""
        self.global_config = config
        self.config = config.get('visualization', {})
        self.experiment_dir = config.get('experiment_dir', 'experiments/default')
        # ...
    
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """添加标量"""
        pass
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int) -> None:
        """添加多个标量"""
        pass
    
    def add_image(self, tag: str, img_tensor: torch.Tensor, global_step: int) -> None:
        """添加图像"""
        pass
    
    # ... 其他方法 ...
```

## TensorBoard可视化器

TensorBoard是一个由TensorFlow团队开发的可视化工具，我们的框架通过`TensorBoardVisualizer`类提供对TensorBoard的支持：

```python
@VisualizerRegistry.register("tensorboard")
class TensorBoardVisualizer(BaseVisualizer):
    """TensorBoard可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化"""
        super().__init__(config)
        
        if self.visualizer_config:
            self.flush_secs = self.visualizer_config.get('flush_secs', 30)
            
            # 创建SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.vis_dir,
                flush_secs=self.flush_secs
            )
    
    # ... 实现各种可视化方法 ...
```

## 配置和使用TensorBoard

### 配置文件示例

创建TensorBoard的可视化配置文件：

```yaml
# configs/visualization/tensorboard.yaml
visualization:
  tensorboard:
    flush_secs: 30  # 多久将数据写入磁盘，默认30秒
    save_figures: true  # 是否保存matplotlib图表到文件
```

### 命令行使用

**重要提示：** 在我们的框架中，可视化功能**只能**通过在训练命令中添加`--vis`参数来开启，而不是在训练配置文件中引用可视化配置。

使用TensorBoard可视化的训练命令：

```bash
python tools/train.py --config configs/training/my_training.yaml --vis configs/visualization/tensorboard.yaml
```
训练开始后，启动TensorBoard查看可视化：

```bash
tensorboard --logdir experiments/my_experiment/logs/tensorboard
```

## 可视化内容

TensorBoard可视化器支持以下类型的可视化：

### 1. 标量可视化

记录训练和验证过程中的标量值，如损失、准确率等：

```python
visualizer.add_scalar('loss/train', train_loss, global_step)
visualizer.add_scalar('accuracy/train', train_accuracy, global_step)
```

### 2. 多标量比较

同时记录多个相关的标量，便于比较：

```python
visualizer.add_scalars('metrics', {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_accuracy,
    'val_acc': val_accuracy
}, global_step)
```

### 3. 图像可视化

可视化模型输入、输出或中间特征图：

```python
visualizer.add_image('input/sample', input_image, global_step)
```

### 4. 特征图网格

以网格形式显示多个特征图：

```python
visualizer.add_images_grid('features/conv1', features, global_step, nrow=8)
```

### 5. 直方图

可视化模型参数或特征分布：

```python
visualizer.add_histogram('weights/conv1', model.conv1.weight, global_step)
```

### 6. 图表

添加matplotlib图表：

```python
fig, ax = plt.subplots()
ax.plot(x, y)
visualizer.add_figure('custom_plot', fig, global_step)
```

### 7. 模型图

可视化模型架构：

```python
visualizer.add_graph(model, sample_input)
```

## 结合Hook系统使用TensorBoard

框架的Hook系统与可视化系统紧密集成，提供了更强大的可视化能力。以下是主要的可视化相关Hook：

### 1. 特征图可视化（FeatureMapHook）

自动捕获并可视化模型中间层的特征图：

```yaml
hooks:
  - type: "FeatureMapHook"
    name: "feature_visualizer"
    frequency: 100  # 每100步可视化一次
    max_features: 16  # 每层最多显示的特征数量
    targets: ["conv1", "conv2", "conv3"]  # 要可视化的层名称
```

### 2. 梯度流可视化（GradientFlowHook）

可视化模型各层梯度的变化，帮助诊断梯度消失或爆炸问题：

```yaml
hooks:
  - type: "GradientFlowHook"
    name: "grad_flow"
    frequency: 100  # 每100步可视化一次
    targets: ["conv1", "conv2", "conv3", "fc1", "fc2"]  # 要跟踪的层
```

### 3. 时间跟踪（TimeTrackingHook）

记录训练时间并预测完成时间：

```yaml
hooks:
  - type: "TimeTrackingHook"
    name: "timer"
    frequency: 1  # 每个epoch记录一次
```

## 完整配置示例

下面是一个完整的训练配置和相应的TensorBoard可视化配置：

### 训练配置文件

```yaml
# configs/training/visualization_example.yaml
experiment_dir: "experiments/visualization_demo"
model_config: "configs/models/cnn.yaml"
dataset_config: "configs/datasets/cifar10.yaml"

device: "cuda"
seed: 42

training:
  epochs: 50
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 1e-5

hooks:
  - type: "FeatureMapHook"
    name: "feature_maps"
    frequency: 100
    max_features: 16
    targets: ["conv1", "conv2", "conv3"]
    
  - type: "GradientFlowHook"
    name: "grad_flow"
    frequency: 200
    targets: ["conv1", "conv2", "conv3", "fc1", "fc2"]
    
  - type: "TimeTrackingHook"
    name: "timer"
    frequency: 1
```

### 可视化配置文件

```yaml
# configs/visualization/tensorboard.yaml
visualization:
  tensorboard:
    enabled: true
    flush_secs: 30
    save_figures: true
```

### 启动训练并开启可视化

```bash
python tools/train.py --config configs/training/visualization_example.yaml --vis configs/visualization/tensorboard.yaml
```

## 查看TensorBoard结果

启动TensorBoard服务器查看可视化结果：

```bash
tensorboard --logdir experiments/my_experiment/logs/tensorboard
```

然后在浏览器中访问：`http://localhost:6006`

## 最佳实践

1. **始终使用--vis参数** - 确保使用`--vis`命令行参数启用可视化功能
2. **有组织的标签命名** - 使用分层结构命名标签，如 'loss/train', 'loss/val'
3. **控制可视化频率** - 对于较大的数据（如特征图），降低可视化频率
4. **选择性可视化** - 只可视化对分析有用的内容，避免过多数据
5. **利用Hook系统** - 使用Hook自动捕获和可视化数据，减少手动代码
6. **保存重要图表** - 对于关键分析，启用`save_figures`选项保存到文件
7. **管理日志目录** - 为不同实验使用不同的日志目录，便于比较
8. **组合多种可视化** - 结合使用标量、图像、直方图等不同类型的可视化 