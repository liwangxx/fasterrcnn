# 深度学习框架 - 自定义模型指南

模型系统是我们框架的核心组件之一，它允许你灵活地创建和注册自定义深度学习模型。本指南将详细介绍如何创建和使用自定义模型。

## 什么是模型注册系统？

模型注册系统是通过`ModelRegistry`类实现的，它允许你将自定义模型类注册到框架中，使其可以通过配置文件轻松调用。这种设计使得模型的创建和使用更加模块化和灵活。

```python
class ModelRegistry:
    """模型注册器，用于注册和获取模型类"""
    _models = {}
    
    @classmethod
    def register(cls, name):
        """注册模型类"""
        def wrapper(model_class):
            cls._models[name] = model_class
            return model_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取模型类"""
        if name not in cls._models:
            raise ValueError(f"未注册的模型: {name}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls):
        """列出所有已注册的模型"""
        return list(cls._models.keys())
```

## 内置模型类型

目前我们的框架提供了以下内置模型：

1. **CNN** - 一个简单的卷积神经网络模型，适用于图像分类任务

## 通过配置文件使用模型

在配置文件中使用模型是最简单的方式。以CNN为例：

```yaml
# configs/models/cnn.yaml
model:
  type: "cnn"
  in_channels: 3
  num_classes: 10
```

然后在训练配置中引用此模型配置：

```yaml
# configs/training/my_training.yaml
model_config: "configs/models/cnn.yaml"
dataset_config: "configs/datasets/cifar10.yaml"

# ... 其他训练参数 ...
```

## 模型基类

所有模型都应该继承自`BaseModel`类，这是一个继承自PyTorch的`nn.Module`的抽象基类：

```python
class BaseModel(nn.Module):
    """所有模型的基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化"""
        super().__init__()
        self.config = config or {}
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        raise NotImplementedError("子类必须实现forward方法")
    
    def get_loss(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """计算损失"""
        raise NotImplementedError("子类必须实现get_loss方法")
    
    def predict(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """预测（用于推理）"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def load_weights(self, checkpoint_path: str) -> None:
        """加载模型权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            self.load_state_dict(checkpoint['model'])
        else:
            self.load_state_dict(checkpoint)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params
        }
```

## 创建自定义模型

创建自定义模型需要三个步骤：

1. 创建模型类（继承`BaseModel`）
2. 注册模型（使用`ModelRegistry.register`）
3. 创建模型配置文件

### 步骤1：创建模型类

首先，创建一个继承自`BaseModel`的新类。以CNN模型为例：

```python
class CNN(BaseModel):
    """简单的CNN模型"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化"""
        super().__init__(config)
        
        # 从配置中获取参数
        self.in_channels = config.get('in_channels', 1)
        self.num_classes = config.get('num_classes', 10)
        
        # 定义卷积层
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算全连接层的输入特征数量
        # 假设输入图像大小为 32x32
        fc_input_features = 128 * (32 // 8) * (32 // 8)  # 三次下采样后
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_features, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        
        # dropout
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 第一个卷积块
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 第三个卷积块
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(outputs, targets)
```

### 步骤2：注册模型

使用装饰器将你的模型注册到系统中：

```python
@ModelRegistry.register('cnn')
class CNN(BaseModel):
    # ... 类定义 ...
```

### 步骤3：创建模型配置

为你的自定义模型创建配置文件：

```yaml
# configs/models/custom_model.yaml
model:
  type: "custom_model"  # 必须与注册名称匹配
  # 模型特定参数
  in_channels: 3
  num_classes: 10
  hidden_dim: 256
  num_layers: 3
```

## 高级模型功能

在CNN模型实现中，我们还可以看到一些高级功能，例如特征提取和可视化：

### 特征提取

CNN模型实现了`forward_with_features`方法，可以在前向传播的同时保存中间特征：

```python
def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """执行前向传播并保存中间特征"""
    features = {}
    
    # 第一个卷积块
    conv1_output = self.conv1(x)
    features['conv1'] = conv1_output
    
    relu1_output = F.relu(conv1_output)
    features['relu1'] = relu1_output
    
    pool1_output = self.pool(relu1_output)
    features['pool1'] = pool1_output
    
    # ... 其他层的特征提取 ...
    
    return fc2_output, features
```

### 特征可视化

CNN模型还实现了`visualize_features`方法，可以将特征图发送到可视化器进行可视化：

```python
def visualize_features(self, visualizer, input_tensor: torch.Tensor, global_step: int) -> None:
    """使用指定的可视化器可视化模型的特征图"""
    # 确保输入是批次形式
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
        
    # 首先可视化输入图像
    visualizer.add_image("input/original", input_tensor[0], global_step)
    
    # 执行前向传播并获取特征
    with torch.no_grad():
        _, features = self.forward_with_features(input_tensor)
        
    # 针对每个卷积层特征图进行可视化
    for layer_name, feature_map in features.items():
        # 只可视化卷积层输出
        if layer_name.startswith('conv') or layer_name.startswith('pool'):
            # ... 特征图可视化代码 ...
```

## 自定义模型示例

下面是一个自定义模型的完整示例，实现了一个简单的多层感知机（MLP）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

from .base_model import BaseModel
from .registry import ModelRegistry

@ModelRegistry.register('mlp')
class MLP(BaseModel):
    """多层感知机模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 从配置中获取参数
        self.input_dim = config.get('input_dim', 784)  # 默认MNIST输入大小
        self.hidden_dims = config.get('hidden_dims', [512, 256])
        self.num_classes = config.get('num_classes', 10)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        
        # 构建层
        layers = []
        prev_dim = self.input_dim
        
        # 添加隐藏层
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        # 创建网络
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 如果输入是图像，需要先展平
        if len(x.shape) > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
        
        return self.network(x)
    
    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        return F.cross_entropy(outputs, targets)
```

## 实现自己的复杂模型

框架允许你实现更复杂的模型，例如带有多个输入或输出的模型：

```python
@ModelRegistry.register('multi_task_model')
class MultiTaskModel(BaseModel):
    """多任务模型示例"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 共享编码器
        self.encoder = nn.Sequential(
            # ... 编码器层 ...
        )
        
        # 任务特定的头部
        self.classification_head = nn.Linear(512, config.get('num_classes', 10))
        self.regression_head = nn.Linear(512, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播，返回多个任务的输出"""
        # 共享特征提取
        features = self.encoder(x)
        
        # 任务特定预测
        class_output = self.classification_head(features)
        reg_output = self.regression_head(features)
        
        return {
            'classification': class_output,
            'regression': reg_output
        }
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算多个任务的联合损失"""
        # 分类损失
        cls_loss = F.cross_entropy(
            outputs['classification'], 
            targets['classification']
        )
        
        # 回归损失
        reg_loss = F.mse_loss(
            outputs['regression'], 
            targets['regression']
        )
        
        # 合并损失（可以添加权重）
        total_loss = cls_loss + 0.5 * reg_loss
        
        return total_loss
```

## 最佳实践

从现有代码实现来看，实现自定义模型时应遵循以下最佳实践：

1. **继承BaseModel类** - 确保利用框架提供的基础功能
2. **实现必要的方法** - 必须实现`forward`和`get_loss`方法
3. **注册模型** - 使用装饰器注册模型以便配置系统能够识别
4. **从配置获取参数** - 使用`self.config.get('key', default_value)`模式获取参数
5. **模块化设计** - 将模型分解为逻辑组件，如编码器、解码器等
6. **提供合理的默认值** - 为配置参数提供合理的默认值，使模型更易于使用
7. **添加辅助方法** - 实现特征提取、可视化等辅助方法以增强模型功能
8. **注意数值稳定性** - 在实现损失函数时注意数值稳定性
9. **维护模型文档** - 清晰地记录模型的输入和输出格式、参数和使用方式 