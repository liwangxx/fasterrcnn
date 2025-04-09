# 深度学习框架 - 损失函数指南

损失函数系统是框架的核心组件之一，它允许你灵活地创建和配置各种损失函数。本指南将详细介绍如何使用和自定义损失函数。

## 什么是损失函数注册系统？

损失函数注册系统是通过`LossRegistry`类实现的，它允许你将自定义损失函数类注册到框架中，使其可以通过配置文件轻松调用。这种设计使得损失函数的创建和使用更加模块化和灵活。

```python
class LossRegistry:
    """损失函数注册器，用于注册和获取损失函数类"""
    _losses = {}
    
    @classmethod
    def register(cls, name):
        """注册损失函数类"""
        def wrapper(loss_class):
            cls._losses[name] = loss_class
            return loss_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取损失函数类"""
        if name not in cls._losses:
            raise ValueError(f"未注册的损失函数: {name}")
        return cls._losses[name]
```

## 内置损失函数类型

目前我们的框架提供了以下内置损失函数：

1. **CrossEntropyLoss** - 交叉熵损失函数，适用于分类任务
2. **MSELoss** - 均方误差损失函数，适用于回归任务
3. **L1Loss** - L1损失函数（绝对误差损失），适用于回归任务

## 配置损失函数的方式

在我们的框架中，你可以通过两种方式配置损失函数：

### 1. 在模型配置中指定损失函数

这种方式将损失函数视为模型的一部分：

```yaml
# configs/models/cnn.yaml
model:
  type: "cnn"
  in_channels: 3
  num_classes: 10
  loss:
    type: "cross_entropy"
    reduction: "mean"
    label_smoothing: 0.1
    weight: 1.0
```

### 2. 在训练配置中指定独立的损失函数

这种方式将损失函数与模型分离：

```yaml
# configs/training/default.yaml
model_config: "configs/models/cnn.yaml"
dataset_config: "configs/datasets/cifar10.yaml"

# 损失函数配置
loss:
  type: "cross_entropy"
  reduction: "mean"
  label_smoothing: 0.1
  weight: 1.0

training:
  # ... 其他训练参数 ...
```

注意：如果同时在训练配置和模型配置中指定了损失函数，训练配置中的损失函数优先级更高。

## 损失函数基类

所有损失函数都应该继承自`BaseLoss`类，这是一个继承自PyTorch的`nn.Module`的抽象基类：

```python
class BaseLoss(nn.Module):
    """损失函数基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化"""
        super().__init__()
        self.config = config or {}
        self._weight = self.config.get('weight', 1.0)
        
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算损失"""
        raise NotImplementedError("子类必须实现forward方法")
    
    @property
    def weight(self) -> float:
        """获取损失权重"""
        return self._weight
```

## 创建自定义损失函数

创建自定义损失函数需要三个步骤：

1. 创建损失函数类（继承`BaseLoss`）
2. 注册损失函数（使用`LossRegistry.register`）
3. 在配置文件中使用自定义损失函数

### 步骤1：创建损失函数类

首先，创建一个继承自`BaseLoss`的新类：

```python
import torch
import torch.nn as nn
from typing import Dict, Any, Union

from dl_framework.losses.base_loss import BaseLoss
from dl_framework.losses.registry import LossRegistry

class FocalLoss(BaseLoss):
    """Focal Loss损失函数"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化"""
        super().__init__(config)
        
        # 提取配置参数
        self.gamma = self.config.get('gamma', 2.0)
        self.alpha = self.config.get('alpha', 0.25)
        self.reduction = self.config.get('reduction', 'mean')
        
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算Focal Loss"""
        # 处理字典输入
        if isinstance(outputs, dict):
            outputs = outputs.get('logits', next(iter(outputs.values())))
        
        if isinstance(targets, dict):
            targets = targets.get('labels', next(iter(targets.values())))
        
        # 实现Focal Loss计算逻辑
        # ...
```

### 步骤2：注册损失函数

使用装饰器将你的损失函数注册到系统中：

```python
@LossRegistry.register('focal_loss')
class FocalLoss(BaseLoss):
    # ... 类定义 ...
```

### 步骤3：在配置文件中使用自定义损失函数

```yaml
# 在模型配置中使用
model:
  type: "cnn"
  in_channels: 3
  num_classes: 10
  loss:
    type: "focal_loss"
    gamma: 2.0
    alpha: 0.25
    reduction: "mean"
    weight: 1.0

# 或在训练配置中使用
loss:
  type: "focal_loss"
  gamma: 2.0
  alpha: 0.25
  reduction: "mean"
  weight: 1.0
```

## 组合多个损失函数

如果你的任务需要组合多个损失函数，可以创建一个组合损失函数：

```python
@LossRegistry.register('combined_loss')
class CombinedLoss(BaseLoss):
    """组合多个损失函数"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化"""
        super().__init__(config)
        
        # 提取子损失函数配置
        loss_configs = self.config.get('losses', [])
        if not loss_configs:
            raise ValueError("组合损失函数必须指定子损失函数")
        
        # 创建子损失函数
        self.losses = []
        for loss_config in loss_configs:
            loss = LossRegistry.create(loss_config)
            self.losses.append(loss)
    
    def forward(self, 
                outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
               ) -> torch.Tensor:
        """计算组合损失"""
        total_loss = 0.0
        for loss in self.losses:
            total_loss += loss(outputs, targets) * loss.weight
        
        return total_loss * self.weight
```

然后在配置中使用：

```yaml
loss:
  type: "combined_loss"
  weight: 1.0
  losses:
    - type: "cross_entropy"
      weight: 1.0
    - type: "l1"
      weight: 0.5
```

## 总结

通过损失函数注册系统，你可以：

1. 轻松地配置和使用不同类型的损失函数
2. 创建自定义损失函数并注册到系统中
3. 组合多个损失函数用于复杂任务
4. 在模型配置或训练配置中灵活指定损失函数 