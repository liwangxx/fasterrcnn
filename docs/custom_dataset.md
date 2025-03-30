# 深度学习框架 - 自定义数据集指南

数据集系统是我们框架的核心组件之一，允许你灵活地创建和使用自定义数据集，无需修改核心训练代码。本指南将详细介绍如何创建和使用自定义数据集。

## 什么是数据集注册系统？

数据集注册系统是通过`DatasetRegistry`类实现的，它允许你将自定义数据集类注册到框架中，使其可以通过配置文件轻松调用。这种设计使得数据集的创建和使用更加模块化和灵活。

```python
class DatasetRegistry:
    """数据集注册器，用于注册和获取数据集类"""
    _datasets = {}
    
    @classmethod
    def register(cls, name):
        """注册数据集类"""
        def wrapper(dataset_class):
            cls._datasets[name] = dataset_class
            return dataset_class
        return wrapper
    
    @classmethod
    def get(cls, name):
        """获取数据集类"""
        if name not in cls._datasets:
            raise ValueError(f"未注册的数据集: {name}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls):
        """列出所有已注册的数据集"""
        return list(cls._datasets.keys())
```

## 内置数据集类型

目前我们的框架提供了以下内置数据集：

1. **CIFAR10Dataset** - CIFAR-10图像分类数据集

## 通过配置文件使用数据集

在配置文件中使用数据集是最简单的方式。以CIFAR-10为例：

```yaml
# configs/datasets/cifar10.yaml
dataset:
  type: "cifar10"
  data_dir: "data/cifar10"
  batch_size: 64
  num_workers: 4
  transforms:
    resize: [32, 32]
    normalize:
      mean: [0.4914, 0.4822, 0.4465]
      std: [0.2023, 0.1994, 0.2010]
```

然后在训练配置中引用此数据集配置：

```yaml
# configs/training/my_training.yaml
model_config: "configs/models/cnn.yaml"
dataset_config: "configs/datasets/cifar10.yaml"  # 引用数据集配置

# ... 其他训练参数 ...
```

## 数据集基类

所有数据集都应该继承自`BaseDataset`类，这是一个继承自PyTorch的`Dataset`的抽象基类：

```python
class BaseDataset(Dataset):
    """所有数据集的基类"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化
        
        Args:
            config: 数据集配置
            is_training: 是否为训练集
        """
        super().__init__()
        self.config = config
        self.is_training = is_training
        self.data_dir = self._get_data_dir()
        self.transform = self._get_transforms()
    
    def _get_data_dir(self) -> str:
        """获取数据目录"""
        data_dir = self.config.get('data_dir', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        return data_dir
    
    def _get_transforms(self) -> Optional[Any]:
        """获取数据变换"""
        # 子类应实现此方法来提供适当的变换
        return None
    
    def _load_data(self) -> None:
        """加载数据"""
        raise NotImplementedError("子类必须实现_load_data方法")
    
    def __len__(self) -> int:
        """获取数据集长度"""
        raise NotImplementedError("子类必须实现__len__方法")
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """获取数据项"""
        raise NotImplementedError("子类必须实现__getitem__方法")
    
    def get_collate_fn(self):
        """获取数据整理函数"""
        # 默认返回None，使用PyTorch默认的collate_fn
        return None
```

## 创建自定义数据集

创建自定义数据集需要三个步骤：

1. 创建数据集类（继承`BaseDataset`）
2. 注册数据集（使用`DatasetRegistry.register`）
3. 创建数据集配置文件

### 步骤1：创建数据集类

首先，创建一个继承自`BaseDataset`的新类。以CIFAR10Dataset为例：

```python
class CIFAR10Dataset(BaseDataset):
    """CIFAR-10数据集"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """初始化"""
        super().__init__(config, is_training)
        self.transform = self._get_transforms()
        self._load_data()
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换"""
        transform_config = self.config.get('transforms', {})
        
        transform_list = []
        
        # 调整大小
        if 'resize' in transform_config:
            size = transform_config['resize']
            transform_list.append(transforms.Resize(size))
        
        # 如果是训练集，添加数据增强
        if self.is_training:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
            ])
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 标准化
        if 'normalize' in transform_config:
            mean = transform_config['normalize'].get('mean', [0.5, 0.5, 0.5])
            std = transform_config['normalize'].get('std', [0.5, 0.5, 0.5])
            transform_list.append(transforms.Normalize(mean, std))
        
        return transforms.Compose(transform_list)
    
    def _load_data(self) -> None:
        """加载数据"""
        # 下载CIFAR-10数据集（如果需要）
        self.dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=self.is_training,
            download=True,
            transform=self.transform
        )
    
    def __len__(self) -> int:
        """获取数据集长度"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据项"""
        image, label = self.dataset[idx]
        return image, label
```

### 步骤2：注册数据集

使用装饰器将你的数据集注册到系统中：

```python
@DatasetRegistry.register('cifar10')
class CIFAR10Dataset(BaseDataset):
    # ... 类定义 ...
```

### 步骤3：创建数据集配置

为你的自定义数据集创建配置文件：

```yaml
# configs/datasets/custom_dataset.yaml
dataset:
  type: "custom_dataset"  # 必须与注册名称匹配
  data_dir: "data/custom_dataset"
  batch_size: 32
  num_workers: 4
  transforms:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

## 自定义数据集示例

下面是一个自定义图像分类数据集的完整示例：

```python
import os
import torch
from PIL import Image
from typing import Dict, Any, Tuple

from .base_dataset import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register('custom_image')
class CustomImageDataset(BaseDataset):
    """自定义图像数据集"""
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        super().__init__(config, is_training)
        self.transform = self._get_transforms()
        self._load_data()
        
    def _get_transforms(self):
        """获取数据变换"""
        # 类似CIFAR10Dataset实现，根据配置构建transform
        transform_config = self.config.get('transforms', {})
        # ... 实现变换逻辑 ...
        
    def _load_data(self):
        """加载数据集"""
        # 实现数据加载逻辑
        image_dir = os.path.join(self.data_dir, 'images')
        label_file = os.path.join(self.data_dir, 'labels.txt')
        
        self.images = []
        self.labels = []
        
        # 读取标签文件
        with open(label_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split(',')
                image_path = os.path.join(image_dir, image_name)
                
                if os.path.exists(image_path):
                    self.images.append(image_path)
                    self.labels.append(int(label))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx: int):
        """获取数据项"""
        # 读取图像
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        
        return image, label
```

## 数据变换

从CIFAR10Dataset的实现中，我们可以看到数据变换是通过`_get_transforms`方法实现的，它支持以下变换：

- `resize`: 调整图像大小
- `normalize`: 标准化

同时对于训练集，还会自动添加以下数据增强：
- `RandomHorizontalFlip`: 随机水平翻转
- `RandomAffine`: 随机仿射变换

## 最佳实践

从现有代码实现来看，实现自定义数据集时应遵循以下最佳实践：

1. **继承BaseDataset类** - 确保利用框架提供的基础功能
2. **实现必要的方法** - 必须实现`_load_data`、`__len__`和`__getitem__`方法
3. **注册数据集** - 使用装饰器注册数据集以便配置系统能够识别
4. **从配置获取参数** - 使用`self.config.get('key', default_value)`模式获取参数
5. **检查文件存在性** - 加载数据前检查文件是否存在，如`os.path.exists(image_path)`
6. **区分训练和测试** - 根据`self.is_training`标志适当地处理数据和变换
7. **提供合理的数据变换** - 针对不同任务提供合适的数据预处理和增强
8. **处理异常情况** - 对于可能的错误情况（如找不到文件）提供合理的处理 