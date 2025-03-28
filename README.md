# DL-Framework - 深度学习训练框架

一个用于深度学习实验和研究的轻量级PyTorch训练框架。该框架提供了一套完整的工具，用于构建、训练和评估深度学习模型，同时支持模型和数据集的注册机制以及可视化功能。

## 特点

- 模块化设计，易于扩展和定制
- 模型和数据集的注册和检索系统
- 灵活的配置系统，支持YAML配置
- 内置可视化功能，支持TensorBoard
- 梯度流等监控工具
- 自动保存检查点和恢复训练
- 支持早停和学习率调度

## 目录结构

```
dl_framework/
├── models/              # 模型定义
│   ├── registry.py      # 模型注册系统
│   ├── base_model.py    # 模型基类
│   └── cnn.py           # CNN模型实现
├── datasets/            # 数据集定义
│   ├── registry.py      # 数据集注册系统
│   ├── base_dataset.py  # 数据集基类
│   └── cifar10.py       # CIFAR-10数据集实现
├── trainers/            # 训练器
│   └── base_trainer.py  # 训练器基类
├── visualization/       # 可视化工具
│   ├── base_visualizer.py  # 可视化器基类
│   ├── tensorboard.py      # TensorBoard可视化器
│   └── hooks/              # 可视化钩子
│       ├── base_hook.py    # 钩子基类
│       └── grad_flow.py    # 梯度流可视化钩子
└── utils/               # 工具函数
    ├── logger.py        # 日志工具
    ├── checkpoint.py    # 检查点工具
    └── config.py        # 配置工具

configs/              # 配置文件
├── models/           # 模型配置
│   └── cnn.yaml      # CNN模型配置
├── datasets/         # 数据集配置
│   └── cifar10.yaml  # CIFAR-10数据集配置
├── training/         # 训练配置
│   └── default.yaml  # 默认训练配置
└── visualization/    # 可视化配置
    └── tensorboard.yaml  # TensorBoard配置

tools/                # 命令行工具
└── train.py          # 训练入口脚本

checkpoints/          # 检查点目录
logs/                 # 日志目录
```

## 安装

### 要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- pyyaml
- matplotlib
- tensorboard

### 步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/dl-framework.git
cd dl-framework
```

2. 安装uv和同步依赖：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## 使用方法

### 训练模型

使用默认配置训练模型：

```bash
python tools/train.py --config configs/training/default.yaml
```

使用TensorBoard可视化：

```bash
python tools/train.py --config configs/training/default.yaml --vis configs/visualization/tensorboard.yaml
```

指定GPU：

```bash
python tools/train.py --config configs/training/default.yaml --device cuda:0
```

### 创建自定义模型

1. 创建模型类：

```python
# dl_framework/models/your_model.py
import torch.nn as nn
from .base_model import BaseModel
from .registry import ModelRegistry

@ModelRegistry.register('your_model')
class YourModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化模型层
        
    def forward(self, x):
        # 实现前向传播
        return x
        
    def get_loss(self, outputs, targets):
        # 实现损失计算
        return loss
```

2. 创建模型配置：

```yaml
# configs/models/your_model.yaml
model:
  type: "your_model"
  # 其他模型参数
```

3. 在训练配置中使用：

```yaml
# configs/training/your_model_training.yaml
model_config: "configs/models/your_model.yaml"
dataset_config: "configs/datasets/cifar10.yaml"
# 其他训练参数
```

### 创建自定义数据集

1. 创建数据集类：

```python
# dl_framework/datasets/your_dataset.py
from .base_dataset import BaseDataset
from .registry import DatasetRegistry

@DatasetRegistry.register('your_dataset')
class YourDataset(BaseDataset):
    def __init__(self, config, is_training=True):
        super().__init__(config, is_training)
        self._load_data()
        
    def _load_data(self):
        # 加载数据
        
    def __len__(self):
        # 返回数据集长度
        
    def __getitem__(self, idx):
        # 返回数据项
```

2. 创建数据集配置：

```yaml
# configs/datasets/your_dataset.yaml
dataset:
  type: "your_dataset"
  # 其他数据集参数
```

## 配置示例

### 模型配置

```yaml
model:
  type: "cnn"
  in_channels: 3
  num_classes: 10
```

### 数据集配置

```yaml
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

### 训练配置

```yaml
model_config: "configs/models/cnn.yaml"
dataset_config: "configs/datasets/cifar10.yaml"

seed: 42
device: "cuda"
output_dir: "checkpoints/default"
log_dir: "logs/default"

training:
  epochs: 50
  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 1e-5
  scheduler:
    type: "cosine"
    T_max: 50
    eta_min: 0.0001
  early_stopping:
    patience: 10
    min_delta: 0.001
```

### 可视化配置

```yaml
tensorboard:
  enabled: true
  log_dir: "logs/tensorboard"
  flush_secs: 30

hooks:
  - name: "grad_flow"
    type: "GradientFlowHook"
    frequency: 100
    targets: ["conv1", "conv2", "conv3", "fc1", "fc2"]
```

## 许可证

MIT
