# 深度学习框架 - 使用指南

本文档介绍DL-Framework的基本使用方法，包括训练模型、配置系统和常见操作。

## 基本用法

### 训练模型

使用默认配置训练模型：

```bash
python tools/train.py --config configs/training/default.yaml
```

使用TensorBoard可视化（必须使用`--vis`参数开启）：

```bash
python tools/train.py --config configs/training/default.yaml --vis configs/visualization/tensorboard.yaml
```

指定GPU设备：

```bash
python tools/train.py --config configs/training/default.yaml --device cuda:0
```

恢复训练：

```bash
python tools/train.py --config configs/training/default.yaml --resume checkpoints/my_experiment/latest.pth
```

## 配置系统

框架使用YAML文件进行配置，支持模块化组合不同的配置文件。

### 配置示例

#### 模型配置

```yaml
# configs/models/cnn.yaml
model:
  type: "cnn"
  in_channels: 3
  num_classes: 10
```

#### 数据集配置

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

#### 训练配置

```yaml
# configs/training/default.yaml
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
  checkpoint:
    save_frequency: 5  # 每5个epoch保存一次检查点
    keep_num: 3  # 保留最近的3个检查点
```

#### 可视化配置

```yaml
# configs/visualization/tensorboard.yaml
visualization:
  tensorboard:
    enabled: true
    flush_secs: 30
    save_figures: true

hooks:
  - type: "GradientFlowHook"
    name: "grad_flow"
    frequency: 100
    targets: ["conv1", "conv2", "conv3", "fc1", "fc2"]
```

## 命令行参数

训练脚本支持以下命令行参数：

- `--config`: 训练配置文件路径（必需）
- `--vis`: 可视化配置文件路径（可选，用于启用可视化）
- `--device`: 训练设备，如 "cuda:0"、"cpu"（可选）
- `--resume`: 恢复训练的检查点路径（可选）
- `--debug`: 启用调试模式（可选）

示例：

```bash
python tools/train.py --config configs/training/default.yaml --vis configs/visualization/tensorboard.yaml --device cuda:0
```

## 目录结构说明

训练过程中生成的文件会按以下结构组织：

```
experiments/<experiment_name>/
├── checkpoints/          # 模型检查点
│   ├── best.pth          # 最佳模型
│   ├── latest.pth        # 最新模型
│   └── epoch_<N>.pth     # 特定epoch的模型
├── logs/                 # 日志文件
│   ├── training.log      # 训练日志
│   └── tensorboard/      # TensorBoard日志
└── samples/              # 生成的样本或可视化结果
```

## 常见问题

### 内存不足怎么办？

- 减小批次大小 (`batch_size`)
- 减少数据加载工作进程 (`num_workers`)
- 考虑使用梯度累积技术

### 如何加速训练？

- 确保使用GPU训练 (`--device cuda:0`)
- 增加数据加载工作进程数 (`num_workers`)
- 使用混合精度训练
- 考虑使用更高效的优化器，如AdamW

### 如何防止过拟合？

- 增加正则化 (如 `weight_decay`)
- 增加Dropout率
- 使用数据增强
- 实现早停机制 (`early_stopping`)

## 后续步骤

了解更多关于框架的高级用法，请参阅以下文档：

- [自定义模型教程](custom_model.md)
- [自定义数据集教程](custom_dataset.md)
- [TensorBoard可视化](tensorboard_visualization.md)
- [钩子系统使用指南](hooks_usage.md) 