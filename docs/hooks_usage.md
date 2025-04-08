# 深度学习框架 - Hook系统使用指南

Hook系统是我们框架中一个强大而灵活的功能，允许你在训练过程的不同阶段注入自定义行为，而无需修改核心训练代码。

## 什么是Hook？

Hook（钩子）是在训练过程的特定时刻执行的回调函数。我们的框架提供了以下Hook点：

- `before_training`: 在训练开始前调用
- `before_epoch`: 在每个epoch开始前调用
- `before_step`: 在每步（batch）开始前调用
- `after_step`: 在每步结束后调用
- `after_epoch`: 在每个epoch结束后调用
- `after_training`: 在训练结束后调用
- `cleanup`: 用于清理资源

## 内置Hook类型

我们的框架提供了以下内置Hook：

1. **FeatureMapHook** - 可视化CNN模型的特征图
2. **GradientFlowHook** - 跟踪和可视化梯度流
3. **TimeTrackingHook** - 跟踪训练时间和预测完成时间
4. **SystemMonitorHook** - 监控CPU、内存和GPU资源使用情况

## 通过配置文件使用Hook

最简单的使用Hook的方式是在配置文件中进行设置。例如：

```yaml
# config.yaml
experiment_dir: "experiments/my_experiment"

model:
  type: "cnn"
  # ... 模型配置 ...

dataset:
  type: "cifar10"
  # ... 数据集配置 ...

training:
  epochs: 50
  # ... 训练配置 ...
  
# 这里定义Hook
hooks:
  - type: "TimeTrackingHook"
    name: "training_timer"
    frequency: 1  # 每个epoch都要触发
    log_to_file: true
    log_path: "time_tracking.log"
    
  - type: "FeatureMapHook"
    name: "feature_visualizer"
    frequency: 100  # 每100步可视化一次
    max_features: 16
    sample_batch_idx: 0
    
  - type: "SystemMonitorHook"
    name: "system_monitor"
    frequency: 10  # 每10步记录一次系统资源使用情况
    interval: 2  # 采样间隔（秒）
    track_cpu: true  # 是否监控CPU使用率
    track_memory: true  # 是否监控内存使用率
    track_gpu: true  # 是否监控GPU使用率
    track_gpu_memory: true  # 是否监控GPU内存使用率
```

然后在代码中加载配置和创建训练器：

```python
from dl_framework.trainers import BaseTrainer
from dl_framework.utils import load_config

# 加载配置
config = load_config("config.yaml")

# 创建训练器（会自动从配置中构建Hook）
trainer = BaseTrainer(config)

# 开始训练
trainer.train()
```

## 在代码中手动注册Hook

你也可以在代码中手动创建和注册Hook：

```python
from dl_framework.trainers import BaseTrainer
from dl_framework.hooks import TimeTrackingHook, FeatureMapHook
from dl_framework.utils import load_config

# 加载配置
config = load_config("config.yaml")

# 创建训练器
trainer = BaseTrainer(config)

# 创建钩子
time_hook = TimeTrackingHook({
    'frequency': 1,
    'log_to_file': True,
    'log_path': 'custom_time_tracking.log'
})

# 注册钩子
trainer.register_hook(time_hook)

# 开始训练
trainer.train()
```

## 自定义Hook示例

创建自定义Hook非常简单。只需继承`BaseHook`类并实现你需要的方法：

```python
from typing import Dict, Any
import torch
import torch.nn as nn

from dl_framework.hooks import BaseHook
from dl_framework.hooks.registry import HookRegistry

@HookRegistry.register("MyCustomHook")
class MyCustomHook(BaseHook):
    """我的自定义钩子"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # 初始化你的钩子
        
    def before_training(self, model: nn.Module) -> None:
        logger = self.get_service("logger")
        if logger:
            logger.info("训练即将开始！")
    
    def after_epoch(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> None:
        # 获取服务
        logger = self.get_service("logger")
        visualizer = self.get_service("visualizer")
        
        # 实现你的逻辑
        if logger:
            logger.info(f"完成第 {epoch+1} 个epoch，指标: {metrics}")
        
        # 可选地使用可视化器
        if visualizer:
            visualizer.add_scalar('custom/metric', 0.5, epoch)
```

## 服务注入

我们的Hook系统支持依赖注入模式，这意味着Hook可以访问Trainer提供的各种"服务"。默认可用的服务包括：

- `trainer` - 训练器实例
- `config` - 全局配置
- `model` - 模型实例
- `train_loader` - 训练数据加载器
- `val_loader` - 验证数据加载器
- `optimizer` - 优化器
- `scheduler` - 学习率调度器
- `device` - 训练设备
- `logger` - 日志记录器
- `experiment_dir` - 实验目录
- `checkpoints_dir` - 检查点目录
- `logs_dir` - 日志目录
- `visualization_dir` - 可视化输出目录
- `visualizer` - 可视化器（如果配置了）

通过`self.get_service()`方法可以获取这些服务：

```python
def after_epoch(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> None:
    logger = self.get_service("logger")
    if logger:
        logger.info("Epoch完成")
```

## 注册自定义服务

你还可以注册自己的自定义服务，供Hook使用：

```python
# 创建一个自定义服务
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# 将其注册到训练器
trainer.register_service("mlflow", mlflow)

# 然后在钩子中使用
class MLFlowHook(BaseHook):
    def after_epoch(self, epoch, model, metrics):
        mlflow = self.get_service("mlflow")
        if mlflow:
            with mlflow.start_run():
                for name, value in metrics.items():
                    mlflow.log_metric(name, value, step=epoch)
```

## Hook频率控制

每个Hook都有一个`frequency`参数，用于控制该Hook被触发的频率：

```python
# 每5步触发一次
hook_config = {
    'frequency': 5
}
```

你可以通过覆盖`should_trigger`方法来创建更复杂的触发逻辑：

```python
def should_trigger(self, step: int) -> bool:
    # 只在前100步每步触发，之后每10步触发一次
    if step < 100:
        return True
    return step % 10 == 0
```

## 最佳实践

1. **按需使用服务** - 不要假设某个服务一定存在，总是在使用前检查
2. **保持Hook轻量级** - Hook应该执行简单的任务，不要在Hook中进行大量计算
3. **使用适当的频率** - 设置合理的触发频率，避免过于频繁的操作降低训练速度
4. **清理资源** - 如果你的Hook分配了资源，确保在`cleanup`方法中释放它们 