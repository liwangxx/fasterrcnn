import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Callable, Optional, Union

from ..models.registry import ModelRegistry
from ..datasets.registry import DatasetRegistry
from ..utils.logger import Logger
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..visualization.base_visualizer import BaseVisualizer
from ..hooks.base_hook import BaseHook
from ..hooks.registry import HookRegistry
from ..visualization.registry import VisualizerRegistry

class BaseTrainer:
    """基础训练器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.logger = self._build_logger()
        self.device = self._get_device()
        
        # 设置随机种子以确保可重复性
        self._set_seed(config.get('seed', 42))
        
        # 构建模型和数据加载器
        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_data_loaders()
        
        # 构建优化器和学习率调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 构建可视化器和钩子
        self.visualizer = self._build_visualizer()
        self.hooks = self._build_hooks()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')  # 用于模型检查点保存
        
        # 输出目录
        self.output_dir = config.get('output_dir', 'checkpoints')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载检查点（如果指定）
        resume_path = config.get('resume')
        if resume_path and os.path.exists(resume_path):
            checkpoint = load_checkpoint(
                self.model,
                resume_path,
                self.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_metric = checkpoint.get('metrics', {}).get('loss', float('inf'))
            self.logger.info(f"从检查点恢复训练: {resume_path}")
            self.logger.info(f"从epoch {self.current_epoch} 继续训练")
    
    def _build_logger(self) -> Logger:
        """构建日志记录器
        
        Returns:
            日志记录器
        """
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        return Logger(log_dir)
    
    def _get_device(self) -> torch.device:
        """获取设备
        
        Returns:
            torch设备
        """
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子
        
        Args:
            seed: 随机种子
        """
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _build_model(self) -> nn.Module:
        """构建模型
        
        Returns:
            模型
        """
        model_config = self.config.get('model', {}).get('model', {})
        model_type = model_config.get('type')
        
        if not model_type:
            raise ValueError("模型类型未指定")
            
        model_class = ModelRegistry.get(model_type)
        model = model_class(model_config)
        
        # 将模型移动到设备
        model = model.to(self.device)
        
        return model
    
    def _build_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """构建数据加载器
        
        Returns:
            训练数据加载器和验证数据加载器
        """
        dataset_config = self.config.get('dataset', {}).get('dataset', {})
        dataset_type = dataset_config.get('type')
        
        if not dataset_type:
            raise ValueError("数据集类型未指定")
            
        dataset_class = DatasetRegistry.get(dataset_type)
        
        # 创建训练和验证数据集
        train_dataset = dataset_class(dataset_config, is_training=True)
        val_dataset = dataset_class(dataset_config, is_training=False)
        
        # 配置数据加载器
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_dataset.get_collate_fn(),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=val_dataset.get_collate_fn(),
        )
        
        return train_loader, val_loader
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器
        
        Returns:
            优化器
        """
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        lr = float(optimizer_config.get('lr', 0.001))
        weight_decay = float(optimizer_config.get('weight_decay', 0))
        
        # 获取模型可训练参数
        params = self.model.get_trainable_parameters()
        
        # 创建优化器
        if optimizer_type == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = float(optimizer_config.get('momentum', 0.9))
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """构建学习率调度器
        
        Returns:
            学习率调度器
        """
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        if not scheduler_config:
            return None
            
        scheduler_type = scheduler_config.get('type', '').lower()
        
        if not scheduler_type:
            return None
            
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 100)
            eta_min = scheduler_config.get('eta_min', 0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'plateau':
            factor = scheduler_config.get('factor', 0.1)
            patience = scheduler_config.get('patience', 10)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def _build_visualizer(self) -> Optional[BaseVisualizer]:
        """构建可视化器
        
        Returns:
            可视化器
        """
        # 检查配置中是否有visualization部分
        vis_config = self.config.get('visualization')
        if not vis_config:
            return None
            
        # 检查配置中是否指定了可视化器类型
        visualizer_type = None
        for key in vis_config.keys():
            if key in VisualizerRegistry.list():
                visualizer_type = key
                break
            
        # 如果找不到指定的可视化器类型，尝试查找默认类型
        if not visualizer_type:
            if 'tensorboard' in vis_config:
                visualizer_type = 'TensorBoard'
            else:
                self.logger.warning("未找到支持的可视化器类型")
                return None
                
        try:
            visualizer_class = VisualizerRegistry.get(visualizer_type)
            visualizer = visualizer_class(vis_config)
            self.logger.info(f"使用可视化器: {visualizer_type}")
            return visualizer
        except KeyError:
            self.logger.warning(f"未知的可视化器类型: {visualizer_type}")
        except Exception as e:
            self.logger.error(f"创建可视化器 {visualizer_type} 时出错: {e}")
            
        return None
    
    def _build_hooks(self) -> List[BaseHook]:
        """构建钩子
        
        Returns:
            钩子列表
        """
        hooks = []
        
        # 检查可视化配置中是否有钩子配置
        vis_config = self.config.get('visualization', {})
        hooks_config = vis_config.get('hooks', [])
        
        if not hooks_config or not self.visualizer:
            return hooks
            
        # 创建钩子实例
        for hook_config in hooks_config:
            hook_type = hook_config.get('type')
            if not hook_type:
                self.logger.warning(f"跳过未指定类型的钩子: {hook_config}")
                continue
                
            try:
                hook_class = HookRegistry.get(hook_type)
                hook = hook_class(hook_config, self.visualizer)
                hooks.append(hook)
                self.logger.info(f"添加钩子: {hook_type} 目标: {hook_config.get('targets', [])}")
            except KeyError:
                self.logger.warning(f"未知的钩子类型: {hook_type}")
            except Exception as e:
                self.logger.error(f"创建钩子 {hook_type} 时出错: {e}")
        
        return hooks
    
    def train(self) -> nn.Module:
        """训练模型
        
        Returns:
            训练完成的模型
        """
        self.logger.info("开始训练...")
        
        # 获取训练配置
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 100)
        
        # 执行钩子的训练前方法
        for hook in self.hooks:
            hook.before_training(self.model)
        
        # 迭代每个epoch
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # 执行钩子的epoch前方法
            for hook in self.hooks:
                hook.before_epoch(epoch, self.model)
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 验证
            val_metrics = self._validate_epoch()
            
            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', float('inf')))
                else:
                    self.scheduler.step()
            
            # 记录指标
            self.logger.info(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_metrics.get('loss', 0):.4f} "
                f"Val Loss: {val_metrics.get('loss', 0):.4f} "
                f"Val Metrics: {val_metrics}"
            )
            
            # 执行钩子的epoch后方法
            for hook in self.hooks:
                hook.after_epoch(epoch, self.model, val_metrics)
            
            # 保存检查点
            self._save_checkpoint(val_metrics)
            
            # 检查早停
            if self._should_stop_early(val_metrics):
                self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break
        
        # 执行钩子的训练后方法
        for hook in self.hooks:
            hook.after_training(self.model, val_metrics)
            hook.cleanup()
        
        # 关闭可视化器
        if self.visualizer is not None:
            self.visualizer.flush()
            self.visualizer.close()
        
        # 加载最佳模型
        best_checkpoint_path = os.path.join(self.output_dir, 'model_best.pth')
        if os.path.exists(best_checkpoint_path):
            load_checkpoint(self.model, best_checkpoint_path, self.device)
        
        self.logger.info("训练完成")
        return self.model
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch
        
        Returns:
            训练指标
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 执行钩子的step前方法
            for hook in self.hooks:
                hook.before_step(self.global_step, batch, self.model)
            
            # 准备数据
            if isinstance(batch, dict):
                # 如果batch是字典，每个键对应一个张量
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                targets = batch['targets'].to(self.device)
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                # 如果batch是元组或列表，假设是(inputs, targets)
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            else:
                raise ValueError(f"不支持的batch格式: {type(batch)}")
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.model.get_loss(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 更新全局步数
            self.global_step += 1
            
            # 累积损失
            total_loss += loss.item()
            
            # 使用可视化器记录每个batch的损失
            if self.visualizer is not None:
                self.visualizer.add_scalar('train/batch_loss', loss.item(), self.global_step)
            
            # 执行钩子的step后方法
            for hook in self.hooks:
                hook.after_step(self.global_step, batch, outputs, loss, self.model)
            
            # 打印进度
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Train Epoch: {self.current_epoch+1} "
                    f"[{batch_idx}/{num_batches} "
                    f"({100. * batch_idx / num_batches:.0f}%)] "
                    f"Loss: {loss.item():.6f}"
                )
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 使用可视化器记录每个epoch的平均损失
        if self.visualizer is not None:
            self.visualizer.add_scalar('train/epoch_loss', avg_loss, self.current_epoch)
        
        return {'loss': avg_loss}
    
    def _validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch
        
        Returns:
            验证指标
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 准备数据
                if isinstance(batch, dict):
                    # 如果batch是字典，每个键对应一个张量
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                    targets = batch['targets'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 如果batch是元组或列表，假设是(inputs, targets)
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    raise ValueError(f"不支持的batch格式: {type(batch)}")
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.model.get_loss(outputs, targets)
                
                # 累积损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算其他指标（子类可以重写此方法添加更多指标）
        metrics = {'loss': avg_loss}
        
        # 使用可视化器记录验证指标
        if self.visualizer is not None:
            self.visualizer.add_scalar('val/loss', avg_loss, self.current_epoch)
            # 如果有其他指标，也记录它们
            for key, value in metrics.items():
                if key != 'loss':  # loss已经记录过了
                    self.visualizer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return metrics
    
    def _save_checkpoint(self, metrics: Dict[str, float]) -> None:
        """保存检查点
        
        Args:
            metrics: 验证指标
        """
        # 保存最新模型
        latest_path = os.path.join(self.output_dir, 'model_latest.pth')
        save_checkpoint(
            self.model, 
            latest_path, 
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics=metrics
        )
        
        # 保存最佳模型（以验证损失为标准）
        current_metric = metrics.get('loss', float('inf'))
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            best_path = os.path.join(self.output_dir, 'model_best.pth')
            save_checkpoint(
                self.model, 
                best_path, 
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                metrics=metrics
            )
            self.logger.info(f"保存最佳模型，指标: {current_metric:.4f}")
    
    def _should_stop_early(self, metrics: Dict[str, float]) -> bool:
        """检查是否应该早停
        
        Args:
            metrics: 验证指标
            
        Returns:
            是否应该早停
        """
        early_stopping_config = self.config.get('training', {}).get('early_stopping', {})
        if not early_stopping_config:
            return False
            
        patience = early_stopping_config.get('patience', 10)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        
        # 子类可以实现早停逻辑
        return False
    
    def test(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """测试模型
        
        Args:
            test_loader: 测试数据加载器，如果为None则使用验证数据加载器
            
        Returns:
            测试指标
        """
        if test_loader is None:
            test_loader = self.val_loader
        
        self.model.eval()
        total_loss = 0
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 准备数据
                if isinstance(batch, dict):
                    # 如果batch是字典，每个键对应一个张量
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                    targets = batch['targets'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    # 如果batch是元组或列表，假设是(inputs, targets)
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                else:
                    raise ValueError(f"不支持的batch格式: {type(batch)}")
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = self.model.get_loss(outputs, targets)
                
                # 累积损失
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算其他指标（子类可以重写此方法添加更多指标）
        metrics = {'loss': avg_loss}
        
        self.logger.info(f"测试结果: {metrics}")
        
        return metrics 