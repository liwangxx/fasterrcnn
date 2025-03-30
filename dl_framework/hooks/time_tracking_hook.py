import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any

from .base_hook import BaseHook
from .registry import HookRegistry
from ..utils.logger import get_logger
@HookRegistry.register("TimeTrackingHook")
class TimeTrackingHook(BaseHook):
    """用于跟踪训练时间和预估完成时间的钩子"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化时间跟踪钩子
        
        Args:
            config: 钩子配置
        """
        super().__init__(config)
        # 初始化时间跟踪变量
        self.epoch_start_time = None
        self.training_start_time = None
        self.epoch_times = []
        self.logger = get_logger('time')
        
    def _setup(self) -> None:
        """设置钩子环境"""
        pass
        
    def before_training(self, model) -> None:
        """训练开始前调用"""
        self.training_start_time = time.time()
        
        # 记录训练开始信息
        self.logger.info(f"训练开始于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def before_epoch(self, epoch: int, model) -> None:
        """每个epoch开始前调用"""
        self.epoch_start_time = time.time()
    
    def after_epoch(self, epoch: int, model, metrics: Dict[str, float]) -> None:
        """每个epoch结束后调用"""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # 获取配置服务
            config = self.get_service("config")
            total_epochs = config.get("training", {}).get("epochs", 100) if config else 100
            
            # 计算平均每个epoch的耗时
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            
            # 预估剩余时间
            remaining_epochs = total_epochs - (epoch + 1)
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            
            # 预计完成时间
            estimated_completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            # 记录信息
            log_message = (
                f"Epoch {epoch+1}/{total_epochs} 耗时: {epoch_time:.2f}秒, "
                f"平均每epoch耗时: {avg_epoch_time:.2f}秒, "
                f"预计剩余时间: {timedelta(seconds=int(estimated_remaining_time))}, "
                f"预计完成时间: {estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # 输出到日志文件
            self.logger.info(f"{log_message}")
            
            
            # 使用可视化服务（如果可用）- 完全可选
            visualizer = self.get_service("visualizer")
            if visualizer:
                visualizer.add_scalar('time/epoch_time', epoch_time, epoch)
                visualizer.add_scalar('time/avg_epoch_time', avg_epoch_time, epoch)
                visualizer.add_scalar('time/estimated_remaining_hours', 
                                     estimated_remaining_time / 3600, epoch)
    
    def after_training(self, model, metrics: Dict[str, float]) -> None:
        """训练结束后调用"""
        total_time = time.time() - self.training_start_time
        log_message = f"训练结束 - 总耗时: {timedelta(seconds=int(total_time))}"
        
        self.logger.info(f"{log_message}")
        