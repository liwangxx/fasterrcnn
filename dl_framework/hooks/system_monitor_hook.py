import os
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import torch

from .base_hook import BaseHook
from .registry import HookRegistry
from ..utils.logger import get_logger

@HookRegistry.register("SystemMonitorHook")
class SystemMonitorHook(BaseHook):
    """系统资源监控钩子，用于监控CPU、内存、GPU使用情况"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化
        
        Args:
            config: 钩子配置
                - interval: 采样间隔（秒），默认为2秒
                - show_plots: 是否显示实时图表，默认为False
                - track_cpu: 是否监控CPU使用率，默认为True
                - track_memory: 是否监控内存使用率，默认为True
                - track_gpu: 是否监控GPU使用率，默认为True（如果有GPU）
                - track_gpu_memory: 是否监控GPU内存使用率，默认为True（如果有GPU）
        """
        super().__init__(config)
        
        # 配置
        self.interval = self.config.get('interval', 2)  # 采样间隔（秒）
        self.show_plots = self.config.get('show_plots', False)  # 是否显示实时图表
        
        # 监控选项
        self.track_cpu = self.config.get('track_cpu', True)
        self.track_memory = self.config.get('track_memory', True)
        self.track_gpu = self.config.get('track_gpu', True)
        self.track_gpu_memory = self.config.get('track_gpu_memory', True)
        
        # 数据存储
        self.timestamps = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []
        
        # 监控线程
        self.stop_event = None
        self.monitor_thread = None
        
            
    def _setup(self) -> None:
        """设置钩子环境"""
        # 可以在这里进行一些初始化工作
        self.logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        try:
            import psutil
        except ImportError:
            self.logger.error("请安装psutil库: uv add psutil")
            raise ImportError("缺少必要的依赖库：psutil")
            
        # 检查GPU可用性
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_count = torch.cuda.device_count()
            self.logger.info(f"检测到 {self.gpu_count} 个GPU设备")
        else:
            self.gpu_count = 0
            self.track_gpu = False
            self.track_gpu_memory = False
            self.logger.info("未检测到GPU设备，禁用GPU监控")
        
        if self.has_gpu:
            try:
                from pynvml import nvmlInit
                nvmlInit()
            except:
                self.logger.warning("无法初始化NVML，GPU监控可能不准确")
                
    def _monitor_resources(self):
        """监控系统资源"""
        try:
            import psutil
        except ImportError:
            self.logger.error("请安装psutil库: uv add psutil")
            return
            
        while not self.stop_event.is_set():
            try:
                # 当前时间戳
                self.timestamps.append(time.time())
                
                # CPU使用率
                if self.track_cpu:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.cpu_usage.append(cpu_percent)
                    
                # 内存使用率
                if self.track_memory:
                    memory_info = psutil.virtual_memory()
                    memory_percent = memory_info.percent
                    self.memory_usage.append(memory_percent)
                
                # GPU使用率和内存
                if self.has_gpu:
                    if self.track_gpu:
                        gpu_percents = []
                        for i in range(self.gpu_count):
                            try:
                                gpu_percent = self._get_gpu_utilization(i)
                                gpu_percents.append(gpu_percent)
                            except Exception as e:
                                self.logger.warning(f"获取GPU {i} 使用率失败: {str(e)}")
                                gpu_percents.append(0)
                        
                        # 添加平均GPU使用率
                        self.gpu_usage.append(np.mean(gpu_percents))
                    
                    if self.track_gpu_memory:
                        gpu_memory_percents = []
                        for i in range(self.gpu_count):
                            try:
                                memory_percent = self._get_gpu_memory_utilization(i)
                                gpu_memory_percents.append(memory_percent)
                            except Exception as e:
                                self.logger.warning(f"获取GPU {i} 内存使用率失败: {str(e)}")
                                gpu_memory_percents.append(0)
                        
                        # 添加平均GPU内存使用率
                        self.gpu_memory_usage.append(np.mean(gpu_memory_percents))
                
                # 间隔
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"资源监控错误: {str(e)}")
                time.sleep(self.interval)  # 出错后继续尝试
    
    def _get_gpu_utilization(self, gpu_id):
        """获取GPU使用率"""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu  # 返回GPU使用百分比
        except ImportError:
            # 如果pynvml不可用，尝试使用nvidia-smi命令
            try:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                return float(result.decode('utf-8').strip())
            except:
                # 如果以上方法都失败，退回到PyTorch
                return 0  # PyTorch无法直接获取GPU使用率
    
    def _get_gpu_memory_utilization(self, gpu_id):
        """获取GPU内存使用率"""
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.used / info.total) * 100
        except ImportError:
            # 尝试使用PyTorch获取内存信息
            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(gpu_id)
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    reserved = torch.cuda.memory_reserved(gpu_id)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory
                    return (allocated / total) * 100
                except:
                    return 0
            return 0
    
    def _visualize_resources(self, global_step):
        """可视化资源使用情况"""
        visualizer = self.get_service("visualizer")
        if not visualizer:
            return
            
        # 创建数据字典
        scalars_dict = {}
        
        # 添加最近的CPU使用率
        if self.track_cpu and self.cpu_usage:
            visualizer.add_scalar('resources/cpu_usage', self.cpu_usage[-1], global_step)
            scalars_dict['cpu'] = self.cpu_usage[-1]
            
        # 添加最近的内存使用率
        if self.track_memory and self.memory_usage:
            visualizer.add_scalar('resources/memory_usage', self.memory_usage[-1], global_step)
            scalars_dict['memory'] = self.memory_usage[-1]
            
        # 添加最近的GPU使用率
        if self.track_gpu and self.gpu_usage:
            visualizer.add_scalar('resources/gpu_usage', self.gpu_usage[-1], global_step)
            scalars_dict['gpu'] = self.gpu_usage[-1]
            
        # 添加最近的GPU内存使用率
        if self.track_gpu_memory and self.gpu_memory_usage:
            visualizer.add_scalar('resources/gpu_memory_usage', self.gpu_memory_usage[-1], global_step)
            scalars_dict['gpu_memory'] = self.gpu_memory_usage[-1]
            
        # 添加所有资源指标到一个图表
        if scalars_dict:
            visualizer.add_scalars('resources/all', scalars_dict, global_step)
            
        # # 每隔一段时间创建并添加趋势图
        # if global_step % (self.frequency * 5) == 0:
        #     # 创建CPU使用率趋势图
        #     if self.track_cpu and len(self.cpu_usage) > 1:
        #         fig_cpu, ax_cpu = plt.subplots(figsize=(10, 4))
        #         ax_cpu.plot(self.timestamps[-100:], self.cpu_usage[-100:])
        #         ax_cpu.set_title('CPU Usage Trend')
        #         ax_cpu.set_ylabel('CPU Usage (%)')
        #         ax_cpu.set_xlabel('Time')
        #         ax_cpu.grid(True)
        #         visualizer.add_figure('resources/cpu_trend', fig_cpu, global_step)
        #         plt.close(fig_cpu)
                
        #     # 创建内存使用率趋势图
        #     if self.track_memory and len(self.memory_usage) > 1:
        #         fig_mem, ax_mem = plt.subplots(figsize=(10, 4))
        #         ax_mem.plot(self.timestamps[-100:], self.memory_usage[-100:])
        #         ax_mem.set_title('Memory Usage Trend')
        #         ax_mem.set_ylabel('Memory Usage (%)')
        #         ax_mem.set_xlabel('Time')
        #         ax_mem.grid(True)
        #         visualizer.add_figure('resources/memory_trend', fig_mem, global_step)
        #         plt.close(fig_mem)
                
        #     # 创建GPU使用率趋势图
        #     if self.track_gpu and len(self.gpu_usage) > 1:
        #         fig_gpu, ax_gpu = plt.subplots(figsize=(10, 4))
        #         ax_gpu.plot(self.timestamps[-100:], self.gpu_usage[-100:])
        #         ax_gpu.set_title('GPU Usage Trend')
        #         ax_gpu.set_ylabel('GPU Usage (%)')
        #         ax_gpu.set_xlabel('Time')
        #         ax_gpu.grid(True)
        #         visualizer.add_figure('resources/gpu_trend', fig_gpu, global_step)
        #         plt.close(fig_gpu)
                
        #     # 创建GPU内存使用率趋势图
        #     if self.track_gpu_memory and len(self.gpu_memory_usage) > 1:
        #         fig_gpu_mem, ax_gpu_mem = plt.subplots(figsize=(10, 4))
        #         ax_gpu_mem.plot(self.timestamps[-100:], self.gpu_memory_usage[-100:])
        #         ax_gpu_mem.set_title('GPU Memory Usage Trend')
        #         ax_gpu_mem.set_ylabel('GPU Memory Usage (%)')
        #         ax_gpu_mem.set_xlabel('Time')
        #         ax_gpu_mem.grid(True)
        #         visualizer.add_figure('resources/gpu_memory_trend', fig_gpu_mem, global_step)
        #         plt.close(fig_gpu_mem)
    
    def before_training(self, model) -> None:
        """训练开始前调用"""
        # 启动监控线程
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True  # 设置为守护线程，主线程结束时会自动结束
        self.monitor_thread.start()
        self.logger.info("系统资源监控已启动")
    
    def after_step(self, step: int, batch: Any, outputs: Any, loss: torch.Tensor, model: torch.nn.Module) -> None:
        """在每步结束后调用"""
        if self.should_trigger(step):
            self._visualize_resources(step)
    
    def after_epoch(self, epoch: int, model: torch.nn.Module, metrics: Dict[str, float]) -> None:
        """每个epoch结束后调用"""
        pass
    
    def after_training(self, model: torch.nn.Module, metrics: Dict[str, float]) -> None:
        """训练结束后调用"""
        # 停止监控线程
        if self.stop_event is not None:
            self.stop_event.set()
            if self.monitor_thread is not None:
                self.monitor_thread.join(timeout=5)
            self.logger.info("系统资源监控已停止")
    
    def cleanup(self) -> None:
        """清理资源"""
        # 确保监控线程已停止
        self.after_training(None, {})
        super().cleanup() 