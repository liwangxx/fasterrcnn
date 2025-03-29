import logging
import os
from typing import Optional
import sys

# 全局日志器实例
_global_logger = None
_log_configured = False

def configure_logging(log_dir: str, experiment_name: Optional[str] = None):
    """配置全局日志系统
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
    """
    global _global_logger, _log_configured
    
    # 确保log_dir是logs子目录
    if os.path.basename(log_dir) != 'logs':
        experiment_dir = log_dir
        log_dir = os.path.join(log_dir, 'logs')
    else:
        experiment_dir = os.path.dirname(log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果未提供实验名称，则使用实验目录的最后一部分
    if experiment_name is None:
        experiment_name = os.path.basename(os.path.normpath(experiment_dir))
        if not experiment_name or experiment_name == '.':
            experiment_name = 'experiment'
    
    # 设置日志文件名
    log_file = os.path.join(log_dir, f'{experiment_name}.log')
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式化器 - 确保日志中包含模块名称
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 创建框架主日志器
    _global_logger = logging.getLogger('dl_framework')
    _global_logger.info(f"实验目录: {experiment_dir}")
    _global_logger.info(f"日志目录: {log_dir}")
    _global_logger.info(f"日志文件: {log_file}")
    _global_logger.info(f"实验名称: {experiment_name}")
    
    _log_configured = True
    
    return _global_logger

def get_logger(module_name: str):
    """获取特定模块的日志器
    
    Args:
        module_name: 模块名称
        
    Returns:
        logging.Logger: 模块日志器
    """
    global _log_configured
    
    if not _log_configured:
        # 如果日志系统未配置，使用默认配置
        default_dir = os.path.join(os.getcwd(), 'experiments', 'default')
        print(f"警告: 日志系统未配置，使用默认配置: {default_dir}")
        configure_logging(default_dir)
    
    # 返回指定名称的日志器，这将自动从父级继承处理器
    logger = logging.getLogger(module_name)
    return logger