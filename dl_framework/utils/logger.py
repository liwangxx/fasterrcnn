import logging
import os
from typing import Optional
import sys

class Logger:
    """日志记录器，用于记录训练过程中的信息"""
    
    def __init__(self, log_dir: str, name: str = 'dl_framework'):
        """初始化
        
        Args:
            log_dir: 日志目录
            name: 日志名称
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建文件处理器
        log_file = os.path.join(log_dir, f'{name}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info(f"日志保存在: {log_file}")
    
    def info(self, msg: str) -> None:
        """记录信息
        
        Args:
            msg: 信息
        """
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """记录警告
        
        Args:
            msg: 警告信息
        """
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """记录错误
        
        Args:
            msg: 错误信息
        """
        self.logger.error(msg)
    
    def debug(self, msg: str) -> None:
        """记录调试信息
        
        Args:
            msg: 调试信息
        """
        self.logger.debug(msg) 