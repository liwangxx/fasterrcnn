#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
from typing import Dict, Any

# 将项目根目录添加到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dl_framework.utils.config import load_config, save_config, merge_configs
from dl_framework.utils.logger import Logger
from dl_framework.trainers.base_trainer import BaseTrainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练深度学习模型')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件路径')
    parser.add_argument('--vis', type=str, help='可视化配置文件路径')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--name', type=str, default=None, help='实验名称')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--resume', type=str, help='断点续训的检查点路径')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 加载训练配置
    config = load_config(args.config)
    
    # 加载可视化配置（如果指定）
    if args.vis:
        vis_config = load_config(args.vis)
        config['visualization'] = vis_config
    
    # 覆盖配置参数
    if args.seed is not None:
        config['seed'] = args.seed
    
    if args.device is not None:
        config['device'] = args.device
        
    # 设置断点续训路径
    if args.resume:
        config['resume'] = args.resume
    
    # 设置实验名称
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.name:
        experiment_name = f"{args.name}_{timestamp}"
    else:
        experiment_name = f"experiment_{timestamp}"
    
    # 设置输出目录和日志目录
    output_dir = os.path.join('checkpoints', experiment_name)
    log_dir = os.path.join('logs', experiment_name)
    
    config['output_dir'] = output_dir
    config['log_dir'] = log_dir
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存完整配置
    save_config(config, os.path.join(output_dir, 'config.yaml'))
    
    # 创建日志记录器
    logger = Logger(log_dir, name=experiment_name)
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {output_dir}")
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
    
    # 创建并训练模型
    trainer = BaseTrainer(config)
    model = trainer.train()
    
    # 测试模型
    metrics = trainer.test()
    logger.info(f"测试指标: {metrics}")
    
    logger.info("训练完成")

if __name__ == '__main__':
    main() 