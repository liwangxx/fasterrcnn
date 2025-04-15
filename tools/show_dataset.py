#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dl_framework.datasets import DatasetRegistry

def main():
    # 加载VOC数据集配置
    config_file = 'configs/datasets/voc2012.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建数据集实例
    dataset_config = config['dataset']
    dataset_class = DatasetRegistry.get(dataset_config['type'])
    dataset = dataset_class(dataset_config, is_training=True)
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    batch_size = 2
    collate_fn = dataset.get_collate_fn()
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 使用0个工作进程，便于调试
        collate_fn=collate_fn
    )
    
    # 获取并打印一个批次
    print("\n获取一个批次的数据:")
    for batch_idx, batch in enumerate(data_loader):
        print(f"批次 {batch_idx + 1} 类型: {type(batch)}")
        
        if isinstance(batch, dict):
            print(f"批次包含的键: {batch.keys()}")
            
            # 打印图像信息
            if 'images' in batch:
                images = batch['images']
                print(f"\n图像信息:")
                print(f"  类型: {type(images)}")
                print(f"  形状: {images.shape}")
            
            # 打印目标信息
            if 'targets' in batch:
                targets = batch['targets']
                print(f"\n目标信息:")
                print(f"  类型: {type(targets)}")
                
                if isinstance(targets, list):
                    print(f"  长度: {len(targets)}")
                    
                    # 打印第一个目标的详细信息
                    if len(targets) > 0:
                        print(f"\n  第一个目标详情:")
                        first_target = targets[0]
                        
                        if isinstance(first_target, dict):
                            print(f"    包含的键: {first_target.keys()}")
                            
                            for k, v in first_target.items():
                                if isinstance(v, torch.Tensor):
                                    print(f"    {k}: 形状={v.shape}, 类型={v.dtype}")
                                    # 如果是边界框，打印第一个框的内容
                                    if k == 'boxes' and v.shape[0] > 0:
                                        print(f"      第一个框: {v[0]}")
                                else:
                                    print(f"    {k}: 类型={type(v)}")
        
        # 只打印第一个批次
        break

if __name__ == '__main__':
    main() 