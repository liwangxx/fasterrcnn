import os
import yaml
from typing import Dict, Any, List, Union, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理引用的子配置
    for key in ['model_config', 'dataset_config', 'visualization_config']:
        if key in config:
            sub_config_path = config[key]
            with open(sub_config_path, 'r', encoding='utf-8') as f:
                sub_config = yaml.safe_load(f)
            config[key.replace('_config', '')] = sub_config
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def merge_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """合并两个配置字典
    
    Args:
        config1: 第一个配置字典
        config2: 第二个配置字典
        
    Returns:
        合并后的配置字典
    """
    merged = config1.copy()
    
    def _merge_dict(d1, d2):
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                _merge_dict(d1[k], v)
            else:
                d1[k] = v
    
    _merge_dict(merged, config2)
    return merged

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """获取配置值，支持嵌套键路径
    
    Args:
        config: 配置字典
        key_path: 键路径，例如 "training.optimizer.lr"
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value 