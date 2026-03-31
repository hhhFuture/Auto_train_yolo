# -*- coding: utf-8 -*-
"""
工具函数模块
提供配置文件读写、路径校验、训练结果目录获取等常用工具函数
"""

import yaml
import os
from pathlib import Path
import shutil

def load_yaml(config_path: str = "config.yaml") -> dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径，默认为config.yaml
    
    Returns:
        dict: 配置字典
    
    Raises:
        RuntimeError: 配置文件加载失败时抛出异常
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)  # 解析YAML文件为字典
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败：{str(e)}")

def save_yaml(config: dict, config_path: str = "config.yaml", backup: bool = True):
    """
    保存YAML配置文件（可选备份原文件）
    
    Args:
        config: 要保存的配置字典
        config_path: 配置文件路径
        backup: 是否备份原文件，默认为True
    
    Raises:
        RuntimeError: 配置文件保存失败时抛出异常
    """
    config_path = Path(config_path)  # 转换为Path对象
    if backup and config_path.exists():  # 需要备份且原文件存在
        backup_path = config_path.with_suffix(".yaml.bak")  # 备份文件名加.bak后缀
        shutil.copy2(config_path, backup_path)  # 复制原文件作为备份
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            # 写入YAML：allow_unicode=True支持中文、不排序、2空格缩进
            yaml.dump(config, f, allow_unicode=True, sort_keys=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"保存配置文件失败：{str(e)}")

def validate_path(path: str, create: bool = False) -> bool:
    """
    校验路径是否有效，可选自动创建目录
    
    Args:
        path: 要校验的路径字符串
        create: 是否在路径不存在时自动创建目录，默认为False
    
    Returns:
        bool: 路径是否有效（存在）
    """
    if not path:  # 空路径直接返回False
        return False
    path_obj = Path(path)
    if create and not path_obj.exists():  # 需要创建且目录不存在
        path_obj.mkdir(parents=True, exist_ok=True)  # 创建目录
    return path_obj.exists()  # 返回路径是否存在

def get_train_result_dir(task_name: str = "face") -> str:
    """
    获取YOLO训练结果目录路径
    
    Args:
        task_name: 训练任务名称，默认为"face"
    
    Returns:
        str: 训练结果目录的绝对路径，如果不存在则返回空字符串
    """
    runs_dir = Path("runs/train") / task_name  # YOLO标准输出路径：runs/train/{task_name}
    if runs_dir.exists():
        return str(runs_dir)
    return ""
