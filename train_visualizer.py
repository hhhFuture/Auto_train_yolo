# -*- coding: utf-8 -*-
"""
YOLO训练可视化模块
提供训练结果的加载、损失曲线绘制、评估指标可视化等功能
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体支持
plt.rcParams["axes.unicode_minus"] = False


def load_train_results(result_dir: str) -> pd.DataFrame:
    """
    加载YOLO训练结果CSV文件
    
    Args:
        result_dir: 训练结果目录路径
    
    Returns:
        pd.DataFrame: 训练结果数据框，包含epoch、损失值、评估指标等
        None: 如果results.csv文件不存在则返回None
    """
    csv_path = Path(result_dir) / "results.csv"  # YOLO训练结果CSV文件路径
    if not csv_path.exists():  # 文件不存在则返回None
        return None
    df = pd.read_csv(csv_path, skiprows=1)  # 跳过第一行注释
    df.columns = [col.strip() for col in df.columns]  # 清理列名空格
    return df


def plot_loss_curve(df: pd.DataFrame):
    """
    绘制训练和验证损失曲线
    
    Args:
        df: 训练结果数据框
    
    Returns:
        matplotlib.figure.Figure: 损失曲线图对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 创建1行2列子图

    # 左图：训练损失曲线
    ax1.plot(df["epoch"], df["train/box_loss"], label="框损失", color="red")  # 边界框回归损失
    ax1.plot(df["epoch"], df["train/cls_loss"], label="分类损失", color="blue")  # 分类损失
    ax1.plot(df["epoch"], df["train/dfl_loss"], label="DFL损失", color="green")  # 分布焦点损失
    ax1.set_title("训练损失曲线")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失值")
    ax1.legend()  # 显示图例
    ax1.grid(True, alpha=0.3)  # 显示网格

    # 右图：验证损失曲线
    ax2.plot(df["epoch"], df["val/box_loss"], label="验证框损失", color="red")
    ax2.plot(df["epoch"], df["val/cls_loss"], label="验证分类损失", color="blue")
    ax2.set_title("验证损失曲线")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("损失值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    return fig


def plot_metrics_curve(df: pd.DataFrame):
    """
    绘制模型评估指标曲线（mAP、Precision、Recall）
    
    Args:
        df: 训练结果数据框
    
    Returns:
        matplotlib.figure.Figure: 评估指标曲线图对象
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # 创建单子图

    # mAP50：IoU阈值为0.5时的平均精度
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="red", linewidth=2)
    # mAP50-95：IoU阈值从0.5到0.95的平均精度
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="orange", linewidth=2)
    ax.plot(df["epoch"], df["metrics/precision(B)"], label="精确率", color="blue", alpha=0.7)  # 精确率
    ax.plot(df["epoch"], df["metrics/recall(B)"], label="召回率", color="green", alpha=0.7)  # 召回率

    ax.set_title("模型评估指标曲线")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("指标值")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def get_confusion_matrix(result_dir: str):
    """
    获取混淆矩阵可视化图片
    
    Args:
        result_dir: 训练结果目录路径
    
    Returns:
        numpy.ndarray: RGB格式的混淆矩阵图片数组
        None: 如果混淆矩阵图片不存在则返回None
    """
    cm_path = Path(result_dir) / "confusion_matrix.png"  # 混淆矩阵图片路径
    if not cm_path.exists():
        return None
    img = cv2.imread(str(cm_path))  # 读取图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
    return img


def get_train_batch_samples(result_dir: str):
    """
    获取训练批次样本可视化图
    
    展示训练过程中每个批次的样本、真实框和预测框的可视化效果
    
    Args:
        result_dir: 训练结果目录路径
    
    Returns:
        numpy.ndarray: RGB格式的训练批次样本图片数组
        None: 如果批次样本图片不存在则返回None
    """
    batch_path = Path(result_dir) / "train_batch0.jpg"  # 第一个训练批次的可视化
    if not batch_path.exists():
        return None
    img = cv2.imread(str(batch_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
