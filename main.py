# -*- coding: utf-8 -*-
"""
AutoYOLO自动化训练主流程模块
整合图像缩放、豆包自动标注、数据集划分和YOLO模型训练的全流程
"""

import os
import sys
import time
import asyncio
import yaml
import logging
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from glob import glob
from data_resize import resize_images_sequential
from doubao2pro_8 import DoubaoDetector
from ultralytics import YOLO


class StreamToLogger:
    """
    标准输出/错误重定向器
    将stdout/stderr重定向到日志回调函数，实现日志实时显示
    """
    
    def __init__(self, log_callback=None, original_stream=None):
        """
        初始化重定向器
        
        Args:
            log_callback: 日志回调函数
            original_stream: 原始流对象（stdout/stderr）
        """
        self.log_callback = log_callback
        self.original_stream = original_stream or sys.stdout
    
    def write(self, message):
        """写入消息，同时输出到原始流和回调"""
        if self.original_stream:
            self.original_stream.write(message)
        if self.log_callback and message.strip():
            self.log_callback(message.strip())
    
    def flush(self):
        """刷新流缓冲区"""
        if self.original_stream:
            self.original_stream.flush()


class AutoYOLOPipeline:
    """
    自动化YOLO训练流程管理器
    
    负责协调和管理图像缩放、目标标注、数据集划分、
    模型训练等完整流程，支持断点续训和实时日志输出
    """

    def __init__(self, config_path="config.yaml", progress_callback=None, log_callback=None):
        """
        初始化训练流程管理器
        
        Args:
            config_path: 配置文件路径
            progress_callback: 进度回调函数 callback(current, total, desc)
            log_callback: 日志回调函数 callback(message)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.doubao_config = self.config["doubao_config"]  # 豆包API配置
        self.resize_data_config = self.config["resize_data_config"]  # 缩放配置
        self.train_config = self.config["train_config"]  # 训练配置
        self.progress_callback = progress_callback
        self.log_callback = log_callback

        # 初始化日志目录：train_log/{timestamp}.log
        self.log_dir = Path("train_log")
        self.log_dir.mkdir(exist_ok=True)

        # 生成带时间戳的日志文件名，避免覆盖
        current_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        self.log_file_path = self.log_dir / f"{current_time}.log"

        # 配置logging模块：同时输出到控制台和文件
        logging.basicConfig(
            level=logging.INFO,
            format="[{asctime}] [{levelname}] {message}",
            style="{",
            handlers=[
                logging.StreamHandler(),  # 控制台输出
                logging.FileHandler(self.log_file_path, encoding="utf-8")  # 文件输出
            ]
        )
        self.logger = logging.getLogger(__name__)
        self._print_log("初始化配置完成")

    def _load_config(self):
        """
        加载YAML配置文件
        
        Returns:
            dict: 配置字典
        
        Raises:
            Exception: 配置文件加载失败时抛出异常
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self._print_log(f"配置文件加载失败：{str(e)}", level="ERROR")
            raise

    def _print_log(self, msg, level="INFO"):
        """
        输出日志消息
        
        Args:
            msg: 日志消息
            level: 日志级别（INFO/ERROR）
        """
        if level.upper() == "INFO":
            self.logger.info(msg)
        elif level.upper() == "ERROR":
            self.logger.error(msg)
        if self.log_callback:
            self.log_callback(msg)

    def _update_progress(self, current, total, desc=""):
        """
        更新训练进度
        
        Args:
            current: 当前进度值
            total: 总进度值
            desc: 进度描述文本
        """
        if self.progress_callback:
            self.progress_callback(current, total, desc)

    def resize_step(self):
        """
        执行图像缩放步骤
        
        根据配置对原始图片进行缩放处理，统一图片尺寸以便后续标注和训练
        """
        cfg = self.resize_data_config
        if not cfg["no_skip"]:  # 检查是否跳过此步骤
            self._print_log("跳过图像缩放")
            return

        self._print_log("开始图像缩放...")
        try:
            resize_images_sequential(
                config_path=self.config_path,
                progress_callback=self._update_progress,
                log_callback=self._print_log
            )
            self._print_log("图像缩放完成！")
        except Exception as e:
            self._print_log(f"缩放失败：{str(e)}", "ERROR")
            raise

    async def annotate_step(self):
        """
        执行豆包自动标注步骤
        
        使用豆包视觉大模型对缩放后的图片进行目标检测标注，
        生成YOLO格式的标注文件
        """
        cfg = self.doubao_config
        if not cfg["no_skip"]:
            self._print_log("跳过豆包标注")
            return

        self._print_log("开始豆包自动标注...")
        try:
            detector = DoubaoDetector(
                config_path=self.config_path,
                progress_callback=self._update_progress,
                log_callback=self._print_log
            )
            await detector.run()  # 异步执行标注
            self._print_log("豆包标注完成！")
        except Exception as e:
            self._print_log(f"标注失败：{str(e)}", "ERROR")
            raise

    def train_val(self):
        """
        执行数据集划分和配置文件生成步骤
        
        将标注好的图片划分为训练集和验证集（8:2），
        并生成YOLO训练所需的数据集配置文件（.yaml）
        """
        # 获取所有标注好的图片路径
        images = glob(os.path.join(self.doubao_config["input_images_path"], "*"))
        # 使用sklearn划分训练集和验证集，80%训练，20%验证
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        
        # 生成训练集和验证集的图片路径列表文件
        train_path = str(Path(os.path.join(os.path.dirname(self.doubao_config["input_images_path"]), "train.txt")))
        val_path = str(Path(os.path.join(os.path.dirname(self.doubao_config["input_images_path"]), "val.txt")))
        
        # 追加写入训练集图片路径
        with open(train_path, "a") as f1:
            for image in train_images:
                f1.write(str(Path(image)) + "\n")
        # 追加写入验证集图片路径
        with open(val_path, "a") as f2:
            for image in val_images:
                f2.write(str(Path(image)) + "\n")

        # 构建YOLO数据集配置字典
        config = {
            'train': train_path if isinstance(train_path, list) else [train_path],
            'val': val_path if isinstance(val_path, list) else [val_path],
            'nc': 1,  # 类别数量
            'names': [self.train_config["name"]],  # 类别名称列表
        }

        # 生成数据集配置文件路径：{任务名}.yaml
        self.output_file = os.path.join(os.path.dirname(self.doubao_config["input_images_path"]),
                                        "{}.yaml".format(self.train_config["name"]))
        self.output_file = str(Path(self.output_file))
        
        if os.path.exists(self.output_file):  # 配置文件已存在
            self._print_log("数据配置文件已存在！")
            if self.output_file == self.train_config["data_yaml"]:
                pass
            else:
                self.train_config["data_yaml"] = self.output_file
                self._print_log("data_yaml文件不一致，已启用{}".format(self.output_file))
        else:  # 创建新的数据集配置文件
            self._print_log("数据配置文件开始生成！")
            with open(self.output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False)
            self._print_log("数据配置文件生成成功！")

        self.train_config["data_yaml"] = str(self.output_file)

    def train_step(self):
        """
        执行YOLO模型训练步骤
        
        加载预训练模型，使用标注好的数据集进行训练，
        支持进度回调和实时日志输出
        """
        cfg = self.train_config
        if not cfg["no_skip"]:
            self._print_log("暂不开启YOLO训练")
            return
        else:
            self._print_log("开始YOLO模型训练...")
            self._print_log(f"模型路径: {cfg['model_path']}")
            self._print_log(f"数据配置: {cfg['data_yaml']}")
            self._print_log(f"训练轮数: {cfg['epochs']}")
            self._print_log(f"批次大小: {cfg['batch']}")
            self._print_log(f"图像尺寸: {cfg['imgsz']}")
            
            try:
                # 保存原始stdout/stderr，用于训练后恢复
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                # 将标准输出重定向到日志回调，实现训练日志实时显示
                if self.log_callback:
                    sys.stdout = StreamToLogger(self.log_callback, original_stdout)
                    sys.stderr = StreamToLogger(self.log_callback, original_stderr)
                
                model = YOLO(cfg["model_path"])  # 加载预训练模型
                
                total_epochs = cfg["epochs"]  # 总训练轮数
                
                class ProgressCallback:
                    """训练进度回调类，用于在每个epoch结束后更新进度"""
                    def __init__(self, outer):
                        self.outer = outer
                    
                    def on_train_epoch_end(self, trainer):
                        epoch = trainer.epoch + 1  # 当前epoch（从0开始）
                        self.outer._update_progress(epoch, total_epochs, f"训练进度: Epoch {epoch}/{total_epochs}")
                        self.outer._print_log(f"Epoch {epoch}/{total_epochs} - 损失: {trainer.loss:.4f}")
                
                model.add_callback("on_train_epoch_end", ProgressCallback(self).on_train_epoch_end)  # 注册回调
                
                # 调用YOLO训练接口
                model.train(
                    data=cfg["data_yaml"],  # 数据集配置
                    imgsz=cfg["imgsz"],  # 图像尺寸
                    device=cfg["device"],  # 训练设备
                    lr0=cfg["lr0"],  # 初始学习率
                    epochs=cfg["epochs"],  # 训练轮数
                    batch=cfg["batch"],  # 批次大小
                    name=cfg["name"],  # 任务名称
                    fliplr=cfg["fliplr"],  # 左右翻转概率
                    flipud=cfg["flipud"],  # 上下翻转概率
                    degrees=cfg["degrees"],  # 旋转角度
                    patience=cfg["patience"],  # 早停耐心值
                    plots=cfg["plots"]  # 是否生成图表
                )
                
                # 恢复原始stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                
                self._print_log("YOLO训练完成！")
            except Exception as e:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                self._print_log(f"训练失败：{str(e)}", "ERROR")
                raise

    def run(self):
        """
        执行完整的AutoYOLO自动化流程
        
        依次执行：图像缩放 → 豆包标注 → 数据集划分 → 模型训练，
        并输出总耗时统计
        """
        self._print_log("=" * 40)
        self._print_log("启动 AutoYOLO 全流程自动化")
        self._print_log("=" * 40)
        start = time.time()  # 记录开始时间

        self.resize_step()  # 1. 图像缩放
        asyncio.run(self.annotate_step())  # 2. 豆包标注（异步）
        self.train_val()  # 3. 数据集划分
        self.train_step()  # 4. 模型训练

        cost = time.time() - start  # 计算总耗时
        self._print_log("=" * 40)
        self._print_log(f"全流程完成！总耗时：{cost:.2f} 秒")
        self._print_log("=" * 40)


if __name__ == "__main__":
    pipeline = AutoYOLOPipeline()
    pipeline.run()
