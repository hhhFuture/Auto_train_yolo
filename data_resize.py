# -*- coding: utf-8 -*-
"""
图片数据预处理模块
支持多进程和单进程两种模式对图片进行缩放处理
"""

import os
import cv2
import yaml
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def _process_single_image(filename, input_dir, output_dir, target_size, valid_extensions):
    """
    单张图片的处理逻辑
    
    Args:
        filename: 图片文件名
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        target_size: 目标尺寸（最长边）
        valid_extensions: 支持的图片扩展名列表
    
    Returns:
        tuple: (成功处理数量, 文件名, 处理是否成功)
    """
    ext = os.path.splitext(filename)[1].lower()  # 获取文件扩展名并转为小写
    if ext not in valid_extensions:  # 检查是否为支持的图片格式
        return 0, filename, False

    img_path = os.path.join(input_dir, filename)  # 拼接完整输入路径
    img = cv2.imread(img_path)  # 读取图片

    if img is None:  # 图片读取失败
        return 0, filename, False

    h, w = img.shape[:2]  # 获取图片高度和宽度
    scale = target_size / max(w, h)  # 计算缩放比例，以最长边为基准

    new_w = int(w * scale)  # 计算新的宽度
    new_h = int(h * scale)  # 计算新的高度

    # 根据缩放比例选择插值方式：缩小用INTER_AREA，放大用INTER_LINEAR
    if scale < 1:
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    save_path = os.path.join(output_dir, filename)  # 拼接完整输出路径
    cv2.imwrite(save_path, resized_img)  # 保存缩放后的图片

    return 1, filename, True


def resize_images(config_path="config.yaml", progress_callback=None, log_callback=None):
    """
    多进程执行图片缩放任务
    
    Args:
        config_path: 配置文件路径
        progress_callback: 进度回调函数 callback(current, total, desc)
        log_callback: 日志回调函数 callback(message)
    
    Returns:
        bool: 是否成功完成
    """

    def _load_config():
        """从配置文件加载参数"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config = _load_config()
    resize_data_config = config["resize_data_config"]
    input_dir = resize_data_config["input_path"]  # 从配置获取输入路径
    output_dir = resize_data_config["output_path"]  # 从配置获取输出路径
    target_size = resize_data_config["target_size"]  # 从配置获取目标尺寸

    if log_callback:
        log_callback(f"输入文件夹: {input_dir}")
        log_callback(f"输出文件夹: {output_dir}")
        log_callback(f"目标尺寸: {target_size}")

    if not os.path.exists(output_dir):  # 如果输出目录不存在则创建
        os.makedirs(output_dir)

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']  # 支持的图片格式列表

    files = os.listdir(input_dir)  # 获取输入目录下的所有文件
    total = len(files)  # 文件总数
    
    if log_callback:
        log_callback(f"共发现 {total} 个文件待处理")

    # 使用partial固定部分参数，创建可并行调用的处理函数
    process_func = partial(
        _process_single_image,
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=target_size,
        valid_extensions=valid_extensions
    )

    counts = int((os.cpu_count()) / 2)  # 使用CPU核心数的一半作为进程数
    
    if progress_callback is None:
        # 无进度回调时，使用tqdm快速处理
        with Pool(counts) as pool:
            results = list(tqdm(pool.imap(process_func, files), total=total, desc="Processing"))
    else:
        # 有进度回调时，逐个获取结果并更新进度
        with Pool(counts) as pool:
            processed = 0
            for result in pool.imap(process_func, files):
                processed += 1
                progress_callback(processed, total, f"处理中: {processed}/{total}")
                if log_callback and result[2]:
                    log_callback(f"已处理: {result[1]}")
            results = list([(r[0], r[1], r[2]) for r in [(0, '', False)] * len(files)])
            for i, r in enumerate(pool.imap(process_func, files)):
                results[i] = r

    count = sum(r[0] for r in results) if results else 0  # 统计成功处理的图片数量

    final_msg = f"全部完成！共处理 {count} 张图片，保存在: {output_dir}"
    print(f"\n{final_msg}")
    
    if log_callback:
        log_callback(final_msg)

    return True


def resize_images_sequential(config_path="config.yaml", progress_callback=None, log_callback=None):
    """
    单进程顺序执行图片缩放任务（更适合Streamlit实时显示进度）
    
    Args:
        config_path: 配置文件路径
        progress_callback: 进度回调函数 callback(current, total, desc)
        log_callback: 日志回调函数 callback(message)
    
    Returns:
        bool: 是否成功完成
    """

    def _load_config():
        """从配置文件加载参数"""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    config = _load_config()
    resize_data_config = config["resize_data_config"]
    input_dir = resize_data_config["input_path"]
    output_dir = resize_data_config["output_path"]
    target_size = resize_data_config["target_size"]

    if log_callback:
        log_callback(f"输入文件夹: {input_dir}")
        log_callback(f"输出文件夹: {output_dir}")
        log_callback(f"目标尺寸: {target_size}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    files = os.listdir(input_dir)
    total = len(files)
    
    if log_callback:
        log_callback(f"共发现 {total} 个文件待处理")

    count = 0  # 成功处理的图片计数器
    for idx, filename in enumerate(files, 1):  # 遍历所有文件
        ext = os.path.splitext(filename)[1].lower()  # 获取扩展名
        if ext not in valid_extensions:  # 跳过非图片文件
            if progress_callback:
                progress_callback(idx, total, f"跳过非图片文件: {filename}")
            continue

        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)  # 读取图片

        if img is None:  # 读取失败
            if log_callback:
                log_callback(f"无法读取图片: {filename}，跳过。")
            if progress_callback:
                progress_callback(idx, total, f"处理中: {idx}/{total}")
            continue

        h, w = img.shape[:2]  # 获取原始尺寸
        scale = target_size / max(w, h)  # 计算缩放比例

        new_w = int(w * scale)  # 新宽度
        new_h = int(h * scale)  # 新高度

        # 根据缩放方向选择插值算法
        if scale < 1:
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        save_path = os.path.join(output_dir, filename)  # 输出路径
        cv2.imwrite(save_path, resized_img)  # 保存图片
        count += 1  # 成功计数加1

        if progress_callback:
            progress_callback(idx, total, f"处理中: {idx}/{total}")
        
        if log_callback:
            log_callback(f"已处理: {filename}")

    final_msg = f"全部完成！共处理 {count} 张图片，保存在: {output_dir}"
    print(f"\n{final_msg}")
    
    if log_callback:
        log_callback(final_msg)

    return True
