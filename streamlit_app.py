# -*- coding: utf-8 -*-
"""
Streamlit Web应用主界面模块

提供图像缩放、豆包标注、模型训练等功能的前端可视化界面，
包含参数配置、进度展示、日志输出和训练结果可视化
"""

import streamlit as st
import asyncio
import os
from pathlib import Path
import sys
from streamlit.runtime.scriptrunner import add_script_run_ctx

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_resize import resize_images_sequential
from doubao2pro_8 import DoubaoDetector
from utils import load_yaml, save_yaml, validate_path, get_train_result_dir
from train_visualizer import (
    load_train_results, plot_loss_curve, plot_metrics_curve,
    get_confusion_matrix, get_train_batch_samples
)

# 设置页面配置：标题、图标、布局宽度
st.set_page_config(page_title="AutoYOLO 可视化平台", page_icon="🤖", layout="wide")
st.title("🤖 AutoYOLO 自动化标注&训练平台")  # 主标题
st.divider()  # 分隔线

CONFIG_PATH = "config.yaml"  # 配置文件路径
config = load_yaml(CONFIG_PATH)  # 加载初始配置

# 创建三个标签页：图像缩放、豆包标注、模型训练
tab1, tab2, tab3 = st.tabs(["📐 图像缩放", "🏷️ 豆包标注", "🚀 模型训练"])

with tab1:
    """
    图像缩放配置标签页
    
    提供图像缩放的参数配置和执行功能，
    包括输入/输出路径设置、目标尺寸设置等
    """
    st.subheader("图像缩放参数配置")
    resize_cfg = config["resize_data_config"]
    
    # 创建表单用于参数输入和提交
    with st.form("resize_form"):
        col1, col2 = st.columns(2)  # 两列布局
        
        with col1:
            resize_no_skip = st.checkbox("执行图像缩放", value=resize_cfg["no_skip"])
            resize_input = st.text_input("原始图片文件夹", value=resize_cfg["input_path"])
            resize_output = st.text_input("输出文件夹", value=resize_cfg["output_path"], placeholder="⚠️ 文件夹名必须以 images 结尾，如：.../data/images")
        with col2:
            resize_target = st.number_input("目标尺寸", value=resize_cfg["target_size"], min_value=320, max_value=1280, step=32)
        
        col_save, col_run = st.columns(2)  # 底部保存和执行按钮
        save_resize_btn = col_save.form_submit_button("💾 保存配置")
        run_resize_btn = col_run.form_submit_button("▶️ 执行缩放")

    # 保存配置按钮逻辑
    if save_resize_btn:
        config["resize_data_config"] = {
            "no_skip": resize_no_skip,
            "input_path": resize_input,
            "output_path": resize_output,
            "target_size": resize_target
        }
        save_yaml(config, CONFIG_PATH)  # 保存配置到YAML文件
        st.success("✅ 配置已保存！")

    # 执行缩放按钮逻辑
    if run_resize_btn:
        if not resize_no_skip:  # 用户取消勾选时跳过
            st.warning("已跳过缩放")
        else:
            progress_bar = st.progress(0, text="准备开始...")  # 创建进度条
            log_container = st.container()  # 创建日志容器
            log_area = log_container.empty()  # 创建可更新的日志区域
            logs = []  # 日志列表
            
            def update_progress(current, total, desc=""):
                """更新进度条回调函数"""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress, text=desc)
            
            def update_log(msg):
                """更新日志显示回调函数"""
                logs.append(msg)
                log_area.code("\n".join(logs[-50:]), language="plaintext")  # 显示最近50条日志
            
            try:
                update_log("🚀 开始执行图像缩放...")
                resize_images_sequential(
                    config_path=CONFIG_PATH,
                    progress_callback=update_progress,
                    log_callback=update_log
                )
                progress_bar.progress(1.0, text="✅ 完成！")
                st.success("✅ 图像缩放完成！")
            except Exception as e:
                progress_bar.progress(1.0, text="❌ 失败")
                st.error(f"缩放失败：{str(e)}")

with tab2:
    """
    豆包自动标注配置标签页
    
    提供豆包API的标注功能配置，包括API密钥、模型选择、
    并发数设置、prompt配置等，并支持执行标注任务
    """
    st.subheader("豆包自动标注配置")
    doubao_cfg = config["doubao_config"]
    
    with st.form("doubao_form"):
        col1, col2 = st.columns(2)
        with col1:
            doubao_no_skip = st.checkbox("执行豆包标注", value=doubao_cfg["no_skip"])
            api_key = st.text_input("API Key", value=doubao_cfg["api_key"], type="password")  # 密码形式隐藏API Key
            model = st.selectbox("模型", ["doubao-seed-2-0-mini-260215", "doubao-seed-2-0-lite-260215"], index=0)
            input_images = st.text_input("待标注图片路径", value=doubao_cfg["input_images_path"])
        with col2:
            label_class = st.text_input("标签类别ID", value=doubao_cfg["label_class"])
            concurrency = st.number_input("并发数", value=doubao_cfg["concurrency"])
            timeout = st.number_input("超时时间", value=doubao_cfg["timeout"])

        st.divider()
        st.markdown("### 📝 标注提示词（Prompt）配置")
        prompt_path = doubao_cfg["prompt_path"]
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            # 上传prompt文件
            uploaded_prompt = st.file_uploader("上传 prompt.txt 文件", type="txt")
            if uploaded_prompt:
                Path(prompt_path).parent.mkdir(exist_ok=True)  # 确保目录存在
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(uploaded_prompt.getvalue().decode("utf-8"))
                st.success(f"文件已保存至：{prompt_path}")

        with p_col2:
            # 在线编辑prompt
            current_prompt = ""
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    current_prompt = f.read()
            new_prompt = st.text_area("在线编辑提示词", value=current_prompt, height=120)

        col_save, col_run = st.columns(2)
        save_doubao_btn = col_save.form_submit_button("💾 保存标注配置")
        run_doubao_btn = col_run.form_submit_button("▶️ 执行标注")

    # 保存标注配置
    if save_doubao_btn:
        config["doubao_config"] = {
            "no_skip": doubao_no_skip, "api_key": api_key, "model": model, "api_url": doubao_cfg["api_url"],
            "input_images_path": input_images, "prompt_path": prompt_path, "label_class": label_class,
            "concurrency": concurrency, "timeout": timeout
        }
        save_yaml(config, CONFIG_PATH)
        st.success("✅ 标注配置保存成功！")

    # 执行豆包标注
    if run_doubao_btn:
        if not doubao_no_skip:
            st.warning("已跳过标注")
        else:
            progress_bar = st.progress(0, text="准备开始...")
            log_container = st.container()
            log_area = log_container.empty()
            logs = []
            
            def update_progress(current, total, desc=""):
                """更新进度条回调函数"""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress, text=desc)
            
            def update_log(msg):
                """更新日志显示回调函数"""
                logs.append(msg)
                log_area.code("\n".join(logs[-50:]), language="plaintext")
            
            try:
                update_log("🚀 开始执行豆包自动标注...")
                detector = DoubaoDetector(
                    config_path=CONFIG_PATH,
                    progress_callback=update_progress,
                    log_callback=update_log
                )
                asyncio.run(detector.run())  # 运行异步标注任务
                progress_bar.progress(1.0, text="✅ 完成！")
                st.success("✅ 标注完成！")
            except Exception as e:
                progress_bar.progress(1.0, text="❌ 失败")
                st.error(f"标注失败：{str(e)}")

with tab3:
    """
    YOLO模型训练配置标签页
    
    提供模型训练的完整参数配置，包括基础参数（模型路径、图像尺寸、
    批次大小、训练轮数等）和高级参数（学习率、损失函数权重、数据增强等），
    并展示训练结果的可视化图表
    """
    st.subheader("YOLO模型训练配置")
    train_cfg = config["train_config"]
    
    with st.form("train_form"):
        train_no_skip = st.checkbox("执行模型训练", value=train_cfg["no_skip"])

        st.markdown("### 🔧 核心参数（推荐修改）")
        col1, col2 = st.columns(2)
        with col1:
            model_path = st.text_input("预训练模型路径", value=train_cfg["model_path"])
            data_yaml = st.text_input("数据集YAML", value=train_cfg["data_yaml"])
            imgsz = st.number_input("训练图像尺寸", value=train_cfg["imgsz"])
            batch = st.number_input("批次大小", value=train_cfg["batch"], min_value=1, max_value=256)
        with col2:
            task_name = st.text_input("训练任务名", value=train_cfg["name"])
            epochs = st.number_input("训练轮数", value=train_cfg["epochs"])
            device = st.selectbox("训练设备", ["CPU", "GPU(0)"], index=1)
            patience = st.number_input("早停耐心值", value=train_cfg["patience"])

        # 高级参数折叠展开
        with st.expander("⚠️ 高级训练参数（不建议修改）", expanded=False):
            st.markdown("##### 学习率参数")
            lr0 = st.number_input("初始学习率 lr0", value=0.01, format="%.4f")
            lrf = st.number_input("最终学习率 lrf", value=0.01, format="%.4f")
            momentum = st.number_input("动量 SGD", value=0.937, format="%.3f")
            weight_decay = st.number_input("权重衰减", value=0.0005, format="%.4f")
            warmup_epochs = st.number_input("预热轮数", value=3.0, format="%.1f")
            warmup_momentum = st.number_input("预热动量", value=0.8, format="%.1f")

            st.markdown("##### 损失函数参数")
            box = st.number_input("框损失增益", value=7.5, format="%.1f")
            cls = st.number_input("分类损失增益", value=0.5, format="%.1f")
            dfl = st.number_input("DFL损失增益", value=1.5, format="%.1f")
            fl_gamma = st.number_input("Focal损失 gamma", value=0.0, format="%.1f")

            st.markdown("##### 数据增强参数")
            hsv_h = st.number_input("HSV色调增强", value=0.015, format="%.3f")
            hsv_s = st.number_input("HSV饱和度增强", value=0.7, format="%.1f")
            hsv_v = st.number_input("HSV亮度增强", value=0.4, format="%.1f")
            degrees = st.number_input("旋转角度", value=train_cfg["degrees"])
            translate = st.number_input("平移比例", value=0.1, format="%.1f")
            scale = st.number_input("缩放比例", value=0.5, format="%.1f")
            shear = st.number_input("剪切角度", value=0.0, format="%.1f")
            perspective = st.number_input("透视变换", value=0.0, format="%.3f")
            flipud = st.number_input("上下翻转概率", value=train_cfg["flipud"])
            fliplr = st.number_input("左右翻转概率", value=train_cfg["fliplr"])
            mosaic = st.number_input("Mosaic增强", value=1.0, format="%.1f")
            mixup = st.number_input("Mixup增强", value=0.0, format="%.1f")
            copy_paste = st.number_input("Copy-Paste增强", value=0.0, format="%.1f")

            st.markdown("##### 其他参数")
            optimizer = st.selectbox("优化器", ["auto", "SGD", "Adam", "AdamW"], index=0)
            plots = st.checkbox("生成可视化图表", value=train_cfg["plots"])
            resume = st.checkbox("断点续训", value=False)
            exist_ok = st.checkbox("覆盖已有任务", value=True)

        col_save, col_run = st.columns(2)
        save_train_btn = col_save.form_submit_button("💾 保存训练配置")
        run_train_btn = col_run.form_submit_button("▶️ 开始训练")

    # 保存训练配置
    if save_train_btn:
        device_val = "cpu" if device == "CPU" else 0  # 转换设备参数
        config["train_config"] = {
            "no_skip": train_no_skip, "model_path": model_path, "data_yaml": data_yaml, "imgsz": imgsz,
            "name": task_name, "epochs": epochs, "batch": batch, "device": device_val, "patience": patience,
            "lr0": lr0, "lrf": lrf, "momentum": momentum, "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs, "warmup_momentum": warmup_momentum,
            "box": box, "cls": cls, "dfl": dfl,
            "hsv_h": hsv_h, "hsv_s": hsv_s, "hsv_v": hsv_v, "degrees": degrees,
            "translate": translate, "scale": scale, "shear": shear, "perspective": perspective,
            "flipud": flipud, "fliplr": fliplr, "mosaic": mosaic, "mixup": mixup, "copy_paste": copy_paste,
            "optimizer": optimizer, "plots": plots, "resume": resume, "exist_ok": exist_ok
        }
        save_yaml(config, CONFIG_PATH)
        st.success("✅ 训练配置保存成功！")

    # 执行训练
    if run_train_btn:
        if not train_no_skip:
            st.warning("已跳过训练")
        else:
            progress_bar = st.progress(0, text="准备开始训练...")
            log_container = st.container()
            log_area = log_container.empty()
            logs = []
            current_task_name = task_name
            
            def update_progress(current, total, desc=""):
                """更新进度条回调函数"""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress, text=desc)
            
            def update_log(msg):
                """更新日志显示回调函数"""
                logs.append(msg)
                log_area.code("\n".join(logs[-100:]), language="plaintext")  # 显示最近100条
            
            try:
                update_log("🚀 开始YOLO模型训练...")
                from main import AutoYOLOPipeline

                pipeline = AutoYOLOPipeline(
                    config_path=CONFIG_PATH,
                    progress_callback=update_progress,
                    log_callback=update_log
                )
                pipeline.resize_step = lambda: None  # 跳过缩放步骤（已提前完成）
                pipeline.annotate_step = lambda: asyncio.sleep(0)  # 跳过标注步骤（已提前完成）
                pipeline.train_val()  # 执行数据集划分
                pipeline.train_step()  # 执行模型训练
                progress_bar.progress(1.0, text="✅ 完成！")
                st.success("✅ 模型训练完成！")
                st.session_state['last_task_name'] = current_task_name  # 记录任务名用于结果展示
            except Exception as e:
                progress_bar.progress(1.0, text="❌ 失败")
                st.error(f"训练失败：{str(e)}")

    st.divider()
    st.subheader("📊 训练结果可视化")
    
    # 获取上一次训练的任务名
    display_task_name = st.session_state.get('last_task_name', task_name)
    result_dir = get_train_result_dir(display_task_name)  # 获取训练结果目录
    
    if validate_path(result_dir):  # 检查结果目录是否存在
        df = load_train_results(result_dir)  # 加载训练结果CSV
        if df is not None:
            st.pyplot(plot_loss_curve(df))  # 绘制损失曲线
            st.pyplot(plot_metrics_curve(df))  # 绘制评估指标曲线
            cm_img = get_confusion_matrix(result_dir)  # 获取混淆矩阵
            if cm_img is not None:
                st.image(cm_img, caption="混淆矩阵")
            batch_img = get_train_batch_samples(result_dir)  # 获取批次样本图
            if batch_img is not None:
                st.image(batch_img, caption="训练批次样本")
        else:
            st.warning("训练结果文件不存在或格式错误，请确认训练已完成")
    else:
        st.info("暂无训练结果，请先执行训练")
