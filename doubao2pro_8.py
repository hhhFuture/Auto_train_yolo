# -*- coding: utf-8 -*-
"""
豆包视觉大模型自动标注模块
使用豆包API对图片进行目标检测标注，生成YOLO格式的标注文件
"""

import os
import base64
import asyncio
import httpx
import cv2
import re
import time
import shutil
import yaml
from pathlib import Path


class DoubaoDetector:
    """
    豆包目标检测标注器
    
    负责调用豆包视觉大模型API对图片进行目标检测，
    将检测结果转换为YOLO格式的标注文件（.txt）
    """

    def __init__(self, config_path="config.yaml", progress_callback=None, log_callback=None):
        """
        初始化豆包标注器
        
        Args:
            config_path: 配置文件路径
            progress_callback: 进度回调函数 callback(current, total, desc)
            log_callback: 日志回调函数 callback(message)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.doubao_config = self.config["doubao_config"]
        self.progress_callback = progress_callback
        self.log_callback = log_callback

        self.model = self.doubao_config["model"]
        self.label_class = self.doubao_config["label_class"]
        self.prompt_path = self.doubao_config["prompt_path"]
        self.api_url = self.doubao_config["api_url"]
        self.api_key = self.doubao_config["api_key"]
        
        # 验证API key是否为有效格式（不能包含中文或特殊字符）
        if not self.api_key or any('\u4e00' <= c <= '\u9fff' for c in self.api_key):
            raise ValueError("API key 无效！请在 config.yaml 中配置正确的豆包 API key（不能包含中文字符）")
        
        self.concurrency = self.doubao_config["concurrency"]
        self.timeout = self.doubao_config["timeout"]
        self.image_folder = self.doubao_config["input_images_path"]

        # 设置输出目录：图片保存目录、标注文件目录、未检测到目标的图片目录
        self.save_root = os.path.dirname(self.image_folder)
        self.img_save_dir = os.path.join(self.save_root, "images_frame")  # 带标注框的原图保存路径
        self.label_save_dir = os.path.join(self.save_root, "labels")  # YOLO标注txt文件保存路径
        self.no_detected_folder = os.path.join(self.save_root, "no_detected")  # 未检测到目标的原图移动路径
        self.log_path = Path(self.save_root) / "log.txt"  # 标注统计日志文件路径

        # 确保所有输出目录存在
        os.makedirs(self.img_save_dir, exist_ok=True)
        os.makedirs(self.label_save_dir, exist_ok=True)
        os.makedirs(self.no_detected_folder, exist_ok=True)

    def _load_config(self):
        """加载YAML配置文件"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _log(self, msg):
        """输出日志信息到控制台和回调"""
        try:
            print(msg)
        except UnicodeEncodeError:
            # Windows控制台编码问题，使用ASCII安全输出
            print(msg.encode('utf-8', errors='replace').decode('utf-8'))
        if self.log_callback:
            self.log_callback(msg)

    def _update_progress(self, current, total, desc=""):
        """更新进度回调"""
        if self.progress_callback:
            self.progress_callback(current, total, desc)

    @staticmethod
    def encode_image(path):
        """
        将图片文件编码为Base64字符串
        
        Args:
            path: 图片文件路径
        
        Returns:
            str: Base64编码的图片数据
        """
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def parse_bboxes(text):
        """
        从API返回的文本中解析出边界框坐标
        
        Args:
            text: API返回的文本内容
        
        Returns:
            list: 边界框坐标字符串列表，格式为"<bbox>x_min y_min x_max y_max</bbox>"
        """
        return re.findall(r"<bbox>(.*?)</bbox>", text, flags=re.S)

    async def process_one_image(self, client, image_path):
        """
        处理单张图片的标注任务
        
        Args:
            client: httpx异步客户端
            image_path: 图片文件路径
        
        Returns:
            tuple: (是否成功, 消耗的token数, 未检测到目标时的图片路径)
        """
        # 读取标注提示词
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            PROMPT = f.read().strip()
        try:
            base64_img = self.encode_image(image_path)  # 将图片转为Base64
            
            # 构建API请求载荷，包含模型名称和消息内容
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{base64_img}"}},  # Base64图片数据
                            {"type": "text", "text": PROMPT},  # 标注指令prompt
                        ],
                    }
                ],
                "thinking": {"type": "disabled"},  # 禁用思考过程，加速返回
            }
            
            # 发送API请求
            resp = await client.post(self.api_url, json=payload)
            resp.raise_for_status()  # 检查HTTP错误
            data = resp.json()  # 解析JSON响应
            content = data["choices"][0]["message"]["content"]  # 获取模型返回内容
            total_tokens = data.get("usage", {}).get("total_tokens", 0)  # 获取消耗的token数

            bboxes = self.parse_bboxes(content)  # 解析返回的边界框

            # 如果没有检测到任何目标，将原图移动到no_detected目录
            if not bboxes:
                dst_path = os.path.join(self.no_detected_folder, os.path.basename(image_path))
                shutil.move(image_path, dst_path)
                return False, total_tokens, dst_path

            # 读取原图获取尺寸，用于坐标转换
            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            # 构建YOLO标注文件路径（与图片同名但扩展名为.txt）
            txt_path = os.path.join(
                self.label_save_dir,
                os.path.splitext(os.path.basename(image_path))[0] + ".txt"
            )

            yolo_lines = []  # YOLO格式行列表

            # 遍历每个检测到的边界框
            for bbox_str in bboxes:
                coords = list(map(int, bbox_str.split()))  # 解析坐标（整数列表）
                x_min, y_min, x_max, y_max = coords

                # 将千分制坐标（0-1000）转换为像素坐标
                x_min = int(x_min * w / 1000)
                y_min = int(y_min * h / 1000)
                x_max = int(x_max * w / 1000)
                y_max = int(y_max * h / 1000)

                # 计算YOLO格式的中心点坐标和宽高（归一化到0-1）
                xc = (x_min + x_max) / 2 / w
                yc = (y_min + y_max) / 2 / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h

                # 写入YOLO格式：类别ID 中心x 中心y 宽度 高度
                yolo_lines.append(f"{self.label_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)  # 绘制红色边界框

            # 保存YOLO标注文件
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            # 保存带标注框的可视化图片
            out_img_path = os.path.join(self.img_save_dir, os.path.basename(image_path))
            cv2.imwrite(out_img_path, img)
            return True, total_tokens, None

        except Exception as e:
            import traceback
            error_msg = f"处理图片 {image_path} 失败：{e}"
            try:
                print(f"❌ {error_msg}")
                print(traceback.format_exc())
            except UnicodeEncodeError:
                print(error_msg.encode('utf-8', errors='replace').decode('utf-8'))
            raise e

    async def run(self):
        """
        启动豆包标注任务的主流程
        
        遍历图片文件夹中的所有图片，使用异步并发方式调用豆包API进行标注，
        生成YOLO格式的标注文件并保存标注结果统计信息
        """
        # 获取待处理图片路径列表
        images = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))  # 只处理图片文件
        ]
        total = len(images)
        self._log(f"共有 {total} 张图片等待处理")

        semaphore = asyncio.Semaphore(self.concurrency)  # 限制并发数量的信号量
        total_tokens_used = 0  # 累计消耗的token数
        no_detected_paths = []  # 未检测到目标的图片路径列表
        processed_count = 0  # 已处理图片计数

        # 创建异步HTTP客户端
        async with httpx.AsyncClient(
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"  # API认证头
                },
                timeout=self.timeout,  # 请求超时时间
        ) as client:

            async def task_wrapper(path):
                """包装任务函数，添加并发控制"""
                async with semaphore:
                    return await self.process_one_image(client, path)

            start = time.time()  # 记录开始时间
            tasks = [task_wrapper(img) for img in images]  # 创建所有任务

            # 使用as_completed实现按完成顺序处理结果
            for coro in asyncio.as_completed(tasks):
                ok, tokens, no_detect_path = await coro
                processed_count += 1
                
                total_tokens_used += tokens  # 累加token消耗
                if no_detect_path:  # 记录未检测到的图片
                    no_detected_paths.append(no_detect_path)
                
                self._update_progress(processed_count, total, f"标注进度: {processed_count}/{total}")
                
                if ok:
                    self._log(f"✅ 已标注: {images[processed_count-1] if processed_count <= len(images) else 'unknown'}")
                else:
                    self._log(f"⚠️ 未检测到目标: {no_detect_path}")

            # 写入统计日志
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write(f"一共处理：{total} 张图片\n")
                f.write(f"消耗总的 token 数量：{total_tokens_used}\n")
                f.write(f"未被检测出目标的图片共：{len(no_detected_paths)} 张\n")
                for p in no_detected_paths:
                    f.write(str(Path(p)) + "\n")
            
            cost_time = time.time() - start  # 计算总耗时
            self._log(f"🔥 全部完成，耗时：{cost_time:.2f} 秒")
            self._log(f"📄 统计信息已写入：{self.log_path}")

        return True
