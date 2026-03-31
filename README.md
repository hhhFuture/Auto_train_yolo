#### 用大模型进行标注，然后进行yolo训练的全流程，无须任何操作，只需要准备数据集即可

##### 需要安装的包

1.需要先安装ultralytics和streamlit 

~~~shell
# Install or upgrade the ultralytics package from PyPI
pip install -U ultralytics
pip install streamlit 
~~~

2.再安装volcengine-python-sdk,这个必须要装！

~~~shell
pip install volcengine-python-sdk[ark]
~~~

3.不同的目标需要自己去写不同的提示词



##### 启动方式：

~~~shell
streamlit run streamlit_app.py
~~~



##### 配置注意地方：

图像缩放、豆包标注、训练yolo，这三个功能都是可以单独运行的，即可以只运行他们独立的py文件夹，并且在配置项中有no_skip，哪些功能不用的，可以直接跳过，无须从头开始运行。

###### config.yaml文件说明

~~~yaml
doubao_config:
  no_skip: true  # 是否跳过豆包标注步骤，True=执行标注，False=跳过
  api_key: cb.....  # 豆包API密钥，用于认证API请求
  model: doubao-seed-2-0-mini-260215  # 豆包视觉大模型名称
  api_url: https://ark.cn-beijing.volces.com/api/v3/chat/completions  # 豆包API调用地址（固定的,无须更改）
  input_images_path: C:\Users\Administrator\Desktop\me\test\images  # 待标注图片所在文件夹路径，文件夹名必须以images结尾
  prompt_path: E:/Auto_YOLO/Data_Process/prompt.txt  # 标注提示词文件路径
  label_class: '0'  # 目标类别ID，YOLO标注文件中的类别编号
  concurrency: 20  # 异步并发数，同时处理的图片数量
  timeout: 60  # API请求超时时间，单位秒
resize_data_config:
  no_skip: true  # 是否跳过缩放步骤，True=执行缩放，False=跳过
  input_path: C:\Users\Administrator\Desktop\me\test\images_org  # 原始图片文件夹路径
  output_path: C:\Users\Administrator\Desktop\me\test\images  # 缩放后图片输出文件夹路径，文件夹名必须以images结尾
  target_size: 640  # 目标尺寸，图片最长边将缩放至此值
train_config:
  no_skip: true  # 是否跳过训练步骤，True=执行训练，False=跳过
  model_path: E:/Auto_YOLO/yolov8n.pt  # 预训练模型权重文件路径
  data_yaml: E:/All_Data/face/f.yaml  # 数据集配置文件路径
  imgsz: 640  # 训练图像尺寸，正方形边长
  name: face  # 训练任务名称，用于区分不同实验
  epochs: 100  # 训练轮数，整个数据集训练遍历次数
  batch: 64  # 批次大小，每次迭代处理的图片数量
  device: 0  # 训练设备，0=GPU:0，cpu=使用CPU训练
  patience: 300  # 早停耐心值，验证集指标未改善的最多epoch数
  lr0: 0.01  # 初始学习率，训练开始时的学习率
  lrf: 0.01  # 最终学习率，训练结束时的学习率
  momentum: 0.937  # SGD优化器的动量参数
  weight_decay: 0.0005  # 权重衰减，防止模型过拟合的正则化参数
  warmup_epochs: 3.0  # 预热轮数，学习率从低到正常值的过渡轮数
  warmup_momentum: 0.8  # 预热阶段动量值
  box: 7.5  # 框损失权重，边界框回归损失在总损失中的占比
  cls: 0.5  # 分类损失权重，分类损失在总损失中的占比
  dfl: 1.5  # DFL损失权重，分布焦点损失在总损失中的占比
  hsv_h: 0.015  # HSV色调增强范围，色相随机扰动系数
  hsv_s: 0.7  # HSV饱和度增强范围，饱和度随机扰动系数
  hsv_v: 0.4  # HSV亮度增强范围，亮度随机扰动系数
  degrees: 0.0  # 随机旋转角度范围，单位度
  translate: 0.1  # 平移变换比例，图片随机平移的比例
  scale: 0.5  # 缩放变换比例，图片随机缩放的比例
  shear: 0.0  # 剪切变换角度，图片随机剪切的角度
  perspective: 0.0  # 透视变换范围，图片随机透视变换的程度
  flipud: 0.5  # 上下翻转概率，图片随机上下翻转的比例
  fliplr: 0.5  # 左右翻转概率，图片随机左右翻转的比例
  mosaic: 1.0  # Mosaic增强概率，将4张图片拼接成1张的比例
  mixup: 0.0  # Mixup增强概率，两张图片混合叠加的比例
  copy_paste: 0.0  # Copy-Paste增强概率，目标复制粘贴的比例
  optimizer: auto  # 优化器类型，auto/SGD/Adam/AdamW自动选择
  plots: true  # 是否生成训练可视化图表
  resume: false  # 是否断点续训，从上次中断处继续训练
  exist_ok: true  # 是否允许同名任务存在，True=覆盖已有结果

~~~

