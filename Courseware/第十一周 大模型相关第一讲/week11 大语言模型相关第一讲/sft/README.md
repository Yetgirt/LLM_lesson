# 大语言模型SFT微调示例

这是一个使用TRL框架进行大语言模型监督式微调（SFT）的简洁示例。

## 特点

- 使用Qwen2-0.5B-Instruct模型（约500M参数），中文支持好，适合普通电脑运行
- 使用主流的TRL框架
- 包含中文问答数据集
- 支持完全离线运行（模型下载到本地）
- 代码简洁，易于理解

## 环境要求

- Python 3.8+
- 至少4GB显存（GPU）或8GB内存（CPU）
- 建议使用GPU加速训练
- 首次运行需要网络下载模型（约1GB），后续可完全离线

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 0. 下载模型（首次运行，需要网络）

首次运行前，需要先下载模型到本地：

```bash
python download_model.py
```

这将下载Qwen2-0.5B-Instruct模型到 `./models/Qwen2-0.5B-Instruct` 目录（约1GB，需要几分钟）。

### 1. 训练模型（可离线运行）

```bash
python sft_train.py
```

训练过程会：
- 从本地加载模型（无需网络）
- 使用内置的中文问答数据进行微调
- 将微调后的模型保存到 `./output` 目录

### 2. 测试模型

```bash
python test_model.py
```

## 离线运行

本脚本设计为完全支持离线运行：

1. **首次运行**（需要网络）：执行 `python download_model.py` 下载模型到 `./models/` 目录
2. **后续运行**（完全离线）：模型已保存在本地，直接运行 `python sft_train.py` 即可，无需网络连接
3. 如果需要将代码复制到其他电脑，将整个项目文件夹（包括 `./models/` 目录）一起复制即可

## 自定义数据

修改 `sft_train.py` 中的 `create_chinese_dataset()` 函数，添加你自己的问答数据。

数据格式：
```python
{
    "instruction": "问题",
    "input": "",  # 可选
    "output": "答案"
}
```

## 注意事项

- 首次运行需要下载模型，需要网络连接
- 如果显存不足，可以减小 `per_device_train_batch_size` 和 `MAX_SEQ_LENGTH`
- CPU训练会非常慢，建议使用GPU
- 训练完成后，可以使用 `test_model.py` 测试模型效果

## 模型信息

本示例使用 **Qwen2-0.5B-Instruct** 模型（约500M参数），这是阿里开源的小型中文语言模型，特点：
- 中文支持好，理解能力强
- 参数量适中，适合普通电脑运行（4GB显存或8GB内存）
- 开源可靠，社区支持好
- 适合学习和实验SFT流程

