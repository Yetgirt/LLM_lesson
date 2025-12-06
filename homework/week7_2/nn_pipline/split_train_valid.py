# -*- coding: utf-8 -*-
"""
文件名: split_train_valid.py
功能: 把 文本分类练习.csv 按 8:2 随机拆分成 train.csv 和 valid.csv
运行方式: python split_train_valid.py
"""

import os
import random
import csv
from pathlib import Path
from config import Config
# ==================== 配置区（改这里就行）====================
TRAIN_RATIO = 0.8                        # 训练集占比（80%）
SEED = 42                                # 随机种子，保证每次结果一样（方便复现）

# 输出文件（会自动创建在同目录下）
# =======================================================

def split_csv(config):
    if not Path(config["origin_data_path"]).exists():
        print(f"错误：找不到文件 {config["origin_data_path"]}")
        return

    print(f"正在读取 {config["origin_data_path"]} ...")
    with open(config["origin_data_path"], encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        print("错误：文件太小或只有标题行")
        return

    # 第一行通常是标题
    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []

    print(f"总数据量: {len(data)} 条（不含标题）")

    # 随机打乱
    random.seed(SEED)
    random.shuffle(data)

    # 计算分割点
    split_idx = int(len(data) * TRAIN_RATIO)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(valid_data)} 条")

    # 确保输出目录存在
    os.makedirs(Path(config["train_data_path"]).parent, exist_ok=True)

    # 写入训练集
    with open(config["train_data_path"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)      # 写标题
        writer.writerows(train_data)

    # 写入验证集
    with open(config["valid_data_path"], "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(valid_data)

    print(f"分割完成！")
    print(f"   训练集 → {config["train_data_path"]}")
    print(f"   验证集 → {config["valid_data_path"]}")

if __name__ == "__main__":
    split_csv(Config)