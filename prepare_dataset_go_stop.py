import os
import shutil
import random
import pandas as pd

# 路径配置

DATASET_ROOT = r"E:\datasets\lisa"    #原始数据集根目录
CSV_PATH = r"E:\datasets\lisa\frameAnnotationsBULB.csv"  #LISA 数据集里保存图片和标签的 CSV 文件
IMAGE_ROOT = r"E:\datasets\lisa\dayTrain"  #实际存放图片的文件夹
OUTPUT_ROOT = r"E:\traffic_light_recognition\data"   #整理后的输出路径（训练集和验证集
TRAIN_RATIO = 0.8   # 80% 训练，20% 验证


def main():
    copied = 0
    missing = 0

    # 读取 CSV（LISA 用 ; 分隔）
    df = pd.read_csv(CSV_PATH, sep=';')

    # 只保留 go / stop
    df = df[df.iloc[:, 1].isin(['go', 'stop'])]

    # 创建输出目录
    for split in ["train", "val"]:
        for label in ["go", "stop"]:
            os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok=True)

    # 建立：src_path -> label
    samples = []

    for _, row in df.iterrows(): #遍历CSV的每一行
        csv_path = row.iloc[0]   # e.g. dayTraining/dayClip1/dayClip1--00001.jpg
        label = row.iloc[1] #go，stop

        # 只取真正的文件名
        filename = os.path.basename(csv_path)

        # 真实图片路径：E:\datasets\lisa\dayTrain\xxx.jpg
        src = os.path.join(IMAGE_ROOT, filename)

        samples.append((src, label)) #列表（图片路径，标签）

    # 打乱 & 划分
    random.shuffle(samples) #随机打乱
    split_idx = int(len(samples) * TRAIN_RATIO)

    train_set = samples[:split_idx]
    val_set = samples[split_idx:]

    # 拷贝训练集
    for src, label in train_set:
        if os.path.exists(src):
            dst = os.path.join(OUTPUT_ROOT, "train", label, os.path.basename(src))
            shutil.copy(src, dst)
            copied += 1 #记录数量
        else:
            missing += 1

    # 拷贝验证集
    for src, label in val_set:
        if os.path.exists(src):
            dst = os.path.join(OUTPUT_ROOT, "val", label, os.path.basename(src))
            shutil.copy(src, dst)
            copied += 1
        else:
            missing += 1

    # 结果统计
    print("数据集整理完成")
    print(f"训练集: {len(train_set)}")
    print(f"验证集: {len(val_set)}")
    print(f"成功拷贝: {copied}")
    print(f"未找到图片: {missing}")


if __name__ == "__main__":
    main()
