# Traffic Light Recognition (Go / Stop)
本项目为深度学习课程大作业，实现了基于 **ResNet18** 的红绿灯（Go / Stop）状态识别任务。  
项目包含完整的 **代码、训练流程、实验结果以及数据集**，可直接复现实验。
---
## 1. 项目简介
针对交通场景中的红绿灯状态识别问题，构建了一个二分类模型（Go / Stop），主要流程包括：

- 数据集预处理与划分
- 基于 ResNet18 的模型训练
- 训练过程可视化（loss / accuracy 曲线）
- 训练完成模型的推理测试
---
## 2. 项目结构说明
```text
traffic_light_recognition
├── data/                     # 数据集（Git LFS 管理）
│   ├── go/                   # 通行状态图片
│   └── stop/                 # 停止状态图片
├── model/                    # 模型相关代码
├── train/                    # 训练过程与日志
├── utils/                    # 工具函数
├── results/                  # 实验结果输出
├── train_resnet18.py         # 模型训练主程序
├── predict_demo.py           # 模型预测示例
├── prepare_dataset_go_stop.py# 数据集预处理脚本
├── resnet18_go_stop.pth      # 训练完成的模型权重
├── loss_curve.png            # Loss 曲线
├── accuracy_curve.png        # Accuracy 曲线
└── readme.md

实验环境：
本实验在 Windows 操作系统环境下完成，采用 Anaconda 作为 Python 环境管理工具，并在 Anaconda 中创建独立的虚拟环境以保证实验环境的可复现性。
实验所使用的虚拟环境名称为 pytorch_env，主要软件环境配置如下（具体版本通过命令行查询获得）：
操作系统：Windows 10
操作系统：Windows
Python 版本：3.11.13
PyTorch 版本：2.5.1（CUDA 12.1）
CUDA 支持：可用（torch.cuda.is_available() = True）
主要依赖库：NumPy、Pandas、Matplotlib、OpenCV、Torchvision 等
通过使用虚拟环境，可以有效避免不同实验之间的依赖冲突，提高实验的稳定性与可重复性。
