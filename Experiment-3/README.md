# 实验三：图像分类 (CIFAR-10)

## 1. 实验说明
本实验基于 PyTorch 设计并实现了一个自定义卷积神经网络 (CNN)，用于 CIFAR-10 数据集的图像分类任务。最终在测试集上取得了 **90.18%** 的准确率。

## 2. 环境依赖
- Python 3.10+
- PyTorch, Torchvision
- Matplotlib, Seaborn, Scikit-learn

## 3. 运行方式
- **训练主程序（含数据增强）**：`python main.py`
- **消融实验程序（无数据增强）**：`python main_no_aug.py`

## 4. 文件说明
- `main.py`: 主训练脚本，包含随机裁剪、水平翻转、学习率衰减与早停机制。
- `main_no_aug.py`: 对比实验脚本，移除了数据增强策略，用于验证过拟合现象。
- `training_curves.png`: 训练与验证的 Loss / Accuracy 曲线。
- `confusion_matrix.png`: 归一化混淆矩阵。
- `misclassified.png`: 误分类样本可视化图。
