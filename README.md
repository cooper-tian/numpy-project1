# Project 1 — NumPy 图像编码器

## 项目简介

使用 NumPy 复现 OpenCLIP ViT-B-32 图像编码器的前向计算，采用课程提供的预训练权重，无需训练模型。输入为预处理后的 RGB 图像 `[B, 3, 224, 224]`，输出为每张图像对应的 512 维单位特征向量 `[B, 512]`。

## 文件说明

- `image_encoder.py`：图像分块、输入嵌入、多头注意力、Transformer 模块和最终特征输出。
- `numpy_homework.py`：NumPy 基础与中等练习（NP-E1–E8、NP-M1–M4）。

课程提供的辅助函数、检查程序、权重、配置及图片资源不在本仓库内，运行主项目时需使用原始课程附件。

## 运行方法

将本仓库的 `image_encoder.py` 复制到解压后的课程附件根目录，与 `helpers.py`、`check.py`、`run.py` 同级。在该目录下执行：

```powershell
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

输出文件包含图片 ID（`ids`）和图像特征（`output_tokens`），可用 `np.load` 读取。

独立练习只需 NumPy，在 `numpy_homework.py` 所在目录执行：

```powershell
python -m pip install numpy
python numpy_homework.py
```

## 实验结果

本地使用课程原版检查器完成全部阶段的数值验证，资源校验通过，各计算阶段结果如下：

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 | 结论 |
|---|---|---:|---:|---|
| preprocess | (10, 3, 224, 224) | 0.000e+00 | 0.000e+00 | PASS |
| patches | (2, 49, 3072) | 0.000e+00 | 0.000e+00 | PASS |
| tokens | (2, 50, 768) | 4.387e-05 | 1.335e-06 | PASS |
| block0 | (2, 50, 768) | 7.147e-06 | 3.348e-07 | PASS |
| last_tokens | (2, 50, 768) | 4.196e-05 | 2.633e-06 | PASS |
| output | (10, 512) | 5.662e-07 | 8.688e-08 | PASS |
| batch | (3, 512) | 4.061e-07 | 7.848e-08 | PASS |

## 课程资源与辅助说明

本项目基于课程提供的原始框架、辅助函数与模型参数，由本人借助AI工具进行逐步学习、测试。
