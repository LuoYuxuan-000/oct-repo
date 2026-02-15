# PESNet — Polar Eyeball Shape Net

> **论文复现**: Zhang et al., *Polar Eyeball Shape Net for 3D Posterior Ocular Shape Representation*, MICCAI 2023

本仓库在没有官方代码的前提下，从零复现 PESNet 的完整 pipeline，使用 OCTA-500 数据集验证"局部 OCT → 完整后眼球形状重建"的可行性。

## 方法概述

<p align="center"><b>OCT 图像 → 极坐标体素化 → 双分支网络 → 2D OSM → 逆变换 → 3D 眼球形状</b></p>

| 组件 | 说明 | 论文对应 |
|------|------|---------|
| **PVB** (Polar Voxelization Block) | 稀疏点云 → 极坐标密集网格 `(R,U,V)` | Sec 2.2, Eq.2-4 |
| **S-branch** (Shape regression) | 局部 subarea 特征提取 (5 层 RB) | Fig.1(b) |
| **A-branch** (Anatomical prior) | 模板先验特征提取 (5 层 RB, 权重不共享) | Fig.1(b) |
| **RFB** (R-wise Fusion Block) | R 维交错融合两分支特征 | Fig.1(c) |
| **Loss** | α·SmoothL1 + VGG16 Perceptual Loss | Sec 2.1, Table 2 |
| **OSM → 3D** | 逆极坐标变换 + 论文去噪策略 | Sec 2.2 |

## 项目结构

```
PESNet_repo_github/
├── train.py                 # 训练入口
├── eval.py                  # 评估入口
├── configs/
│   └── pesnet_octa500.yaml  # 训练配置
├── pesnet/                  # 核心代码包
│   ├── geometry/polar.py    # 极坐标变换 (Eq.2-4)
│   ├── models/pesnet.py     # PESNet 双分支网络 (Fig.1)
│   ├── losses/pesnet_loss.py# 损失函数 (α·L1+Perceptual)
│   ├── datasets/processed.py# 数据加载器
│   └── utils/               # 指标/点云处理/可视化
├── tools/
│   └── prepare_octa500.py   # OCTA-500 数据预处理
└── tests/                   # 单元测试
```

## 快速开始

### 0. 环境

```bash
conda create -n pesnet python=3.10
conda activate pesnet
pip install -r requirements.txt
```

### 1. 数据准备

```bash
python tools/prepare_octa500.py \
  --octa500_root /path/to/OCTA-500 \
  --out_root data/processed/octa500_usage1 \
  --overwrite
```

生成:
- `splits.json` — train/val/test 划分 (70%/20%/10%)
- `samples/{id}.npz` — 每个 volume 的 GT OSM + 5 个 subarea 极坐标网格
- `template_grid_uint8.npy` — 模板（top-7 训练样本平均 + 高斯平滑）

### 2. 训练

```bash
python train.py --config configs/pesnet_octa500.yaml
```

覆盖参数:
```bash
python train.py --config configs/pesnet_octa500.yaml --set training.lr=5e-4 --set training.epochs=200
```

输出: `outputs/runs/<timestamp>/checkpoints/{best,last}.pt`

### 3. 评估

```bash
python eval.py --config configs/pesnet_octa500.yaml --run_dir outputs/runs/<run>
```

输出: `eval/metrics.json` + 每个样本的预测 OSM / 点云 PLY 文件

### 4. 测试

```bash
python -m pytest tests/ -v
```

## 论文关键设定对齐

| 设定 | 论文 | 本复现 |
|------|------|--------|
| 极坐标网格 | R=64, U=180, V=180 | R=64, U=180, V=180 |
| 极坐标原点 | top en-face 中心 (W/2, H/2, 0) | 同上 |
| 网络 | 双分支 unshared + RFB 融合 | 同上 |
| 损失 | α·SmoothL1 + Perceptual(VGG16) | 同上 |
| α 衰减 | 前 50 epoch 固定，之后线性→0 | 同上 |
| 训练 | Adam, lr=0.001, batch=10, 300 epochs | 同上 |
| 评估指标 | CD / SSIM / Density | 同上 |
| 去噪策略 | OSM(u,v) != OSM(5,5) | 同上 |

## 评估指标

| 指标 | 含义 | 方向 |
|------|------|------|
| **CD** (Chamfer Distance) | 预测与 GT 点云的平均最近邻距离 | ↓ 越小越好 |
| **SSIM** | 2D OSM 结构相似性 | ↑ 越大越好 |
| **Density** | \|1 - N_pred/N_gt\|，有效区域完整性 | ↓ 越小越好 |

## 参考文献

```
@inproceedings{zhang2023pesnet,
  title={Polar Eyeball Shape Net for 3D Posterior Ocular Shape Representation},
  author={Zhang, Jiaqi and Hu, Yan and Qi, Xiaojuan and Meng, Ting and Wang, Lihui and Fu, Huazhu and Yang, Mingming and Liu, Jiang},
  booktitle={MICCAI},
  year={2023}
}
```
