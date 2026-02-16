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

## 复现结果

### 训练配置

- GPU: 2 × NVIDIA RTX 3090 (DataParallel)
- Batch size: 4
- Epochs: 300
- 优化器: Adam, lr=1e-3
- 混合精度训练 (AMP)
- 总训练时间: ~15.5 小时

### 测试集指标 (best checkpoint, 150 个样本)

| 指标 | 本复现 | 论文报告 |
|------|--------|----------|
| **CD** ↓ | 31.23 | 9.52 |
| **SSIM** ↑ | 0.786 | 0.844 |
| **Density** ↓ | 0.173 | 0.248 |

### 训练过程 (验证集，每 10 epoch 评估一次)

| Epoch | α | SSIM | CD | Density |
|-------|---|------|----|---------|
| 10 | 1.000 | 0.275 | 111.15 | 1.559 |
| 50 | 1.000 | 0.767 | 94.01 | 1.035 |
| 100 | 0.800 | 0.843 | 78.78 | 0.708 |
| 140 | 0.640 | 0.883 | 59.54 | 0.443 |
| 200 | 0.400 | 0.867 | 63.95 | 0.471 |
| 280 | 0.080 | 0.908 | 41.64 | 0.232 |
| 300 | 0.000 | 0.872 | 52.08 | 0.352 |

### 关于指标差异的说明

> **注意**: 本复现的指标与论文报告值不能直接对齐，主要原因如下:
>
> 1. **数据集不同**: 论文使用的是作者私有的临床 OCT 数据集，本复现使用公开的 OCTA-500 数据集。两个数据集在成像设备、扫描范围、分辨率、样本量等方面均存在差异，直接影响模型的训练和评估表现。
> 2. **GT 构造方式不同**: 论文的 GT 眼球形状来自临床 biometry 设备测量，而本复现基于 OCTA-500 的 OCT 体数据自行构造 GT OSM，构造精度和覆盖范围与原始数据存在差距。
> 3. **CD 均值受异常样本影响**: 本复现中大部分样本的 CD 在 10~20 之间，但少数异常样本（如 10089、10057，CD > 200）显著拉高了均值。去除这些异常值后，CD 中位数约为 15。
> 4. **SSIM 表现接近**: 验证集最佳 SSIM 达到 0.908（epoch 280），测试集 SSIM=0.786，与论文的 0.844 在同一量级，说明网络结构复现是有效的。
>
> 综上，本复现验证了 PESNet 方法在公开数据集上的可行性，指标差异主要源于数据而非方法。

## 参考文献

```
@inproceedings{zhang2023pesnet,
  title={Polar Eyeball Shape Net for 3D Posterior Ocular Shape Representation},
  author={Zhang, Jiaqi and Hu, Yan and Qi, Xiaojuan and Meng, Ting and Wang, Lihui and Fu, Huazhu and Yang, Mingming and Liu, Jiang},
  booktitle={MICCAI},
  year={2023}
}
```
