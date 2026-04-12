<div align="center">

# 🌊 From Degradation Guidance to Latent Purification: A Unified Network for Underwater Image Restoration

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

</div>

---



## Table of Contents
- [Environment](#environment)
- [Datasets](#数据集--datasets)
- [Dataset Structure](#数据集目录结构--dataset-structure)
- [Quick Start](#快速开始--quick-start)
---
## Environment
### 安装步骤
1. **克隆仓库**
   ```bash
   git clone https://github.com/House-yuyu/PROTEUS.git
   cd PROTEUS
   ```
2. **创建虚拟环境（推荐）**
   ```bash
   conda create -n PROTEUS python=3.10 -y
   conda activate PROTEUS
   ```
3. **安装其余依赖**
   ```bash
   pip install -r requirements.txt
   ```
### 主要依赖库
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
scikit-image>=0.18.0
matplotlib>=3.4.0
tqdm>=4.62.0
tensorboard>=2.7.0
PyYAML>=5.4.0
```
---
## Datasets
本项目支持以下常用低光照图像增强数据集：
### 1. LOL Dataset
- **来源**: [Retinex-Net (Wei et al., BMVC 2018)](https://daooshee.github.io/BMVC2018website/)
- **训练集**: 485 对图像
- **测试集**: 15 对图像
- **图像分辨率**: 400×600
- **下载地址**: [Google Drive](https://drive.google.com/file/d/157bjO1_cFoSJ2IaUNnEkBDsXUOXajUTa/view)
### 2. VE-LOL Dataset
- **来源**: [From Noise to Signal (Liu et al., IJCV 2021)](https://flyywh.github.io/IJCV2021LowLight_VELOL/)
- **合成训练集**: 2,500 对图像
- **真实测试集**: 100 张图像
### 3. MIT-Adobe FiveK Dataset
- **来源**: [MIT-Adobe FiveK (Bychkovsky et al., CVPR 2011)](https://data.csail.mit.edu/graphics/fivek/)
- **图像数量**: 5,000 张（训练 4,500 / 测试 500）
### 4. LSRW Dataset
- **来源**: [LSRW (Hai et al., ACM MM 2023)](https://github.com/JianghaiSCU/R2RNet)
- **图像数量**: 5,650 对（训练 5,200 / 测试 450）
---
## Dataset Structure
请将数据集下载后按如下目录结构组织：
```
LLIEmm/
├── data/
│   ├── LOL/
│   │   ├── train/
│   │   │   ├── low/          # 低光照输入图像 (485 张)
│   │   │   │   ├── 1.png
│   │   │   │   ├── 2.png
│   │   │   │   └── ...
│   │   │   └── high/         # 对应正常光照参考图像 (485 张)
│   │   │       ├── 1.png
│   │   │       ├── 2.png
│   │   │       └── ...
│   │   └── test/
│   │       ├── low/          # 低光照测试图像 (15 张)
│   │       │   ├── 1.png
│   │       │   └── ...
│   │       └── high/         # 参考图像 (15 张)
│   │           ├── 1.png
│   │           └── ...
│   │
│   ├── VE-LOL/
│   │   ├── train/
│   │   │   ├── low/
│   │   │   └── high/
│   │   └── test/
│   │       ├── low/
│   │       └── high/
│   │
│   ├── FiveK/
│   │   ├── train/
│   │   │   ├── input/        # 原始低曝光图像
│   │   │   └── expert_C/     # Expert C 精修图像（常用参考）
│   │   └── test/
│   │       ├── input/
│   │       └── expert_C/
│   │
│   └── LSRW/
│       ├── train/
│       │   ├── low/
│       │   └── high/
│       └── test/
│           ├── low/
│           └── high/
│
├── models/                   # 模型定义
├── utils/                    # 工具函数
├── configs/                  # 配置文件
├── train.py                  # 训练脚本
├── test.py                   # 测试脚本
├── requirements.txt          # Python 依赖
└── README.md
```
---
## Quick Start
### 训练
```bash
python train.py --config configs/lol.yaml --dataset LOL --data_root data/LOL
```
### 测试
```bash
python test.py --config configs/lol.yaml --checkpoint checkpoints/best_model.pth \
               --input data/LOL/test/low --output results/
```
### 评估指标
本项目使用以下指标评估增强效果：
| 指标 | 说明 |
|------|------|
| PSNR | 峰值信噪比（越高越好）|
| SSIM | 结构相似性（越高越好）|
| LPIPS | 感知图像相似度（越低越好）|
---
## 引用 / Citation
如果本项目对您的研究有帮助，请引用：
```bibtex
@article{lliemm2026,
  title   = {LLIEmm: Low-Light Image Enhancement},
  author  = {Xu Zhang},
  journal = {arXiv preprint},
  year    = {2026}
}
```
---
## 致谢 / Acknowledgements
感谢以下数据集和开源项目的贡献：
- [LOL Dataset](https://daooshee.github.io/BMVC2018website/)
- [VE-LOL Dataset](https://flyywh.github.io/IJCV2021LowLight_VELOL/)
- [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/)
- [LSRW Dataset](https://github.com/JianghaiSCU/R2RNet)
---
## 联系方式 / Contact
如有问题，请联系：**zhangx0802@whu.edu.cn**
