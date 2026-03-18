# LLIEmm
# LLIEmm — Low-Light Image Enhancement
LLIEmm is a PyTorch-based framework for low-light image enhancement (LLIE). It restores visibility, color fidelity, and perceptual quality in images captured under poor illumination conditions.
---
## 目录 / Table of Contents
- [环境要求 / Environment](#环境要求--environment)
- [数据集 / Datasets](#数据集--datasets)
- [数据集目录结构 / Dataset Structure](#数据集目录结构--dataset-structure)
- [快速开始 / Quick Start](#快速开始--quick-start)
---
## Environment
### 系统要求
| 组件 | 版本要求 |
|------|---------|
| 操作系统 | Ubuntu 18.04 / 20.04 / 22.04 (推荐), Windows 10/11 |
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.10.0 |
| torchvision | ≥ 0.11.0 |
| CUDA | ≥ 11.1（GPU 训练推荐）|
| cuDNN | ≥ 8.0 |
### 安装步骤
1. **克隆仓库**
   ```bash
   git clone https://github.com/House-yuyu/LLIEmm.git
   cd LLIEmm
   ```
2. **创建虚拟环境（推荐）**
   ```bash
   conda create -n lliemm python=3.9 -y
   conda activate lliemm
   ```
3. **安装 PyTorch（根据 CUDA 版本选择）**
   ```bash
   # CUDA 11.3
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   # CUDA 11.6
   pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
   # CPU only
   pip install torch torchvision
   ```
4. **安装其余依赖**
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
## 数据集 / Datasets
本项目支持以下常用低光照图像增强数据集：
### 1. LOL 数据集 (Low-Light)
LOL 数据集包含真实场景下采集的低光照/正常光照图像对，是 LLIE 领域最常用的基准数据集之一。
- **来源**: [Retinex-Net (Wei et al., BMVC 2018)](https://daooshee.github.io/BMVC2018website/)
- **训练集**: 485 对图像
- **测试集**: 15 对图像
- **图像分辨率**: 400×600
- **下载地址**: [Google Drive](https://drive.google.com/file/d/157bjO1_cFoSJ2IaUNnEkBDsXUOXajUTa/view)
### 2. VE-LOL 数据集
VE-LOL (Visible and Enhanced Low-Light) 包含合成与真实两种子集，规模更大。
- **来源**: [From Noise to Signal (Liu et al., IJCV 2021)](https://flyywh.github.io/IJCV2021LowLight_VELOL/)
- **合成训练集**: 2,500 对图像
- **真实测试集**: 100 张图像
### 3. MIT-Adobe FiveK 数据集
包含 5,000 张由专业摄影师精修的 RAW 图像，常用于曝光校正与图像增强研究。
- **来源**: [MIT-Adobe FiveK (Bychkovsky et al., CVPR 2011)](https://data.csail.mit.edu/graphics/fivek/)
- **图像数量**: 5,000 张（训练 4,500 / 测试 500）
### 4. LSRW 数据集
LSRW (Large-Scale Real-World) 是大规模真实低光照数据集，图像来自手机相机。
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
