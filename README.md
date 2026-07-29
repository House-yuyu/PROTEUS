<div align="center">

# 🌊 Degradation-Guided Underwater Image Restoration with Task-Oriented Latent Control

![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=House-yuyu.PROTEUS)

</div>

---


This is the official PyTorch codes for the paper:

>**From Degradation Guidance to Latent Purification: A Unified Network for Underwater Image Restoration**<br>  [Xu Zhang<sup>1</sup>](https://house-yuyu.github.io/), Xuhui Cao<sup>1</sup>, Kangzhe Yuan<sup>1</sup>, [Laibin Chang<sup>1</sup>](https://scholar.google.com.hk/citations?user=1l8X8PgAAAAJ&hl=zh-CN), [Huan Zhang<sup>2</sup>](https://scholar.google.com.hk/citations?user=bJjd_kMAAAAJ&hl=zh-CN), [Lefei Zhang<sup>1📧</sup>](https://scholar.google.com.hk/citations?user=BLKHwNwAAAAJ&hl=zh-CN)<br>
> <sup>1</sup>Wuhan University, <sup>2</sup>Guangdong University of Technology<br>
> <sup>📧</sup>Corresponding author.

![teaser_img](fig/model.png)


:star: If PROTEUS is helpful to your images or projects, please help star this repo. Thank you! :point_left:


## Table of Contents
- [Environment](#Environment)
- [Datasets](#Datasets)
- [Dataset Structure](#Dataset-Structure)
- [Quick Start](#Quick-Start)
---
## Environment
### **Installation Steps**
1. **Clone the repository**
   ```bash
   git clone https://github.com/House-yuyu/PROTEUS.git
   cd PROTEUS
   ```
2. **Create a virtual environment**
   ```bash
   conda create -n PROTEUS python=3.10 -y
   conda activate PROTEUS
   ```
3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```
---


## Datasets

The Datasets are hosted on Baidu Netdisk (百度网盘):

-  [Link / 链接](https://pan.baidu.com/s/1qaMhFXSfK3iSk1E9M8qYOA)  
- **Extraction code / 提取码**: `1222`

We conduct experiments on several standard benchmarks for underwater image restoration, which can be grouped into two categories:

### 1) Paired datasets with synthetic references

| Dataset | Source | Train | Val | Test |
|---------|--------|-------|-----|------|
| UIEB | [Li et al., TIP 2020](https://li-chongyi.github.io/proj_benchmark.html) | 800 | — | 90 |
| LSUI | [Peng et al., TIP 2023](https://lintaopeng.github.io/) | 3,879 | — | 400 |
| UFO | [Islam et al., RAL 2020](https://irvlab.cs.umn.edu/resources) | 1,200 | 300 | 120 |
| EUVP-Scene | [Islam et al., RAL 2020](https://irvlab.cs.umn.edu/resources/euvp-dataset) | 1,748 | 218 | 218 |
| EUVP-Dark | [Islam et al., RAL 2020](https://irvlab.cs.umn.edu/resources/euvp-dataset) | 4,440 | 555 | 555 |

### 2) Non-referenced datasets (test only)

| Dataset | No. of Images | Description |
|---------|--------------|-------------|
| Challenging-60 | 60 | Real-world subset of UIEB without ground-truth references |
| U45 | 45 | Diverse collection of real underwater images |
| UCCS | 300 | Underwater color cast scenes (blue, blue-green, green) |
| EUVP-330 | 330 | Non-reference subset from EUVP |

### Evaluation Metrics

- **Reference-based**: PSNR, SSIM, LPIPS
- **Non-reference**: UCIQE, UIQM, URanker

---

## Dataset-Structure

Please organize the downloaded datasets as follows:

```
PROTEUS/
├── data/
│   ├── UIEB/
│   │   ├── train/
│   │   │   ├── input/        # 800 degraded underwater images
│   │   │   └── GT/           # 800 reference images
│   │   └── test/
│   │       ├── U90/          # 90 ref-based test images
│   │       ├── Challenge-60/ # 60 non-ref test images
│   │       └── GT/           # GT for U90 only
│   │
│   ├── LSUI/
│   │   ├── train/
│   │   │   ├── input/        # 3,879 images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 400 images
│   │       └── GT/
│   │
│   ├── UFO/
│   │   ├── train/
│   │   │   ├── input/        # 1,200 images
│   │   │   └── GT/
│   │   ├── val/
│   │   │   ├── input/        # 300 images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 120 images
│   │       └── GT/
│   │
│   ├── EUVP-Scene/
│   │   ├── train/
│   │   │   ├── input/        # 1,748 images
│   │   │   └── GT/
│   │   ├── val/
│   │   │   ├── input/        # 218 images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 218 images
│   │       └── GT/
│   │
│   ├── EUVP-Dark/
│   │   ├── train/
│   │   │   ├── input/        # 4,440 images
│   │   │   └── GT/
│   │   ├── val/
│   │   │   ├── input/        # 555 images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 555 images
│   │       └── GT/
│   │
│   └── test_sets/
│       ├── U45/              # 45 images (non-ref)
│       ├── UCCS/             # 300 images (non-ref)
│       └── EUVP-330/         # 330 images (non-ref)
│
├── Neg_dir/                  # Negative sample data directory
├── utils/
├── train.py
├── test.py
├── requirements.txt
└── README.md
```
---
## Quick-Start
### train
```bash
./train.sh
```
### test
```bash
./test.sh
```

## Contact

If you have any questions, please feel free to reach us out at <a href="zhangx0802@whu.edu.cn">zhangx0802@whu.edu.cn</a>.
