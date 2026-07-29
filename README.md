<div align="center">

# 🌊 Degradation-Guided Underwater Image Restoration with Task-Oriented Latent Control

![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=House-yuyu.PROTEUS)

</div>

---

This repository provides the official PyTorch implementation of the paper:

> **Degradation-Guided Underwater Image Restoration with Task-Oriented Latent Control**
>
> [Xu Zhang<sup>1</sup>](https://house-yuyu.github.io/), Xuhui Cao<sup>1</sup>, Kangzhe Yuan<sup>1</sup>, [Laibin Chang<sup>1</sup>](https://scholar.google.com.hk/citations?user=1l8X8PgAAAAJ&hl=zh-CN), [Huan Zhang<sup>2</sup>](https://scholar.google.com.hk/citations?user=bJjd_kMAAAAJ&hl=zh-CN), [Lefei Zhang<sup>1📧</sup>](https://scholar.google.com.hk/citations?user=BLKHwNwAAAAJ&hl=zh-CN)
>
> <sup>1</sup>Wuhan University, <sup>2</sup>Guangdong University of Technology <sup>📧</sup>Corresponding author.

![teaser\_img](fig/model.png)

:star: If PROTEUS is helpful to your research or projects, please consider giving this repository a star. Thank you! :point_left:

## Table of Contents

* [Environment](#environment)
* [Datasets](#datasets)
* [Pre-trained Models](#pre-trained-models)
* [Dataset Structure](#dataset-structure)
* [Quick Start](#quick-start)
* [Contact](#contact)

---

## Environment

### Installation Steps

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

The datasets used in this project are hosted on Baidu Netdisk (百度网盘):

* [Download Link / 下载链接](https://pan.baidu.com/s/1qaMhFXSfK3iSk1E9M8qYOA)
* **Extraction code / 提取码**: `1222`

We conduct experiments on several standard benchmarks for underwater image restoration. These datasets can be divided into two categories.

### 1. Paired Datasets with Reference Images

| Dataset    | Source                                                                     | Train | Val | Test |
| :--------- | :------------------------------------------------------------------------- | ----: | --: | ---: |
| UIEB       | [Li et al., TIP 2020](https://li-chongyi.github.io/proj_benchmark.html)    |   800 |   — |   90 |
| LSUI       | [Peng et al., TIP 2023](https://lintaopeng.github.io/)                     | 3,879 |   — |  400 |
| UFO        | [Islam et al., RAL 2020](https://irvlab.cs.umn.edu/resources)              | 1,200 | 300 |  120 |
| EUVP-Scene | [Islam et al., RAL 2020](https://irvlab.cs.umn.edu/resources/euvp-dataset) | 1,748 | 218 |  218 |
| EUVP-Dark  | [Islam et al., RAL 2020](https://irvlab.cs.umn.edu/resources/euvp-dataset) | 4,440 | 555 |  555 |

### 2. Non-reference Datasets for Testing

| Dataset        | No. of Images | Description                                                      |
| :------------- | ------------: | :--------------------------------------------------------------- |
| Challenging-60 |            60 | A challenging real-world subset of UIEB without reference images |
| U45            |            45 | A diverse collection of real-world underwater images             |
| UCCS           |           300 | Underwater scenes with blue, blue-green, and green color casts   |
| EUVP-330       |           330 | A non-reference subset collected from EUVP                       |

### Evaluation Metrics

* **Reference-based metrics**: PSNR, SSIM, and LPIPS
* **Non-reference metrics**: UCIQE, UIQM, and URanker

---

## Pre-trained Models

The pre-trained weights of PROTEUS are available on Baidu Netdisk (百度网盘):

* [Download Link / 下载链接](https://pan.baidu.com/s/1hVvuud6MCkTjdDXaRDSj8Q?pwd=iwa5)
* **Extraction code / 提取码**: `iwa5`

Please download and extract the pre-trained weights before running the testing scripts. We recommend organizing the downloaded checkpoints as follows:

```text
PROTEUS/
└── pretrained_models/
    └── checkpoints/
```

After downloading the weights, please modify the checkpoint path in `test.sh` or the corresponding configuration file according to the actual filename and storage location.

The released checkpoints are intended for academic research and non-commercial use. If you use the pre-trained models in your research, please cite our paper and acknowledge this repository.

---

## Dataset Structure

Please organize the downloaded datasets and pre-trained weights as follows:

```text
PROTEUS/
├── data/
│   ├── UIEB/
│   │   ├── train/
│   │   │   ├── input/        # 800 degraded underwater images
│   │   │   └── GT/           # 800 reference images
│   │   └── test/
│   │       ├── U90/          # 90 reference-based test images
│   │       ├── Challenge-60/ # 60 non-reference test images
│   │       └── GT/           # Reference images for U90 only
│   │
│   ├── LSUI/
│   │   ├── train/
│   │   │   ├── input/        # 3,879 degraded underwater images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 400 test images
│   │       └── GT/
│   │
│   ├── UFO/
│   │   ├── train/
│   │   │   ├── input/        # 1,200 training images
│   │   │   └── GT/
│   │   ├── val/
│   │   │   ├── input/        # 300 validation images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 120 test images
│   │       └── GT/
│   │
│   ├── EUVP-Scene/
│   │   ├── train/
│   │   │   ├── input/        # 1,748 training images
│   │   │   └── GT/
│   │   ├── val/
│   │   │   ├── input/        # 218 validation images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 218 test images
│   │       └── GT/
│   │
│   ├── EUVP-Dark/
│   │   ├── train/
│   │   │   ├── input/        # 4,440 training images
│   │   │   └── GT/
│   │   ├── val/
│   │   │   ├── input/        # 555 validation images
│   │   │   └── GT/
│   │   └── test/
│   │       ├── input/        # 555 test images
│   │       └── GT/
│   │
│   └── test_sets/
│       ├── U45/              # 45 non-reference images
│       ├── UCCS/             # 300 non-reference images
│       └── EUVP-330/         # 330 non-reference images
│
├── pretrained_models/
│   └── checkpoints/          # Pre-trained PROTEUS checkpoints
│
├── Neg_dir/                  # Negative sample data directory
├── utils/
├── train.py
├── test.py
├── train.sh
├── test.sh
├── requirements.txt
└── README.md
```

---

## Quick Start

### Training

Before training, please check the dataset paths and other experimental settings in `train.sh`.

```bash
bash train.sh
```

Alternatively, if the script has executable permission, run:

```bash
./train.sh
```

### Testing

Download the pre-trained weights and specify the correct checkpoint path in `test.sh`. Then run:

```bash
bash test.sh
```

Alternatively, if the script has executable permission, run:

```bash
./test.sh
```

The restored images and evaluation results will be saved to the output directory specified in the testing script.

---

## Contact

If you have any questions, please feel free to contact us at [zhangx0802@whu.edu.cn](mailto:zhangx0802@whu.edu.cn).
