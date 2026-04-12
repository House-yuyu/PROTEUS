#!/bin/bash

# ["UIEB", "LSUI", "UFO", "EUVP-s", "EUVP-d"]
# EUVP 200; UIEB 4卡1000和1300epoch性能差别不大
# EUVP-d 学习率得小点

CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 train.py \
  --dataset UIEB \
  --epoch 1000 \
  --lr 1e-4 \
  --train_batch_size 2 \
  --model_name PROTEUS \
  --w_l1 1.0 \
  --w_hvi 0.5 \
  --w_ssim 0.1 \
  --w_vgg 0.1 \
  --w_edge 0.1 \
  --w_bary 1.0 \
  --w_msr 1.0 \
  --ext_aug \
  --ssl_aug \
  --eval_interval 4 && \
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 train.py \
  --dataset LSUI \
  --epoch 300 \
  --lr 1e-4 \
  --train_batch_size 2 \
  --model_name PROTEUS \
  --w_l1 1.0 \
  --w_hvi 0.5 \
  --w_ssim 0.1 \
  --w_vgg 0.1 \
  --w_edge 0.1 \
  --w_bary 1.0 \
  --w_msr 1.0 \
  --ext_aug \
  --ssl_aug \
  --eval_interval 2 && \
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 train.py \
  --dataset UFO \
  --epoch 300 \
  --lr 1e-4 \
  --train_batch_size 2 \
  --model_name PROTEUS \
  --w_l1 1.0 \
  --w_hvi 0.5 \
  --w_ssim 0.1 \
  --w_vgg 0.1 \
  --w_edge 0.1 \
  --w_bary 1.0 \
  --w_msr 1.0 \
  --ext_aug \
  --ssl_aug \
  --eval_interval 2 && \
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 train.py \
  --dataset EUVP-s \
  --epoch 200 \
  --lr 1e-4 \
  --train_batch_size 2 \
  --model_name PROTEUS \
  --w_l1 1.0 \
  --w_hvi 0.5 \
  --w_ssim 0.1 \
  --w_vgg 0.1 \
  --w_edge 0.1 \
  --w_bary 1.0 \
  --w_msr 1.0 \
  --ext_aug \
  --ssl_aug \
  --eval_interval 2 && \
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 train.py \
  --dataset EUVP-d \
  --epoch 200 \
  --lr 5e-5 \
  --train_batch_size 2 \
  --model_name PROTEUS \
  --w_l1 1.0 \
  --w_hvi 0.5 \
  --w_ssim 0.1 \
  --w_vgg 0.1 \
  --w_edge 0.1 \
  --w_bary 1.0 \
  --w_msr 1.0 \
  --ext_aug \
  --ssl_aug \
  --eval_interval 2