CKPT_PATH=/data2/users/zhangxu/ISP/6_UIR/W_UIR/output/PROTEUS/UIEB/202603021_bs2/best_model_test.pth

CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset "UIEB" \
  --ckpt $CKPT_PATH && \
CUDA_VISIBLE_DEVICES=0 python test_nr.py \
  --dataset "C60,U45,SQUID,UCCS,EUVP-330" \
  --ckpt $CKPT_PATH
