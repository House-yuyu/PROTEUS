# BaryIR.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

# 引用你现有的 dataset 工具
from utils.dataset import load_img, get_patch, augment, is_image_file

class BaryDataset(Dataset):
    def __init__(self, data_root, data_size, neg_dirs=None, transform=ToTensor(), train=True, resize=False):
        self.train = train
        self.data_size = data_size
        self.transform = transform
        self.resize = resize
        self.neg_dirs = neg_dirs if neg_dirs is not None else []

        self.input_dir = os.path.join(data_root, "input")
        self.target_dir = os.path.join(data_root, "GT")
        
        self.image_filenames = [x for x in os.listdir(self.input_dir) if is_image_file(x)]
        self.image_filenames.sort()

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        input_img = load_img(os.path.join(self.input_dir, filename))
        target_img = load_img(os.path.join(self.target_dir, filename))

        if self.resize:
            input_img = input_img.resize((self.data_size, self.data_size), Image.BILINEAR)
            target_img = target_img.resize((self.data_size, self.data_size), Image.BILINEAR)
        elif target_img.size != input_img.size:
            target_img = target_img.resize(input_img.size, Image.BILINEAR)

        # 获取裁剪参数
        input_patch, target_patch, patch_info = get_patch(input_img, target_img, self.data_size)
        input_patch, target_patch, aug_info = augment(input_patch, target_patch)

        # 加载负样本
        neg_patches = []
        for neg_dir in self.neg_dirs:
            # 假设负样本文件名与原图一致
            neg_path = os.path.join(neg_dir, filename)
            if os.path.exists(neg_path):
                neg_img = load_img(neg_path)
                if self.resize:
                    neg_img = neg_img.resize((self.data_size, self.data_size), Image.BILINEAR)
                elif neg_img.size != input_img.size:
                     neg_img = neg_img.resize(input_img.size, Image.BILINEAR)
                
                # 使用相同的裁剪
                ix, iy, ip = patch_info["ix"], patch_info["iy"], patch_info["ip"]
                neg_patch = neg_img.crop((iy, ix, iy + ip, ix + ip))
                
                # 应用增强 (简易版，需确保与dataset.py augment逻辑一致)
                neg_patch = self._apply_augment(neg_patch, aug_info)
                neg_patches.append(self.transform(neg_patch))
            else:
                # 缺失则用输入代替
                neg_patches.append(self.transform(input_patch))

        if self.transform:
            input_patch = self.transform(input_patch)
            target_patch = self.transform(target_patch)

        if len(neg_patches) > 0:
            neg_tensor = torch.stack(neg_patches, dim=0)
        else:
            neg_tensor = torch.empty(0)

        return input_patch, target_patch, neg_tensor, filename

    def _apply_augment(self, img, info):
        from PIL import ImageOps
        if info["flip_h"]: img = ImageOps.flip(img)
        if info["flip_v"]: img = ImageOps.mirror(img)
        # 注意：dataset.py 的augment里旋转角度是随机的，无法在此完美复现
        # 如果需要精确复现，请修改 dataset.py 让 augment 返回具体角度
        return img

    def __len__(self):
        return len(self.image_filenames)

class BaryLoss(nn.Module):
    def __init__(self, lambda_anchor=0.1, lambda_orth=0.05, lambda_contrast=0.05, contrast_margin=0.1):
        super(BaryLoss, self).__init__()
        self.lambda_anchor = lambda_anchor
        self.lambda_orth = lambda_orth
        self.lambda_contrast = lambda_contrast
        self.margin = contrast_margin
        self.mse = nn.MSELoss()

    def forward(self, z_bary, z_gt, z_res, z_negs_list=None):
        loss_dict = {}
        
        # 1. 锚定 Loss: z_bary 应该靠近 z_gt
        loss_anchor = self.mse(z_bary, z_gt)
        loss_dict['anchor'] = loss_anchor

        # 2. 正交 Loss: z_bary 与 z_res 不相关
        b_vec = F.normalize(z_bary.reshape(z_bary.size(0), -1), dim=1)
        r_vec = F.normalize(z_res.reshape(z_res.size(0), -1), dim=1)
        loss_orth = torch.mean(torch.abs(torch.sum(b_vec * r_vec, dim=1)))
        loss_dict['orth'] = loss_orth

        # 3. 对比 Loss: z_bary 离 GT 近，离 Negative 远
        loss_contrast = torch.tensor(0.0).to(z_bary.device)
        if z_negs_list and len(z_negs_list) > 0:
            gt_vec = F.normalize(z_gt.reshape(z_gt.size(0), -1), dim=1)
            dist_pos = F.pairwise_distance(b_vec, gt_vec)
            
            for z_neg in z_negs_list:
                n_vec = F.normalize(z_neg.reshape(z_neg.size(0), -1), dim=1)
                dist_neg = F.pairwise_distance(b_vec, n_vec)
                loss_contrast += torch.mean(torch.relu(dist_pos - dist_neg + self.margin))
            loss_contrast /= len(z_negs_list)
        
        loss_dict['contrast'] = loss_contrast

        total_loss = (self.lambda_anchor * loss_anchor + 
                      self.lambda_orth * loss_orth + 
                      self.lambda_contrast * loss_contrast)
        return total_loss, loss_dict