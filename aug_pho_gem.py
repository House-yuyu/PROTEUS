import random
import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F

class AugNoneOpt(nn.Module):
    """
    内部增强模块 (保持不变或微调)
    用于自监督学习的一致性约束。
    """
    def __init__(self):
        super(AugNoneOpt, self).__init__()
        # 弱增强：中心裁剪，模拟视场变化
        self.weak_aug = nn.Sequential(T.CenterCrop(16))
        # 强增强：高斯模糊，模拟水下混浊导致的边缘模糊 (Turbidity)
        # 加大了一点 sigma 上限以模拟浑浊水域
        self.aggr_aug = nn.Sequential(T.GaussianBlur(kernel_size=(9,21), sigma=(0.1, 8.0)))

    def forward(self, source_img):
        augweak_source_img = self.weak_aug(source_img)
        augaggr_source_img = self.aggr_aug(augweak_source_img)
        return augweak_source_img, augaggr_source_img


class AugExternal(nn.Module):
    """
    外增强模块 - 针对水下图像优化 (Underwater Optimization)
    
    设计原则：
    1. 模拟色差 (Chromatic Aberration)：水下折射导致RGB通道在边缘处发生空间错位。
    2. 模拟光谱衰减 (Spectral Attenuation)：红光在水下衰减最快，蓝/绿光主导。
    3. 模拟对比度拉伸：水下散射导致直方图分布变化。
    """
    def __init__(self, 
                 prob=0.8, 
                 shift_limit=2, 
                 attenuation_range=(0.8, 1.2)):
        """
        Args:
            prob: 应用增强的概率
            shift_limit: 色差模拟的最大像素位移 (pixel shift)
            attenuation_range: 颜色通道增益的波动范围
        """
        super(AugExternal, self).__init__()
        self.prob = prob
        self.shift_limit = shift_limit
        self.attenuation_range = attenuation_range
        
    def chromatic_aberration(self, x):
        """
        模拟色差：对 R 和 B 通道进行随机的空间平移，G 通道保持不动作为基准。
        这模拟了透镜在水介质中对不同波长光线的折射差异。
        """
        B, C, H, W = x.shape
        if C < 3: return x # 单通道图像无法做色差
        
        # 为 R 和 B 通道生成随机位移量 (-limit, +limit)
        rx = random.randint(-self.shift_limit, self.shift_limit)
        ry = random.randint(-self.shift_limit, self.shift_limit)
        bx = random.randint(-self.shift_limit, self.shift_limit)
        by = random.randint(-self.shift_limit, self.shift_limit)
        
        # 分离通道
        r, g, b = x.chunk(3, dim=1)
        
        # 使用 torch.roll 进行循环移位 (比 grid_sample 快，适合微小位移)
        # 注意：边缘部分可能会有伪影，但对于增强鲁棒性是可以接受的
        r_shift = torch.roll(r, shifts=(ry, rx), dims=(2, 3))
        b_shift = torch.roll(b, shifts=(by, bx), dims=(2, 3))
        
        # 重新组合
        return torch.cat([r_shift, g, b_shift], dim=1)

    def spectral_attenuation(self, x):
        """
        模拟水下颜色衰减/色偏：
        水下图像通常 R 通道信号最弱，B/G 通道信号较强。
        这里随机调整各通道的增益，迫使模型学习颜色恒常性。
        """
        B, C, H, W = x.shape
        if C < 3: return x
        
        low, high = self.attenuation_range
        
        # 生成通道增益 [B, C, 1, 1]
        # 专门针对水下：给 Red 通道一个倾向于衰减的 bias，给 Blue/Green 倾向于保留
        # 这里的逻辑是：如果是做 Restoration，我们希望输入更加多样化（有的偏蓝，有的偏绿）
        
        # 随机生成 RGB 的增益因子
        r_gain = torch.empty(B, 1, 1, 1, device=x.device).uniform_(low * 0.8, high * 1.0) # R 倾向于更暗
        g_gain = torch.empty(B, 1, 1, 1, device=x.device).uniform_(low, high)
        b_gain = torch.empty(B, 1, 1, 1, device=x.device).uniform_(low, high * 1.1)       # B 倾向于更亮
        
        gains = torch.cat([r_gain, g_gain, b_gain], dim=1)
        
        return x * gains

    def forward(self, x):
        # x: [B, C, H, W], expected range [0, 1]
        if not self.training:
            return x
        
        if random.random() > self.prob:
            return x
        
        # 1. 应用色差 (空间域变换)
        out = self.chromatic_aberration(x)
        
        # 2. 应用光谱衰减 (颜色域变换)
        out = self.spectral_attenuation(out)
        
        return out.clamp(0.0, 1.0)