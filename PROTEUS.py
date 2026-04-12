import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import einops


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.laynorm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# --- Attention Modules ---
class Channel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Channel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        out = self.project_out(out)
        return out


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_head):
        super(CrossAttnBlock, self).__init__()
        self.norm11 = LayerNorm(dim, 'WithBias')
        self.norm12 = LayerNorm(dim, 'WithBias')
        self.attn = Channel_Cross_Attention(dim=dim, num_head=num_head, bias=False)

    def forward(self, x, kv):
        x = x + self.attn(self.norm11(x), self.norm12(kv))
        return x


class DeformAttn(nn.Module):
    def __init__(self, dim, n_heads=1, n_groups=1, stride=1, ksize=3, offset_range_factor=4):
        super(DeformAttn, self).__init__()
        self.dim = dim
        self.offset_range_factor = offset_range_factor
        self.n_head_channels = dim // n_heads
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.n_groups = n_groups

        self.n_group_channels = self.dim // self.n_groups
        self.ksize = ksize
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0
        self.conv_offset = nn.Sequential(nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
                                         LayerNorm(self.n_group_channels, 'WithBias'),
                                         nn.GELU(),
                                         nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False))
        self.proj_q = nn.Conv2d(dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.dim, dim, kernel_size=1, stride=1, padding=0)
        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.conv_offset[-1], nn.Conv2d):
            nn.init.constant_(self.conv_offset[-1].weight, 0.0)
            if self.conv_offset[-1].bias is not None:
                nn.init.constant_(self.conv_offset[-1].bias, 0.0)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, prompt, kv):
        B, C, Hp, Wp = kv.size()
        B, C, H, W = prompt.size()
        dtype, device = kv.dtype, kv.device
        q = self.proj_q(prompt)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()

        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = offset + reference
        x_sampled = F.grid_sample(input=kv.reshape(B * self.n_groups, self.n_group_channels, Hp, Wp),
                                  grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        return self.proj_out(out)


class DeformAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=1, n_groups=1, stride=1, ksize=3, offset_range_factor=4):
        super(DeformAttnBlock, self).__init__()
        self.norm11 = LayerNorm(dim, 'WithBias')
        self.norm12 = LayerNorm(dim, 'WithBias')
        self.attn = DeformAttn(dim=dim, n_heads=n_heads, n_groups=n_groups, stride=stride,
                               ksize=ksize, offset_range_factor=offset_range_factor)

    def forward(self, x, kv):
        x = x + self.attn(self.norm11(x), self.norm12(kv))
        return x


class GaussianBasisExpansion(nn.Module):
    def __init__(self, min_v, max_v, num_k):
        super(GaussianBasisExpansion, self).__init__()
        self.min_v = min_v
        self.max_v = max_v
        self.num_k = num_k
        center = torch.linspace(min_v, max_v, num_k)
        self.center = nn.Parameter(center, requires_grad=False)
        self.denominator = (max_v - min_v) / (num_k - 1)

    def gaussian_basis_func(self, x):
        return torch.exp(-((x[..., None] - self.center) / self.denominator) ** 2)

    def forward(self, x):
        return self.gaussian_basis_func(x)


class GDAB(nn.Module): # Guided Degradation-Adaptive Block
    def __init__(self, dim, num_head=4, hidden_dim=128, max_value=2., min_value=-2., num_k=8,
                 n_groups=2, offset_range_factor=1, ksize=3, stride=1, input_channels=3):
        super(GDAB, self).__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.num_k = num_k
        self.hidden_dim = hidden_dim

        # conv_x 负责处理输入的 Guide Image (通常是降质原图)
        self.conv_x = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        # conv_y 负责处理当前的 Feature Map
        self.conv_y = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False)

        # DaCA (Deformable and Cross Attention)
        self.DaCA = DeformAttnBlock(dim=hidden_dim, n_heads=num_head,
                                    n_groups=n_groups, offset_range_factor=offset_range_factor, ksize=ksize,
                                    stride=stride)

        GBE_dim = hidden_dim // 2
        self.GBE_dim = GBE_dim
        self.transformation = nn.Linear(hidden_dim, GBE_dim)
        self.norm_GBE = nn.LayerNorm(GBE_dim)
        self.GBE_expansion = GaussianBasisExpansion(min_v=min_value, max_v=max_value, num_k=num_k)
        self.dynamic_W1 = nn.Linear(hidden_dim, num_k * GBE_dim)
        self.dynamic_W2 = nn.Linear(GBE_dim, hidden_dim)

        self.compression = nn.Linear(hidden_dim, hidden_dim // 2)
        self.act = nn.GELU()
        self.GBE_linear, self.emb_linear = nn.Linear(hidden_dim // 2, hidden_dim), nn.Linear(hidden_dim // 2, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.conv_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.cross_attn = CrossAttnBlock(dim=dim, num_head=num_head)

    def forward(self, x, y):
        """
        x: Guide Image (原始输入, shape: B, 3, H, W)
        y: Feature Map (当前特征, shape: B, C, H', W')
        """
        B, C, Hp, Wp = y.shape
        u = y.clone()
        x = self.conv_x(x)
        y = self.conv_y(y)

        # 【修复】原代码此处写了 self.DCCA，已修正为 self.DaCA
        deg_feat = self.DaCA(y, x) 

        emb = y.mean(dim=(-2, -1))
        deg_emb = deg_feat.mean(dim=(-2, -1))

        trans_emb = self.transformation(emb)
        basis = self.GBE_expansion(self.norm_GBE(trans_emb)).reshape(B, -1)

        W1 = self.act(self.dynamic_W1(deg_emb))
        W2 = self.act(self.dynamic_W2(trans_emb))
        dynamic_W = W2.unsqueeze(-1) @ W1.unsqueeze(1)

        GBE_out = (basis.unsqueeze(1) @ dynamic_W.transpose(-2, -1)).squeeze(1)
        compression = self.act(self.compression(GBE_out + emb))
        sel_w = F.softmax(torch.stack([self.emb_linear(compression), self.GBE_linear(compression)], dim=1), dim=1)

        out = emb * sel_w[:, 0] + GBE_out * sel_w[:, 1]
        out = self.act(self.out_linear(out)).unsqueeze(-1).unsqueeze(-1) * deg_feat

        out = F.interpolate(self.conv_out(out), size=(Hp, Wp), mode='bilinear')
        out = self.cross_attn(u, out)
        return out


class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=True, padding_mode="zeros"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=kernel_size // 2,
                               groups=in_channel, bias=bias, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super(BasicBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out = out + self.identity(x)
        return out


class GetGradient(nn.Module):
    def __init__(self, dim=3, mode="sobel"):
        super(GetGradient, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == "sobel":
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            kernel_y = torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kernel_x = torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_y", kernel_y.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kernel_x.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kernel_laplace = [[0.25, 1, 0.25], [1, -5, 1], [0.25, 1, 0.25]]
            kernel_laplace = torch.tensor(kernel_laplace, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_laplace", kernel_laplace.repeat(self.dim, 1, 1, 1))

    def forward(self, x):
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)
            grad_magnitude = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-6)
        elif self.mode == "laplacian":
            grad_magnitude = F.conv2d(x, self.kernel_laplace, padding=1, groups=self.dim)
            grad_magnitude = torch.abs(grad_magnitude)
        return grad_magnitude


class GFB(nn.Module): # Gradient Fusion Block
    def __init__(self, feature_channels=48):
        super(GFB, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad = self.get_gradient(x)
        grad = self.conv_grad(grad)
        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        x = self.frdb2(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class GEPM(nn.Module): # Gray-Edge Prior Module
    def __init__(self, eps=1e-6, p=6.0, gain_clip=(0.5, 2.5)):
        super(GEPM, self).__init__()
        self.eps = eps
        self.p = p
        self.gain_min, self.gain_max = gain_clip

        # Gray-Edge with Scharr filters for robust edge-based channel statistics.
        kernel_y = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
        kernel_x = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        kernel_y = torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_x = torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel_y_base", kernel_y)
        self.register_buffer("kernel_x_base", kernel_x)

    def forward(self, x):
        b, c, _, _ = x.shape
        x = torch.clamp(x, min=self.eps)

        kernel_x = self.kernel_x_base.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        kernel_y = self.kernel_y_base.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)

        # Gray-Edge: match channel edge statistics instead of raw pixel means.
        edge_stat = torch.pow(
            torch.mean(torch.pow(grad_mag, self.p), dim=(2, 3), keepdim=True) + self.eps,
            1.0 / self.p,
        )
        gray_edge = edge_stat.mean(dim=1, keepdim=True)
        gain = gray_edge / (edge_stat + self.eps)
        gain = torch.clamp(gain, min=self.gain_min, max=self.gain_max)
        x = x * gain

        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out


class AttentionGate(nn.Module):
    def __init__(self, latent_channels, skip_channels):
        super(AttentionGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(latent_channels, skip_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels // 2, skip_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, skip_feat, latent_feat):
        weights = self.mlp(latent_feat)
        return skip_feat * weights


class GDFMB(nn.Module): # Guided Dynamic Feature Modulation Block
    # 1. 增加 stride 参数，默认为 4
    def __init__(self, feature_channels=48, input_channels=3, stride=4):
        super(GDFMB, self).__init__()
        
        # 2. 将 stride 传入 GDAB
        self.gdfm = GDAB(
            dim=feature_channels, 
            num_head=4, 
            hidden_dim=feature_channels * 2,
            input_channels=input_channels,
            stride=stride  # <--- 关键修改：使用传入的 stride
        )
        self.sgfb = GFB(feature_channels)

    def forward(self, x, input_img):
        res = x
        x = self.gdfm(input_img, x) + x
        x = self.sgfb(x)
        return 0.5 * x + 0.5 * res


class myModel(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32, use_white_balance=False):
        super(myModel, self).__init__()
        self.use_white_balance = use_white_balance
        if self.use_white_balance:
            self.wb = GEPM()
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        self.first = nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1)
        
        # 为不同层设置不同的 stride ===
        # Encoder1: 256x256 分辨率极高，stride=8 (降低 64 倍计算量)
        self.encoder1 = GDFMB(feature_channels, input_channels=in_channels, stride=8)
        self.down1 = Downsample(feature_channels)
        
        # Encoder2: 128x128，stride=4
        self.encoder2 = GDFMB(feature_channels * 2**1, input_channels=in_channels, stride=4)
        self.down2 = Downsample(feature_channels * 2**1)
        
        # Bottleneck: 64x64，stride=2 或 1 (显存允许可以用 1)
        self.bottleneck = GDFMB(feature_channels * 2**2, input_channels=in_channels, stride=2)
        

        latent_dim = feature_channels * (2**2)
        self.bary_mapper = nn.Sequential(
            BasicBlock(latent_dim, latent_dim),
            BasicBlock(latent_dim, latent_dim)
        )
        self.gate1 = AttentionGate(latent_dim, feature_channels * 2**1)
        self.gate2 = AttentionGate(latent_dim, feature_channels)

        self.up1 = Upsample(feature_channels * 2**2)
        
        # Decoder1: 对应 Encoder2，stride=4
        self.decoder1 = GDFMB(feature_channels * 2**1, input_channels=in_channels, stride=4)
        
        self.up2 = Upsample(feature_channels * 2**1)
        
        # Decoder2: 对应 Encoder1，stride=8
        self.decoder2 = GDFMB(feature_channels, input_channels=in_channels, stride=8)
        
        self.out = nn.Conv2d(feature_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def _get_latent(self, x):
        # 内部处理：需要保存原始 x (或 WB 后的 x) 传递给层
        img_input = x 
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            img_input = alpha * self.wb(x) + (1 - alpha) * x
            x = img_input # 让主网络处理 WB 后的图
        else:
            # 如果不开启 WB，x 就是原始输入
            img_input = x

        x0 = self.first(x)

        x1 = self.encoder1(x0, img_input)
        
        x2 = self.encoder2(self.down1(x1), img_input)
        
        x3 = self.bottleneck(self.down2(x2), img_input)
        
        return x3, x2, x1, img_input # 返回 img_input 供后续 decoder 使用

    def forward(self, x):
        res = x
        z_source, x2, x1, img_input = self._get_latent(x)
        
        # 1. 强力映射
        z_bary = self.bary_mapper(z_source)
        z_res = z_source - z_bary
        
        # 2. 过滤
        x2_clean = self.gate1(x2, z_bary) 
        x1_clean = self.gate2(x1, z_bary)
        
        # 3. 解码 (同样需要传递 img_input)
        z_recon = z_bary 
        
        x = self.up1(z_recon) + x2_clean
        x = self.decoder1(x, img_input)
        
        x = self.up2(x) + x1_clean
        x = self.decoder2(x, img_input)
        
        out = self.out(x) + res
        
        return out, z_source, z_bary, z_res

    def extract_latent(self, x):
        z_source, _, _, _ = self._get_latent(x)
        return z_source