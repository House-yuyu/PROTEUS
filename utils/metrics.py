import numpy as np
import math
from .niqe_utils import calculate_niqe
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import color, exposure
import torch
import cv2
from scipy import ndimage  # 新增依赖

"""URanker"""
def preprocessing(d_img_org):
    d_img_org = padding_img(d_img_org)
    x_his = build_historgram(d_img_org)
    return {"x": d_img_org, "x_his": x_his}

def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32

    left_pad = (w_out - w) // 2
    right_pad = w_out - w - left_pad
    top_pad = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad

    img = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)

    return img

def build_historgram(img):
    with torch.no_grad():
        b, _, _, _ = img.shape

        r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)

        historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)

        for i in range(1, b):
            r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
            g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
            b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)

            historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
            historgram = torch.cat((historgram, historgram_temp), dim=0)

    return historgram


def getURanker(image: np.array, uranker_model):
    inputs = torch.from_numpy(image).float()
    inputs = inputs.permute(0, 3, 1, 2)  # B, H, W, C => B, C, H, W
    inputs = preprocessing(inputs)
    uiqa = 0.0
    with torch.no_grad():
        uiqa += torch.sum(
            uranker_model(**inputs)["final_result"].squeeze(-1).squeeze(-1)
        ).item()
    return uiqa

"""
UCIQE
======================================
https://ieeexplore.ieee.org/document/7300447
Compute the Underwater Color Image Quality Evaluation (UCIQE) score.
"""
def get_uciqe_single(image):
    """
    计算单张图像的 UCIQE 指标 (严格遵循用户提供的参考代码逻辑)
    Args:
        image: Numpy array, HWC格式, 范围 [0, 255] (uint8)
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # 2. 转换到 Lab 空间 (假设输入是 RGB)
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) 
    
    # 3. 归一化 (遵循参考代码逻辑，直接除以255，不进行中心化处理)
    img_lum = img_lab[:, :, 0] / 255.0
    img_a   = img_lab[:, :, 1] / 255.0
    img_b   = img_lab[:, :, 2] / 255.0

    # 4. Chroma (色度) 计算
    # 参考代码: sqrt(a^2 + b^2)
    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    
    # 5. Saturation (饱和度) 计算
    # 参考代码: chr / sqrt(chr^2 + lum^2)
    # 添加 1e-10 避免除零
    denom = np.sqrt(np.square(img_chr) + np.square(img_lum))
    img_sat = img_chr / (denom + 1e-10)
    
    aver_sat = np.mean(img_sat) # Average Saturation
    aver_chr = np.mean(img_chr) # Average Chroma

    # 6. Variance of Chroma
    # 你的参考代码使用了特殊公式，而非普通标准差:
    # var_chr = sqrt(mean(abs(1 - (avg/val)^2)))
    # 注意处理除零异常
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = aver_chr / (img_chr + 1e-10)
        diff = 1 - np.square(ratio)
        var_chr = np.sqrt(np.mean(np.abs(diff)))

    # 7. Contrast of Luminance (亮度对比度)
    # 参考代码使用直方图 CDF 找 1% 和 99% 分位点
    # 使用 percentile 提速，逻辑数学等价
    bottom_1 = np.percentile(img_lum, 1)
    top_1    = np.percentile(img_lum, 99)
    con_lum  = top_1 - bottom_1

    # 8. 加权求和 (参考代码系数)
    # coe_metric = [0.4680, 0.2745, 0.2576]
    quality_val = 0.4680 * var_chr + 0.2745 * con_lum + 0.2576 * aver_sat
    
    return quality_val


def getUCIQE(images):
    """
    Args:
        images: PyTorch Tensor [B, C, H, W] (范围 0-1) 或 Numpy [B, H, W, C]
    Return:
        平均 UCIQE 分数
    """
    # 1. 格式统一化处理
    if isinstance(images, torch.Tensor):
        if images.device.type != 'cpu':
            images = images.cpu()
        images = images.detach().numpy()
        # [B, C, H, W] -> [B, H, W, C]
        if images.ndim == 4 and images.shape[1] in [1, 3]: 
            images = images.transpose(0, 2, 3, 1)
            
    # 单张图扩充为 Batch
    if images.ndim == 3:
        images = images[np.newaxis, ...]
        
    scores = []
    for i in range(images.shape[0]):
        try:
            score = get_uciqe_single(images[i])
            # 过滤掉 NaN 或 Inf
            if np.isfinite(score):
                scores.append(score)
            else:
                scores.append(0.0)
        except Exception as e:
            print(f"Error calc UCIQE: {e}")
            scores.append(0.0)
    
    if len(scores) == 0:
        return 0.0
        
    return np.mean(scores)
# def get_uciqe(image):
#     hsv = cv2.cvtColor(np.array(image * 255, dtype=np.uint8), cv2.COLOR_RGB2HSV)
#     H, S, V = cv2.split(hsv)
#     delta = np.std(H) / 180
#     mu = np.mean(S) / 255
#     n, m = np.shape(V)
#     number = math.floor(n * m / 100)
#     Maxsum, Minsum = 0, 0
#     V1, V2 = V / 255, V / 255

#     for i in range(1, number + 1):
#         Maxvalue = np.amax(np.amax(V1))
#         x, y = np.where(V1 == Maxvalue)
#         Maxsum = Maxsum + V1[x[0], y[0]]
#         V1[x[0], y[0]] = 0

#     top = Maxsum / number

#     for i in range(1, number + 1):
#         Minvalue = np.amin(np.amin(V2))
#         X, Y = np.where(V2 == Minvalue)
#         Minsum = Minsum + V2[X[0], Y[0]]
#         V2[X[0], Y[0]] = 1

#     bottom = Minsum / number

#     conl = top - bottom
#     uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
#     return uciqe


# def getUCIQE(image):
#     # image:  B, H, W, C
#     UCIQE = 0
#     for i in range(image.shape[0]):
#         UCIQE += get_uciqe(image[i, :, :, :])
#     return UCIQE


### NIQE ### 
def getNIQE(image):
    # image:  B, H, W, C
    NIQE = 0
    for i in range(image.shape[0]):
        NIQE += calculate_niqe(image[i, :, :, :][:, :, ::-1] * 255)
    return NIQE


"""
UIQM Implementation
======================================
"""
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    # calculate mu_alpha weight
    weight = (1 / (K - T_a_L - T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)

def _uicm(x):
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)

def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    max_mag = np.max(mag)
    if max_mag > 1e-12:
        mag *= 255.0 / max_mag
    else:
        mag = np.zeros_like(mag)
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = 2. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)

            # bound checks, can't do log(0)
            if min_ == 0.0:
                val += 0
            elif max_ == 0.0:
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]

    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    # get eme for each channel
    r_eme = eme(R_edge_map, 8)
    g_eme = eme(G_edge_map, 8)
    b_eme = eme(B_edge_map, 8)

    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def plip_g(x, mu=1026.0):
    return mu - x


def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))


def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / (gamma))


def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta))


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
    """

    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0

    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = -1. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    # entropy scale - higher helps with randomness
    alpha = 1

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)
            min_ = np.min(block)

            top = max_ - min_
            bot = max_ + min_

            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)

            # try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w * val


def get_uiqm_single(x):
    """
      Function to return UIQM to be called from other programs
      x: image (H, W, C)
    """
    x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    # c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    #return uiqm,uicm,uism,uiconm
    return uiqm

def getUIQM(image):
    # image:  B, H, W, C in range [0, 1]
    UIQM = 0
    for i in range(image.shape[0]):
        # UIQM logic is designed for pixel values in general (0-255 scale)
        # Similar to UCIQE, we scale up.
        img_single = image[i, :, :, :] * 255.0
        UIQM += get_uiqm_single(img_single)
    return UIQM


##############################################################################
def getPSNR(img, imclean, data_range):
    # Img = img.data.detach().cpu().numpy().astype(np.float32) # B, H, W, C
    # Iclean = imclean.data.detach().cpu().numpy().astype(np.float32) # B, H, W, C
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += peak_signal_noise_ratio(
            imclean[i, :, :, :], img[i, :, :, :], data_range=data_range
        )
    return PSNR


def getSSIM(img, imclean, data_range):
    # Img = img.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1) # B, H, W, C
    # Iclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0,2,3,1) # B, H, W, C

    SSIM = 0
    for i in range(img.shape[0]):
        SSIM += structural_similarity(   
            imclean[i, :, :, :],
            img[i, :, :, :],
            data_range=data_range,
            channel_axis=-1,
            win_size=5,
        )
    return SSIM


class Evaluator:
    def __init__(self, no_ref=False, uranker_model=None):
        self.no_ref = no_ref
        self.uranker_model = uranker_model
        self.reset()

    def reset(
        self,
    ):
        if self.no_ref:
            self.niqe = 0.0
            self.uciqe = 0.0
            self.uranker = 0
            self.uiqm = 0.0
        else:
            self.ssim = 0.0
            self.psnr = 0.0
        self.count = 0

    def evaluation(self, pred, label):
        if self.no_ref:
            self.niqe += getNIQE(pred)
            # getUCIQE returns batch mean; convert to sum for consistent averaging by self.count.
            self.uciqe += getUCIQE(pred) * pred.shape[0]
            self.uranker += getURanker(pred, self.uranker_model)
            self.uiqm += getUIQM(pred)
        else:
            self.psnr += getPSNR(pred, label, 1.0)
            self.ssim += getSSIM(pred, label, 1.0)
        self.count += pred.shape[0]

    def getMean(self):
        if self.no_ref:
            self.niqe /= self.count
            self.uciqe /= self.count
            self.uranker /= self.count
            self.uiqm /= self.count
            return self.niqe, self.uciqe, self.uranker, self.uiqm
        else:
            self.ssim /= self.count
            self.psnr /= self.count
            return self.ssim, self.psnr