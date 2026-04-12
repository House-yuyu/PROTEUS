import argparse
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image
from datetime import datetime  # 确保导入 datetime

from utils.dataset import get_loader
from model import myModel
from utils.metrics import Evaluator
from utils.uranker.uranker_utils import build_model, get_option


class Tester(object):
    def __init__(self, args):
        self.args = args

        self.dataset_roots = {
            "C60": "/data2/users/zhangxu/ISP/6_UIR/Datasets/Challenging-60/test/",
            "EUVP-330": "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP/test/",
            "U45": "/data2/users/zhangxu/ISP/6_UIR/Datasets/U45/test/",
            "SQUID": "/data2/users/zhangxu/ISP/6_UIR/Datasets/SQUID-16/",
            "UCCS": "/data2/users/zhangxu/ISP/6_UIR/Datasets/UCCS/",
        }
        self.dataset_list = self._parse_dataset_arg(args.dataset)

        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        )

        options = get_option(r"utils/uranker/URanker.yaml")
        options["model"]["resume_ckpt_path"] = r"utils/uranker/URanker_ckpt.pth"
        self.uranker_model = build_model(options["model"])
        self.uranker_model = self.uranker_model.cpu()
        self.uranker_model.eval()

        self.evaluator = Evaluator(no_ref=True, uranker_model=self.uranker_model)

        if os.path.isfile(args.ckpt):
            checkpoint = torch.load(args.ckpt, weights_only=False)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.ckpt))

        self.deep_model = self.deep_model.to("cuda")
        self.deep_model.eval()

    def _parse_dataset_arg(self, dataset_arg):
        arg = dataset_arg.strip()
        if arg.upper() == "ALL":
            return list(self.dataset_roots.keys())

        selected = []
        for name in [x.strip() for x in arg.split(",") if x.strip()]:
            if name not in self.dataset_roots:
                valid = ", ".join(self.dataset_roots.keys())
                raise ValueError(f"Unknown dataset: {name}. Valid choices: {valid}, or ALL")
            if name not in selected:
                selected.append(name)

        if not selected:
            valid = ", ".join(self.dataset_roots.keys())
            raise ValueError(f"No valid dataset provided. Valid choices: {valid}, or ALL")

        return selected

    def testing(self):
        #以此确定保存路径的父目录
        ckpt_dir = "/".join(self.args.ckpt.split("/")[:-1])

        dummy = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Params: {params}, FLOPs: {flops}")

        # === 【新增】保存测试结果到 txt ===
        result_txt_path = os.path.join(ckpt_dir, "result_nr.txt")
        
        # 获取当前时间
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with torch.no_grad():
            for dataset_name in self.dataset_list:
                self.evaluator.reset()
                torch.cuda.empty_cache()

                dataloader = get_loader(
                    self.dataset_roots[dataset_name],
                    1,
                    self.args.datasize,
                    train=False,
                    resize=not self.args.no_resize,
                    num_workers=1,
                    shuffle=False,
                    pin_memory=True,
                    non_ref=True,
                )

                loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
                for _, (x, fn) in loop:
                    x = x.to("cuda")

                    # Force unified test resolution (datasize x datasize).
                    if x.shape[-2] != self.args.datasize or x.shape[-1] != self.args.datasize:
                        x = F.interpolate(
                            x,
                            size=(self.args.datasize, self.args.datasize),
                            mode="bilinear",
                            align_corners=False,
                        )

                    pred, _, _, _ = self.deep_model(x)
                    pred = torch.clamp(pred, 0.0, 1.0)
                    pred = (
                        pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                    )  # B, H, W, C

                    self.evaluator.evaluation(pred, None)

                    save_pred_dir = os.path.join(ckpt_dir, f"pred/{dataset_name}")
                    if not os.path.exists(save_pred_dir):
                        os.makedirs(save_pred_dir)

                    base_name = os.path.splitext(fn[0])[0]
                    Image.fromarray((pred[0] * 255).astype(np.uint8)).save(
                        os.path.join(save_pred_dir, base_name + ".png")
                    )
                    loop.set_description(f"[Testing {dataset_name}]")

                niqe_, uciqe_, uranker_, uiqm_ = self.evaluator.getMean()
                print(
                    "[Testing %s] NIQE: %.4f, UCIQE: %.4f, UIQM: %.4f, URanker: %.4f"
                    % (dataset_name, niqe_, uciqe_, uiqm_, uranker_)
                )

                # 写入内容
                with open(result_txt_path, "a") as f:
                    f.write(f"[{curr_time}] Dataset: {dataset_name}\n")
                    f.write(f"Checkpoint: {self.args.ckpt}\n")
                    f.write(f"Metrics: NIQE: {niqe_:.4f}, UCIQE: {uciqe_:.4f}, UIQM: {uiqm_:.4f}, URanker: {uranker_:.4f}\n")
                    f.write(f"Model: Params: {params}, FLOPs: {flops}\n")
                    f.write("-" * 50 + "\n")
        
        print(f"Results saved to: {result_txt_path}")

        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="C60,U45,SQUID,UCCS,EUVP-330",
        help="Single dataset name, comma-separated list (e.g. C60,U45,SQUID), or ALL",
    )
    parser.add_argument("--datasize", type=int, default=256)
    parser.add_argument(
        "--no_resize",
        action="store_true",
        default=False,
        help="Disable resize and keep original image resolution (DatasetFromFolder_NR still enforces multiples of 4).",
    )

    args = parser.parse_args()

    tester = Tester(args)

    start = time.time()

    tester.testing()

    end = time.time()
    print("Testing time:", end - start, "sec")


if __name__ == "__main__":
    main()