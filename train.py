import argparse
import os
import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from thop import profile, clever_format
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta

from utils.dataset import get_loader
from PROTEUS import myModel
from utils.metrics import Evaluator
from utils.loss_funcs import (
    EdgeAwareLoss,
    SSIMLoss,
    L1_Charbonnier_loss,
    PerceptualLoss,
)
from utils.CIDNet import CIDNet
from utils.BaryIR import BaryLoss, BaryDataset
from aug_pho_gem import AugExternal, AugNoneOpt


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def print_network_stats(model, input_size=256):
    try:
        original_mode = model.training
        model.eval()

        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        flops_str, params_str = clever_format([flops, params], "%.3f")

        print_rank0("-" * 50)
        print_rank0(f"Model Stats (Input: {input_size}x{input_size}):")
        print_rank0(f"  - Params: {params_str}")
        print_rank0(f"  - FLOPs : {flops_str}")
        print_rank0("-" * 50)

        model.train(original_mode)

        return flops, params

    except Exception as e:
        print_rank0(f"Warning: Failed to calculate FLOPs/Params: {e}")
        return 0, 0


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        self.evaluator = Evaluator()

        # ---- Model ----
        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        ).to(self.device)  # 启用白平衡

        # ---- CIDNet (frozen, used for HVI loss) ----
        self.hvi_net = CIDNet().to(self.device)
        pth = r"utils/CIDNet_weight_LOLv2_bestSSIM.pth"
        self.hvi_net.load_state_dict(torch.load(pth, map_location=self.device))
        self.hvi_net.eval()
        for p in self.hvi_net.parameters():
            p.requires_grad = False

        # ---- Resume ----
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=self.device)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
            print_rank0("=> loaded checkpoint '{}'".format(args.resume))

        # ---- Wrap with DDP ----
        self.deep_model = DDP(
            self.deep_model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

        # ---- Augmentation ----
        self.ext_aug = None
        if args.ext_aug:
            self.ext_aug = AugExternal(
                prob=0.6,
                shift_limit=2,
                attenuation_range=(0.8, 1.2),
            ).to(self.device)
            print_rank0("===> External Augmentation Enabled (Chromatic Aberration & Spectral Attenuation)")

        self.aug_opt = None
        if args.ssl_aug:
            self.aug_opt = AugNoneOpt().to(self.device)
            print_rank0("===> Internal SSL Augmentation Enabled (MSR Loss)")

        # ---- Save Path (rank 0 creates dirs) ----
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batchsize = args.train_batch_size
        self.model_save_path = os.path.join(
            args.save_path, args.model_name, args.dataset, now_str + f"_bs{batchsize}"
        )
        if is_main_process():
            os.makedirs(self.model_save_path, exist_ok=True)
        dist.barrier()

        # ---- TensorBoard & Step (rank 0 only) ----
        self.writer = None
        if is_main_process():
            log_dir = os.path.join(self.model_save_path, "tb_logs")
            self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # ---- Save Config (rank 0 only) ----
        if is_main_process():
            config_file_path = os.path.join(self.model_save_path, "config_log.txt")
            with open(config_file_path, "w") as f:
                f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"World Size: {dist.get_world_size()}\n")
                f.write("==================================================\n")
                f.write("               LOSS HYPER-PARAMETERS              \n")
                f.write("==================================================\n")
                f.write("1. BaryLoss Settings:\n")
                f.write("   - lambda_anchor:   0.1\n")
                f.write("   - lambda_orth:     0.05\n")
                f.write("   - lambda_contrast: 0.05 (Default)\n\n")

                f.write("2. Final Loss Composition:\n")
                f.write("   final_loss = (\n")
                f.write(f"       {args.w_l1:.4f} * l1_loss\n")
                f.write(f"     + {args.w_hvi:.4f} * hvi_loss\n")
                f.write(f"     + {args.w_ssim:.4f} * ssim_loss\n")
                f.write(f"     + {args.w_vgg:.4f} * vgg_loss\n")
                f.write(f"     + {args.w_edge:.4f} * edge_loss\n")
                f.write(f"     + {args.w_bary:.4f} * bary_loss_val\n")
                f.write(f"     + {args.w_msr:.4f} * msr_loss\n")
                f.write("   )\n\n")

                f.write("==================================================\n")
                f.write("                 RUNNING ARGUMENTS                \n")
                f.write("==================================================\n")
                for k, v in vars(args).items():
                    f.write(f"{k.ljust(20)}: {v}\n")

            print(f"Configuration saved to: {config_file_path}")

        # ---- Optimizer & Scheduler ----
        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        if args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optim, args.epoch, eta_min=args.lr * 1e-4
            )
        elif args.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optim, step_size=args.decay_epoch, gamma=args.decay_rate
            )
        else:
            self.scheduler = None
            print_rank0("===> Scheduler Disabled (warmup only, then constant LR)")

        # ---- Dataset Paths ----
        if args.dataset == "EUVP-d":
            args.train_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP-Dark/train/"
            args.val_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP-Dark/val/"
            args.test_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP-Dark/test/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "EUVP-s":
            args.train_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP-Scene/train/"
            args.val_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP-Scene/val/"
            args.test_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/EUVP-Scene/test/"
            args.datasize = 256
            args.resize = True
        elif args.dataset == "UIEB":
            args.train_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/UIEB_L/"
            args.test_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/U90"
            args.datasize = 256
            args.resize = True
        elif args.dataset == "UFO":
            args.train_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/UFO-120/train/"
            args.val_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/UFO-120/val/"
            args.test_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/UFO-120/test/"
            args.datasize = 256
            args.resize = True
        elif args.dataset == "LSUI":
            args.train_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/LSUI/train/"
            args.test_root = "/data2/users/zhangxu/ISP/6_UIR/Datasets/LSUI/test/"
            args.datasize = 256
            args.resize = True

        # ---- Loss Functions ----
        self.vggL = PerceptualLoss()
        self.L1L = L1_Charbonnier_loss()
        self.ssimL = SSIMLoss(device="cuda", window_size=5)
        self.edgeL = EdgeAwareLoss(loss_type="l2", device="cuda")
        self.msr_loss_fn = torch.nn.MSELoss()

    @staticmethod
    def _format_ckpt_name(psnr, epoch):
        return f"PSNR_{psnr:.4f}_epoch_{epoch}.pth"

    def _save_checkpoint(self, psnr, epoch):
        ckpt_name = self._format_ckpt_name(psnr, epoch)
        ckpt_path = os.path.join(self.model_save_path, ckpt_name)
        torch.save(self.deep_model.module.state_dict(), ckpt_path)
        return ckpt_path

    def calculate_beta(self, current_step, total_steps):
        beta = self.args.beta_base * (
            1 / 2 * (math.cos(math.pi * current_step / total_steps) + 1)
        )
        return beta

    def training(self):
        best_psnr_test = 0.0
        best_round_test = {}
        best_ckpt_path = None

        torch.cuda.empty_cache()

        neg_dirs = [
            os.path.join(self.args.negdir_root, "BaryIR"),
            os.path.join(self.args.negdir_root, "BaryIR03"),
            os.path.join(self.args.negdir_root, "FUnIE"),
            os.path.join(self.args.negdir_root, "USUIR"),
        ]
        train_dataset = BaryDataset(
            data_root=self.args.train_root,
            data_size=self.args.datasize,
            neg_dirs=neg_dirs,
            resize=self.args.resize,
        )

        # ---- DistributedSampler ----
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.bary_criterion = BaryLoss(lambda_anchor=0.1, lambda_orth=0.05).to(self.device)

        total_steps = self.args.epoch * len(train_data_loader)

        # ---- Print Network Stats (rank 0, on unwrapped model) ----
        if is_main_process():
            print("Calculated Model Complexity:")
            print_network_stats(self.deep_model.module, input_size=self.args.datasize)

        self.deep_model.train()

        warmup_epochs = 10
        target_lr = self.args.lr

        for epoch in range(1, self.args.epoch + 1):
            train_sampler.set_epoch(epoch)

            if epoch <= warmup_epochs:
                warmup_lr = target_lr * (epoch / warmup_epochs)
                for param_group in self.optim.param_groups:
                    param_group["lr"] = warmup_lr
                print_rank0(f"--- [Warmup] Epoch {epoch}: LR set to {warmup_lr:.2e} ---")

            loop = tqdm(
                enumerate(train_data_loader),
                total=len(train_data_loader),
                leave=False,
                disable=not is_main_process(),
            )
            loss_mean = 0.0
            bary_loss_mean = 0.0

            for _, (x, label, negatives, _) in loop:
                x = x.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                if negatives.numel() > 0:
                    negatives = negatives.to(self.device, non_blocking=True)

                self.optim.zero_grad(set_to_none=True)

                if self.ext_aug is not None:
                    x = self.ext_aug(x)

                pred, z_source, z_bary, z_res = self.deep_model(x)

                # Access the unwrapped model for extract_latent
                base_model = self.deep_model.module

                with torch.no_grad():  # 不同的z_source
                    z_gt = base_model.extract_latent(label)
                    z_negs_list = []
                    if negatives.numel() > 0:
                        for i in range(negatives.shape[1]):
                            z_n = base_model.extract_latent(negatives[:, i, ...])
                            z_negs_list.append(z_n)

                with torch.no_grad():
                    label_hvi = self.hvi_net.trans.HVIT(label)
                    pred_hvi = self.hvi_net.trans.HVIT(pred.clamp(0.0, 1.0))

                hvi_loss = self.L1L(pred_hvi, label_hvi)
                l1_loss = self.L1L(pred, label)
                vgg_loss = self.vggL(pred, label)
                ssim_loss = self.ssimL(pred, label)
                edge_loss = self.edgeL(pred, label)
                bary_loss_val, bary_loss_dict = self.bary_criterion(
                    z_bary, z_gt, z_res, z_negs_list
                )

                msr_loss = torch.tensor(0.0, device=self.device)
                if self.aug_opt is not None:
                    weak_output, aggr_output = self.aug_opt(pred)
                    msr_loss = self.msr_loss_fn(weak_output, aggr_output)
                    beta = self.calculate_beta(self.global_step, total_steps)
                    msr_loss = beta * msr_loss

                final_loss = (
                    self.args.w_l1 * l1_loss
                    + self.args.w_hvi * hvi_loss
                    + self.args.w_ssim * ssim_loss
                    + self.args.w_vgg * vgg_loss
                    + self.args.w_edge * edge_loss
                    + self.args.w_bary * bary_loss_val
                    + self.args.w_msr * msr_loss
                )

                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.deep_model.parameters(), max_norm=1.0)
                self.optim.step()

                loss_mean += final_loss.item()
                bary_loss_mean += bary_loss_val.item()

                # TensorBoard logging (rank 0 only)
                if self.writer is not None:
                    self.writer.add_scalar("Train/Loss_Step", final_loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_L1", l1_loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_HVI", hvi_loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_SSIM", ssim_loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_VGG", vgg_loss.item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_Bary_Total", bary_loss_val.item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_Bary_Anchor", bary_loss_dict["anchor"].item(), self.global_step)
                    self.writer.add_scalar("Train/Loss_Bary_Orth", bary_loss_dict["orth"].item(), self.global_step)
                    if self.aug_opt is not None:
                        self.writer.add_scalar("Train/Loss_MSR", msr_loss.item(), self.global_step)

                self.global_step += 1

                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                loop.set_postfix(loss=final_loss.item())

            avg_loss = loss_mean / len(train_data_loader)
            avg_bary_loss = bary_loss_mean / len(train_data_loader)
            current_lr = self.optim.param_groups[0]["lr"]

            print_rank0(
                f"[{epoch}/{self.args.epoch}] "
                f"Avg Total: {avg_loss:.5f} | "
                f"Avg Bary: {avg_bary_loss:.5f} | "
                f"LR: {current_lr:.2e}"
            )

            if self.writer is not None:
                self.writer.add_scalar("Train/Loss_Epoch_Avg", avg_loss, epoch)
                self.writer.add_scalar("Train/Loss_Epoch_Bary", avg_bary_loss, epoch)
                self.writer.add_scalar("Train/Learning_Rate", current_lr, epoch)

            # ---- Evaluation (rank 0 only): test set only every eval_interval epochs ----
            if epoch % self.args.eval_interval == 0:
                if is_main_process():
                    self.deep_model.eval()

                    print(f"--- Evaluating on TEST set ({self.args.test_root}) ---")
                    ssim_test, psnr_test = self.validation(self.args.test_root)
                    if self.writer is not None:
                        self.writer.add_scalar("Test/PSNR", psnr_test, epoch)
                        self.writer.add_scalar("Test/SSIM", ssim_test, epoch)

                    save_uieb_threshold = (
                        self.args.dataset == "UIEB"
                        and bool(self.args.enable_uieb_psnr_threshold_save)
                        and psnr_test > self.args.uieb_psnr_save_threshold
                    )

                    if save_uieb_threshold:
                        thr_ckpt = self._save_checkpoint(psnr_test, epoch)
                        print(
                            f"*** UIEB Threshold Checkpoint Saved: {thr_ckpt} "
                            f"(PSNR {psnr_test:.4f} > {self.args.uieb_psnr_save_threshold:.2f}) ***"
                        )
                        if psnr_test > best_psnr_test:
                            best_psnr_test = psnr_test
                            best_round_test = {"epoch": epoch, "psnr": psnr_test, "ssim": ssim_test}
                    else:
                        # Not in UIEB-threshold-save condition: keep only one best checkpoint.
                        if psnr_test > best_psnr_test:
                            best_psnr_test = psnr_test
                            best_round_test = {"epoch": epoch, "psnr": psnr_test, "ssim": ssim_test}
                            if best_ckpt_path is not None and os.path.isfile(best_ckpt_path):
                                os.remove(best_ckpt_path)
                            best_ckpt_path = self._save_checkpoint(psnr_test, epoch)
                            print(
                                f"*** New Best TEST Model Saved: {best_ckpt_path} "
                                f"| PSNR: {psnr_test:.4f} ***"
                            )

                    with open(os.path.join(self.model_save_path, "records.txt"), "a") as f:
                        f.write(f"\n[Epoch {epoch}]\n")
                        f.write(f"  TEST -> PSNR: {psnr_test:.4f}, SSIM: {ssim_test:.4f}\n")
                        f.write(f"  Best Test so far: {best_round_test}\n")

                    self.deep_model.train()

                # All ranks wait for rank 0 to finish validation
                dist.barrier()

            if epoch > warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

        print_rank0("Training Finished.")
        print_rank0("Best Test Round:", best_round_test)
        if self.writer is not None:
            self.writer.close()

    def validation(self, data_root):
        self.evaluator.reset()
        val_data_loader = get_loader(
            data_root,
            self.args.eval_batch_size,
            self.args.datasize,
            train=False,
            resize=self.args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(
                enumerate(val_data_loader),
                total=len(val_data_loader),
                leave=False,
            )
            for _, (x, label, _) in loop:
                x = x.to(self.device)
                label = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)

                pred, _, _, _ = self.deep_model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred = (
                    pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )

                self.evaluator.evaluation(pred, label)
                loop.set_description(
                    f"[Eval] {os.path.basename(os.path.normpath(data_root))}"
                )

        ssim_, psnr_ = self.evaluator.getMean()
        print(f"   Score: SSIM: {ssim_:.4f}, PSNR: {psnr_:.4f}")
        return ssim_, psnr_


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=1000, help="epoch number")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5,
        help="run test evaluation every N epochs",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, default=2, help="per-GPU batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2)

    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "step"],
        help="LR scheduler type; 'none' keeps LR constant after warmup",
    )

    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument(
        "--dataset",
        type=str,
        default="UIEB",
        choices=["UIEB", "LSUI", "UFO", "EUVP-s", "EUVP-d"],
    )
    parser.add_argument("--model_name", type=str, default="wwe1")
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--resume", type=str)

    parser.add_argument(
        "--negdir_root",
        type=str,
        default="/data2/users/zhangxu/ISP/6_UIR/W_UIR/Neg_dir",
        help="Root dir containing negative folders: BaryIR, BaryIR03, FUnIE, USUIR",
    )

    parser.add_argument(
        "--ext_aug",
        action="store_true",
        default=True,
        help="Enable external augmentation (Input Gamma)",
    )
    parser.add_argument(
        "--ssl_aug",
        action="store_true",
        default=True,
        help="Enable internal SSL augmentation (Output Consistency)",
    )
    parser.add_argument(
        "--beta_base", type=float, default=0.1, help="Base weight for SSL MSR loss"
    )
    parser.add_argument("--w_l1", type=float, default=1.0, help="Weight for L1 loss")
    parser.add_argument("--w_hvi", type=float, default=0.5, help="Weight for HVI loss")
    parser.add_argument("--w_ssim", type=float, default=0.1, help="Weight for SSIM loss")
    parser.add_argument("--w_vgg", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--w_edge", type=float, default=0.1, help="Weight for edge loss")
    parser.add_argument("--w_bary", type=float, default=1.0, help="Weight for bary loss")
    parser.add_argument("--w_msr", type=float, default=1.0, help="Weight for MSR consistency loss")
    parser.add_argument(
        "--enable_uieb_psnr_threshold_save",
        type=int,
        choices=[0, 1],
        default=1,
        help="If 1, save UIEB checkpoints when PSNR is above threshold; if 0, disable this feature.",
    )
    parser.add_argument(
        "--uieb_psnr_save_threshold",
        type=float,
        default=25.35,
        help="UIEB checkpoint save threshold on PSNR when threshold saving is enabled.",
    )

    args = parser.parse_args()

    # ---- DDP Init ----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    seed_everything(7 + local_rank)

    trainer = Trainer(args)
    trainer.training()

    dist.destroy_process_group()


if __name__ == "__main__":
    start = time.time()
    main()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        end = time.time()
        total_seconds = end - start
        print(f"The total training time is: {str(timedelta(seconds=int(total_seconds)))}")
        # 577