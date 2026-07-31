#!/usr/bin/env python3
"""Same-checkpoint interventions for the spatial guide used by PROTEUS."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import lpips
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEVERITY_ROOT = PROJECT_ROOT / "review_gap_runs/degradation_severity_20260729"
if str(SEVERITY_ROOT) not in sys.path:
    sys.path.insert(0, str(SEVERITY_ROOT))

from paired_eval_common import (  # noqa: E402
    DATASETS,
    bootstrap_ci,
    image_metrics,
    load_gt_batch,
    load_proteus,
    paired_samples,
)


MODES = OrderedDict(
    [
        ("Zero guide", "zero"),
        ("Uniform guide", "uniform"),
        ("Raw-input guide", "raw"),
        ("Cross-image guide", "cross"),
        ("Full spatial guide", "full"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=["U90", "LSUI"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser.parse_args()


def gepm_blend(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    if not model.use_white_balance:
        return image
    alpha = torch.sigmoid(model.alpha)
    return alpha * model.wb(image) + (1.0 - alpha) * image


def forward_with_guide(
    model: torch.nn.Module,
    image: torch.Tensor,
    mode: str,
    cross_image: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "full":
        return model(image)[0].clamp(0.0, 1.0)

    residual = image
    img_input = gepm_blend(model, image)
    if mode == "zero":
        guide = torch.zeros_like(img_input)
    elif mode == "uniform":
        guide = img_input.mean(dim=(-2, -1), keepdim=True).expand_as(img_input)
    elif mode == "raw":
        guide = image
    elif mode == "cross":
        if cross_image is None:
            raise ValueError("cross_image is required for cross-guide intervention")
        guide = gepm_blend(model, cross_image)
    else:
        raise ValueError(f"Unknown guide mode: {mode}")

    x0 = model.first(img_input)
    x1 = model.encoder1(x0, guide)
    x2 = model.encoder2(model.down1(x1), guide)
    z_source = model.bottleneck(model.down2(x2), guide)
    z_control = model.bary_mapper(z_source)
    x = model.up1(z_control) + model.gate1(x2, z_control)
    x = model.decoder1(x, guide)
    x = model.up2(x) + model.gate2(x1, z_control)
    x = model.decoder2(x, guide)
    return (model.out(x) + residual).clamp(0.0, 1.0)


def lpips_values(
    metric: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    values = metric(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
    return values.detach().cpu().reshape(-1).numpy().astype(np.float64)


def evaluate_dataset(
    dataset_name: str,
    args: argparse.Namespace,
    device: torch.device,
    perceptual_metric: torch.nn.Module,
) -> tuple[list[dict], dict]:
    configuration = DATASETS[dataset_name]
    samples = paired_samples(configuration, args.limit)
    model = load_proteus(configuration["proteus_ckpt"], device)
    rows: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            names = [sample["name"] for sample in batch_samples]
            inputs = load_gt_batch(
                [sample["input"] for sample in batch_samples],
                args.size,
                device,
            )
            targets = load_gt_batch(
                [sample["gt"] for sample in batch_samples],
                args.size,
                device,
            )
            guide_samples = [
                samples[(start + offset + 1) % len(samples)]
                for offset in range(len(batch_samples))
            ]
            cross_inputs = load_gt_batch(
                [sample["input"] for sample in guide_samples],
                args.size,
                device,
            )

            for label, mode in MODES.items():
                prediction = forward_with_guide(
                    model,
                    inputs,
                    mode,
                    cross_image=cross_inputs if mode == "cross" else None,
                )
                psnr, ssim = image_metrics(prediction, targets)
                perceptual = lpips_values(perceptual_metric, prediction, targets)
                for index, name in enumerate(names):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "filename": name,
                            "setting": label,
                            "psnr": float(psnr[index]),
                            "ssim": float(ssim[index]),
                            "lpips": float(perceptual[index]),
                        }
                    )
            print(
                f"[{dataset_name}] {min(start + args.batch_size, len(samples))}/"
                f"{len(samples)}",
                flush=True,
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    summary: dict[str, dict] = OrderedDict()
    full_by_name = {
        row["filename"]: row
        for row in rows
        if row["setting"] == "Full spatial guide"
    }
    for setting in MODES:
        setting_rows = [row for row in rows if row["setting"] == setting]
        summary[setting] = {
            "count": len(setting_rows),
            "psnr_mean": float(np.mean([row["psnr"] for row in setting_rows])),
            "ssim_mean": float(np.mean([row["ssim"] for row in setting_rows])),
            "lpips_mean": float(np.mean([row["lpips"] for row in setting_rows])),
        }
        if setting != "Full spatial guide":
            differences = np.array(
                [
                    full_by_name[row["filename"]]["psnr"] - row["psnr"]
                    for row in setting_rows
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_ci(differences, args.bootstrap_samples, rng)
            summary[setting]["full_minus_setting_psnr"] = {
                "mean": float(differences.mean()),
                "paired_bootstrap_95_ci": [low, high],
                "positive_images": int(np.sum(differences > 0)),
            }
    return rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    perceptual_metric = lpips.LPIPS(net="alex").to(device).eval()

    all_rows: list[dict] = []
    results: dict[str, dict] = {}
    for dataset in args.datasets:
        rows, summary = evaluate_dataset(
            dataset,
            args,
            device,
            perceptual_metric,
        )
        all_rows.extend(rows)
        results[dataset] = summary

    with (args.output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    metadata = {
        "protocol": {
            "seed": args.seed,
            "size": args.size,
            "bootstrap_samples": args.bootstrap_samples,
            "cross_image_mapping": (
                "next filename in deterministic sorted dataset order with "
                "circular wraparound"
            ),
            "interventions": {
                "Zero guide": "all-zero image supplied to every GDFMB",
                "Uniform guide": (
                    "per-channel spatial mean of the normal full guide, "
                    "broadcast to image size"
                ),
                "Raw-input guide": (
                    "original underwater input supplied as guide while the "
                    "main path retains the normal GEPM blend"
                ),
                "Cross-image guide": (
                    "normal full guide computed from another test image"
                ),
                "Full spatial guide": "unmodified checkpoint forward pass",
            },
            "note": (
                "All settings use the same dataset-matched checkpoint. These "
                "are inference interventions, not retrained component ablations."
            ),
        },
        "datasets": {
            dataset: {
                "checkpoint": str(DATASETS[dataset]["proteus_ckpt"]),
                "count": len(paired_samples(DATASETS[dataset], args.limit)),
            }
            for dataset in args.datasets
        },
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
