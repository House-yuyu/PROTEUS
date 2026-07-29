#!/usr/bin/env python3
"""Evaluate restoration across real spatial-degradation strata.

This analysis uses only the original paired benchmark images.  It does not
synthesize degradations.  For each input/reference pair, a model-independent
spatial-degradation score is computed from low-frequency CIELAB residuals:

1. resize the aligned pair to the evaluation size;
2. average Lab(input)-Lab(reference) over an 8x8 grid;
3. subtract the image-wide residual, removing uniform colour bias;
4. take the RMS magnitude of the remaining block residuals.

Samples are rank-split into equal-sized within-dataset tertiles.  The labels
Mild/Moderate/Severe therefore mean relative spatial non-uniformity within a
dataset, not absolute physical degradation levels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from skimage.color import rgb2lab


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
if str(RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT))

from degradation_severity_eval import (  # noqa: E402
    DATASETS,
    bootstrap_ci,
    image_metrics,
    load_gt_batch,
    load_proteus,
    load_wwe,
    paired_samples,
    proteus_uniform_guide,
    save_uint8,
)


LEVELS = ("Mild", "Moderate", "Severe")
METHODS = ("Input", "WWE-UIE", "Uniform guide", "PROTEUS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["U90", "LSUI"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Compute strata and representative real images without model inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "results_real_spatial",
    )
    return parser.parse_args()


def load_rgb(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def spatial_degradation_statistics(
    input_rgb: np.ndarray,
    target_rgb: np.ndarray,
    grid_size: int,
) -> dict[str, float | np.ndarray]:
    if input_rgb.shape != target_rgb.shape:
        raise ValueError(f"Shape mismatch: {input_rgb.shape} vs {target_rgb.shape}")
    height, width, _ = input_rgb.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by grid {grid_size}"
        )

    residual = rgb2lab(input_rgb) - rgb2lab(target_rgb)
    block_height = height // grid_size
    block_width = width // grid_size
    block_means = residual.reshape(
        grid_size,
        block_height,
        grid_size,
        block_width,
        3,
    ).mean(axis=(1, 3))
    global_residual = residual.mean(axis=(0, 1))
    centred_blocks = block_means - global_residual[None, None, :]
    block_magnitudes = np.linalg.norm(centred_blocks, axis=2)
    pixel_delta_e = np.linalg.norm(residual, axis=2)
    return {
        "spatial_score": float(np.sqrt(np.mean(block_magnitudes**2))),
        "global_cast_score": float(np.linalg.norm(global_residual)),
        "mean_delta_e": float(pixel_delta_e.mean()),
        "block_magnitudes": block_magnitudes.astype(np.float32),
    }


def assign_balanced_tertiles(records: list[dict]) -> dict[str, dict]:
    scores = np.array([record["spatial_score"] for record in records])
    order = np.argsort(scores, kind="mergesort")
    split_indices = np.array_split(order, len(LEVELS))
    metadata: dict[str, dict] = {}
    for level, indices in zip(LEVELS, split_indices):
        level_scores = scores[indices]
        for index in indices:
            records[int(index)]["severity"] = level
        metadata[level] = {
            "count": int(len(indices)),
            "score_min": float(level_scores.min()),
            "score_median": float(np.median(level_scores)),
            "score_max": float(level_scores.max()),
        }
    return metadata


def build_score_records(
    dataset_name: str,
    size: int,
    grid_size: int,
    limit: int | None,
) -> tuple[list[dict], dict]:
    samples = paired_samples(DATASETS[dataset_name], limit)
    records: list[dict] = []
    for sample in samples:
        input_rgb = load_rgb(sample["input"], size)
        target_rgb = load_rgb(sample["gt"], size)
        statistics = spatial_degradation_statistics(
            input_rgb, target_rgb, grid_size
        )
        records.append(
            {
                "dataset": dataset_name,
                "filename": sample["name"],
                "input_path": str(sample["input"]),
                "gt_path": str(sample["gt"]),
                "spatial_score": statistics["spatial_score"],
                "global_cast_score": statistics["global_cast_score"],
                "mean_delta_e": statistics["mean_delta_e"],
            }
        )
    strata = assign_balanced_tertiles(records)
    return records, strata


def representative_records(records: list[dict]) -> dict[str, dict]:
    representatives: dict[str, dict] = {}
    for level in LEVELS:
        candidates = [record for record in records if record["severity"] == level]
        median_score = float(np.median([item["spatial_score"] for item in candidates]))
        representatives[level] = min(
            candidates,
            key=lambda item: (
                abs(item["spatial_score"] - median_score),
                item["filename"],
            ),
        )
    return representatives


def write_score_outputs(
    all_records: list[dict],
    strata: dict[str, dict],
    output_dir: Path,
    size: int,
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "filename",
        "severity",
        "spatial_score",
        "global_cast_score",
        "mean_delta_e",
        "input_path",
        "gt_path",
    ]
    with (output_dir / "spatial_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    representatives: dict[str, dict] = {}
    for dataset_name in sorted({record["dataset"] for record in all_records}):
        dataset_records = [
            record for record in all_records if record["dataset"] == dataset_name
        ]
        representatives[dataset_name] = representative_records(dataset_records)
        example_root = output_dir / "examples" / dataset_name
        example_root.mkdir(parents=True, exist_ok=True)
        for level, record in representatives[dataset_name].items():
            input_tensor = torch.from_numpy(
                load_rgb(Path(record["input_path"]), size)
            ).permute(2, 0, 1)
            target_tensor = torch.from_numpy(
                load_rgb(Path(record["gt_path"]), size)
            ).permute(2, 0, 1)
            save_uint8(example_root / f"{level}_input.png", input_tensor)
            save_uint8(example_root / f"{level}_GT.png", target_tensor)
        (example_root / "representatives.json").write_text(
            json.dumps(representatives[dataset_name], indent=2),
            encoding="utf-8",
        )
    return representatives


def bootstrap_standardized_slope(
    scores: np.ndarray,
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    standardized = (scores - scores.mean()) / max(scores.std(), 1e-12)

    def slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
        centred_x = x_values - x_values.mean()
        centred_y = y_values - y_values.mean()
        denominator = np.sum(centred_x**2)
        return float(np.sum(centred_x * centred_y) / max(denominator, 1e-12))

    observed = slope(standardized, differences)
    bootstrapped = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(scores), size=(count, len(scores)))
        for offset, row in enumerate(indices):
            bootstrapped[start + offset] = slope(
                standardized[row], differences[row]
            )
    low, high = np.percentile(bootstrapped, [2.5, 97.5])
    return observed, (float(low), float(high))


def evaluate_dataset(
    dataset_name: str,
    score_records: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    configuration = DATASETS[dataset_name]
    samples = paired_samples(configuration, args.limit)
    score_by_name = {record["filename"]: record for record in score_records}
    proteus = load_proteus(configuration["proteus_ckpt"], device)
    wwe = load_wwe(configuration["wwe_ckpt"], device)
    rows: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            names = [sample["name"] for sample in batch_samples]
            inputs = load_gt_batch(
                [sample["input"] for sample in batch_samples], args.size, device
            )
            targets = load_gt_batch(
                [sample["gt"] for sample in batch_samples], args.size, device
            )
            predictions = {
                "Input": inputs,
                "WWE-UIE": wwe(inputs).clamp(0.0, 1.0),
                "Uniform guide": proteus_uniform_guide(proteus, inputs),
                "PROTEUS": proteus(inputs)[0].clamp(0.0, 1.0),
            }
            for method, prediction in predictions.items():
                psnr_values, ssim_values = image_metrics(prediction, targets)
                for index, name in enumerate(names):
                    score_record = score_by_name[name]
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "filename": name,
                            "severity": score_record["severity"],
                            "spatial_score": score_record["spatial_score"],
                            "method": method,
                            "psnr": float(psnr_values[index]),
                            "ssim": float(ssim_values[index]),
                        }
                    )
            print(
                f"[{dataset_name}] {min(start + args.batch_size, len(samples))}/"
                f"{len(samples)}",
                flush=True,
            )

    del proteus, wwe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    summary: dict[str, dict] = OrderedDict()
    for level in LEVELS:
        level_rows = [row for row in rows if row["severity"] == level]
        per_method = {
            method: [row for row in level_rows if row["method"] == method]
            for method in METHODS
        }
        summary[level] = {}
        for method, method_rows in per_method.items():
            summary[level][method] = {
                "count": len(method_rows),
                "psnr_mean": float(np.mean([row["psnr"] for row in method_rows])),
                "ssim_mean": float(np.mean([row["ssim"] for row in method_rows])),
            }
        full_by_name = {
            row["filename"]: row for row in per_method["PROTEUS"]
        }
        for baseline in ("WWE-UIE", "Uniform guide"):
            differences = np.array(
                [
                    full_by_name[row["filename"]]["psnr"] - row["psnr"]
                    for row in per_method[baseline]
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_ci(differences, args.bootstrap_samples, rng)
            summary[level][f"PROTEUS_minus_{baseline}"] = {
                "psnr_mean_difference": float(differences.mean()),
                "paired_bootstrap_95_ci": [low, high],
                "positive_images": int(np.sum(differences > 0)),
                "count": int(len(differences)),
            }

    full_rows = {
        row["filename"]: row for row in rows if row["method"] == "PROTEUS"
    }
    uniform_rows = {
        row["filename"]: row for row in rows if row["method"] == "Uniform guide"
    }
    ordered_names = sorted(full_rows)
    scores = np.array(
        [score_by_name[name]["spatial_score"] for name in ordered_names],
        dtype=np.float64,
    )
    differences = np.array(
        [
            full_rows[name]["psnr"] - uniform_rows[name]["psnr"]
            for name in ordered_names
        ],
        dtype=np.float64,
    )
    slope, slope_ci = bootstrap_standardized_slope(
        scores, differences, args.bootstrap_samples, rng
    )
    correlation = spearmanr(scores, differences)
    summary["trend"] = {
        "comparison": "PROTEUS minus Uniform guide",
        "standardized_score_slope_db": slope,
        "paired_bootstrap_95_ci": list(slope_ci),
        "spearman_rho": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
    }
    return rows, summary


def build_figure(
    results: dict,
    representatives: dict,
    output_dir: Path,
) -> None:
    colours = {
        "WWE-UIE": "#4C78A8",
        "Uniform guide": "#F58518",
        "PROTEUS": "#2A9D8F",
    }
    markers = {"WWE-UIE": "o", "Uniform guide": "s", "PROTEUS": "^"}
    figure = plt.figure(figsize=(7.0, 3.35))
    grid = figure.add_gridspec(
        2, 6, height_ratios=[1.0, 1.2], hspace=0.42, wspace=0.42
    )
    for index, level in enumerate(LEVELS):
        record = representatives["U90"][level]
        axis = figure.add_subplot(grid[0, index * 2 : (index + 1) * 2])
        axis.imshow(Image.open(record["input_path"]).convert("RGB"))
        axis.set_title(
            f"{level} real input\nscore={record['spatial_score']:.2f}",
            fontsize=7.2,
            pad=2,
        )
        axis.axis("off")

    for index, dataset in enumerate(("U90", "LSUI")):
        axis = figure.add_subplot(grid[1, index * 3 : (index + 1) * 3])
        x_values = np.arange(len(LEVELS))
        for method in ("WWE-UIE", "Uniform guide", "PROTEUS"):
            values = [
                results[dataset][level][method]["psnr_mean"] for level in LEVELS
            ]
            axis.plot(
                x_values,
                values,
                color=colours[method],
                marker=markers[method],
                linewidth=1.6,
                markersize=4.5,
                label=method,
            )
        axis.set_title(dataset, fontsize=8, pad=3)
        axis.set_xticks(x_values, LEVELS, fontsize=6.8)
        axis.tick_params(axis="y", labelsize=7)
        axis.set_ylabel("PSNR (dB)", fontsize=7.5)
        axis.set_facecolor("#F3F4F6")
        axis.grid(color="white", linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_color("#6B7280")
            spine.set_linewidth(0.6)
        if index == 0:
            legend = axis.legend(
                fontsize=6.1,
                frameon=True,
                fancybox=True,
                shadow=True,
                ncol=1,
            )
            legend.get_frame().set_facecolor("white")

    figure.savefig(
        output_dir / "real_spatial_severity_analysis.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "real_spatial_severity_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_score_records: list[dict] = []
    strata: dict[str, dict] = {}
    for dataset in args.datasets:
        records, dataset_strata = build_score_records(
            dataset, args.size, args.grid_size, args.limit
        )
        all_score_records.extend(records)
        strata[dataset] = dataset_strata
        print(f"[{dataset}] scored {len(records)} real pairs", flush=True)
    representatives = write_score_outputs(
        all_score_records, strata, args.output_dir, args.size
    )

    metadata: dict = {
        "protocol": {
            "seed": args.seed,
            "size": args.size,
            "grid_size": args.grid_size,
            "score": (
                "RMS magnitude of 8x8 block-mean CIELAB input-reference "
                "residuals after subtracting the image-wide residual"
            ),
            "stratification": (
                "stable rank split into balanced within-dataset tertiles; "
                "labels indicate relative spatial non-uniformity, not absolute "
                "physical degradation severity"
            ),
            "uses_model_outputs_for_stratification": False,
            "methods": {
                "Input": "unprocessed benchmark input",
                "WWE-UIE": "dataset-matched checkpoint used in the main paper",
                "Uniform guide": (
                    "same PROTEUS checkpoint; each GDFMB guide replaced by its "
                    "per-channel spatial mean at inference"
                ),
                "PROTEUS": "unmodified dataset-matched checkpoint",
            },
        },
        "strata": strata,
        "representatives": representatives,
    }
    (args.output_dir / "score_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if args.score_only:
        return

    device = torch.device(args.device)
    results: dict[str, dict] = {}
    all_metric_rows: list[dict] = []
    for dataset in args.datasets:
        dataset_records = [
            record for record in all_score_records if record["dataset"] == dataset
        ]
        metric_rows, summary = evaluate_dataset(
            dataset, dataset_records, args, device
        )
        all_metric_rows.extend(metric_rows)
        results[dataset] = summary

    with (args.output_dir / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metric_rows[0]))
        writer.writeheader()
        writer.writerows(all_metric_rows)

    metadata["results"] = results
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if set(args.datasets) == {"U90", "LSUI"}:
        build_figure(results, representatives, args.output_dir)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
