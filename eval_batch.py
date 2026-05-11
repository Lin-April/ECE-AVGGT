"""Run a matrix of VGGT/AVGGT evaluations with one model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import eval_7scenes
import eval_re10k
from vggt.models.vggt import VGGT


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME_OR_PATH = "facebook/VGGT-1B"
DEFAULT_FRAMES = (20, 100, 200)
DEFAULT_FACTORS = (1, 2, 4, 6, 9)
DATASETS = ("re10k", "7scenes")


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline and AVGGT evals with one loaded VGGT model.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Root containing datasets and results/.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS), help="Datasets to evaluate.")
    parser.add_argument("--frames", nargs="+", type=int, default=list(DEFAULT_FRAMES), help="Frame counts to evaluate.")
    parser.add_argument(
        "--factors",
        nargs="+",
        type=int,
        choices=(1, 2, 4, 6, 9),
        default=list(DEFAULT_FACTORS),
        help="AVGGT subsampling factors to evaluate.",
    )
    parser.add_argument("--tearly", type=int, default=9, help="Number of early global blocks converted to frame attention.")
    parser.add_argument("--preserve-diagonal", action="store_true", help="Use explicit diagonal self-attention in AVGGT.")
    parser.add_argument(
        "--sample-mode",
        choices=("uniform", "sequential"),
        default="uniform",
        help="RealEstate10K frame sampling mode.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs whose accuracy and profile files already exist.")
    parser.add_argument("--warmup-samples", type=int, default=0, help="Unrecorded warmup samples per config.")
    parser.add_argument("--no-baseline", action="store_true", help="Only run AVGGT configs.")
    args = parser.parse_args()
    if any(frame_count < 2 for frame_count in args.frames):
        parser.error("--frames values must be at least 2")
    if args.warmup_samples < 0:
        parser.error("--warmup-samples must be non-negative")
    return args


def make_eval_args(args, frames, avggt=False, factor=4):
    return SimpleNamespace(
        data_root=args.data_root,
        frames=frames,
        sample_mode=args.sample_mode,
        profile=True,
        avggt=avggt,
        subsample_factor=factor,
        tearly=args.tearly,
        preserve_diagonal=args.preserve_diagonal,
    )


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_inference_dtype(device):
    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def calculate_means(all_r, all_t, eval_module):
    all_r = np.array(all_r)
    all_t = np.array(all_t)
    m30, _ = eval_module.calculate_auc_np(all_r, all_t, 30)
    m15, _ = eval_module.calculate_auc_np(all_r, all_t, 15)
    m5, _ = eval_module.calculate_auc_np(all_r, all_t, 5)
    m3, _ = eval_module.calculate_auc_np(all_r, all_t, 3)
    return {"auc30": float(m30), "auc15": float(m15), "auc5": float(m5), "auc3": float(m3)}


def files_exist(paths):
    return paths["results"].exists() and paths["profile"].exists()


def warmup_re10k(model, samples, data_root, device, dtype, count):
    for sample in samples[:count]:
        eval_re10k.evaluate_sample(model, sample, data_root, device, dtype, profile=False)


def warmup_7scenes(model, samples, data_root, device, dtype, count):
    for sample in samples[:count]:
        eval_7scenes.evaluate_sample(model, sample, data_root, device, dtype, profile=False)


def run_re10k(model, data_root, args, device, dtype):
    eval_args = make_eval_args(args, args.current_frames, args.current_avggt, args.current_factor)
    paths = eval_re10k.output_paths(data_root, eval_args)
    if args.skip_existing and files_exist(paths):
        print(f"Skipping existing re10k frames={eval_args.frames} {label(eval_args)}")
        return "skipped"

    samples = eval_re10k.read_manifest(data_root, eval_args.frames, eval_args.sample_mode)
    print(f"\n[re10k] frames={eval_args.frames} {label(eval_args)}")
    print(f"Loaded {len(samples)} scenes from {eval_re10k.dataset_dir(data_root) / eval_re10k.MANIFEST_NAME}")

    if args.warmup_samples:
        warmup_re10k(model, samples, data_root, device, dtype, args.warmup_samples)

    all_r, all_t = [], []
    results = {}
    eval_manifest_rows = []
    profile_rows = []
    for sample in samples:
        scene_id = sample["scene_id"]
        print(f"  {scene_id}")
        r_err, t_err, profile_result = eval_re10k.evaluate_sample(model, sample, data_root, device, dtype, profile=True)
        auc30, _ = eval_re10k.calculate_auc_np(r_err, t_err, 30)
        auc15, _ = eval_re10k.calculate_auc_np(r_err, t_err, 15)
        auc5, _ = eval_re10k.calculate_auc_np(r_err, t_err, 5)
        auc3, _ = eval_re10k.calculate_auc_np(r_err, t_err, 3)
        print(f"    AUC@30={auc30:.4f} inference={profile_result['inference_seconds']:.4f}s")
        all_r.extend(r_err)
        all_t.extend(t_err)
        profile_rows.append({"key": scene_id, **profile_result})
        results[scene_id] = {
            "num_frames": len(sample["frames"]),
            "auc30": float(auc30),
            "auc15": float(auc15),
            "auc5": float(auc5),
            "auc3": float(auc3),
        }
        eval_manifest_rows.extend(sample["frames"])

    results["__mean__"] = calculate_means(all_r, all_t, eval_re10k)
    paths["results"].parent.mkdir(parents=True, exist_ok=True)
    with paths["results"].open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    eval_re10k.write_csv(paths["manifest"], eval_manifest_rows)
    eval_re10k.write_profile(paths["profile"], profile_rows, eval_args, device, dtype)
    print(f"Saved {paths['results']}")
    print(f"Saved {paths['profile']}")
    return "done"


def run_7scenes(model, data_root, args, device, dtype):
    eval_args = make_eval_args(args, args.current_frames, args.current_avggt, args.current_factor)
    paths = eval_7scenes.output_paths(data_root, eval_args)
    if args.skip_existing and files_exist(paths):
        print(f"Skipping existing 7scenes frames={eval_args.frames} {label(eval_args)}")
        return "skipped"

    samples = eval_7scenes.read_manifest(data_root, eval_args.frames)
    print(f"\n[7scenes] frames={eval_args.frames} {label(eval_args)}")
    print(f"Loaded {len(samples)} windows from {eval_7scenes.dataset_dir(data_root) / eval_7scenes.MANIFEST_NAME}")

    if args.warmup_samples:
        warmup_7scenes(model, samples, data_root, device, dtype, args.warmup_samples)

    all_r, all_t = [], []
    results = {}
    eval_manifest_rows = []
    profile_rows = []
    for sample in samples:
        key = f"{sample['scene']}/{sample['window']}"
        print(f"  {key}")
        r_err, t_err, profile_result = eval_7scenes.evaluate_sample(model, sample, data_root, device, dtype, profile=True)
        auc30, _ = eval_7scenes.calculate_auc_np(r_err, t_err, 30)
        auc15, _ = eval_7scenes.calculate_auc_np(r_err, t_err, 15)
        auc5, _ = eval_7scenes.calculate_auc_np(r_err, t_err, 5)
        auc3, _ = eval_7scenes.calculate_auc_np(r_err, t_err, 3)
        print(f"    AUC@30={auc30:.4f} inference={profile_result['inference_seconds']:.4f}s")
        all_r.extend(r_err)
        all_t.extend(t_err)
        profile_rows.append({"key": key, **profile_result})
        results[key] = {
            "scene": sample["scene"],
            "window": sample["window"],
            "num_frames": len(sample["frames"]),
            "auc30": float(auc30),
            "auc15": float(auc15),
            "auc5": float(auc5),
            "auc3": float(auc3),
        }
        eval_manifest_rows.extend(sample["frames"])

    results["__mean__"] = calculate_means(all_r, all_t, eval_7scenes)
    paths["results"].parent.mkdir(parents=True, exist_ok=True)
    with paths["results"].open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    eval_7scenes.write_csv(paths["manifest"], eval_manifest_rows)
    eval_7scenes.write_profile(paths["profile"], profile_rows, eval_args, device, dtype)
    print(f"Saved {paths['results']}")
    print(f"Saved {paths['profile']}")
    return "done"


def label(eval_args):
    if eval_args.avggt:
        return f"AVGGT-{eval_args.subsample_factor}"
    return "VGGT"


def apply_avggt(model, args, factor):
    from avggt import apply_avggt as patch_model

    patch_model(
        model,
        subsample_factor=factor,
        tearly=args.tearly,
        preserve_diagonal=args.preserve_diagonal,
    )
    print(
        "Applied AVGGT patch: "
        f"factor={factor}, tearly={args.tearly}, preserve_diagonal={args.preserve_diagonal}"
    )


def run_config(model, data_root, args, device, dtype, dataset, frames, avggt, factor):
    args.current_frames = frames
    args.current_avggt = avggt
    args.current_factor = factor
    try:
        if dataset == "re10k":
            return run_re10k(model, data_root, args, device, dtype)
        return run_7scenes(model, data_root, args, device, dtype)
    except ValueError as exc:
        print(f"Skipping {dataset} frames={frames} {label(make_eval_args(args, frames, avggt, factor))}: {exc}")
        return "skipped"


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    torch.manual_seed(eval_re10k.SEED)
    np.random.seed(eval_re10k.SEED)

    device = select_device()
    dtype = select_inference_dtype(device)
    print(f"Using device: {device} ({dtype})")
    print("Loading model once...")
    model = VGGT.from_pretrained(MODEL_NAME_OR_PATH)
    model.eval().to(device)
    print("Model loaded.")

    counts = {"done": 0, "skipped": 0}
    if not args.no_baseline:
        for dataset in args.datasets:
            for frames in args.frames:
                status = run_config(model, data_root, args, device, dtype, dataset, frames, False, 4)
                counts[status] += 1

    for factor in args.factors:
        apply_avggt(model, args, factor)
        for dataset in args.datasets:
            for frames in args.frames:
                status = run_config(model, data_root, args, device, dtype, dataset, frames, True, factor)
                counts[status] += 1

    print(f"\nBatch complete: {counts['done']} finished, {counts['skipped']} skipped.")
    print("Generate plots with:")
    print(
        "  uv run python plot_results.py "
        f"--frames {' '.join(str(frame) for frame in args.frames)} "
        f"--factors {' '.join(str(factor) for factor in args.factors)}"
    )


if __name__ == "__main__":
    main()
