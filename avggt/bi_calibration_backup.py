"""Compute Block Influence scores for VGGT global blocks.

Backup copy of the manifest-driven BI calibration used for the BI/Pyramid
experiments. It keeps richer metadata than the root-level calibration script:
calibration sample IDs, derived routing, and per-sample per-layer scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_7scenes
import eval_re10k
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


MODEL_NAME_OR_PATH = "facebook/VGGT-1B"
DEFAULT_OUTPUT = Path("results/bi.json")
DEFAULT_RE10K_SCENES = 5
DEFAULT_7SCENES_WINDOWS = 3
DEFAULT_FRAMES = 20
DEFAULT_N_SKIP = 5
DEFAULT_N_FRAME = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Compute VGGT Block Influence routing JSON.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT, help="Root containing final datasets and results/.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output BI JSON path, relative to data-root if not absolute.")
    parser.add_argument("--re10k-scenes", type=int, default=DEFAULT_RE10K_SCENES, help="Number of RE10K scenes for calibration.")
    parser.add_argument(
        "--7scenes-windows",
        dest="seven_scenes_windows",
        type=int,
        default=DEFAULT_7SCENES_WINDOWS,
        help="Number of 7-Scenes windows for calibration.",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help="Frames per calibration sample.")
    parser.add_argument("--n-skip", type=int, default=DEFAULT_N_SKIP, help="Lowest-BI layers routed to skip.")
    parser.add_argument("--n-frame", type=int, default=DEFAULT_N_FRAME, help="Next-lowest-BI layers routed to frame.")
    return parser.parse_args()


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


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def load_calibration_samples(data_root, frames, n_re10k, n_7scenes):
    samples = []

    if n_re10k:
        re10k_samples = eval_re10k.read_manifest(data_root, frames, eval_re10k.DEFAULT_SAMPLE_MODE)
        for sample in re10k_samples[:n_re10k]:
            image_paths, _ = eval_re10k.load_sample(sample, data_root)
            samples.append(
                {
                    "dataset": "re10k",
                    "key": sample["scene_id"],
                    "image_paths": image_paths,
                }
            )

    if n_7scenes:
        seven_samples = eval_7scenes.read_manifest(data_root, frames)
        selected_scenes = set()
        selected = []
        for sample in seven_samples:
            if sample["scene"] in selected_scenes:
                continue
            selected_scenes.add(sample["scene"])
            selected.append(sample)
            if len(selected) >= n_7scenes:
                break
        for sample in selected:
            image_paths, _ = eval_7scenes.load_sample(sample, data_root)
            samples.append(
                {
                    "dataset": "7scenes",
                    "key": f"{sample['scene']}/{sample['window']}",
                    "image_paths": image_paths,
                }
            )

    return samples


def compute_bi_scores(model, samples, device, dtype):
    blocks = model.aggregator.global_blocks
    layer_scores = [[] for _ in blocks]
    current_frames = {"count": None}
    patch_start_idx = int(getattr(model.aggregator, "patch_start_idx", 0))

    def make_hook(layer_idx):
        def hook(_module, inputs, output):
            x_in = inputs[0].detach().float()
            x_out = output.detach().float() if isinstance(output, torch.Tensor) else output[0].detach().float()
            x_in, x_out = _select_patch_tokens(x_in, x_out, current_frames["count"], patch_start_idx)
            score = 1.0 - F.cosine_similarity(x_in, x_out, dim=-1).mean().item()
            layer_scores[layer_idx].append(float(score))

        return hook

    handles = [block.register_forward_hook(make_hook(layer_idx)) for layer_idx, block in enumerate(blocks)]
    try:
        for sample in samples:
            current_frames["count"] = len(sample["image_paths"])
            print(f"[calib] {sample['dataset']} {sample['key']} frames={current_frames['count']}")
            images = load_and_preprocess_images(sample["image_paths"]).to(device)
            synchronize_device(device)
            with torch.no_grad():
                if device.type in {"cuda", "mps"}:
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        model(images)
                else:
                    model(images)
            synchronize_device(device)
    finally:
        for handle in handles:
            handle.remove()

    if any(not scores for scores in layer_scores):
        empty = [idx for idx, scores in enumerate(layer_scores) if not scores]
        raise RuntimeError(f"No BI hook values collected for layers: {empty}")
    return [float(np.mean(scores)) for scores in layer_scores], layer_scores


def _select_patch_tokens(x_in, x_out, frames, patch_start_idx):
    if frames is None or frames <= 0:
        return x_in, x_out
    token_count = x_in.shape[1]
    if token_count % frames != 0:
        return x_in, x_out
    tokens_per_frame = token_count // frames
    if patch_start_idx <= 0 or patch_start_idx >= tokens_per_frame:
        return x_in, x_out

    mask = torch.ones(token_count, device=x_in.device, dtype=torch.bool)
    for frame_idx in range(frames):
        start = frame_idx * tokens_per_frame
        mask[start : start + patch_start_idx] = False
    return x_in[:, mask, :], x_out[:, mask, :]


def routing_from_bi(bi_scores, n_skip, n_frame):
    scores = np.array(bi_scores, dtype=float)
    if n_skip < 0 or n_frame < 0 or n_skip + n_frame > len(scores):
        raise ValueError("n_skip and n_frame must fit within the number of layers")
    order = np.argsort(scores)
    routing = ["kv"] * len(scores)
    for layer_idx in order[:n_skip]:
        routing[int(layer_idx)] = "skip"
    for layer_idx in order[n_skip : n_skip + n_frame]:
        routing[int(layer_idx)] = "frame"
    return routing


def output_path(data_root, path):
    path = path.expanduser()
    if path.is_absolute():
        return path
    return data_root / path


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    torch.manual_seed(eval_re10k.SEED)
    np.random.seed(eval_re10k.SEED)

    device = select_device()
    dtype = select_inference_dtype(device)
    print(f"Using device: {device} ({dtype})")

    samples = load_calibration_samples(
        data_root=data_root,
        frames=args.frames,
        n_re10k=args.re10k_scenes,
        n_7scenes=args.seven_scenes_windows,
    )
    if not samples:
        raise ValueError("No calibration samples selected")
    print(f"Loaded {len(samples)} calibration samples")

    model = VGGT.from_pretrained(MODEL_NAME_OR_PATH)
    model.eval().to(device)

    bi_scores, raw_scores = compute_bi_scores(model, samples, device, dtype)
    routing = routing_from_bi(bi_scores, args.n_skip, args.n_frame)

    payload = {
        "bi_scores": bi_scores,
        "routing": routing,
        "n_skip": args.n_skip,
        "n_frame": args.n_frame,
        "frames": args.frames,
        "calibration_samples": [
            {"dataset": sample["dataset"], "key": sample["key"], "num_frames": len(sample["image_paths"])}
            for sample in samples
        ],
        "per_sample_scores": raw_scores,
    }

    out = output_path(data_root, args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
