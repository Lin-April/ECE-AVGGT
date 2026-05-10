"""
7-Scenes camera pose evaluation for VGGT using the final dataset manifest.

Path convention:
  SERVER_ROOT / DATASET_REL_DIR points to the copied 7-scenes-final directory.
  frame_manifest.csv stores paths relative to DATASET_REL_DIR.
"""
import csv
import json
import logging
import os
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.rotation import mat_to_quat


# Local path configuration. Defaults to this repository; override with EVAL_CODE_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = Path(os.environ.get("EVAL_CODE_ROOT", PROJECT_ROOT)).expanduser().resolve()
DATASET_REL_DIR = Path("7-scenes-final")
MANIFEST_NAME = "frame_manifest.csv"
MODEL_NAME_OR_PATH = "facebook/VGGT-1B"
SEED = 0
USE_AVGGT = os.environ.get("USE_AVGGT", "0").lower() in {"1", "true", "yes", "on"}
AVGGT_SUBSAMPLE_FACTOR = int(os.environ.get("AVGGT_SUBSAMPLE_FACTOR", "4"))
AVGGT_TEARLY = int(os.environ.get("AVGGT_TEARLY", "9"))
AVGGT_PRESERVE_DIAGONAL = os.environ.get("AVGGT_PRESERVE_DIAGONAL", "0").lower() in {"1", "true", "yes", "on"}
OUTPUT_SUFFIX = f"_avggt{AVGGT_SUBSAMPLE_FACTOR}" if USE_AVGGT else ""
OUTPUT_JSON = SERVER_ROOT / f"results/7scenes_manifest_eval{OUTPUT_SUFFIX}.json"
OUTPUT_EVAL_MANIFEST = SERVER_ROOT / f"results/7scenes_manifest_eval_frames{OUTPUT_SUFFIX}.csv"


logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False


def dataset_dir() -> Path:
    return SERVER_ROOT / DATASET_REL_DIR


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


def autocast_context(device, dtype):
    if device.type in {"cuda", "mps"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def metric_dtype(device):
    # MPS does not support float64 tensors; CPU/CUDA keep the original precision.
    return torch.float32 if device.type == "mps" else torch.float64


def build_pair_index(n):
    i1, i2 = torch.combinations(torch.arange(n), 2, with_replacement=False).unbind(-1)
    return i1, i2


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)
    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)
    return err_q * 180 / np.pi


def translation_angle(tvec_gt, tvec_pred, eps=1e-15):
    def norm(t):
        return t / (torch.norm(t, dim=1, keepdim=True) + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(norm(tvec_pred) * norm(tvec_gt), dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t)) * 180.0 / np.pi
    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = 1e6
    return torch.min(err_t, (180 - err_t).abs())


def calculate_auc_np(r_error, t_error, max_threshold=30):
    max_errors = np.max(np.stack([r_error, t_error], axis=1), axis=1)
    bins = np.arange(max_threshold + 1)
    hist, _ = np.histogram(max_errors, bins=bins)
    norm_hist = hist.astype(float) / len(max_errors)
    return np.mean(np.cumsum(norm_hist)), norm_hist


def se3_to_relative_pose_error(pred_se3, gt_se3, n):
    i1, i2 = build_pair_index(n)
    rel_gt = gt_se3[i1].bmm(closed_form_inverse_se3(gt_se3[i2]))
    rel_pred = pred_se3[i1].bmm(closed_form_inverse_se3(pred_se3[i2]))
    r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_err = translation_angle(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
    return r_err, t_err


def read_manifest():
    manifest_path = dataset_dir() / MANIFEST_NAME
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["window"])].append(row)

    samples = []
    for (scene, window), frame_rows in sorted(groups.items()):
        frame_rows.sort(key=lambda r: int(r["selected_index"]))
        samples.append({"scene": scene, "window": window, "frames": frame_rows})
    return samples


def load_sample(sample):
    image_paths = []
    gt_extris = []
    for row in sample["frames"]:
        image_path = dataset_dir() / row["color_path"]
        pose_path = dataset_dir() / row["pose_path"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not pose_path.exists():
            raise FileNotFoundError(pose_path)

        c2w = np.loadtxt(pose_path).reshape(4, 4)
        if not np.isfinite(c2w).all():
            raise ValueError(f"Invalid pose: {pose_path}")
        r_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        r_w2c = r_c2w.T
        t_w2c = -r_w2c @ t_c2w
        gt_extris.append(np.hstack([r_w2c, t_w2c[:, None]]))
        image_paths.append(str(image_path))

    return image_paths, np.stack(gt_extris)


def run_inference(model, image_paths, device, dtype):
    images = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        with autocast_context(device, dtype):
            predictions = model(images)
    extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    return extrinsic[0]


def evaluate_sample(model, sample, device, dtype):
    image_paths, gt_extris = load_sample(sample)
    pred_extri = run_inference(model, image_paths, device, dtype)
    n = len(image_paths)
    eval_dtype = metric_dtype(device)

    gt_t = torch.from_numpy(gt_extris).to(device, dtype=eval_dtype)
    pad = torch.tensor([0, 0, 0, 1], device=device, dtype=eval_dtype).expand(n, 1, 4)
    pred_se3 = torch.cat([pred_extri.to(eval_dtype), pad], dim=1)
    gt_se3 = torch.cat([gt_t, pad], dim=1)
    r_err, t_err = se3_to_relative_pose_error(pred_se3, gt_se3, n)

    return r_err.cpu().numpy(), t_err.cpu().numpy()


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples = read_manifest()
    print(f"Loaded {len(samples)} windows from {dataset_dir() / MANIFEST_NAME}")

    device = select_device()
    dtype = select_inference_dtype(device)
    print(f"Using device: {device} ({dtype})")

    print("Loading model...")
    model = VGGT.from_pretrained(MODEL_NAME_OR_PATH)
    model.eval().to(device)
    if USE_AVGGT:
        from avggt import apply_avggt

        apply_avggt(
            model,
            subsample_factor=AVGGT_SUBSAMPLE_FACTOR,
            tearly=AVGGT_TEARLY,
            preserve_diagonal=AVGGT_PRESERVE_DIAGONAL,
        )
        print(f"Applied AVGGT patch: factor={AVGGT_SUBSAMPLE_FACTOR}, tearly={AVGGT_TEARLY}")
    print("Model loaded.")

    all_r, all_t = [], []
    results = {}
    eval_manifest_rows = []

    for sample in samples:
        key = f"{sample['scene']}/{sample['window']}"
        print(f"\n[{key}]")
        r_err, t_err = evaluate_sample(model, sample, device, dtype)
        auc30, _ = calculate_auc_np(r_err, t_err, 30)
        auc15, _ = calculate_auc_np(r_err, t_err, 15)
        auc5, _ = calculate_auc_np(r_err, t_err, 5)
        auc3, _ = calculate_auc_np(r_err, t_err, 3)
        print(f"  AUC@30={auc30:.4f}  AUC@15={auc15:.4f}  AUC@5={auc5:.4f}  AUC@3={auc3:.4f}")

        all_r.extend(r_err)
        all_t.extend(t_err)
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

    all_r = np.array(all_r)
    all_t = np.array(all_t)
    m30, _ = calculate_auc_np(all_r, all_t, 30)
    m15, _ = calculate_auc_np(all_r, all_t, 15)
    m5, _ = calculate_auc_np(all_r, all_t, 5)
    m3, _ = calculate_auc_np(all_r, all_t, 3)
    results["__mean__"] = {"auc30": float(m30), "auc15": float(m15), "auc5": float(m5), "auc3": float(m3)}

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_csv(OUTPUT_EVAL_MANIFEST, eval_manifest_rows)
    print(f"\nResults saved to {OUTPUT_JSON}")
    print(f"Eval manifest saved to {OUTPUT_EVAL_MANIFEST}")


if __name__ == "__main__":
    main()
