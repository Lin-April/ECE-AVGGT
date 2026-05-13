import torch
import torch.nn.functional as F
import json
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from eval_re10k import (
    read_manifest as read_re10k,
    load_sample as load_re10k,
    select_device,
    select_inference_dtype
)
from eval_7scenes import (
    read_manifest as read_s7,
    load_sample as load_s7
)


def compute_bi_scores(model, calibration_samples, data_root, device, dtype, num_register=5):
    bi_per_layer = [[] for _ in range(24)]
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, outputs):
            x_in = inputs[0].detach().float()
            x_out = outputs[0].detach().float() if isinstance(outputs, (tuple, list)) else outputs.detach().float()

            x_in_p = x_in[:, num_register:]
            x_out_p = x_out[:, num_register:]

            cos = F.cosine_similarity(x_in_p, x_out_p, dim=-1).mean().item()
            bi_per_layer[layer_idx].append(1.0 - cos)

        return hook

    for l, block in enumerate(model.aggregator.global_blocks):
        handles.append(block.register_forward_hook(make_hook(l)))

    print(f"Running forward pass on {len(calibration_samples)} scenes...")
    model.eval()
    with torch.no_grad():
        for sample in calibration_samples:
            if 'source' in sample and sample['source'] == '7scenes':
                image_paths, _ = load_s7(sample, data_root)
            else:
                image_paths, _ = load_re10k(sample, data_root)

            images = load_and_preprocess_images(image_paths).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                model(images)

    for h in handles:
        h.remove()

    return [sum(scores) / len(scores) for scores in bi_per_layer]


if __name__ == "__main__":
    DATA_ROOT = Path(".").resolve()
    DEVICE = select_device()
    DTYPE = select_inference_dtype(DEVICE)

    # 1. Selected 5+3 scenario IDs
    re10k_ids = ["0c6b149da098b121", "00e8df74b6805da7", "004dd4b46a06e5be", "094fd37f09dc318c", "0588138dfec165a1"]
    s7_ids = ["chess/window-00", "fire/window-01", "pumpkin/window-02"]

    # 2. Read RE10K samples
    all_re10k = read_re10k(DATA_ROOT, frames_per_scene=20, sample_mode="sequential")
    re10k_calib = [s for s in all_re10k if s['scene_id'] in re10k_ids]
    for s in re10k_calib: s['source'] = 're10k'

    # 3. Read 7-Scenes samples
    all_s7 = read_s7(DATA_ROOT, frames_per_window=20)
    s7_calib = []
    for s in all_s7:
        # 7-Scenes key: "scene/window"
        key = f"{s['scene']}/{s['window']}"
        if key in s7_ids:
            s['source'] = '7scenes'
            s7_calib.append(s)

    calib_samples = re10k_calib + s7_calib
    print(f"Calibration set: {len(re10k_calib)} RE10K + {len(s7_calib)} 7-Scenes = {len(calib_samples)} total.")

    # 4. Compute and Save
    print("Loading VGGT-1B for calibration...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(DEVICE)

    bi_scores = compute_bi_scores(model, calib_samples, DATA_ROOT, DEVICE, DTYPE)

    output_path = Path("results/bi_scores.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bi_scores, f, indent=2)

    print(f"Success! Corrected BI scores saved to {output_path}")