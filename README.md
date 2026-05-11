# VGGT Evaluation

This project evaluates `facebook/VGGT-1B` camera pose predictions on prepared
RealEstate10K and 7-Scenes subsets.

## Setup

```bash
uv sync
```

The scripts default to this repository as the data root. To use a copied dataset
elsewhere, pass `--data-root`:

```bash
uv run python eval_re10k.py --data-root /path/to/eval_code
```

## Scripts

Use separate scripts for evaluation, batch experiments, and plotting:

```text
eval_re10k.py      # Run one RealEstate10K eval configuration.
eval_7scenes.py    # Run one 7-Scenes eval configuration.
eval_batch.py      # Recommended: load VGGT once and run many configurations.
plot_results.py    # Read results/ and generate interactive Plotly charts.
```

## Experiment Checklist

Recommended: use `eval_batch.py` to load VGGT once and run the full experiment
matrix in one process. This is much faster than launching one command per
configuration.

```bash
uv run python eval_batch.py --skip-existing
```

By default, the batch runner evaluates:

```text
7-Scenes:       frames 5, 10, 20
RealEstate10K:  frames 5, 10, 20, 60
AVGGT factors:  1, 2, 4, 6, 9
```

Useful options:

```bash
uv run python eval_batch.py --datasets 7scenes --7scenes-frames 5 10 20 --skip-existing
uv run python eval_batch.py --datasets re10k --re10k-frames 5 10 20 60 --skip-existing
uv run python eval_batch.py --frames 20 --factors 1 2 4 6 9 --skip-existing
uv run python eval_batch.py --frames 20 --factors 4 --warmup-samples 1
```

The batch runner always writes both accuracy and inference-time profile files.
It runs baseline VGGT first, then applies the AVGGT patch and runs each
subsampling factor.

After the batch run finishes, generate plots with:

```bash
uv run python plot_results.py
```

Important: `--frames` cannot exceed the number of frames available in each
prepared RealEstate10K scene or 7-Scenes window. The prepared manifests in this
repository support 7-Scenes up to 20 frames per window and RealEstate10K up to
60 frames per scene.

The commands below are equivalent single-run commands. Use them only when you
need to rerun one configuration manually.

### 1. Baseline VGGT

Frames 5:

```bash
uv run python eval_re10k.py --profile --frames 5
uv run python eval_7scenes.py --profile --frames 5
```

Frames 10:

```bash
uv run python eval_re10k.py --profile --frames 10
uv run python eval_7scenes.py --profile --frames 10
```

Frames 20:

```bash
uv run python eval_re10k.py --profile --frames 20
uv run python eval_7scenes.py --profile --frames 20
```

RealEstate10K frames 60:

```bash
uv run python eval_re10k.py --profile --frames 60
```

### 2. AVGGT Factor 1

Frames 5:

```bash
uv run python eval_re10k.py --profile --frames 5 --avggt --subsample-factor 1
uv run python eval_7scenes.py --profile --frames 5 --avggt --subsample-factor 1
```

Frames 10:

```bash
uv run python eval_re10k.py --profile --frames 10 --avggt --subsample-factor 1
uv run python eval_7scenes.py --profile --frames 10 --avggt --subsample-factor 1
```

Frames 20:

```bash
uv run python eval_re10k.py --profile --frames 20 --avggt --subsample-factor 1
uv run python eval_7scenes.py --profile --frames 20 --avggt --subsample-factor 1
```

RealEstate10K frames 60:

```bash
uv run python eval_re10k.py --profile --frames 60 --avggt --subsample-factor 1
```

### 3. AVGGT Factor 2

Frames 5:

```bash
uv run python eval_re10k.py --profile --frames 5 --avggt --subsample-factor 2
uv run python eval_7scenes.py --profile --frames 5 --avggt --subsample-factor 2
```

Frames 10:

```bash
uv run python eval_re10k.py --profile --frames 10 --avggt --subsample-factor 2
uv run python eval_7scenes.py --profile --frames 10 --avggt --subsample-factor 2
```

Frames 20:

```bash
uv run python eval_re10k.py --profile --frames 20 --avggt --subsample-factor 2
uv run python eval_7scenes.py --profile --frames 20 --avggt --subsample-factor 2
```

RealEstate10K frames 60:

```bash
uv run python eval_re10k.py --profile --frames 60 --avggt --subsample-factor 2
```

### 4. AVGGT Factor 4

Frames 5:

```bash
uv run python eval_re10k.py --profile --frames 5 --avggt --subsample-factor 4
uv run python eval_7scenes.py --profile --frames 5 --avggt --subsample-factor 4
```

Frames 10:

```bash
uv run python eval_re10k.py --profile --frames 10 --avggt --subsample-factor 4
uv run python eval_7scenes.py --profile --frames 10 --avggt --subsample-factor 4
```

Frames 20:

```bash
uv run python eval_re10k.py --profile --frames 20 --avggt --subsample-factor 4
uv run python eval_7scenes.py --profile --frames 20 --avggt --subsample-factor 4
```

RealEstate10K frames 60:

```bash
uv run python eval_re10k.py --profile --frames 60 --avggt --subsample-factor 4
```

### 5. AVGGT Factor 6

Frames 5:

```bash
uv run python eval_re10k.py --profile --frames 5 --avggt --subsample-factor 6
uv run python eval_7scenes.py --profile --frames 5 --avggt --subsample-factor 6
```

Frames 10:

```bash
uv run python eval_re10k.py --profile --frames 10 --avggt --subsample-factor 6
uv run python eval_7scenes.py --profile --frames 10 --avggt --subsample-factor 6
```

Frames 20:

```bash
uv run python eval_re10k.py --profile --frames 20 --avggt --subsample-factor 6
uv run python eval_7scenes.py --profile --frames 20 --avggt --subsample-factor 6
```

RealEstate10K frames 60:

```bash
uv run python eval_re10k.py --profile --frames 60 --avggt --subsample-factor 6
```

### 6. AVGGT Factor 9

Frames 5:

```bash
uv run python eval_re10k.py --profile --frames 5 --avggt --subsample-factor 9
uv run python eval_7scenes.py --profile --frames 5 --avggt --subsample-factor 9
```

Frames 10:

```bash
uv run python eval_re10k.py --profile --frames 10 --avggt --subsample-factor 9
uv run python eval_7scenes.py --profile --frames 10 --avggt --subsample-factor 9
```

Frames 20:

```bash
uv run python eval_re10k.py --profile --frames 20 --avggt --subsample-factor 9
uv run python eval_7scenes.py --profile --frames 20 --avggt --subsample-factor 9
```

RealEstate10K frames 60:

```bash
uv run python eval_re10k.py --profile --frames 60 --avggt --subsample-factor 9
```

Results are written under `results/`.

Accuracy files:

```text
results/7scenes_manifest_eval*.json
results/re10k_manifest_eval*.json
```

Inference-time profile files:

```text
results/7scenes_profile*.json
results/re10k_profile*.json
```

Used-frame manifests:

```text
results/7scenes_manifest_eval_frames*.csv
results/re10k_manifest_eval_frames*.csv
```

## AVGGT Patch

The local `avggt/` package monkey-patches the loaded VGGT model at inference
time. It does not modify model weights.

```bash
uv run python eval_re10k.py --avggt --subsample-factor 4
uv run python eval_7scenes.py --avggt --subsample-factor 4
```

Supported `--subsample-factor` values are `1`, `2`, `4`, `6`, and `9`.
`--tearly` defaults to `9`, matching the paper's VGGT setting.

By default the patch uses PyTorch scaled-dot-product attention with subsampled
K/V tokens and one mean K/V token for dropped patches. Pass
`--preserve-diagonal` to also add explicit diagonal self-attention terms; this is
closer to the paper but can use much more memory.

Use `--frames` to choose how many frames are evaluated per RealEstate10K scene or
7-Scenes window. Non-default frame counts add an `_fN` suffix to result files.

## Plot Results

Generate interactive Plotly charts from the default `results/` directory:

```bash
uv run python plot_results.py
```

Plot the full experiment grid listed above:

```bash
uv run python plot_results.py
```

If files are missing, the script prints the exact eval commands needed to
generate them. Plots and `summary.csv` are written to `results/plots/`.
