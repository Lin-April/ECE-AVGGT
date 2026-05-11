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

## Run

```bash
uv run python eval_re10k.py
uv run python eval_7scenes.py
```

Results are written under `results/`.

## Profiling

Pass `--profile` to write inference-time metrics to a separate profile file
while keeping the accuracy output unchanged.

```bash
uv run python eval_re10k.py --profile
uv run python eval_7scenes.py --profile
uv run python eval_re10k.py --profile --avggt --subsample-factor 4
uv run python eval_7scenes.py --profile --avggt --subsample-factor 4
```

Profile results are written under `results/`, for example
`re10k_profile.json`, `7scenes_profile.json`, and `_avggt4` variants. The
profile files record inference time only.

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

By default this expects baseline plus AVGGT factors `2`, `4`, `6`, and `9` for
both datasets at `--frames 20`. If files are missing, the script prints the exact
eval commands needed to generate them. Plots and `summary.csv` are written to
`results/plots/`.
