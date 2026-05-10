# VGGT Evaluation

This project evaluates `facebook/VGGT-1B` camera pose predictions on prepared
RealEstate10K and 7-Scenes subsets.

## Setup

```bash
uv sync
```

The scripts default to this repository as the data root. To use a copied dataset
elsewhere, set `EVAL_CODE_ROOT`:

```bash
EVAL_CODE_ROOT=/path/to/eval_code uv run python eval_re10k.py
```

## Run

```bash
uv run python eval_re10k.py
uv run python eval_7scenes.py
```

Results are written under `results/`.

## AVGGT Patch

The local `avggt/` package monkey-patches the loaded VGGT model at inference
time. It does not modify model weights.

```bash
USE_AVGGT=1 AVGGT_SUBSAMPLE_FACTOR=4 uv run python eval_re10k.py
USE_AVGGT=1 AVGGT_SUBSAMPLE_FACTOR=4 uv run python eval_7scenes.py
```

Supported `AVGGT_SUBSAMPLE_FACTOR` values are `1`, `2`, `4`, `6`, and `9`.
`AVGGT_TEARLY` defaults to `9`, matching the paper's VGGT setting.

By default the patch uses PyTorch scaled-dot-product attention with subsampled
K/V tokens and one mean K/V token for dropped patches. Set
`AVGGT_PRESERVE_DIAGONAL=1` to also add explicit diagonal self-attention terms;
this is closer to the paper but can use much more memory.
