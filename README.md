# Optimizing and Analyzing VGGT

This repository contains the evaluation code for our CS543 final project on
training-free acceleration of VGGT for multi-view camera pose estimation.

We evaluate the original VGGT model and AVGGT-inspired inference-time attention
patches on prepared RealEstate10K and 7-Scenes subsets. The final experiments
cover static K/V subsampling, frame-count scaling, BI-guided layer routing, and
Pyramid K/V budget schedules.

## Setup

The project uses `uv`:

```bash
uv sync
```

Run commands through `uv run` so the project virtual environment is used:

```bash
uv run python eval_batch.py --skip-existing
```

The scripts default to this repository as the data root. To use the copied
datasets elsewhere, pass `--data-root`:

```bash
uv run python eval_batch.py --data-root /path/to/ECE-AVGGT --skip-existing
```

## Main Scripts

```text
eval_re10k.py                  # Run one RealEstate10K configuration.
eval_7scenes.py                # Run one 7-Scenes configuration.
eval_batch.py                  # Load VGGT once and run multiple configurations.
plot_results.py                # Generate plots from result files.
bi_calibration.py              # Compute BI scores for calibration scenes.
a4_analysis.ipynb              # BI score vs. layer sensitivity analysis.
analyze_bi_pyramid_results.py  # Summarize BI/Pyramid ablation results.
```

## Standard Evaluation

The recommended entry point is `eval_batch.py`, because it loads VGGT once and
then runs the requested configurations.

```bash
uv run python eval_batch.py --skip-existing
```

By default, the batch runner evaluates:

```text
7-Scenes:       frames 5, 10, 20
RealEstate10K:  frames 5, 10, 20, 60
AVGGT factors:  1, 2, 4, 6, 9
```

Useful variants:

```bash
uv run python eval_batch.py --datasets 7scenes --7scenes-frames 5 10 20 --skip-existing
uv run python eval_batch.py --datasets re10k --re10k-frames 5 10 20 60 --skip-existing
uv run python eval_batch.py --frames 20 --factors 4 --warmup-samples 1 --skip-existing
```

Single-dataset scripts are still useful for debugging:

```bash
uv run python eval_re10k.py --profile --frames 20
uv run python eval_7scenes.py --profile --frames 20
uv run python eval_re10k.py --profile --frames 60 --avggt --subsample-factor 4
```

## BI Routing and Pyramid Budgets

The local `avggt/` package monkey-patches the loaded VGGT model at inference
time. It does not modify model weights.

Static AVGGT-style K/V subsampling:

```bash
uv run python eval_batch.py --frames 20 --factors 4 --skip-existing
```

BI-guided routing and Pyramid budget modes:

```bash
uv run python eval_batch.py \
  --frames 20 \
  --factors 4 \
  --bi-routing results/bi.json \
  --budget-fn uniform \
  --results-dir results/new_bi_pyramid/uniform \
  --skip-existing
```

Budget shapes supported by `--budget-fn`:

```text
uniform
linear
exp
```

The BI routing file should contain either a `routing` list or layer scores such
as `bi_scores`. When scores are provided, the script builds a skip/frame/global
routing schedule using the configured tier sizes.

## Results and Plots

Evaluation outputs are written under `results/` by default, or under the folder
passed to `--results-dir`.

Common result files:

```text
results/*manifest_eval*.json          # Pose AUC summaries.
results/*manifest_eval_frames*.csv    # Per-frame/pair evaluation records.
results/*profile*.json                # Inference-time profiles.
results/plots/                        # Generated plots and summary tables.
```

Generate the standard result plots:

```bash
uv run python plot_results.py
```

Summarize BI/Pyramid ablation results:

```bash
uv run python analyze_bi_pyramid_results.py
```

## Final Report

The final report source is in:

```text
report/main.tex
report/references.bib
report/results/plots/
```

The compiled PDF is:

```text
report/Optimizing_and_Analyzing_VGGT_for_Multi_View_3D_Reconstruction.pdf
```

## Notes

- RealEstate10K supports up to 60 frames per prepared scene.
- 7-Scenes supports up to 20 frames per prepared window.
- Use `--skip-existing` to avoid overwriting completed result files.
- Use `--results-dir` for new experiments so final results do not mix with
  older runs.
