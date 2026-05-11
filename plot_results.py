"""Plot VGGT/AVGGT evaluation results from the default results directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go


DEFAULT_FRAMES = 20
SUPPORTED_FACTORS = (1, 2, 4, 6, 9)
DEFAULT_FACTORS = (2, 4, 6, 9)
DATASET_TO_SCRIPT = {
    "7scenes": "eval_7scenes.py",
    "re10k": "eval_re10k.py",
}
AUC_METRICS = (
    ("auc30", "AUC@30", "#2563eb"),
    ("auc15", "AUC@15", "#16a34a"),
    ("auc5", "AUC@5", "#f97316"),
    ("auc3", "AUC@3", "#dc2626"),
)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    frames: int
    factor: int | None = None

    @property
    def is_avggt(self) -> bool:
        return self.factor is not None

    @property
    def label(self) -> str:
        if self.is_avggt:
            return f"AVGGT-{self.factor}"
        return "VGGT"

    @property
    def suffix(self) -> str:
        suffix = f"_avggt{self.factor}" if self.is_avggt else ""
        if self.frames != DEFAULT_FRAMES:
            suffix += f"_f{self.frames}"
        return suffix

    def accuracy_path(self, results_dir: Path) -> Path:
        return results_dir / f"{self.dataset}_manifest_eval{self.suffix}.json"

    def profile_path(self, results_dir: Path) -> Path:
        return results_dir / f"{self.dataset}_profile{self.suffix}.json"

    def command(self) -> str:
        parts = ["uv", "run", "python", DATASET_TO_SCRIPT[self.dataset], "--profile"]
        if self.frames != DEFAULT_FRAMES:
            parts.extend(["--frames", str(self.frames)])
        if self.is_avggt:
            parts.extend(["--avggt", "--subsample-factor", str(self.factor)])
        return " ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot VGGT/AVGGT accuracy and speed results.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory containing result JSON files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plots. Defaults to results/plots.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_TO_SCRIPT),
        default=list(DATASET_TO_SCRIPT),
        help="Datasets to include.",
    )
    parser.add_argument("--frames", nargs="+", type=int, default=[DEFAULT_FRAMES], help="Frame counts to include.")
    parser.add_argument(
        "--factors",
        nargs="+",
        type=int,
        choices=SUPPORTED_FACTORS,
        default=list(DEFAULT_FACTORS),
        help="AVGGT subsampling factors to include.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Plot available complete runs instead of failing.")
    return parser.parse_args()


def build_specs(datasets, frames, factors):
    specs = []
    for dataset in datasets:
        for frame_count in frames:
            specs.append(RunSpec(dataset=dataset, frames=frame_count))
            for factor in factors:
                specs.append(RunSpec(dataset=dataset, frames=frame_count, factor=factor))
    return specs


def missing_files(specs, results_dir):
    missing = []
    for spec in specs:
        paths = [spec.accuracy_path(results_dir), spec.profile_path(results_dir)]
        missing_paths = [path for path in paths if not path.exists()]
        if missing_paths:
            missing.append((spec, missing_paths))
    return missing


def print_missing(missing):
    print("Missing required result files.\n", file=sys.stderr)
    for spec, paths in missing:
        print(f"- {spec.dataset} {spec.label} frames={spec.frames}", file=sys.stderr)
        for path in paths:
            print(f"  missing: {path}", file=sys.stderr)
        print(f"  generate with: {spec.command()}\n", file=sys.stderr)

    commands = list(dict.fromkeys(spec.command() for spec, _ in missing))
    print("Copy and run all missing experiments:", file=sys.stderr)
    print("", file=sys.stderr)
    print("```bash", file=sys.stderr)
    for command in commands:
        print(command, file=sys.stderr)
    print("```", file=sys.stderr)
    print("", file=sys.stderr)
    print("After generating the missing files, rerun:", file=sys.stderr)
    print("  uv run python plot_results.py", file=sys.stderr)
    print("Or pass --allow-missing to plot only complete available runs.", file=sys.stderr)


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_run(spec, results_dir):
    accuracy_path = spec.accuracy_path(results_dir)
    profile_path = spec.profile_path(results_dir)
    accuracy = load_json(accuracy_path)
    profile = load_json(profile_path)
    mean = accuracy.get("__mean__")
    if mean is None:
        raise ValueError(f"{accuracy_path} is missing __mean__ accuracy metrics")

    return {
        "dataset": spec.dataset,
        "frames": spec.frames,
        "method": "avggt" if spec.is_avggt else "baseline",
        "factor": spec.factor,
        "label": spec.label,
        "order": spec.factor if spec.factor is not None else 0,
        "auc30": float(mean["auc30"]),
        "auc15": float(mean["auc15"]),
        "auc5": float(mean["auc5"]),
        "auc3": float(mean["auc3"]),
        "mean_inference_seconds": float(profile["mean_inference_seconds"]),
        "median_inference_seconds": float(profile["median_inference_seconds"]),
        "total_inference_seconds": float(profile["total_inference_seconds"]),
        "num_samples": int(profile["num_samples"]),
        "accuracy_file": str(accuracy_path),
        "profile_file": str(profile_path),
        "command": spec.command(),
    }


def load_runs(specs, results_dir, allow_missing):
    rows = []
    for spec in specs:
        if allow_missing and (not spec.accuracy_path(results_dir).exists() or not spec.profile_path(results_dir).exists()):
            continue
        rows.append(load_run(spec, results_dir))
    if not rows:
        raise ValueError("No complete result/profile pairs found.")
    return rows


def write_summary(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    fieldnames = [
        "dataset",
        "frames",
        "method",
        "factor",
        "label",
        "auc30",
        "auc15",
        "auc5",
        "auc3",
        "mean_inference_seconds",
        "median_inference_seconds",
        "total_inference_seconds",
        "num_samples",
        "accuracy_file",
        "profile_file",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def figure_layout(fig, title, x_title, y_title):
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", "size": 14},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 70, "r": 30, "t": 85, "b": 70},
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def write_figure(fig, path):
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def plot_auc(rows, dataset, frames, output_dir):
    labels = [row["label"] for row in rows]
    fig = go.Figure()
    for key, name, color in AUC_METRICS:
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[row[key] for row in rows],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 3},
                marker={"size": 9},
            )
        )
    figure_layout(fig, f"{dataset} Frames={frames}: Accuracy vs AVGGT Factor", "Method", "AUC")
    fig.update_yaxes(range=[0, 1])
    return write_figure(fig, output_dir / f"{dataset}_f{frames}_auc_vs_factor.html")


def plot_time(rows, dataset, frames, output_dir):
    labels = [row["label"] for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[row["mean_inference_seconds"] for row in rows],
            name="Mean",
            marker_color="#6366f1",
            opacity=0.9,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[row["median_inference_seconds"] for row in rows],
            mode="lines+markers",
            name="Median",
            line={"color": "#f97316", "width": 3},
            marker={"size": 9},
        )
    )
    figure_layout(fig, f"{dataset} Frames={frames}: Inference Time", "Method", "Seconds / sample")
    return write_figure(fig, output_dir / f"{dataset}_f{frames}_time_vs_factor.html")


def plot_speedup(rows, dataset, frames, output_dir):
    baseline = next((row for row in rows if row["method"] == "baseline"), None)
    if baseline is None:
        return None

    labels = [row["label"] for row in rows]
    speedups = [baseline["mean_inference_seconds"] / row["mean_inference_seconds"] for row in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=speedups,
            text=[f"{value:.2f}x" for value in speedups],
            textposition="outside",
            name="Speedup",
            marker_color=["#94a3b8" if row["method"] == "baseline" else "#10b981" for row in rows],
        )
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#64748b")
    figure_layout(fig, f"{dataset} Frames={frames}: Speedup vs Baseline", "Method", "Speedup")
    return write_figure(fig, output_dir / f"{dataset}_f{frames}_speedup_vs_factor.html")


def plot_tradeoff(rows, dataset, frames, output_dir):
    fig = go.Figure()
    for key, name, color in (("auc30", "AUC@30", "#2563eb"), ("auc15", "AUC@15", "#16a34a")):
        fig.add_trace(
            go.Scatter(
                x=[row["mean_inference_seconds"] for row in rows],
                y=[row[key] for row in rows],
                mode="markers+text",
                text=[row["label"] for row in rows],
                textposition="top center",
                name=name,
                marker={"size": 14, "color": color, "line": {"width": 1, "color": "#ffffff"}},
                customdata=[[row["median_inference_seconds"], row["total_inference_seconds"]] for row in rows],
                hovertemplate=(
                    "%{text}<br>"
                    "Mean: %{x:.4f}s<br>"
                    f"{name}: " + "%{y:.4f}<br>"
                    "Median: %{customdata[0]:.4f}s<br>"
                    "Total: %{customdata[1]:.2f}s<extra></extra>"
                ),
            )
        )
    figure_layout(fig, f"{dataset} Frames={frames}: Accuracy-Time Tradeoff", "Mean inference seconds", "AUC")
    fig.update_yaxes(range=[0, 1])
    return write_figure(fig, output_dir / f"{dataset}_f{frames}_accuracy_time_tradeoff.html")


def plot_group(rows, dataset, frames, output_dir):
    group_rows = [row for row in rows if row["dataset"] == dataset and row["frames"] == frames]
    group_rows.sort(key=lambda row: row["order"])
    if not group_rows:
        return []

    plot_files = [
        plot_auc(group_rows, dataset, frames, output_dir),
        plot_time(group_rows, dataset, frames, output_dir),
        plot_tradeoff(group_rows, dataset, frames, output_dir),
    ]
    speedup = plot_speedup(group_rows, dataset, frames, output_dir)
    if speedup is not None:
        plot_files.append(speedup)
    return plot_files


def write_index(plot_files, summary_path, output_dir):
    links = "\n".join(f'<li><a href="{path.name}">{path.name}</a></li>' for path in sorted(plot_files))
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VGGT/AVGGT Result Plots</title>
  <style>
    body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 40px; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>VGGT/AVGGT Result Plots</h1>
  <p>Summary CSV: <code>{summary_path.name}</code></p>
  <ul>
    {links}
  </ul>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    specs = build_specs(args.datasets, args.frames, args.factors)
    missing = missing_files(specs, results_dir)
    if missing and not args.allow_missing:
        print_missing(missing)
        return 2

    rows = load_runs(specs, results_dir, args.allow_missing)
    summary_path = write_summary(rows, output_dir)

    plot_files = []
    for dataset in args.datasets:
        for frame_count in args.frames:
            plot_files.extend(plot_group(rows, dataset, frame_count, output_dir))

    index_path = write_index(plot_files, summary_path, output_dir)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plot index: {index_path}")
    for path in sorted(plot_files):
        print(f"Wrote plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
