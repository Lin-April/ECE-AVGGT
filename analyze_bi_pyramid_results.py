"""Summarize VGGT/AVGGT/BI/Pyramid result JSON files.

The script reads baseline/AVGGT results from ``results/`` and BI/Pyramid
results from ``results/new_bi_pyramid``. It writes compact CSV tables for the
final report and prints a missing-file checklist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_NEW_DIR = DEFAULT_RESULTS_DIR / "new_bi_pyramid"
DEFAULT_OUTPUT_DIR = DEFAULT_NEW_DIR / "summary"

DATASETS = ("7scenes", "re10k")
MAIN_METHODS = ("baseline", "avggt4", "bi", "pyramid_linear", "full_linear")
A1_METHODS = (
    "pyramid_uniform",
    "pyramid_linear",
    "pyramid_exp",
    "full_uniform",
    "full_linear",
    "full_exp",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize BI/Pyramid experiment results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def accuracy_name(dataset, method):
    if method == "baseline":
        return f"{dataset}_manifest_eval.json"
    return f"{dataset}_manifest_eval_{method}.json"


def profile_name(dataset, method):
    if method == "baseline":
        return f"{dataset}_profile.json"
    return f"{dataset}_profile_{method}.json"


def candidate_dirs(method, results_dir, new_dir):
    if method in {"baseline", "avggt4"}:
        return [results_dir, new_dir, new_dir / "exp", new_dir / "uniform"]
    if method.endswith("_uniform"):
        return [new_dir / "uniform", new_dir]
    if method.endswith("_exp"):
        return [new_dir / "exp", new_dir]
    return [new_dir, new_dir / "uniform", new_dir / "exp"]


def find_file(filename, method, results_dir, new_dir):
    for directory in candidate_dirs(method, results_dir, new_dir):
        path = directory / filename
        if path.exists():
            return path
    return None


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_row(dataset, method, results_dir, new_dir):
    accuracy_path = find_file(accuracy_name(dataset, method), method, results_dir, new_dir)
    profile_path = find_file(profile_name(dataset, method), method, results_dir, new_dir)
    if accuracy_path is None:
        return None, f"missing accuracy: {dataset} {method}"
    if profile_path is None:
        return None, f"missing profile: {dataset} {method}"

    accuracy = read_json(accuracy_path)
    mean = accuracy.get("__mean__", {})
    profile = read_json(profile_path)
    row = {
        "dataset": dataset,
        "method": method,
        "auc30": mean.get("auc30"),
        "auc15": mean.get("auc15"),
        "auc5": mean.get("auc5"),
        "auc3": mean.get("auc3"),
        "mean_inference_seconds": profile.get("mean_inference_seconds"),
        "median_inference_seconds": profile.get("median_inference_seconds"),
        "total_inference_seconds": profile.get("total_inference_seconds"),
        "num_samples": profile.get("num_samples"),
        "accuracy_file": str(accuracy_path),
        "profile_file": str(profile_path),
    }
    return row, None


def load_table(methods, results_dir, new_dir):
    rows = []
    missing = []
    for dataset in DATASETS:
        for method in methods:
            row, error = load_row(dataset, method, results_dir, new_dir)
            if error:
                missing.append(error)
            else:
                rows.append(row)
    return rows, missing


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def print_table(title, rows):
    print(f"\n{title}")
    print("dataset   method             auc30   auc15   auc5    auc3    mean_s  median_s")
    for row in rows:
        print(
            f"{row['dataset']:<8} "
            f"{row['method']:<18} "
            f"{format_float(row['auc30']):>6} "
            f"{format_float(row['auc15']):>7} "
            f"{format_float(row['auc5']):>7} "
            f"{format_float(row['auc3']):>7} "
            f"{format_float(row['mean_inference_seconds']):>7} "
            f"{format_float(row['median_inference_seconds']):>8}"
        )


def add_speedups(rows):
    baseline_by_dataset = {
        row["dataset"]: row
        for row in rows
        if row["method"] == "baseline" and row.get("mean_inference_seconds")
    }
    for row in rows:
        baseline = baseline_by_dataset.get(row["dataset"])
        if baseline and row.get("mean_inference_seconds"):
            row["speedup_vs_baseline_mean"] = float(baseline["mean_inference_seconds"]) / float(
                row["mean_inference_seconds"]
            )
        else:
            row["speedup_vs_baseline_mean"] = None
    return rows


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    new_dir = args.new_dir.resolve()
    output_dir = args.output_dir.resolve()

    main_rows, main_missing = load_table(MAIN_METHODS, results_dir, new_dir)
    a1_rows, a1_missing = load_table(A1_METHODS, results_dir, new_dir)
    main_rows = add_speedups(main_rows)
    a1_rows = add_speedups(a1_rows)

    write_csv(output_dir / "main_table.csv", main_rows)
    write_csv(output_dir / "a1_budget_shapes.csv", a1_rows)

    print_table("Main Table", main_rows)
    print_table("A1 Budget Shape Ablation", a1_rows)

    missing = main_missing + a1_missing
    if missing:
        print("\nMissing")
        for item in missing:
            print(f"- {item}")

    print(f"\nWrote {output_dir / 'main_table.csv'}")
    print(f"Wrote {output_dir / 'a1_budget_shapes.csv'}")


if __name__ == "__main__":
    main()
