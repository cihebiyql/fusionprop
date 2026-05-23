#!/usr/bin/env python3
"""Recompute Table 5 mean and standard deviation from split-level metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="source_data/table5_thermostability/table5_thermostability_split_metrics.csv",
        help="CSV with subset-level split metrics.",
    )
    parser.add_argument("--out", default="source_data/table5_thermostability/metrics_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    summary = {}
    for subset, group in df.groupby("subset", sort=False):
        summary[subset] = {"n": int(len(group))}
        for metric in ["spearman", "pearson", "rmse", "mae", "r2"]:
            summary[subset][metric] = {
                "mean": float(group[metric].mean()),
                "sample_sd": float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0,
                "population_sd": float(group[metric].std(ddof=0)) if len(group) > 1 else 0.0,
            }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
