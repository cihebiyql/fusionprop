#!/usr/bin/env python3
"""Recompute FusionProp-Tox ensemble metrics from archived predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="source_data/table4_toxicity/ensemble_predictions_detailed.csv",
        help="CSV with true_label, ensemble_avg_prob, and ensemble_avg_pred columns.",
    )
    parser.add_argument("--out", default="source_data/table4_toxicity/recomputed_metrics_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    y_true = df["true_label"].astype(int)
    y_prob = df["ensemble_avg_prob"].astype(float)
    y_pred = df["ensemble_avg_pred"].astype(int)
    cm = confusion_matrix(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    metrics = {
        "n": int(len(df)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "confusion_matrix_tn_fp_fn_tp": [int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
