#!/usr/bin/env python3
"""Recompute FusionProp-Sol eSOL secondary classification metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="source_data/table2_solubility/ensemble_predictions_test.csv",
        help="CSV with Actual and Predicted columns.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="source_data/table2_solubility/metrics_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    y_true = (df["Actual"] >= args.threshold).astype(int)
    y_pred = (df["Predicted"] >= args.threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "n": int(len(df)),
        "threshold": args.threshold,
        "actual_positive_count": int(y_true.sum()),
        "predicted_positive_count": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_continuous_score": float(roc_auc_score(y_true, df["Predicted"])),
        "confusion_matrix_tn_fp_fn_tp": [int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
