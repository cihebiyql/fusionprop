# Source data for FusionProp JCIM tables

This directory contains compact source-data artifacts used to reproduce the
FusionProp values in the manuscript and Supporting Information.

## Contents

- `table2_solubility/ensemble_predictions_test.csv`
  - Archived FusionProp-Sol ensemble predictions on the eSOL held-out test set.
  - Columns: `Actual`, `Predicted`, `Error`.
  - Reproduce metrics with:
    ```bash
    python scripts/reproduce_table2_solubility.py
    ```

- `table4_toxicity/ensemble_predictions_detailed.csv`
  - Archived FusionProp-Tox five-checkpoint ensemble predictions on the independent toxicity test set.
  - Columns include per-checkpoint probabilities/predictions and `ensemble_avg_prob`, `ensemble_avg_pred`.
  - Reproduce metrics with:
    ```bash
    python scripts/reproduce_table4_toxicity.py
    ```

- `table4_toxicity/ensemble_metrics_summary.json`
  - Metrics exported from the original bx3 ensemble evaluation run.
  - Original model-inference evaluator: `train_script/toxicity/ensemble_evaluate.py`.

- `table5_thermostability/table5_thermostability_split_metrics.csv`
  - Split/version-level FusionProp-Thermo metrics used to summarize Table 5.
  - Reproduce mean and SD summaries with:
    ```bash
    python scripts/reproduce_table5_thermostability.py
    ```

- `table5_thermostability/raw_training_results/`
  - Raw `training_results.json` files for each Table 5 split/version.
  - Matching selected checkpoints are in `models/thermostability/table5_selected_checkpoints/`.
  - Original submission scripts are archived in `train_script/thermostability/`.

## Notes

- For eSOL secondary classification, labels and predictions are binarized at 0.5; ROC-AUC uses continuous prediction scores.
- For toxicity, the reported ensemble is average-pooling over five ESMC+ESM2 fold checkpoints at a fixed 0.5 threshold.
- For thermostability, the values summarize split/version-level selected-checkpoint evaluations, not fold-level or seed-level repeats.
