# Model artifacts

This directory contains model artifacts added for reproducibility of the reported FusionProp results.

## Toxicity ensemble

`models/toxicity/ensemble_20250513_105204/` contains the five ESMC+ESM2 fold checkpoints used for the reported FusionProp-Tox average-pooling ensemble:

- `fold_1_epoch_2.pt`
- `fold_2_epoch_1.pt`
- `fold_3_epoch_2.pt`
- `fold_4_epoch_7.pt`
- `fold_5_epoch_8.pt`

The archived source-data predictions and metrics for this ensemble are in `source_data/table4_toxicity/`.

## Thermostability Table 5 checkpoints

`models/thermostability/table5_selected_checkpoints/` contains the selected
checkpoints corresponding to the split-level FusionProp-Thermo results
summarized in Table 5:

- `HP-S2C2/s2c2_0` through `HP-S2C2/s2c2_9`
- `HP-S2C5/s2c5_0` through `HP-S2C5/s2c5_9`
- `HP-S/S_train23_1` through `HP-S/S_train23_4`

The matching raw `training_results.json` files are archived under
`source_data/table5_thermostability/raw_training_results/`.

## Web-facing models

The optional web interface loads endpoint-specific model files from `web/protein_predictor/models/`:

- Solubility: five fold checkpoints in `web/protein_predictor/models/solubility/`.
- Toxicity: a single checkpoint in `web/protein_predictor/models/toxicity/best_model.pt`.
- Thermostability: a single checkpoint in `web/protein_predictor/models/thermostability/best_model.pth`.

The web interface is a convenience layer. The manuscript tables should be reproduced from `source_data/` and the corresponding training/evaluation scripts.

## Checksums

See `CHECKSUMS.sha256` for SHA-256 checksums of model, data, source-data, and key script artifacts.
