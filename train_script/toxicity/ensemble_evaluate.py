#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from scipy import stats # For majority voting mode
import fcntl # For potential file locking if needed, though less critical here
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, confusion_matrix,
    matthews_corrcoef as mcc_score
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import time

# Assuming train_12_2.py is in the same directory or accessible via PYTHONPATH
# Make sure train_12_2.py exists and contains these imports
try:
    from train_12_2 import (
        ExperimentConfig, Logger, FeatureManager, ProteinFeatureDataset,
        collate_protein_features, SingleModelClassifier, FusionModelClassifier,
        WeightedFusionClassifier, ModelConfig, ESM2Config, ESMCConfig, SPLMConfig
    )
except ImportError as e:
    print(f"Error importing from train_12_2: {e}")
    print("Please ensure train_12_2.py is in the Python path.")
    exit(1)

def parse_ensemble_args():
    """Parses command-line arguments for ensemble evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate an ensemble of 5 pre-trained protein language models.")
    parser.add_argument("--model_paths", type=str, required=True, nargs=5,
                        help="Paths to the 5 saved model (.pt) files for the ensemble.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the *single* experiment config.json file shared by the models.")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to the test CSV data file (combined, with 'sequence' and 'label' columns).")
    parser.add_argument("--sequence_column", type=str, default="sequence",
                        help="Name of the column containing protein sequences in the test CSV.")
    parser.add_argument("--target_column", type=str, default="label",
                        help="Name of the column containing true labels in the test CSV.")
    parser.add_argument("--output_dir", type=str, default="./ensemble_evaluation_results",
                        help="Directory to save ensemble evaluation results.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading.")
    parser.add_argument("--feature_cache_size", type=int, default=1000,
                        help="Feature cache size for the dataset.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for evaluation (e.g., 'cuda:0', 'cpu'). Autodetects if None.")
    return parser.parse_args()

def _calculate_metrics(true_labels, pred_labels, probabilities, logger, prefix=""):
    """Calculates evaluation metrics, adding a prefix to metric names."""
    metrics = {}
    if true_labels is None or len(true_labels) == 0:
        logger.warning(f"{prefix}: No true labels provided or empty true labels. Skipping metrics calculation.")
        return {f"{prefix}status": "no_labels", f"{prefix}mcc": np.nan, f"{prefix}auc": np.nan}

    if len(true_labels) != len(pred_labels):
        logger.error(f"{prefix}: Mismatch between true labels ({len(true_labels)}) and predicted labels ({len(pred_labels)}). Skipping metrics.")
        return {f"{prefix}status": "length_mismatch", f"{prefix}mcc": np.nan, f"{prefix}auc": np.nan}

    try:
        metrics[f"{prefix}accuracy"] = accuracy_score(true_labels, pred_labels)
        metrics[f"{prefix}precision"] = precision_score(true_labels, pred_labels, zero_division=0)
        metrics[f"{prefix}recall"] = recall_score(true_labels, pred_labels, zero_division=0)
        metrics[f"{prefix}f1"] = f1_score(true_labels, pred_labels, zero_division=0)
        metrics[f"{prefix}mcc"] = mcc_score(true_labels, pred_labels)

        # Check for AUC/PR-AUC calculation validity
        can_calc_auc = False
        if probabilities is not None and len(probabilities) == len(true_labels):
            if len(np.unique(true_labels)) > 1:
                 can_calc_auc = True
            else:
                logger.warning(f"{prefix}: Only one class present in true labels. AUC is not defined.")
        else:
            logger.warning(f"{prefix}: Probabilities missing, have incorrect length ({len(probabilities) if probabilities is not None else 'None'} vs {len(true_labels)} labels), or only one class. Skipping AUC/PR-AUC.")

        if can_calc_auc:
            metrics[f"{prefix}auc"] = roc_auc_score(true_labels, probabilities)
            precision_vals, recall_vals, _ = precision_recall_curve(true_labels, probabilities)
            metrics[f"{prefix}pr_auc"] = auc(recall_vals, precision_vals)
        else:
            metrics[f"{prefix}auc"] = np.nan
            metrics[f"{prefix}pr_auc"] = np.nan

        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics[f"{prefix}tn"] = int(tn)
            metrics[f"{prefix}fp"] = int(fp)
            metrics[f"{prefix}fn"] = int(fn)
            metrics[f"{prefix}tp"] = int(tp)
        else:
            logger.warning(f"{prefix}: Could not compute full confusion matrix. Shape: {cm.shape}")
            metrics[f"{prefix}tn"] = metrics[f"{prefix}fp"] = metrics[f"{prefix}fn"] = metrics[f"{prefix}tp"] = np.nan

    except Exception as e:
        logger.error(f"{prefix}: Error calculating metrics: {e}")
        metrics[f"{prefix}error"] = str(e)
        # Ensure default numeric keys exist if error occurs mid-calculation
        for key in ["mcc", "auc", "accuracy", "precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp"]:
            full_key = f"{prefix}{key}"
            if full_key not in metrics: metrics[full_key] = np.nan
    return metrics

def main_ensemble_evaluate():
    """Main function for ensemble model evaluation."""
    args = parse_ensemble_args()
    num_models = len(args.model_paths)
    if num_models != 5:
        # This check is redundant due to nargs=5, but good practice
        print(f"Error: Expected 5 model paths, but received {num_models}.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "ensemble_evaluation.log")
    logger = Logger(log_file=log_file, console=True)
    logger.info("Starting ENSEMBLE evaluation.")
    logger.info(f"Models: {args.model_paths}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Arguments: {vars(args)}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading shared experiment configuration from: {args.config_path}")
    try:
        exp_config = ExperimentConfig.load_config(args.config_path)
        exp_config.training_config.feature_extraction_device = device
        exp_config.training_config.training_device = device
        logger.info("Experiment configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load experiment configuration: {e}")
        return

    logger.info("Initializing FeatureManager...")
    feature_manager = None
    try:
        feature_manager = FeatureManager(exp_config, logger)
        feature_manager.preload_models()
        logger.info("FeatureManager initialized and models preloaded.")
    except Exception as e:
        logger.error(f"Failed to initialize FeatureManager: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info(f"Loading test data from: {args.test_csv}")
    try:
        test_df = pd.read_csv(args.test_csv)
        if not {args.sequence_column, args.target_column}.issubset(test_df.columns):
            logger.error(f"Test CSV must contain '{args.sequence_column}' and '{args.target_column}' columns. Found: {test_df.columns.tolist()}")
            if feature_manager: feature_manager.cleanup()
            return
        test_df[args.target_column] = pd.to_numeric(test_df[args.target_column], errors='coerce')
        test_df[args.sequence_column] = test_df[args.sequence_column].astype(str)
        logger.info(f"Test data loaded: {len(test_df)} records.")
        # Filter out rows where the label could not be parsed
        initial_len = len(test_df)
        test_df = test_df.dropna(subset=[args.target_column])
        if len(test_df) < initial_len:
            logger.warning(f"Removed {initial_len - len(test_df)} rows with invalid target labels.")

    except Exception as e:
        logger.error(f"Failed to load or process test data: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info("Creating test dataset and dataloader...")
    try:
        test_dataset = ProteinFeatureDataset(
            df=test_df,
            feature_manager=feature_manager,
            config=exp_config,
            target_col=args.target_column,
            sequence_col=args.sequence_column,
            cache_size=args.feature_cache_size,
            logger=logger
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_protein_features,
            pin_memory=True if device.type == 'cuda' else False
        )
        logger.info("Test dataset and dataloader created.")
    except Exception as e:
        logger.error(f"Failed to create dataset/dataloader: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info("Loading trained models...")
    models = []
    try:
        train_mode = exp_config.training_config.train_mode
        model_class = None
        if train_mode == "fusion":
            fusion_type = getattr(exp_config.training_config, 'fusion_type', 'default')
            model_class = WeightedFusionClassifier if fusion_type == "weighted" else FusionModelClassifier
        elif train_mode == "single":
            model_class = SingleModelClassifier
        else:
            raise ValueError(f"Unsupported train_mode '{train_mode}' in config.")

        for i, model_path in enumerate(args.model_paths):
            logger.info(f"  Loading model {i+1}/{num_models} from: {model_path}")
            model = model_class.load_model(model_path, device=device)
            model.eval()
            models.append(model)
        logger.info(f"All {num_models} models loaded successfully ({model_class.__name__}) and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Failed to load one or more models: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info("Generating predictions from individual models...")
    all_individual_probs = [[] for _ in range(num_models)]
    all_true_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Ensemble"):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "labels" in batch_data:
                 all_true_labels.extend(batch_data["labels"].cpu().numpy())

            for i, model in enumerate(models):
                outputs = model(batch_data)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_individual_probs[i].extend(probs)
    logger.info("Finished generating individual predictions.")

    # Combine results into numpy arrays
    # Shape: (num_samples, num_models)
    individual_probs_np = np.stack(all_individual_probs, axis=-1)
    true_labels_np = np.array(all_true_labels)

    # Ensure we have valid labels before proceeding
    valid_indices = ~np.isnan(true_labels_np)
    if not np.any(valid_indices):
        logger.error("No valid true labels found after processing test data. Cannot calculate metrics.")
        if feature_manager: feature_manager.cleanup()
        return

    true_labels_np = true_labels_np[valid_indices]
    individual_probs_np = individual_probs_np[valid_indices, :]
    individual_preds_np = (individual_probs_np >= 0.5).astype(int)

    logger.info(f"Calculating metrics for {len(true_labels_np)} samples with valid labels.")

    # --- Ensemble Strategy 1: Average Pooling ---
    logger.info("Applying Average Pooling ensemble strategy...")
    avg_probs_np = np.mean(individual_probs_np, axis=1)
    avg_preds_np = (avg_probs_np >= 0.5).astype(int)

    # --- Ensemble Strategy 2: Majority Voting ---
    logger.info("Applying Majority Voting ensemble strategy...")
    # Get mode (most frequent prediction) across models for each sample
    # Note: stats.mode returns mode and count. We only need the mode ([0]).
    # If using scipy < 1.9.0, keepdims=False is default and output shape needs handling.
    # For scipy >= 1.9.0, keepdims=True is default, output shape is (n_samples, 1).
    # Using keepdims=False and squeezing simplifies compatibility.
    majority_preds_np, _ = stats.mode(individual_preds_np, axis=1, keepdims=False)
    # No direct probability for majority vote, use average probability for AUC calculation if needed,
    # but it's conceptually distinct. We'll pass None for probabilities for voting metrics.
    # Or, calculate AUC based on the *proportion* of models voting positive? Let's use avg_probs for AUC.


    # --- Calculate Metrics ---
    all_metrics = {}
    logger.info("Calculating metrics for individual models...")
    for i in range(num_models):
        model_metrics = _calculate_metrics(true_labels_np, individual_preds_np[:, i], individual_probs_np[:, i], logger, prefix=f"model_{i+1}_")
        all_metrics.update(model_metrics)

    logger.info("Calculating metrics for Average Pooling ensemble...")
    avg_pool_metrics = _calculate_metrics(true_labels_np, avg_preds_np, avg_probs_np, logger, prefix="ensemble_avg_pool_")
    all_metrics.update(avg_pool_metrics)

    logger.info("Calculating metrics for Majority Voting ensemble...")
    # Using avg_probs for AUC calculation for majority vote, as there isn't a direct probability output.
    majority_vote_metrics = _calculate_metrics(true_labels_np, majority_preds_np, avg_probs_np, logger, prefix="ensemble_majority_vote_")
    all_metrics.update(majority_vote_metrics)

    # --- Save Results ---
    logger.info("Saving detailed prediction results...")
    results_df = test_df.iloc[valid_indices][[args.sequence_column, args.target_column]].copy()
    results_df['true_label'] = true_labels_np # Use the validated true labels

    for i in range(num_models):
        results_df[f'model_{i+1}_prob'] = individual_probs_np[:, i]
        results_df[f'model_{i+1}_pred'] = individual_preds_np[:, i]

    results_df['ensemble_avg_prob'] = avg_probs_np
    results_df['ensemble_avg_pred'] = avg_preds_np
    results_df['ensemble_majority_pred'] = majority_preds_np

    predictions_path = os.path.join(args.output_dir, "ensemble_predictions_detailed.csv")
    try:
        results_df.to_csv(predictions_path, index=False)
        logger.info(f"Detailed predictions saved to: {predictions_path}")
    except Exception as e:
        logger.error(f"Failed to save detailed predictions: {e}")

    logger.info("Saving summary metrics...")
    metrics_path = os.path.join(args.output_dir, "ensemble_metrics_summary.json")
    try:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for k, v in all_metrics.items():
            if isinstance(v, (np.generic, np.ndarray)):
                 serializable_metrics[k] = v.item() if v.size == 1 else v.tolist()
            elif isinstance(v, (float, int)):
                 serializable_metrics[k] = v
            else: # Handle non-numeric types like status strings or NaN
                 serializable_metrics[k] = float('nan') if isinstance(v, float) and np.isnan(v) else str(v)


        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Summary metrics saved to: {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save summary metrics: {e}")

    logger.info("--- Metrics Summary ---")
    # Pretty print the metrics from the dictionary
    current_prefix = ""
    for k, v in sorted(serializable_metrics.items()):
         # Detect prefix change to group output
         prefix_parts = k.split('_')
         new_prefix = "_".join(prefix_parts[:-1]) # Everything before the last underscore
         if new_prefix != current_prefix:
              print("-" * 10)
              current_prefix = new_prefix
              print(f"Results for: {current_prefix}")

         metric_name = prefix_parts[-1]
         print(f"  {metric_name}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {metric_name}: {v}")
    logger.info("--- End Summary ---")


    if feature_manager:
        logger.info("Cleaning up FeatureManager resources...")
        feature_manager.cleanup()
    logger.info(f"Ensemble evaluation completed. Results in: {args.output_dir}")

if __name__ == "__main__":
    try:
        # Setting spawn method might be necessary depending on the system and libraries used
        # Check if it's already set or if setting it causes issues
        if torch.multiprocessing.get_start_method(allow_none=True) is None:
             torch.multiprocessing.set_start_method('spawn')
    except RuntimeError as e:
         print(f"Could not set multiprocessing start method 'spawn': {e}")
         # Continue anyway, might work or might fail later if multiprocessing is used heavily internally
         pass
    main_ensemble_evaluate()