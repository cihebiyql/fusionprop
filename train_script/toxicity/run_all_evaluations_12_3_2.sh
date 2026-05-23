#!/bin/bash
#SBATCH --gpus=1
#SBATCH -x g[0014-0016,0023,0025-0032,0034-0035,0039,0042-0043,0045,0047,0053,0060,0065,0159,0162,0164,0166,0170,0171,0172]
module purge
module load anaconda/2020.11 gcc/9.3
module load cuda/12.1
module load cudnn/8.9.7_cuda12.x
source activate dev240430
# export PATH=/HOME/scz0brz/run/anaconda3/bin:$PATH
export PYTHONUNBUFFERED=1
which python
# ldd /data/apps/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn_cnn_train.so.8
# strings /data/apps/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn_cnn_infer.so |grep libcudnn_cnn_infer.so.8
env
# Script to evaluate all .pt models in a specified experiment folder structure.

# --- Configuration ---
# Base directory where your training output (e.g., train_12_3_2) is located.
# The script assumes it's run from the EE_toxicity directory or that paths below are adjusted.
BASE_EXPERIMENT_DIR_PARENT="/HOME/scz0brz/run/EE_toxicity/train_12_3_2"

# Test data files (assuming they are in the current working directory or provide full paths)
TEST_POS_CSV="filtered_toxin_0.7.csv"
TEST_NEG_CSV="filtered_notoxin_0.7.csv"

# Column name in the above CSVs that contains the sequences (after header)
# This script assumes the input CSVs have a header, and the first column after header is the sequence.
# The combined file will have headers "sequence,label"

# Output directory for all evaluations
MAIN_OUTPUT_DIR="./evaluation_output_all_models_12_3_2"

# Path to the aggregated results JSON file
AGGREGATE_JSON_PATH="${MAIN_OUTPUT_DIR}/all_metrics_summary_12_3_2.json"

# Path to the evaluation script
EVAL_SCRIPT_PATH="./evaluate_model.py" # Assumes evaluate_model.py is in the same dir as this script

# --- End Configuration ---

# Ensure the main output directory exists
mkdir -p "$MAIN_OUTPUT_DIR"

# Initialize the aggregate results file as an empty JSON object
echo "Initializing aggregate results file: $AGGREGATE_JSON_PATH"
echo "{}" > "$AGGREGATE_JSON_PATH"

# Temporary combined test file
COMBINED_TEST_CSV="${MAIN_OUTPUT_DIR}/temp_combined_test_data.csv"

# Check if test data files exist
if [ ! -f "$TEST_POS_CSV" ]; then
    echo "ERROR: Positive test CSV file not found: $TEST_POS_CSV"
    exit 1
fi
if [ ! -f "$TEST_NEG_CSV" ]; then
    echo "ERROR: Negative test CSV file not found: $TEST_NEG_CSV"
    exit 1
fi

# Create the combined test CSV with "sequence,label" header
echo "Creating combined test file: $COMBINED_TEST_CSV"
echo "sequence,label" > "$COMBINED_TEST_CSV"

# Append data from positive CSV (skip header, take the 3rd column as sequence, add ,1 for label)
# Remove potential quotes from the sequence string itself and trailing CR
tail -n +2 "$TEST_POS_CSV" | awk -F',' -v q='"' '{gsub(/^"|"$/, "", $3); gsub(/""/, q, $3); sub(/\r$/, "", $3); print $3",1"}' >> "$COMBINED_TEST_CSV"

# Append data from negative CSV (skip header, take the 3rd column as sequence, add ,0 for label)
# Remove potential quotes from the sequence string itself and trailing CR
tail -n +2 "$TEST_NEG_CSV" | awk -F',' -v q='"' '{gsub(/^"|"$/, "", $3); gsub(/""/, q, $3); sub(/\r$/, "", $3); print $3",0"}' >> "$COMBINED_TEST_CSV"

echo "Combined test file created."

# Find all experiment subdirectories (like ESMC+ESM2_20250501_222948)
find "$BASE_EXPERIMENT_DIR_PARENT" -mindepth 1 -maxdepth 1 -type d | while read experiment_path; do
    experiment_name=$(basename "$experiment_path")
    echo ""
    echo "Processing experiment: $experiment_name"

    config_json_path="${experiment_path}/config.json"

    if [ ! -f "$config_json_path" ]; then
        echo "WARNING: config.json not found in $experiment_path. Skipping this experiment."
        continue
    fi

    # Find all .pt model files in this experiment directory
    find "$experiment_path" -name "*.pt" | while read model_file_path; do
        model_file_name=$(basename "$model_file_path")
        # Create a clean name for the output subdirectory, removing .pt
        model_eval_output_name=${model_file_name%.pt}
        # Create a unique identifier for the aggregate JSON key
        run_identifier="${experiment_name}/${model_eval_output_name}"

        echo "  Evaluating model: $model_file_name with identifier: $run_identifier"

        # Define a unique output directory for this specific model's evaluation
        eval_output_subdir="${MAIN_OUTPUT_DIR}/${experiment_name}/${model_eval_output_name}"
        mkdir -p "$eval_output_subdir"

        # Log the command being run for easier debugging
        echo "    Running command:"
        echo "    python $EVAL_SCRIPT_PATH \\"
        echo "        --model_path \"$model_file_path\" \\"
        echo "        --config_path \"$config_json_path\" \\"
        echo "        --test_csv \"$COMBINED_TEST_CSV\" \\"
        echo "        --output_dir \"$eval_output_subdir\" \\"
        echo "        --sequence_column sequence \\"
        echo "        --target_column label \\"
        echo "        --aggregate_results_file \"$AGGREGATE_JSON_PATH\" \\"
        echo "        --run_identifier \"$run_identifier\" \\"
        echo "        --batch_size 16 \\"
        echo "        --num_workers 2"
        # Add other parameters for evaluate_model.py if needed, e.g., --device cuda:0

        # Execute the command directly, ensuring variables are quoted
        python "$EVAL_SCRIPT_PATH" \
            --model_path "$model_file_path" \
            --config_path "$config_json_path" \
            --test_csv "$COMBINED_TEST_CSV" \
            --output_dir "$eval_output_subdir" \
            --sequence_column "sequence" \
            --target_column "label" \
            --aggregate_results_file "$AGGREGATE_JSON_PATH" \
            --run_identifier "$run_identifier" \
            --batch_size "16" \
            --num_workers "2"
            # Add other parameters for evaluate_model.py if needed, e.g., --device cuda:0

        echo "    Finished evaluation for $model_file_name. Results in $eval_output_subdir"
        echo "    --------------------------------------------------"
    done
done

# Clean up the temporary combined test file
if [ -f "$COMBINED_TEST_CSV" ]; then
    echo "Cleaning up temporary test file: $COMBINED_TEST_CSV"
    rm "$COMBINED_TEST_CSV"
fi

echo ""
echo "Aggregate results saved to: $AGGREGATE_JSON_PATH"
echo "All evaluations complete. Check subdirectories in $MAIN_OUTPUT_DIR for logs and predictions."