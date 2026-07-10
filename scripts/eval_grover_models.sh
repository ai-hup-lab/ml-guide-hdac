#!/bin/bash
#
# Benchmark the GROVER-based classifiers on the held-out test set.
# Reports accuracy, precision, recall, macro F1 and ROC-AUC for
# {pretrained, finetuned} GROVER x {Random Forest, XGBoost, MLP}.
#
# Usage:
#   scripts/eval_grover_models.sh
#
# Missing combinations are skipped with a warning rather than aborting.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

MODEL_DIR="$RESULTS_DIR/models"
SAVE_DIR="$RESULTS_DIR/benchmark"

require_dir  "$DATA_DIR"  "Dataset is available from the authors on request."
require_file "$DATA_DIR/test_set.csv" "Dataset is available from the authors on request."
require_dir  "$MODEL_DIR" "Train the models first: scripts/train_predictive_model.sh <fp> <model>"
mkdir -p "$SAVE_DIR"

python "$REPO_ROOT/src/eval_grover_models.py" \
    --data_dir  "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --save_dir  "$SAVE_DIR"

echo "Metrics written to $SAVE_DIR/grover_model_performance.{csv,xlsx}"
