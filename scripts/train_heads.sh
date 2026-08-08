#!/bin/bash
#
# Train one (representation, classifier) combination. ILLUSTRATIVE -- see src/train_classifiers.py.
#
# Usage:
#   scripts/train_heads.sh <fp_type> <model_type>
#     <fp_type>    ecfp4 | rdkit | mordred | padel | base_grover | finetuned_grover
#     <model_type> rf | xgb | mlp
#
# Example:
#   scripts/train_heads.sh finetuned_grover rf
#
# To train all six combinations:
#   for fp in base_grover finetuned_grover; do
#     for m in rf xgb mlp; do scripts/train_heads.sh $fp $m; done
#   done
#
# Embeddings must already exist -- see scripts/gen_grover_fingerprint.sh.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

FP_TYPE="${1:?usage: $0 <ecfp4|rdkit|mordred|padel|base_grover|finetuned_grover> <rf|xgb|mlp>}"
MODEL_TYPE="${2:?usage: $0 <ecfp4|rdkit|mordred|padel|base_grover|finetuned_grover> <rf|xgb|mlp>}"

require_file "$DATA_DIR/train_set.csv" "Dataset is available from the authors on request."
require_file "$DATA_DIR/test_set.csv"  "Dataset is available from the authors on request."
require_dir  "$DATA_DIR/${FP_TYPE}_fpts" \
    "Generate embeddings first: scripts/gen_grover_fingerprint.sh <split> <base|finetuned>"
mkdir -p "$RESULTS_DIR"

# Hyperparameters as used in the paper.
RANDOM_SEED=42
N_ESTIMATORS=200          # RandomForest
XGB_LEARNING_RATE=0.001
EPOCHS=300                # MLP
BATCH_SIZE=32
LR=0.001
HIDDEN_DIM=512
DROPOUT=0

# Fraction of the TRAINING set held out for MLP early stopping.
# 0 uses the test set for validation. 0.1 holds out a validation split.
MLP_VAL_SPLIT="${MLP_VAL_SPLIT:-0.1}"

echo "Training $FP_TYPE + $MODEL_TYPE -> $RESULTS_DIR/models/"

python "$REPO_ROOT/src/train_classifiers.py" \
    --train_csv "$DATA_DIR/train_set.csv" \
    --test_csv  "$DATA_DIR/test_set.csv" \
    --fpts_dir  "$DATA_DIR" \
    --save_dir  "$RESULTS_DIR" \
    --fp_type "$FP_TYPE" \
    --model_type "$MODEL_TYPE" \
    --random_seed $RANDOM_SEED \
    --n_estimators $N_ESTIMATORS \
    --xgb_learning_rate $XGB_LEARNING_RATE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --dropout $DROPOUT \
    --mlp_val_split $MLP_VAL_SPLIT \
    --scheduler plateau \
    --scheduler_factor 0.5 \
    --scheduler_patience 10 \
    --scheduler_min_lr 1e-6
