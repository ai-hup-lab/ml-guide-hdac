#!/bin/bash
#
# Finetune a pretrained GROVER checkpoint on the HDAC activity classification task.
#
# Usage:
#   TRAIN_CSV=$DATA_DIR/fit_set.csv VAL_CSV=$DATA_DIR/val_set.csv scripts/finetune_grover.sh
#
# VAL_CSV is required and must be disjoint from the test set. GROVER uses it for
# early stopping and best-checkpoint selection, so pointing it at the test set makes
# the reported test performance optimistically biased.
#
# The dataset distributed by the authors already contains the split used here:
#   fit_set.csv    1629 molecules  training proper
#   val_set.csv     182 molecules  checkpoint selection
#   test_set.csv    201 molecules  held out, never seen during training
# fit_set.csv and val_set.csv partition train_set.csv; both are disjoint from the
# test set. Set TRAIN_CSV to the fit split, not to train_set.csv, or the validation
# molecules will also be trained on.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# Checked first: it costs nothing and it is the mistake most likely to be made.
if [ -z "${VAL_CSV:-}" ]; then
    echo "Error: VAL_CSV is not set." >&2
    echo "  A validation set disjoint from the test set is required: GROVER selects its" >&2
    echo "  best checkpoint on it, so reusing the test set biases the reported result." >&2
    echo "  Create one with src/make_validation_split.py -- see the header of this script." >&2
    exit 1
fi

require_dir  "$GROVER_ROOT"       "Clone https://github.com/tencent-ailab/grover into it, or set GROVER_ROOT."
require_file "$GROVER_PRETRAINED" "Download grover_large.pt from the GROVER repo, or set GROVER_PRETRAINED."
TRAIN_INPUT="${TRAIN_CSV:-$DATA_DIR/train_set.csv}"
require_file "$TRAIN_INPUT"            "Dataset is available from the authors on request."
require_file "$DATA_DIR/test_set.csv"  "Dataset is available from the authors on request."
require_file "$VAL_CSV"                "VAL_CSV was set but does not exist."

PREPARED_DIR="$DATA_DIR/grover_prepared"
SAVE_DIR="${SAVE_DIR:-$DATA_DIR/finetuned_model_grover}"
mkdir -p "$PREPARED_DIR" "$SAVE_DIR"

TRAIN_PREPARED="$PREPARED_DIR/train_set.csv"; TRAIN_NPZ="$PREPARED_DIR/train_set.npz"
TEST_CSV="$PREPARED_DIR/test_set.csv";        TEST_NPZ="$PREPARED_DIR/test_set.npz"
VAL_PREPARED="$PREPARED_DIR/val_set.csv";     VAL_NPZ="$PREPARED_DIR/val_set.npz"

# --- Step 1: numeric labels for GROVER ---
echo "[1/3] Preparing CSVs..."
python "$REPO_ROOT/grover_addons/prepare_grover_training.py" --input_csv "$TRAIN_INPUT"          --output_csv "$TRAIN_PREPARED"
python "$REPO_ROOT/grover_addons/prepare_grover_training.py" --input_csv "$DATA_DIR/test_set.csv" --output_csv "$TEST_CSV"
python "$REPO_ROOT/grover_addons/prepare_grover_training.py" --input_csv "$VAL_CSV"              --output_csv "$VAL_PREPARED"

# --- Step 2: RDKit 2D normalized features (217-dim) ---
echo "[2/3] Generating RDKit features..."
gen_features() {  # $1 = prepared csv, $2 = output npz
    [ -f "$2" ] && { echo "  reusing $2"; return; }
    python "$GROVER_ROOT/scripts/save_features.py" \
        --data_path "$1" --save_path "$2" \
        --features_generator rdkit_2d_normalized --restart
}
cd "$GROVER_ROOT"
gen_features "$TRAIN_PREPARED" "$TRAIN_NPZ"
gen_features "$TEST_CSV"  "$TEST_NPZ"
gen_features "$VAL_PREPARED" "$VAL_NPZ"

# --- Hyperparameters (as used in the paper) ---
INIT_LR=0.0002; MAX_LR=0.00001; FINAL_LR=0.00001
DROPOUT=0.01;   BOND_DROP_RATE=0.2
BATCH_SIZE=32;  EPOCHS=200
ATTEN_HIDDEN=128; ATTEN_OUT=8; DIST_COFF=0.02
FFN_NUM_LAYERS=3; FFN_HIDDEN_SIZE=200; ENSEMBLE_SIZE=1

# --- Step 3: finetune ---
echo "[3/3] Finetuning GROVER..."
python -W ignore::DeprecationWarning main.py finetune \
    --data_path "$TRAIN_PREPARED" \
    --features_path "$TRAIN_NPZ" \
    --separate_val_path "$VAL_PREPARED" \
    --separate_val_features_path "$VAL_NPZ" \
    --separate_test_path "$TEST_CSV" \
    --separate_test_features_path "$TEST_NPZ" \
    --save_dir "$SAVE_DIR" \
    --checkpoint_path "$GROVER_PRETRAINED" \
    --dataset_type classification \
    --split_type scaffold_balanced \
    --init_lr $INIT_LR --max_lr $MAX_LR --final_lr $FINAL_LR \
    --dropout $DROPOUT --no_features_scaling \
    --attn_hidden $ATTEN_HIDDEN --attn_out $ATTEN_OUT --dist_coff $DIST_COFF \
    --ffn_num_layers $FFN_NUM_LAYERS --ffn_hidden_size $FFN_HIDDEN_SIZE \
    --ensemble_size $ENSEMBLE_SIZE --bond_drop_rate $BOND_DROP_RATE \
    --epochs $EPOCHS --batch_size $BATCH_SIZE

echo "Done. Checkpoint: $SAVE_DIR/fold_0/model_0/model.pt"
