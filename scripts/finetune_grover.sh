#!/bin/bash
#
# Finetune a pretrained GROVER checkpoint on the HDAC activity classification task.
#
# Usage:
#   scripts/finetune_grover.sh                       # uses VAL_CSV if present (recommended)
#   VAL_CSV=data/val_set.csv scripts/finetune_grover.sh
#
# Validation set setup:
#   You can set VAL_CSV to held-out validation molecules.
#   Otherwise, it defaults to using the test set.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

require_dir  "$GROVER_ROOT"       "Clone https://github.com/tencent-ailab/grover into it, or set GROVER_ROOT."
require_file "$GROVER_PRETRAINED" "Download grover_large.pt from the GROVER repo, or set GROVER_PRETRAINED."
require_file "$DATA_DIR/train_set.csv" "Dataset is available from the authors on request."
require_file "$DATA_DIR/test_set.csv"  "Dataset is available from the authors on request."

PREPARED_DIR="$DATA_DIR/grover_prepared"
SAVE_DIR="${SAVE_DIR:-$DATA_DIR/finetuned_model_grover}"
mkdir -p "$PREPARED_DIR" "$SAVE_DIR"

TRAIN_CSV="$PREPARED_DIR/train_set.csv"; TRAIN_NPZ="$PREPARED_DIR/train_set.npz"
TEST_CSV="$PREPARED_DIR/test_set.csv";   TEST_NPZ="$PREPARED_DIR/test_set.npz"

# --- Resolve the validation set ---
VAL_CSV="${VAL_CSV:-}"
if [ -n "$VAL_CSV" ]; then
    require_file "$VAL_CSV" "VAL_CSV was set but does not exist."
    VAL_PREPARED="$PREPARED_DIR/val_set.csv"; VAL_NPZ="$PREPARED_DIR/val_set.npz"
else
    VAL_PREPARED="$TEST_CSV"; VAL_NPZ="$TEST_NPZ"
fi

# --- Step 1: numeric labels for GROVER ---
echo "[1/3] Preparing CSVs..."
python "$REPO_ROOT/grover_addons/prepare_grover_training.py" --input_csv "$DATA_DIR/train_set.csv" --output_csv "$TRAIN_CSV"
python "$REPO_ROOT/grover_addons/prepare_grover_training.py" --input_csv "$DATA_DIR/test_set.csv"  --output_csv "$TEST_CSV"
if [ -n "$VAL_CSV" ]; then
    python "$REPO_ROOT/grover_addons/prepare_grover_training.py" --input_csv "$VAL_CSV" --output_csv "$VAL_PREPARED"
fi

# --- Step 2: RDKit 2D normalized features (217-dim) ---
echo "[2/3] Generating RDKit features..."
gen_features() {  # $1 = prepared csv, $2 = output npz
    [ -f "$2" ] && { echo "  reusing $2"; return; }
    python "$GROVER_ROOT/scripts/save_features.py" \
        --data_path "$1" --save_path "$2" \
        --features_generator rdkit_2d_normalized --restart
}
cd "$GROVER_ROOT"
gen_features "$TRAIN_CSV" "$TRAIN_NPZ"
gen_features "$TEST_CSV"  "$TEST_NPZ"
[ -n "$VAL_CSV" ] && gen_features "$VAL_PREPARED" "$VAL_NPZ"

# --- Hyperparameters (as used in the paper) ---
INIT_LR=0.0002; MAX_LR=0.00001; FINAL_LR=0.00001
DROPOUT=0.01;   BOND_DROP_RATE=0.2
BATCH_SIZE=32;  EPOCHS=200
ATTEN_HIDDEN=128; ATTEN_OUT=8; DIST_COFF=0.02
FFN_NUM_LAYERS=3; FFN_HIDDEN_SIZE=200; ENSEMBLE_SIZE=1

# --- Step 3: finetune ---
echo "[3/3] Finetuning GROVER..."
python -W ignore::DeprecationWarning main.py finetune \
    --data_path "$TRAIN_CSV" \
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
