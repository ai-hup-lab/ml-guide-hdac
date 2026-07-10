#!/bin/bash
#
# Extract 5017-dimensional GROVER embeddings (4800 atom+bond readout || 217 RDKit descriptors).
#
# Usage:
#   scripts/gen_grover_fingerprint.sh <split> <variant>
#     <split>   train_set | test_set   (a <split>.csv must exist in $DATA_DIR)
#     <variant> base | finetuned       (which checkpoint to embed with)
#
# Example:
#   scripts/gen_grover_fingerprint.sh train_set finetuned
#
# Always regenerate embeddings with the SAME checkpoint you will use downstream. Mixing
# embeddings from one checkpoint with heads trained on another silently degrades predictions.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SPLIT="${1:?usage: $0 <train_set|test_set> <base|finetuned>}"
VARIANT="${2:?usage: $0 <train_set|test_set> <base|finetuned>}"

case "$VARIANT" in
    base)      CHECKPOINT="$GROVER_PRETRAINED"; OUT_DIR="$DATA_DIR/base_grover_fpts" ;;
    finetuned) CHECKPOINT="$GROVER_FINETUNED";  OUT_DIR="$DATA_DIR/finetuned_grover_fpts" ;;
    *) echo "Error: variant must be 'base' or 'finetuned'" >&2; exit 1 ;;
esac

require_dir  "$GROVER_ROOT" "Clone the GROVER repo, or set GROVER_ROOT."
require_file "$CHECKPOINT"  "Checkpoint missing for variant '$VARIANT'."
require_file "$DATA_DIR/$SPLIT.csv" "Dataset is available from the authors on request."

PREPARED_DIR="$DATA_DIR/grover_prepared"
mkdir -p "$PREPARED_DIR" "$OUT_DIR"

CSV_PREPARED="$PREPARED_DIR/$SPLIT.csv"
NPZ_FEATURES="$PREPARED_DIR/$SPLIT.npz"
OUTPUT="$OUT_DIR/${SPLIT}_fingerprint.npz"

echo "[1/3] Preparing input CSV..."
if [ ! -f "$CSV_PREPARED" ]; then
    python "$REPO_ROOT/grover_addons/prepare_grover_input.py" \
        --input_csv "$DATA_DIR/$SPLIT.csv" --output_csv "$CSV_PREPARED" --smiles_column smiles
else
    echo "  reusing $CSV_PREPARED"
fi

cd "$GROVER_ROOT"

echo "[2/3] Generating RDKit 2D normalized features..."
if [ ! -f "$NPZ_FEATURES" ]; then
    python "$GROVER_ROOT/scripts/save_features.py" \
        --data_path "$CSV_PREPARED" --save_path "$NPZ_FEATURES" \
        --features_generator rdkit_2d_normalized --restart
else
    echo "  reusing $NPZ_FEATURES"
fi

echo "[3/3] Extracting GROVER fingerprints ($VARIANT)..."
python main.py fingerprint \
    --data_path "$CSV_PREPARED" \
    --features_path "$NPZ_FEATURES" \
    --checkpoint_path "$CHECKPOINT" \
    --fingerprint_source both \
    --output "$OUTPUT"

echo "Fingerprints written to: $OUTPUT   (key: 'fps')"
