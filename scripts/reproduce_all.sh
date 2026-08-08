#!/bin/bash
#
# Reproduce every published result and check it against expected_results/.
#
# Usage:
#   scripts/reproduce_all.sh
#
# Prerequisite: unzip the dataset archive supplied by the authors and copy its
# data/ and results/ directories into the repository root. Nothing is retrained;
# the saved models are loaded and scored.
#
# Exits non-zero if any output disagrees with the published numbers.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

require_dir  "$DATA_DIR"              "Copy data/ from the dataset archive into the repository root."
require_dir  "$RESULTS_DIR"           "Copy results/ from the dataset archive into the repository root."
require_file "$DATA_DIR/test_set.csv" "Copy data/ from the dataset archive into the repository root."
require_dir  "$MODEL_DIR"             "The archive supplies results/final-classification-models."
require_dir  "$CV_MODEL_DIR"          "The archive supplies results/final-cv-classifier-result."
require_dir  "$EXPECTED_DIR"          "expected_results/ is part of this repository."

mkdir -p "$REPRODUCTION_DIR"

echo "==> 1/3  Held-out test set (18 models)"
python "$REPO_ROOT/src/reproduce_heldout.py" \
    --data_dir  "$DATA_DIR" \
    --model_dir "$MODEL_DIR" \
    --save_dir  "$REPRODUCTION_DIR"

echo
echo "==> 2/3  Five-fold cross-validation (90 fold heads)"
python "$REPO_ROOT/src/reproduce_cv.py" \
    --data_dir       "$DATA_DIR" \
    --splits_dir     "$SPLITS_DIR" \
    --model_dir      "$CV_MODEL_DIR" \
    --grover_emb_dir "$CV_EMB_DIR" \
    --save_dir       "$REPRODUCTION_DIR"

echo
echo "==> 3/3  Virtual screening (60 compounds)"
python "$REPO_ROOT/src/screen.py" \
    --input_csv "$SCREEN_CSV" \
    --fpts_path "$SCREEN_FPTS" \
    --model_dir "$MODEL_DIR" \
    --save_dir  "$REPRODUCTION_DIR"

echo
echo "==> Checking against the published results"
python "$REPO_ROOT/src/check_expected.py" \
    --expected_dir "$EXPECTED_DIR" \
    --actual_dir   "$REPRODUCTION_DIR"

echo
echo "Outputs in $REPRODUCTION_DIR"
