#!/bin/bash
#
# Build the rule-based descriptor caches for the train and test splits.
#
# Usage:
#   scripts/prepare_features.sh                      # ecfp4,rdkit -- RDKit only
#   REPRESENTATIONS=ecfp4,rdkit,mordred,padel scripts/prepare_features.sh
#
# Mordred needs scikit-fingerprints; PaDEL needs padel-pywrapper and a Java runtime.
# Neither is installed by requirements.txt -- see the "optional" block there. The
# generators raise a clear ImportError if a backend is missing.
#
# The dataset archive supplied by the authors already contains these caches. This
# script is for featurising new molecules, or rebuilding from scratch.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

REPRESENTATIONS="${REPRESENTATIONS:-ecfp4,rdkit}"

require_dir  "$DATA_DIR"               "Dataset is available from the authors on request."
require_file "$DATA_DIR/train_set.csv" "Dataset is available from the authors on request."
require_file "$DATA_DIR/test_set.csv"  "Dataset is available from the authors on request."

for split in train test; do
    echo "[$split] building: $REPRESENTATIONS"
    python "$REPO_ROOT/src/prepare_features.py" \
        --csv             "$DATA_DIR/${split}_set.csv" \
        --split           "$split" \
        --fpts_dir        "$DATA_DIR" \
        --representations "$REPRESENTATIONS" \
        "$@"
done

echo "Caches written to \$DATA_DIR/<representation>_fpts/{train,test}_set_fingerprint.npz"
