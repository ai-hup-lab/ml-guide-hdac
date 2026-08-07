#!/bin/bash
# Shared configuration. Override any of these in your shell before running a script, e.g.
#   export GROVER_ROOT=/opt/grover
#   export DATA_DIR=$PWD/data/my_split
#
# Nothing here is machine-specific; every default is relative to the repo root.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Where the GROVER source tree lives (https://github.com/tencent-ailab/grover).
: "${GROVER_ROOT:=$REPO_ROOT/external/grover}"

# Pretrained GROVER checkpoint (grover_base.pt or grover_large.pt), downloaded from the
# GROVER repository. Not distributed here.
: "${GROVER_PRETRAINED:=$REPO_ROOT/checkpoints/grover_large.pt}"

# Dataset root. Must contain train_set.csv and test_set.csv with columns: smiles,labels
# Available from the authors on request -- see README.
: "${DATA_DIR:=$REPO_ROOT/data}"

# Where trained heads, metrics and logs are written.
: "${RESULTS_DIR:=$REPO_ROOT/results}"

# Finetuned GROVER checkpoint produced by scripts/finetune_grover.sh
: "${GROVER_FINETUNED:=$DATA_DIR/finetuned_model_grover/fold_0/model_0/model.pt}"

# --- Reproduction inputs -------------------------------------------------------
# All of these come from the dataset archive supplied by the authors: unzip it and
# copy its data/ and results/ directories into the repository root.

# The 18 final classifier heads, and the 90 per-fold heads.
: "${MODEL_DIR:=$RESULTS_DIR/final-classification-models}"
: "${CV_MODEL_DIR:=$RESULTS_DIR/final-cv-classifier-result}"

# Per-fold finetuned GROVER embeddings. Each fold's encoder never saw that fold's
# hold-out, which is what makes the cross-validation leak-free.
: "${CV_EMB_DIR:=$RESULTS_DIR/cv_representation_finetuned_grover}"
: "${SPLITS_DIR:=$DATA_DIR/5-fold CV splits}"

# The screened compound series and its cached embeddings.
: "${SCREEN_CSV:=$DATA_DIR/virutal_screening_series/6i-hdac-vs.csv}"
: "${SCREEN_FPTS:=$RESULTS_DIR/final-virtual-screening-result/6i-hdac-grover-fingerprints.npz}"

# Where reproduction outputs are written.
: "${REPRODUCTION_DIR:=$REPO_ROOT/reproduction_output}"
: "${EXPECTED_DIR:=$REPO_ROOT/expected_results}"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::DeprecationWarning}"
export RDKIT_DEPRECATION_WARNING=0

require_dir() {
    if [ ! -d "$1" ]; then
        echo "Error: directory not found: $1" >&2
        echo "  $2" >&2
        exit 1
    fi
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "Error: file not found: $1" >&2
        echo "  $2" >&2
        exit 1
    fi
}
