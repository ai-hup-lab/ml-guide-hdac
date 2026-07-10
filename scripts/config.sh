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
