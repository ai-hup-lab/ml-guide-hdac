# Machine learning-guided design, synthesis and biological evaluation of novel benzimidazole-based HDAC inhibitors

Code accompanying the manuscript on an integrated workflow in which a fine-tuned [GROVER](https://github.com/tencent-ailab/grover) graph transformer prioritises synthetically accessible HDAC inhibitor candidates, which are then docked, simulated, synthesised and evaluated biologically.

## What the code does

Two GROVER representations × three classifiers, evaluated on a held-out test set:

| Representation | Dim | Source |
|---|---|---|
| Pretrained GROVER (zero-shot) | 5017 | `grover_large.pt`, `--fingerprint_source both` |
| Fine-tuned GROVER | 5017 | checkpoint from `scripts/finetune_grover.sh` |

Each embedding is 4800 atom+bond readout dimensions concatenated with 217 RDKit 2D normalized descriptors, used unscaled.

Classifiers: `RandomForest`, `XGBoost`, and a single-hidden-layer MLP (`src/models/simple_mlp.py`).

Metrics: accuracy, precision, recall, macro F1, ROC-AUC.

## Layout

```
src/
  train_predictive_model.py   train one (representation, classifier) combination
  eval_grover_models.py       benchmark all six combinations -> csv + xlsx
  models/simple_mlp.py        SimpleMLP + Trainer
grover_addons/
  prepare_grover_input.py     SMILES -> GROVER inference CSV (dummy labels)
  prepare_grover_training.py  labelled CSV -> GROVER finetuning CSV
scripts/
  config.sh                   shared paths, override via environment variables
  finetune_grover.sh          finetune a pretrained GROVER checkpoint
  gen_grover_fingerprint.sh   extract 5017-d embeddings from a checkpoint
  train_predictive_model.sh   train one combination
  eval_grover_models.sh       benchmark grover model
```

## Installation

```bash
git clone <this-repo> ml-guide-hdac && cd ml-guide-hdac
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# GROVER is not a pip package
git clone https://github.com/tencent-ailab/grover external/grover
pip install -r external/grover/requirements.txt
```

Then download the pretrained `grover_large.pt` from the GROVER repository into `checkpoints/`.

Every path is configurable through the environment; see `scripts/config.sh`.

```bash
export GROVER_ROOT=/opt/grover
export GROVER_PRETRAINED=/weights/grover_large.pt
export DATA_DIR=$PWD/data
```

## Data format

`$DATA_DIR/train_set.csv` and `$DATA_DIR/test_set.csv`, each with at least:

```csv
smiles,labels
ONC(/C=C/c1cccc(...)c1)=O,1
COc(cc(...))c1OCC(...)=O,0
```

`labels`: 1 = active, 0 = inactive. Additional columns are ignored.

## Usage

```bash
# 1. Finetune GROVER
VAL_CSV=data/val_set.csv scripts/finetune_grover.sh

# 2. Extract embeddings with each checkpoint
scripts/gen_grover_fingerprint.sh train_set base
scripts/gen_grover_fingerprint.sh test_set  base
scripts/gen_grover_fingerprint.sh train_set finetuned
scripts/gen_grover_fingerprint.sh test_set  finetuned

# 3. Train all six combinations
for fp in base_grover finetuned_grover; do
  for m in rf xgb mlp; do scripts/train_predictive_model.sh "$fp" "$m"; done
done

# 4. Benchmark
scripts/eval_grover_models.sh
```

Step 4 writes `results/benchmark/grover_model_performance.{csv,xlsx}`. The workbook carries a `Summary` sheet, per-metric pivots, and per-molecule predictions.

## Citation

This code is for reviewing purpose.

## Acknowledgement

We thank you GROVER for distributed the code under the MIT License, please clone it from [tencent-ailab/grover](https://github.com/tencent-ailab/grover) when using our code. The two scripts in `grover_addons/` are ours and are designed to run against that tree.

## License and third-party code

This repository is released under the [MIT License](LICENSE).
