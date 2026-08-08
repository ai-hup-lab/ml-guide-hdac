# Machine learning-guided design and synthesis of benzimidazole-based HDAC inhibitors: prospective validation of a fine-tuned pretrained graph transformer

![Graphical abstract: molecular 2D graphs are encoded by a fine-tuned GROVER representation model and classified by random forest, XGBoost and MLP classifiers; 60 designed compounds are screened down to 6 synthesised compounds of series IX; these are evaluated for HDAC inhibition, cytotoxicity, cell-cycle effects and apoptosis, with IXc inhibiting HDAC at 19.33 nM against 79.95 nM for SAHA.](documents/toc-graphic.png)

Code accompanying the manuscript. A pretrained [GROVER](https://github.com/tencent-ailab/grover)
graph transformer is fine-tuned on HDAC activity data and used to prioritise synthetically
accessible candidates from a designed series. The predictions are then validated prospectively:
the selected compounds were synthesised, and evaluated by docking, molecular dynamics and
biological assay. The dataset and the trained model weights are available from the authors on request, see [ARCHIVE.md](ARCHIVE.md).

## What the code does

Six molecular representations × three classifiers, evaluated on a held-out test set and by
five-fold cross-validation, then used to screen a 60-compound design series.

| Representation | Dimensions | Source |
|---|---:|---|
| Mordred | 1613 | descriptor set, standardised |
| PaDEL | 881 | PubChem substructure fingerprint |
| RDKit | 2048 | RDKit topological fingerprint |
| ECFP4 | 2048 | circular fingerprint, radius 2 |
| Pretrained GROVER (zero-shot) | 5017 | `grover_large.pt` |
| Fine-tuned GROVER | 5017 | checkpoint from `scripts/finetune_grover.sh` |

Each GROVER embedding is 4800 atom+bond readout dimensions concatenated with 217 RDKit 2D
normalized descriptors, used unscaled.

Classifiers: random forest, XGBoost, and a single-hidden-layer MLP (`src/models/simple_mlp.py`).
Metrics: accuracy, precision, recall, macro F1, ROC-AUC.

## Reproducing the published results

This is the path to check our numbers. Nothing is retrained — the distributed models are loaded
and scored.

```bash
git clone <this-repo> ml-guide-hdac && cd ml-guide-hdac
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Unzip the dataset archive, verify it, and copy its two directories in.
scripts/verify_archive.sh /path/to/hdac-reproduction-archive
cp -r /path/to/hdac-reproduction-archive/{data,results} .

scripts/reproduce_all.sh
```

`scripts/reproduce_all.sh` runs the held-out evaluation, the cross-validation and the screening,
then compares all three against `expected_results/`. **It exits non-zero if anything disagrees**,
so a silent partial success is not possible. Outputs land in `reproduction_output/`.

`expected_results/` is committed here, so you can inspect the published numbers without the
archive.

No GPU is required: the cached embeddings mean the GROVER encoder is never run.

## Layout

```
src/
  hdac/                    shared building blocks; each convention defined once
    io.py                  CSV delimiter handling, feature-cache layouts, row-count checks
    models.py              locating classifiers, Mordred scaling, the label decision rule
    metrics.py             the five reported metrics, fold summary statistics
    features.py            fingerprint generators (Mordred/PaDEL optional)
  reproduce_heldout.py     18 models on the held-out test set
  reproduce_cv.py          90 fold classifiers on their fold hold-outs
  screen.py                score a compound series
  check_expected.py        compare a run against expected_results/
  prepare_features.py      build rule-based caches from SMILES
  train_classifiers.py     ILLUSTRATIVE: how the classifiers were fitted
  models/simple_mlp.py     the MLP architecture
scripts/                   shell wrappers; paths configured in config.sh
grover_addons/             SMILES -> GROVER input conversion
expected_results/          the published summary tables
```

## Installation

```bash
pip install -r requirements.txt

# GROVER is not a pip package, and is only needed to run the encoder yourself.
git clone https://github.com/tencent-ailab/grover external/grover
pip install -r external/grover/requirements.txt
```

Then download the pretrained `grover_large.pt` from the GROVER repository into `checkpoints/`,
or use the copy in the dataset archive.

Every path is configurable through the environment; see `scripts/config.sh`.

```bash
export GROVER_ROOT=/opt/grover
export GROVER_PRETRAINED=/weights/grover_large.pt
export DATA_DIR=$PWD/data
```

Mordred and PaDEL generation needs extra backends that are **not** installed by default — see the
optional block in `requirements.txt`. They are only required to featurise new molecules; the
archive ships the caches used in the paper.

## Data format

The dataset provides four splits in `$DATA_DIR`:

| File | Molecules | Role |
|---|---:|---|
| `train_set.csv` | 1811 | Full training set; the union of the two below |
| `fit_set.csv` | 1629 | Training proper, for GROVER fine-tuning |
| `val_set.csv` | 182 | Checkpoint selection during fine-tuning |
| `test_set.csv` | 201 | Held out; never seen during training |

Each has at least:

```csv
smiles,labels
ONC(/C=C/c1cccc(...)c1)=O,1
COc(cc(...))c1OCC(...)=O,0
```

`labels`: 1 = active, 0 = inactive. Additional columns are ignored. Comma- and
semicolon-separated files are both accepted; the delimiter is detected automatically.

### Validation protocol

`fit_set.csv` and `val_set.csv` partition `train_set.csv`, and both are disjoint from
`test_set.csv`. The separation is load-bearing: GROVER selects its best checkpoint on the
validation set, so validating on the test set would make the reported test performance
optimistically biased. `scripts/finetune_grover.sh` requires `VAL_CSV` and will not fall back to
the test set.

Downstream classifiers are fitted on the full `train_set.csv`; only the GROVER fine-tuning step
uses the fit/val separation.

## Training

**Retraining will not reproduce the distributed weights bit for bit.** Results depend on library
versions, thread counts and GPU non-determinism, none of which are pinned here. The training
entry points document the procedure and the published hyperparameters; the models themselves are
distributed directly. To verify our numbers, use the reproduction path above.

```bash
# 1. Fine-tune GROVER on the fit split, selecting on the validation split
TRAIN_CSV=data/fit_set.csv VAL_CSV=data/val_set.csv scripts/finetune_grover.sh

# 2. Extract embeddings with each checkpoint
scripts/gen_grover_fingerprint.sh train_set base
scripts/gen_grover_fingerprint.sh test_set  base
scripts/gen_grover_fingerprint.sh train_set finetuned
scripts/gen_grover_fingerprint.sh test_set  finetuned

# 3. Build the rule-based descriptor caches
REPRESENTATIONS=ecfp4,rdkit,mordred,padel scripts/prepare_features.sh

# 4. Train a classifier
scripts/train_classifiers.sh finetuned_grover rf
```

## Applying the models to new molecules

```bash
python src/prepare_features.py --csv new_compounds.csv --split new \
    --fpts_dir data --representations ecfp4,rdkit

python src/screen.py --input_csv new_compounds.csv \
    --fpts_path <embeddings.npz> --model_dir results/final-classification-models \
    --save_dir screening_output
```

GROVER embeddings for new molecules require a forward pass through the encoder
(`scripts/gen_grover_fingerprint.sh`), so that step does need the checkpoint and a GPU.

**The feature caches are positional.** Row 7 of a `.npz` corresponds to row 7 of the matching CSV,
and nothing inside the file records that. A mismatched cache produces confident predictions for
the wrong molecules rather than an error, so every loader here asserts row counts — do not bypass
those checks.

## Contact

Questions about the code, the dataset, or requests for the archive of trained models:

- congnguyen.research@outlook.com
- dungdtm@hup.edu.vn

## Citation

This code is for reviewing purpose.

## Acknowledgement

We thank the GROVER authors for distributing their code under the MIT License; please clone it
from [tencent-ailab/grover](https://github.com/tencent-ailab/grover) when using our code. The two
scripts in `grover_addons/` are ours and are designed to run against that tree.

## License and third-party code

This repository is released under the [MIT License](LICENSE).
