"""
Reproduce the five-fold cross-validation results.

Loads the 90 saved fold heads (5 folds x 6 representations x 3 classifiers) and
scores each on its own fold hold-out. Nothing is retrained.

Feature routing mirrors how the heads were trained:

* Mordred, PaDEL, RDKit, ECFP4 and pretrained GROVER index the full training-set
  cache by each hold-out molecule's position in train_set.csv.
* Finetuned GROVER instead reads that fold's own emb_holdout.npz, produced by an
  encoder that never saw the fold's hold-out. This is what makes the finetuned
  GROVER cross-validation leak-free.

Writes, under --save_dir:
  cv_metrics.csv    90 rows, one per fold and model
  cv_summary.csv    18 rows, mean and standard deviation across folds
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdac import io, metrics, models  # noqa: E402

FOLD_GROVER = 'finetuned_grover'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True,
                   help='Holds train_set.csv and <representation>_fpts/')
    p.add_argument('--splits_dir', required=True,
                   help='Holds fold_0..fold_4/holdout.csv')
    p.add_argument('--model_dir', required=True, help='Holds Fold1..Fold5/')
    p.add_argument('--grover_emb_dir', required=True,
                   help='Holds fold_0..fold_4/emb_holdout.npz')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    params = parse_args()
    device = params.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(params.save_dir, exist_ok=True)

    train_smiles, _ = io.read_smiles_and_labels(
        os.path.join(params.data_dir, 'train_set.csv'))
    row_of = {smiles: index for index, smiles in enumerate(train_smiles)}
    if len(row_of) != len(train_smiles):
        raise ValueError('Duplicate SMILES in train_set.csv; row lookup is ambiguous')

    caches = {}
    for representation in models.REPRESENTATIONS:
        if representation == FOLD_GROVER:
            continue
        matrix = io.load_feature_matrix(
            io.feature_path(params.data_dir, representation, 'train'))
        io.assert_rows_match(matrix, len(train_smiles), f'{representation} train features')
        caches[representation] = matrix

    print(f'Device for MLP inference: {device}')
    rows = []
    for fold in range(1, params.folds + 1):
        holdout_smiles, y_true = io.read_smiles_and_labels(
            os.path.join(params.splits_dir, f'fold_{fold - 1}', 'holdout.csv'))
        fold_dir = os.path.join(params.model_dir, f'Fold{fold}')
        print(f'\n=== Fold {fold}: {len(y_true)} hold-out molecules ===')

        for representation in models.REPRESENTATIONS:
            if representation == FOLD_GROVER:
                features = io.load_feature_matrix(
                    os.path.join(params.grover_emb_dir, f'fold_{fold - 1}', 'emb_holdout.npz'))
                io.assert_rows_match(
                    features, len(y_true), f'fold {fold} finetuned GROVER embedding')
            else:
                features = caches[representation][[row_of[s] for s in holdout_smiles]]

            for model in models.MODELS:
                y_pred, y_prob = models.predict(
                    fold_dir, representation, model, features,
                    model_set='cv', device=device)
                scores = metrics.compute_metrics(y_true, y_pred, y_prob)
                rows.append({'fold': fold,
                             'representation': models.display_name(representation),
                             'model': models.model_name(model), **scores})
                print(f'  {models.display_name(representation):18s} '
                      f'{models.model_name(model):8s} '
                      f'accuracy={scores["accuracy"]:.6f} '
                      f'macro_f1={scores["macro_f1"]:.6f}')

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(os.path.join(params.save_dir, 'cv_metrics.csv'), index=False)
    summary = metrics.summarise_folds(fold_metrics, ['representation', 'model'])
    summary.to_csv(os.path.join(params.save_dir, 'cv_summary.csv'), index=False)

    print(f'\nWrote {len(fold_metrics)} fold rows and {len(summary)} summary rows '
          f'to {params.save_dir}')
    print()
    print(summary[['representation', 'model', 'accuracy_mean', 'accuracy_std',
                   'macro_f1_mean', 'macro_f1_std']].to_string(index=False))


if __name__ == '__main__':
    main()
