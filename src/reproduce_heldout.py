"""
Reproduce the held-out test-set results.

Loads the 18 saved heads (6 representations x 3 classifiers) and scores them on
the held-out test set. Nothing is retrained.

Writes, under --save_dir:
  heldout_metrics.csv                     one row per model, five metrics
  predictions/<rep>_<model>.csv           per-molecule predictions

Compare the metrics against expected_results/heldout_metrics.csv, or run
scripts/reproduce_all.sh which does that for you.
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdac import io, metrics, models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True,
                   help='Holds test_set.csv and <representation>_fpts/')
    p.add_argument('--model_dir', required=True, help='Holds the 18 saved heads')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--device', default=None,
                   help='Torch device for the MLP heads; defaults to cuda:0 when available')
    return p.parse_args()


def main():
    params = parse_args()
    device = params.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    predictions_dir = os.path.join(params.save_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    test_csv = os.path.join(params.data_dir, 'test_set.csv')
    smiles, y_true = io.read_smiles_and_labels(test_csv)
    print(f'{len(smiles)} molecules from {test_csv}')
    print(f'Device for MLP inference: {device}\n')

    rows = []
    for representation in models.REPRESENTATIONS:
        features = io.load_feature_matrix(
            io.feature_path(params.data_dir, representation, 'test'))
        io.assert_rows_match(features, len(smiles), f'{representation} test features')

        for model in models.MODELS:
            y_pred, y_prob = models.predict(
                params.model_dir, representation, model, features,
                model_set='final', device=device)

            pd.DataFrame({
                'smiles': smiles,
                'true_label': y_true,
                'predicted_label': y_pred,
                'active_probability': y_prob,
                'inactive_probability': 1.0 - y_prob,
            }).to_csv(os.path.join(predictions_dir,
                                   f'{representation}_{model}.csv'), index=False)

            scores = metrics.compute_metrics(y_true, y_pred, y_prob)
            rows.append({'representation': models.display_name(representation),
                         'model': models.model_name(model), **scores})
            print(f'  {models.display_name(representation):18s} '
                  f'{models.model_name(model):8s} '
                  + '  '.join(f'{k}={scores[k]:.4f}' for k in metrics.METRIC_NAMES))

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(params.save_dir, 'heldout_metrics.csv')
    summary.to_csv(summary_path, index=False, float_format='%.6f')
    print(f'\nWrote {len(summary)} rows to {summary_path}')
    print(f'Wrote {len(summary)} prediction files to {predictions_dir}')


if __name__ == '__main__':
    main()
