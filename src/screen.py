"""
Screen a compound series with the finetuned-GROVER classifier heads.

Scores every molecule in the input CSV with the random forest, XGBoost and MLP
heads and writes one row per compound, with a probability and a label per head.

GROVER embeddings cannot be produced from SMILES here -- they need a checkpoint
and a forward pass -- so --fpts_path must point at a cache whose row order
matches the input CSV. The archive supplies one for the published series.
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdac import io, models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input_csv', required=True, help='Screening set with a smiles column')
    p.add_argument('--fpts_path', required=True,
                   help='Cached embeddings; row order must match --input_csv')
    p.add_argument('--model_dir', required=True, help='Holds the final classifier heads')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--output_name', default='screening_result.csv')
    p.add_argument('--representation', default='finetuned_grover',
                   choices=sorted(models.REPRESENTATIONS))
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Probability at or above which a compound is called active')
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    params = parse_args()
    device = params.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(params.save_dir, exist_ok=True)

    frame, smiles_column, smiles = io.read_smiles(params.input_csv)
    id_column = next((c for c in frame.columns
                      if c.strip().lower() in {'code', 'id', 'name', 'compound'}), None)
    if id_column is None:
        raise ValueError(f'{params.input_csv}: no identifier column. '
                         f'Found: {frame.columns.tolist()}')

    features = io.load_feature_matrix(params.fpts_path)
    io.assert_rows_match(features, len(smiles), 'screening embeddings')
    print(f'Screening {len(smiles)} compounds with '
          f'{models.display_name(params.representation)} heads on {device}')

    label = 'GROVER' if 'grover' in params.representation else \
        models.display_name(params.representation)
    result = pd.DataFrame({id_column: frame[id_column], smiles_column: smiles})
    summary = []
    for model in models.MODELS:
        _, probability = models.predict(
            params.model_dir, params.representation, model, features,
            model_set='final', device=device)
        if not ((0.0 <= probability) & (probability <= 1.0)).all():
            raise ValueError(f'{model}: probabilities outside [0, 1]')
        labels = (probability >= params.threshold).astype(int)
        name = models.model_name(model)
        result[f'{name}-{label} Prob.'] = probability
        result[f'{name}-{label} Labels'] = labels
        summary.append((f'{name}-{label}', int(labels.sum()), len(labels)))

    output = os.path.join(params.save_dir, params.output_name)
    result.to_csv(output, index=False, float_format='%.6f')
    print(f'\nWrote {len(result)} rows to {output}')
    print(f'\nPredicted active at probability >= {params.threshold}:')
    for name, actives, total in summary:
        print(f'  {name:24s} {actives:3d}/{total}')


if __name__ == '__main__':
    main()
