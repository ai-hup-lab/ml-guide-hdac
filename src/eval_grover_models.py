"""
Benchmark the GROVER-based classifiers on the held-out test set.

Representations : base_grover (pretrained, zero-shot) and finetuned_grover
Classifiers     : Random Forest, XGBoost, MLP
Metrics         : accuracy, precision, recall, macro F1, ROC-AUC

Embeddings are read unscaled from <data_dir>/<fp_type>_fpts/test_set_fingerprint.npz,
matching how the heads were trained.
"""
import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.simple_mlp import SimpleMLP  # noqa: E402

warnings.filterwarnings('ignore')

FP_TYPES = ['base_grover', 'finetuned_grover']
MODEL_TYPES = ['rf', 'xgb', 'mlp']

FP_LABEL = {
    'base_grover': 'Pretrained GROVER (Zero-shot)',
    'finetuned_grover': 'Finetuned GROVER',
}
MODEL_LABEL = {'rf': 'Random Forest', 'xgb': 'XGBoost', 'mlp': 'MLP'}

# Evaluation Variables

MAIN_METRICS = ['accuracy', 'precision', 'recall', 'macro_f1', 'roc_auc']


def caveat_for(fp_type, model_type):
    return ''


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark GROVER-based classifiers')
    p.add_argument('--data_dir', required=True, help='holds test_set.csv and <fp_type>_fpts/')
    p.add_argument('--model_dir', required=True, help='holds <fp_type>_<model>.{joblib,pt}')
    p.add_argument('--save_dir', required=True)
    return p.parse_args()


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df['smiles'].str.strip().values, df['labels'].values.astype(int)


def load_mlp(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model = SimpleMLP(
        input_dim=int(ckpt.get('input_dim', state['net.0.weight'].shape[1])),
        hidden_dim=int(ckpt.get('hidden_dim', state['net.0.weight'].shape[0])),
        output_dim=int(ckpt.get('output_dim', 1)),
        dropout=float(ckpt.get('dropout', 0.1)),
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_proba_active(model_path, model_type, X, device):
    """Probability of the positive (active = 1) class."""
    if model_type == 'mlp':
        model = load_mlp(model_path, device)
        with torch.no_grad():
            xb = torch.from_numpy(np.asarray(X)).float().to(device)
            return model.predict_proba(xb).cpu().numpy().flatten()

    model = joblib.load(model_path)
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba[:, 0] if list(model.classes_) == [1] else 1.0 - proba[:, 0]


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'roc_auc': roc_auc,
        'f1_active': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    test_smiles, y_test = load_csv(os.path.join(args.data_dir, 'test_set.csv'))
    print(f'Device: {device} | test set: {len(y_test)} molecules '
          f'({int((y_test == 1).sum())} active / {int((y_test == 0).sum())} inactive)\n')

    rows, per_mol = [], {}
    for fp_type in FP_TYPES:
        fp_path = os.path.join(args.data_dir, f'{fp_type}_fpts', 'test_set_fingerprint.npz')
        if not os.path.exists(fp_path):
            print(f'! missing {fp_path} -- skipped')
            continue
        X_test = np.load(fp_path)['fps']
        if len(X_test) != len(y_test):
            raise ValueError(f'{fp_path}: {len(X_test)} rows vs {len(y_test)} labels. '
                             'Regenerate the embeddings for this exact test_set.csv.')
        print(f'{FP_LABEL[fp_type]:32s} test features {X_test.shape}')

        for model_type in MODEL_TYPES:
            ext = 'pt' if model_type == 'mlp' else 'joblib'
            model_path = os.path.join(args.model_dir, f'{fp_type}_{model_type}.{ext}')
            if not os.path.exists(model_path):
                print(f'  ! missing {model_path} -- skipped')
                continue

            y_prob = predict_proba_active(model_path, model_type, X_test, device)
            metrics = compute_metrics(y_test, y_prob)
            rows.append({'representation': FP_LABEL[fp_type], 'model': MODEL_LABEL[model_type],
                         'fp_type': fp_type, 'model_type': model_type, **metrics,
                         'caveat': caveat_for(fp_type, model_type)})
            per_mol[f'{fp_type}_{model_type}'] = pd.DataFrame({
                'smiles': test_smiles,
                'true_label': y_test,
                'predicted_label': (y_prob >= 0.5).astype(int),
                'active_probability': y_prob,
            })
            print(f'  {MODEL_LABEL[model_type]:14s} acc={metrics["accuracy"]:.4f} '
                  f'prec={metrics["precision"]:.4f} rec={metrics["recall"]:.4f} '
                  f'macroF1={metrics["macro_f1"]:.4f} auc={metrics["roc_auc"]:.4f}')

    if not rows:
        raise SystemExit('No models evaluated. Train them first with '
                         'scripts/train_predictive_model.sh <fp_type> <model_type>')

    results = pd.DataFrame(rows)
    csv_path = os.path.join(args.save_dir, 'grover_model_performance.csv')
    results.to_csv(csv_path, index=False)

    summary = results[['representation', 'model'] + MAIN_METRICS]
    present_fp = [FP_LABEL[f] for f in FP_TYPES if FP_LABEL[f] in set(results['representation'])]
    present_md = [MODEL_LABEL[m] for m in MODEL_TYPES if MODEL_LABEL[m] in set(results['model'])]
    pivots = {m: results.pivot(index='representation', columns='model', values=m)
                 .reindex(index=present_fp, columns=present_md)
              for m in MAIN_METRICS}

    xlsx_path = os.path.join(args.save_dir, 'grover_model_performance.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        summary.to_excel(w, sheet_name='Summary', index=False)
        results.to_excel(w, sheet_name='Full_Metrics', index=False)
        for m, table in pivots.items():
            table.round(4).to_excel(w, sheet_name=f'by_{m}'[:31])
        for name, df in per_mol.items():
            df.to_excel(w, sheet_name=f'pred_{name}'[:31], index=False)

    print('\n' + '=' * 78)
    print('TEST-SET PERFORMANCE  (positive class = Active)')
    print('=' * 78)
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.round(4).to_string(index=False))
    for m in MAIN_METRICS:
        print(f'\n--- {m} ---')
        print(pivots[m].round(4).to_string())

    best = results.loc[results['roc_auc'].idxmax()]
    print(f'\nBest ROC-AUC: {best["representation"]} + {best["model"]} = {best["roc_auc"]:.4f}')
    print(f'\nSaved:\n  {csv_path}\n  {xlsx_path}')


if __name__ == '__main__':
    main()
