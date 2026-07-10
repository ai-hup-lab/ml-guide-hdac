"""
Train a downstream classifier on GROVER embeddings.

Representations (--fp_type):
  base_grover        5017-d embeddings from the pretrained GROVER checkpoint (zero-shot)
  finetuned_grover   5017-d embeddings from the HDAC-finetuned GROVER checkpoint

Classifiers (--model_type): rf | xgb | mlp

Embeddings are produced by scripts/gen_grover_fingerprint.sh and are read from
  <fpts_dir>/<fp_type>_fpts/{train,test}_set_fingerprint.npz   (key: 'fps')

They are used unscaled, matching the finetuning pipeline.
"""
import argparse
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.simple_mlp import SimpleMLP, Trainer  # noqa: E402

FP_TYPES = ['base_grover', 'finetuned_grover']
MODEL_TYPES = ['rf', 'xgb', 'mlp']


class TeeStream:
    """Write output to multiple streams (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_training_log(save_dir, fp_type, model_type):
    logs_dir = os.path.join(save_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(logs_dir, f'{fp_type}_{model_type}_{stamp}.log')
    log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path


def parse_args():
    p = argparse.ArgumentParser(description='Train an HDAC classifier on GROVER embeddings')
    p.add_argument('--train_csv', required=True, help='CSV with columns smiles,labels')
    p.add_argument('--test_csv', required=True, help='CSV with columns smiles,labels')
    p.add_argument('--fpts_dir', required=True, help='Directory holding <fp_type>_fpts/')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--fp_type', default='finetuned_grover', choices=FP_TYPES)
    p.add_argument('--model_type', default='rf', choices=MODEL_TYPES)
    p.add_argument('--random_seed', type=int, default=42)

    # MLP
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--scheduler', default='plateau', choices=['plateau', 'cosine', 'none'])
    p.add_argument('--scheduler_factor', type=float, default=0.5)
    p.add_argument('--scheduler_patience', type=int, default=10)
    p.add_argument('--scheduler_min_lr', type=float, default=1e-6)
    p.add_argument('--scheduler_t_max', type=int, default=100)
    p.add_argument('--scheduler_eta_min', type=float, default=1e-6)
    p.add_argument('--mlp_val_split', type=float, default=0.1,
                   help='Fraction of the TRAINING set held out for MLP early stopping.')

    # RF / XGB
    p.add_argument('--n_estimators', type=int, default=200)
    p.add_argument('--xgb_n_estimators', type=int, default=100)
    p.add_argument('--xgb_learning_rate', type=float, default=0.001)
    return p.parse_args()


def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'smiles' not in df.columns or 'labels' not in df.columns:
        raise ValueError(f"CSV must contain 'smiles' and 'labels'. Found: {df.columns.tolist()}")
    smiles = df['smiles'].str.strip().values
    labels = df['labels']
    labels = labels.str.strip().values if labels.dtype == 'object' else labels.values
    return smiles, labels


def load_grover_fingerprints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'GROVER fingerprints not found: {path}\n'
            'Generate them first: scripts/gen_grover_fingerprint.sh <split> <base|finetuned>'
        )
    print(f'Loading GROVER fingerprints: {path}')
    return np.load(path)['fps']


def build_mlp_scheduler(optimizer, params, epochs):
    if params.scheduler == 'none':
        return None
    if params.scheduler == 'cosine':
        t_max = max(1, int(params.scheduler_t_max or epochs))
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=float(params.scheduler_eta_min))
    return ReduceLROnPlateau(optimizer, mode='min', factor=float(params.scheduler_factor),
                             patience=max(1, int(params.scheduler_patience)),
                             min_lr=float(params.scheduler_min_lr))


def report(y_true, y_pred, y_prob, target_names, split_name):
    print('\n' + '=' * 50)
    print(f'{split_name.upper()} SET EVALUATION')
    print('=' * 50)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')
    print(f'Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'Overall Precision: {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}')
    print(f'Overall Recall: {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}')
    print(f'Overall Macro F1: {f1_score(y_true, y_pred, average="macro", zero_division=0):.4f}')
    print(f'Overall ROC-AUC: {roc_auc:.4f}')


def main():
    params = parse_args()
    os.makedirs(params.save_dir, exist_ok=True)
    print(f'Training log file: {setup_training_log(params.save_dir, params.fp_type, params.model_type)}')

    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\nRepresentation: {params.fp_type}\nClassifier: {params.model_type}')

    train_smiles, train_labels = load_csv_data(params.train_csv)
    test_smiles, test_labels = load_csv_data(params.test_csv)

    label_encoder = LabelEncoder().fit(np.concatenate([train_labels, test_labels]))
    y_train = label_encoder.transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    target_names = [str(c) for c in label_encoder.classes_]
    print(f'\nClasses: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}')
    print(f'Train distribution: {np.bincount(y_train)} | Test distribution: {np.bincount(y_test)}')

    fp_dir = os.path.join(params.fpts_dir, f'{params.fp_type}_fpts')
    X_train = load_grover_fingerprints(os.path.join(fp_dir, 'train_set_fingerprint.npz'))
    X_test = load_grover_fingerprints(os.path.join(fp_dir, 'test_set_fingerprint.npz'))
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(
            f'Row mismatch: fingerprints ({len(X_train)}/{len(X_test)}) vs labels '
            f'({len(y_train)}/{len(y_test)}). Regenerate the embeddings for these exact CSVs.'
        )
    print(f'\nTrain fingerprints: {X_train.shape} | Test fingerprints: {X_test.shape}')

    trainer = None
    if params.model_type == 'rf':
        model = RandomForestClassifier(n_estimators=params.n_estimators, random_state=params.random_seed)
        model.fit(X_train, y_train)
    elif params.model_type == 'xgb':
        model = XGBClassifier(n_estimators=params.xgb_n_estimators,
                              learning_rate=params.xgb_learning_rate,
                              eval_metric='logloss', random_state=params.random_seed)
        model.fit(X_train, y_train)
    else:
        # Choose the early-stopping validation set.
        if params.mlp_val_split > 0:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y_train, test_size=params.mlp_val_split,
                stratify=y_train, random_state=params.random_seed)
            print(f'\nMLP early stopping on a {params.mlp_val_split:.0%} split of the training set '
                  f'({len(y_fit)} fit / {len(y_val)} val). Test set untouched.')
        else:
            X_fit, y_fit, X_val, y_val = X_train, y_train, X_test, y_test
            print('\nMLP early stopping on the TEST set.')

        model = SimpleMLP(input_dim=X_train.shape[1], hidden_dim=params.hidden_dim,
                          output_dim=1, dropout=params.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        trainer = Trainer(model, torch.nn.BCEWithLogitsLoss(), optimizer, device=device,
                          scheduler=build_mlp_scheduler(optimizer, params, params.epochs))
        os.makedirs(os.path.join(params.save_dir, 'models'), exist_ok=True)
        trainer.train(
            X_fit, y_fit, X_val=X_val, y_val=y_val,
            epochs=params.epochs, batch_size=params.batch_size, lr=params.lr,
            best_checkpoint_path=os.path.join(params.save_dir, 'models',
                                              f'{params.fp_type}_{params.model_type}.pt'),
            checkpoint_payload={
                'model_name': 'SimpleMLP', 'model_type': params.model_type,
                'fp_type': params.fp_type, 'input_dim': int(X_train.shape[1]),
                'hidden_dim': int(params.hidden_dim), 'output_dim': 1,
                'dropout': float(params.dropout),
                'label_classes': [str(x) for x in label_encoder.classes_],
            },
        )

    def predict(X):
        """Return (y_pred, p_active)."""
        if params.model_type == 'mlp':
            xb = torch.from_numpy(np.asarray(X)).float().to(device)
            p = trainer.model.predict_proba(xb).cpu().numpy().flatten()
        else:
            p = model.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int), p

    results = {}
    for split, X, y, smiles, labels in [('Train', X_train, y_train, train_smiles, train_labels),
                                        ('Test', X_test, y_test, test_smiles, test_labels)]:
        y_pred, y_prob = predict(X)
        report(y, y_pred, y_prob, target_names, split)
        results[split] = pd.DataFrame({
            'smiles': smiles,
            'true_label': labels,
            'predicted_label': label_encoder.inverse_transform(y_pred),
            'active_probability': y_prob,
            'inactive_probability': 1.0 - y_prob,
        })

    eval_dir = os.path.join(params.save_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    predictions_path = os.path.join(eval_dir, f'{params.fp_type}_{params.model_type}.xlsx')
    with pd.ExcelWriter(predictions_path) as writer:
        results['Train'].to_excel(writer, sheet_name='Train Set', index=False)
        results['Test'].to_excel(writer, sheet_name='Test Set', index=False)
    print(f'\nPredictions saved to {predictions_path}')

    models_dir = os.path.join(params.save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    if params.model_type in ('rf', 'xgb'):
        out = os.path.join(models_dir, f'{params.fp_type}_{params.model_type}.joblib')
        joblib.dump(model, out, compress=5)
        print(f'Model saved to {out}')
    else:
        print(f'Best MLP checkpoint saved to {models_dir}/{params.fp_type}_{params.model_type}.pt')


if __name__ == '__main__':
    main()
