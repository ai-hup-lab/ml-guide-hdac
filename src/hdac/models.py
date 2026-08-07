"""Loading trained heads and turning features into predictions.

Every convention that differs between the two shipped model sets is expressed
here as data, so no caller has to remember it.

Two of them are easy to get wrong and fail quietly rather than loudly:

1. **Mordred scaling is inverted between the model sets.** In the final heads,
   the random forest and XGBoost are plain estimators and the scaler must be
   applied by the caller -- for all three classifiers. In the cross-validation
   heads they are sklearn Pipelines carrying a fold-fitted scaler, so they take
   raw descriptors and only the MLP needs external scaling. Feeding unscaled
   descriptors to a model that expects scaled ones yields chance accuracy with no
   error raised.

2. **Labels come from ``predict()``, not from thresholding the probability.**
   For the tree ensembles, ``predict`` takes an argmax, which breaks an exact
   0.5 tie toward the inactive class; ``p >= 0.5`` breaks it the other way. With
   200 trees exact ties do occur, and the published numbers use ``predict``.
"""
import os

import joblib
import numpy as np
import torch

from .simple_mlp_loader import load_mlp

# Display name, and the stem used in each model set's filenames.
REPRESENTATIONS = {
    'mordred': ('Mordred', 'mordred', 'Mordred'),
    'padel': ('PaDEL', 'padel', 'PaDEL'),
    'rdkit': ('RDKit', 'rdkit', 'RDKit'),
    'ecfp4': ('ECFP4', 'ecfp4', 'ECFP4'),
    'base_grover': ('Pretrained GROVER', 'base_grover', 'PretrainedGROVERZeroShot'),
    'finetuned_grover': ('Finetuned GROVER', 'finetune_grover', 'FinetunedGROVER'),
}
MODELS = {
    'rf': ('RF', 'rf', 'RandomForest'),
    'xgb': ('XGBoost', 'xgb', 'XGBoost'),
    'mlp': ('MLP', 'mlp', 'MLP'),
}


def display_name(representation):
    return REPRESENTATIONS[representation][0]


def model_name(model):
    return MODELS[model][0]


def head_path(model_dir, representation, model, model_set='final'):
    """Locate a trained head. The two model sets use different filename styles."""
    extension = 'pt' if model == 'mlp' else 'joblib'
    if model_set == 'final':
        stem = REPRESENTATIONS[representation][1]
        return os.path.join(model_dir, f'{stem}_{MODELS[model][1]}.{extension}')
    stem = REPRESENTATIONS[representation][2]
    return os.path.join(model_dir, f'{MODELS[model][2]}-{stem}.{extension}')


def scaler_path(model_dir, representation, model, model_set='final'):
    """Return the scaler to apply before this head, or None if none is needed."""
    if representation != 'mordred':
        return None
    if model_set == 'final':
        # Plain estimators throughout: every classifier needs external scaling.
        return os.path.join(model_dir, 'mordred_scaler.joblib')
    if model == 'mlp':
        # In the CV set only the MLP is unwrapped; RF and XGBoost are Pipelines.
        return os.path.join(model_dir, 'MLP-Mordred-scaler.joblib')
    return None


def prepare_features(features, model_dir, representation, model, model_set='final'):
    """Apply whatever preprocessing this head expects."""
    path = scaler_path(model_dir, representation, model, model_set)
    if path is None:
        return features
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{representation}/{model} needs its scaler, missing: {path}. '
            'Without it the descriptors are on the wrong scale and accuracy '
            'collapses to chance without any error being raised.'
        )
    return joblib.load(path).transform(features)


def predict(model_dir, representation, model, features, model_set='final', device='cpu'):
    """Return (predicted_label, active_probability) for one head.

    Labels follow the estimator's own decision rule rather than a threshold on
    the probability -- see the module docstring.
    """
    path = head_path(model_dir, representation, model, model_set)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Model not found: {path}')
    X = prepare_features(features, model_dir, representation, model, model_set)

    if model == 'mlp':
        net = load_mlp(path, device)
        with torch.no_grad():
            probability = net.predict_proba(
                torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            ).cpu().numpy().ravel()
        return (probability >= 0.5).astype(int), probability

    estimator = joblib.load(path)
    labels = np.asarray(estimator.predict(X)).astype(int)
    probabilities = estimator.predict_proba(X)
    classes = list(getattr(estimator, 'classes_', [0, 1]))
    active = classes.index(1) if 1 in classes else 1
    return labels, probabilities[:, active]
