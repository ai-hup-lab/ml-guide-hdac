"""The reported metric set, defined once.

Two of the five have more than one defensible definition, so they are pinned
here to match the manuscript:

* precision and recall are for the **active class only** (``pos_label=1``);
* F1 is **macro-averaged** over both classes, so it is not flattered by whichever
  class is larger.

ROC-AUC uses the continuous probability, not the thresholded label.
"""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ('accuracy', 'precision', 'recall', 'macro_f1', 'roc_auc')

# Column headings used in the published tables.
PUBLISHED_HEADINGS = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'macro_f1': 'Macro F1-Score',
    'roc_auc': 'ROC-AUC',
}


def compute_metrics(y_true, y_pred, y_prob):
    """Return the five reported metrics.

    Takes predicted labels explicitly rather than thresholding ``y_prob``: the
    tree ensembles decide by argmax, which breaks an exact 0.5 tie toward the
    inactive class, and re-deriving labels here would silently disagree.
    """
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float('nan')
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'roc_auc': roc_auc,
    }


def summarise_folds(frame, group_columns):
    """Mean and standard deviation of each metric across folds.

    ``ddof=0`` is the population standard deviation, matching how the published
    tables were computed. The sample form would inflate every reported spread by
    a factor of sqrt(5/4), about 1.118, on five folds.
    """
    import numpy as np

    aggregations = {}
    for metric in METRIC_NAMES:
        if metric in frame.columns:
            aggregations[f'{metric}_mean'] = (metric, 'mean')
            aggregations[f'{metric}_std'] = (metric, lambda x: np.std(x, ddof=0))
    return frame.groupby(group_columns, as_index=False, sort=False).agg(**aggregations)
