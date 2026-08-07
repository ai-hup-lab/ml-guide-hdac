"""Reading the inputs, with the file-format quirks handled in one place.

Two things here have bitten this pipeline before and are easy to get wrong
independently in each script, which is why they live together:

* The labelled CSVs do not all use the same delimiter.
* The feature caches do not all use the same internal layout.
"""
import os

import numpy as np
import pandas as pd


def read_labelled_csv(path):
    """Return (dataframe, smiles_column, labels_column).

    ``sep=None`` lets the csv sniffer choose the delimiter. Comma- and
    semicolon-separated splits both occur in this dataset, and a plain
    ``read_csv`` turns a semicolon file into a single column named
    ``smiles;labels`` -- which surfaces as a confusing "column not found" error
    rather than as a parsing failure.
    """
    frame = pd.read_csv(path, sep=None, engine='python')
    frame.columns = frame.columns.str.strip()
    smiles_column = next((c for c in frame.columns if c.lower() == 'smiles'), None)
    labels_column = next((c for c in frame.columns if c.lower().startswith('label')), None)
    if smiles_column is None:
        raise ValueError(f"{path}: no 'smiles' column. Found: {frame.columns.tolist()}")
    if labels_column is None:
        raise ValueError(f"{path}: no 'labels' column. Found: {frame.columns.tolist()}")
    return frame, smiles_column, labels_column


def read_smiles_and_labels(path):
    """Return (smiles list, integer label array)."""
    frame, smiles_column, labels_column = read_labelled_csv(path)
    smiles = frame[smiles_column].astype(str).str.strip().tolist()
    labels = frame[labels_column].astype(int).values
    return smiles, labels


def read_smiles(path):
    """Return the SMILES column of a CSV that need not carry labels."""
    frame = pd.read_csv(path, sep=None, engine='python')
    frame.columns = frame.columns.str.strip()
    smiles_column = next((c for c in frame.columns if c.lower() == 'smiles'), None)
    if smiles_column is None:
        raise ValueError(f"{path}: no 'smiles' column. Found: {frame.columns.tolist()}")
    return frame, smiles_column, frame[smiles_column].astype(str).str.strip().tolist()


def load_feature_matrix(path):
    """Load a feature cache, tolerating both layouts these files come in.

    GROVER embeddings are true ``.npz`` archives keyed ``fps``. The rule-based
    descriptor caches are a raw ``.npy`` payload written to a ``.npz`` filename,
    which ``np.load`` returns as a plain array. Assuming either layout crashes on
    the other.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Feature file not found: {path}\n'
            'GROVER embeddings: scripts/gen_grover_fingerprint.sh <split> <base|finetuned>\n'
            'Rule-based descriptors: scripts/prepare_features.sh\n'
            'Or use the caches in the dataset archive supplied by the authors.'
        )
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if 'fps' in loaded.files:
            return loaded['fps']
        if len(loaded.files) == 1:
            return loaded[loaded.files[0]]
        raise ValueError(f"{path} holds multiple arrays and no 'fps' key: {loaded.files}")
    return loaded


def feature_path(fpts_dir, representation, split):
    """The canonical location of a cached representation."""
    return os.path.join(fpts_dir, f'{representation}_fpts', f'{split}_set_fingerprint.npz')


def assert_rows_match(features, expected, context):
    """Fail loudly when a cache and a molecule list disagree on length.

    The caches are positional: row 7 corresponds to row 7 of the CSV and nothing
    inside the file records that. A mismatch therefore scores the wrong molecules
    with entirely plausible-looking output, so this check is the only thing
    standing between a stale cache and a silently wrong result.
    """
    if len(features) != expected:
        raise ValueError(
            f'{context}: {len(features)} feature rows for {expected} molecules. '
            'These caches are positional, so this would associate features with the '
            'wrong molecules. Regenerate them for this exact CSV.'
        )
