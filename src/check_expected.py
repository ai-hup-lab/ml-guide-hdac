"""
Compare a reproduction run against the published numbers in expected_results/.

Exits non-zero if anything disagrees beyond the stated tolerance, so it can gate
a script or a CI job.

Tolerances are not uniform, because the three outputs are not equally
constrained:

  screening    exact -- the same models scoring the same molecules
  cross-val    exact -- likewise, and the fold definitions are fixed
  held-out     1e-06 -- the expected file is written at six decimals, so
                        agreement cannot be asserted more tightly than that

A difference larger than these means the pipeline is not reproducing the paper,
not that floating point drifted.
"""
import argparse
import os
import sys

import pandas as pd

TOLERANCES = {'heldout_metrics.csv': 1e-6, 'cv_summary.csv': 0.0, 'screening_result.csv': 0.0}
KEYS = {
    'heldout_metrics.csv': ['representation', 'model'],
    'cv_summary.csv': ['representation', 'model'],
    'screening_result.csv': None,   # positional: same order, same molecules
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--expected_dir', required=True)
    p.add_argument('--actual_dir', required=True,
                   help='Directory holding the reproduction outputs')
    return p.parse_args()


def read(path):
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [c.strip() for c in frame.columns]
    for column in frame.columns:
        if frame[column].dtype == object or str(frame[column].dtype) == 'str':
            frame[column] = frame[column].astype(str).str.strip()
    return frame


def compare(name, expected_path, actual_path):
    tolerance = TOLERANCES[name]
    expected, actual = read(expected_path), read(actual_path)

    missing = [c for c in expected.columns if c not in actual.columns]
    if missing:
        return [f'{name}: missing column(s) {missing}']
    if len(expected) != len(actual):
        return [f'{name}: {len(actual)} rows, expected {len(expected)}']

    keys = KEYS[name]
    if keys:
        merged = actual.merge(expected, on=keys, how='outer',
                              suffixes=('_actual', '_expected'), indicator=True)
        if (merged['_merge'] != 'both').any():
            return [f'{name}: row keys do not line up']
        pairs = [(c, f'{c}_actual', f'{c}_expected')
                 for c in expected.columns
                 if c not in keys and pd.api.types.is_numeric_dtype(expected[c])]
        frame = merged
    else:
        pairs = [(c, c, c) for c in expected.columns
                 if pd.api.types.is_numeric_dtype(expected[c])]
        frame = None

    problems = []
    for label, left, right in pairs:
        if frame is not None:
            difference = (frame[left] - frame[right]).abs().max()
        else:
            difference = (actual[label] - expected[label]).abs().max()
        if difference > tolerance:
            problems.append(f'{name}: {label} differs by {difference:.3e} '
                            f'(tolerance {tolerance:.0e})')
    return problems


def main():
    params = parse_args()
    problems, checked = [], 0
    for name in TOLERANCES:
        expected_path = os.path.join(params.expected_dir, name)
        actual_path = os.path.join(params.actual_dir, name)
        if not os.path.isfile(expected_path):
            problems.append(f'{name}: no expected file at {expected_path}')
            continue
        if not os.path.isfile(actual_path):
            problems.append(f'{name}: not produced -- expected at {actual_path}')
            continue
        found = compare(name, expected_path, actual_path)
        checked += 1
        print(f'  {name:24s} {"OK" if not found else "MISMATCH"}'
              f'   (tolerance {TOLERANCES[name]:.0e})')
        problems.extend(found)

    print()
    if problems:
        print(f'{len(problems)} problem(s):')
        for problem in problems:
            print(f'  - {problem}')
        sys.exit(1)
    print(f'All {checked} output(s) match the published results.')


if __name__ == '__main__':
    main()
