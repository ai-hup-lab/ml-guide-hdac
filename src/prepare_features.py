"""
Build the rule-based descriptor caches from SMILES.

Writes <fpts_dir>/<rep>_fpts/<split>_set_fingerprint.npz for each requested
representation, which is the layout the training and evaluation scripts expect.

  ecfp4     2048-bit circular fingerprint   RDKit only
  rdkit     2048-bit RDKit fingerprint      RDKit only
  mordred   1613 descriptors                needs scikit-fingerprints
  padel     881-bit PubChem fingerprint     needs padel-pywrapper and a Java runtime

The GROVER embeddings are not built here -- they need a checkpoint and a GPU pass.
Use scripts/gen_grover_fingerprint.sh for those.

If you received the dataset archive from the authors, these caches are already in
it and this script is only needed to featurise new molecules.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hdac.features import (  # noqa: E402
    gen_ecfp4_fpts,
    gen_mordred_fpts,
    gen_padel_fpts,
    gen_rdkit_fpts,
)
from hdac.io import read_smiles  # noqa: E402

# representation -> (generator, takes a `bits` argument)
GENERATORS = {
    'ecfp4': (gen_ecfp4_fpts, True),
    'rdkit': (gen_rdkit_fpts, True),
    'mordred': (gen_mordred_fpts, False),
    'padel': (gen_padel_fpts, False),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True, help='CSV with a smiles column')
    p.add_argument('--split', required=True,
                   help="Split name used in the output filename, e.g. 'train' or 'test'")
    p.add_argument('--fpts_dir', required=True, help='Root directory for <rep>_fpts/')
    p.add_argument('--representations', default='ecfp4,rdkit',
                   help=f'Comma-separated subset of: {", ".join(GENERATORS)}')
    p.add_argument('--bit_size', type=int, default=2048,
                   help='Bit size for ECFP4 and RDKit fingerprints')
    p.add_argument('--overwrite', action='store_true',
                   help='Rebuild caches that already exist')
    return p.parse_args()


def main():
    params = parse_args()
    requested = [r.strip() for r in params.representations.split(',') if r.strip()]
    unknown = [r for r in requested if r not in GENERATORS]
    if unknown:
        raise SystemExit(f'Unknown representation(s): {", ".join(unknown)}. '
                         f'Choose from {", ".join(GENERATORS)}')

    _, _, smiles = read_smiles(params.csv)
    print(f'{len(smiles)} molecules from {params.csv}')

    for representation in requested:
        generator, takes_bits = GENERATORS[representation]
        out_dir = os.path.join(params.fpts_dir, f'{representation}_fpts')
        out_path = os.path.join(out_dir, f'{params.split}_set_fingerprint.npz')
        if os.path.exists(out_path) and not params.overwrite:
            print(f'  {representation}: {out_path} exists, skipping (--overwrite to rebuild)')
            continue

        print(f'  {representation}: generating...')
        features = np.asarray(
            generator(smiles, bits=params.bit_size, verbose=True) if takes_bits
            else generator(smiles, verbose=True)
        )
        if features.shape[0] != len(smiles):
            raise ValueError(
                f'{representation}: generated {features.shape[0]} rows for {len(smiles)} '
                'molecules. These caches are positional, so a mismatch would silently '
                'associate features with the wrong molecules.'
            )

        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(out_path, fps=features)
        print(f'  {representation}: {features.shape} -> {out_path}')


if __name__ == '__main__':
    main()
