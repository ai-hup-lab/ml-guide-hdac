import numpy as np
import importlib
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from typing import List, Optional, Dict, Any

def _process_molecules(data: List[str], process_func, verbose: bool = True) -> np.ndarray:
    """Generic function to process molecules and generate fingerprints."""
    result_fpts = []
    iterator = tqdm(enumerate(data), total=len(data), desc='Generating fingerprints', disable=not verbose)
    for idx, smiles in iterator:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(f"Failed to parse SMILES at index {idx}: {smiles}")
                continue
            fpts = process_func(mol)
            result_fpts.append(fpts)
        except Exception as e:
            if verbose:
                print(f"An exception occurred at index {idx}: {str(e)}")
    return np.array(result_fpts)

def _validate_bits(bits: int) -> None:
    if bits not in [1024, 2048]:
        raise ValueError("Invalid value for bits. Must be either 1024 or 2048.")

def _process_molecules_with_transformer(
    data: List[str],
    transformer,
    verbose: bool = True,
) -> np.ndarray:
    """Process molecules then apply a scikit-fingerprints transformer."""
    mols = []
    iterator = tqdm(enumerate(data), total=len(data), desc='Preparing molecules', disable=not verbose)
    for idx, smiles in iterator:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(f"Failed to parse SMILES at index {idx}: {smiles}")
                continue
            mols.append(mol)
        except Exception as e:
            if verbose:
                print(f"An exception occurred at index {idx}: {str(e)}")

    if len(mols) == 0:
        return np.array([])

    result = transformer.transform(mols)
    if hasattr(result, "toarray"):
        result = result.toarray()
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(result)

def gen_maccs_fpts(data: List[str], verbose: bool = True) -> np.ndarray:
    return _process_molecules(data, lambda mol: np.array(MACCSkeys.GenMACCSKeys(mol)), verbose=verbose)

def gen_ecfp4_fpts(data: List[str], bits: int = 2048, verbose: bool = True) -> np.ndarray:
    _validate_bits(bits)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    return _process_molecules(data, lambda mol: np.array(mfpgen.GetFingerprint(mol)), verbose=verbose)

def gen_ecfp6_fpts(data: List[str], bits: int = 2048, verbose: bool = True) -> np.ndarray:
    _validate_bits(bits)
    return _process_molecules(data, 
    lambda mol: np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 3, bits)), verbose=verbose)

def gen_rdkit_fpts(data: List[str], bits: int = 2048, verbose: bool = True) -> np.ndarray:
    _validate_bits(bits)
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=bits)
    return _process_molecules(data, lambda mol: np.array(rdkit_gen.GetFingerprint(mol)), verbose=verbose)

def gen_mordred_fpts(data: List[str], verbose: bool = True) -> np.ndarray:
    try:
        from skfp.fingerprints import MordredFingerprint
    except Exception as exc:
        raise ImportError(
            "MordredFingerprint backend is unavailable. "
            "This is often caused by mordred/networkx incompatibility in Python 3.12."
        ) from exc
    mordred_gen = MordredFingerprint()
    return _process_molecules_with_transformer(data, mordred_gen, verbose=verbose)

def gen_rdkit2d_fpts(data: List[str], verbose: bool = True) -> np.ndarray:
    try:
        from skfp.fingerprints import RDKit2DDescriptorsFingerprint
    except Exception as exc:
        raise ImportError(
            "RDKit2DDescriptorsFingerprint backend is unavailable. "
            "Please ensure scikit-fingerprints and its dependencies are installed correctly."
        ) from exc
    rdkit2d_gen = RDKit2DDescriptorsFingerprint()
    return _process_molecules_with_transformer(data, rdkit2d_gen, verbose=verbose)

def gen_pubchem_fpts(data: List[str], verbose: bool = True) -> np.ndarray:
    try:
        from skfp.fingerprints import PubChemFingerprint
    except Exception as exc:
        raise ImportError(
            "PubChemFingerprint backend is unavailable. "
            "Please ensure scikit-fingerprints and its dependencies are installed correctly."
        ) from exc
    pubchem_gen = PubChemFingerprint()
    return _process_molecules_with_transformer(data, pubchem_gen, verbose=verbose)

def gen_padel_fpts(
    data: List[str],
    fp_name: str = "PubchemFP",
    fp_params: Optional[Dict[str, Any]] = {"size": 2048, "searchDepth": 8},
    ignore_3D: bool = False,
    njobs: int = 1,
    chunksize: int = 100,
    verbose: bool = True,
) -> np.ndarray:
    """Generate PaDEL fingerprints using PaDEL_pywrapper.

    Args:
        data: List of SMILES strings.
        fp_name: Fingerprint class name from `PaDEL_pywrapper.descriptor`
            (e.g., `GraphOnlyFP`, `PubchemFP`, `SubstructureFPCount`).
        fp_params: Optional keyword arguments used to instantiate the
            fingerprint class (e.g., `{"size": 2048, "searchDepth": 8}`).
        ignore_3D: Whether to skip 3D-only descriptors/fingerprints.
        njobs: Number of parallel processes used by PaDEL.
        chunksize: Molecules per process chunk.
        verbose: Whether to print progress/errors.

    Returns:
        np.ndarray: Array of shape (n_valid_molecules, n_features).
    """
    try:
        padel_module = importlib.import_module("PaDEL_pywrapper")
        padel_descriptor = importlib.import_module("PaDEL_pywrapper.descriptor")
        PaDEL = getattr(padel_module, "PaDEL")
    except ImportError as exc:
        raise ImportError(
            "PaDEL_pywrapper is required for gen_padel_fpts. "
            "Install it with: pip install padel-pywrapper"
        ) from exc

    mols = []
    for idx, smiles in tqdm(
        enumerate(data),
        total=len(data),
        desc="Preparing molecules for PaDEL",
        disable=not verbose,
    ):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if verbose:
                print(f"Failed to parse SMILES at index {idx}: {smiles}")
            continue
        mols.append(mol)

    if len(mols) == 0:
        return np.array([])

    descriptor_cls = getattr(padel_descriptor, fp_name, None)
    if descriptor_cls is None:
        raise ValueError(
            f"Unknown PaDEL descriptor/fingerprint: {fp_name}. "
            "Check names in PaDEL_pywrapper.descriptor"
        )

    descriptor = descriptor_cls(**fp_params) if fp_params else descriptor_cls
    padel = PaDEL([descriptor], ignore_3D=ignore_3D)
    fingerprints = padel.calculate(
        mols,
        show_banner=verbose,
        njobs=njobs,
        chunksize=chunksize,
    )
    return np.asarray(fingerprints)