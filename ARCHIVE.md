# The dataset archive

This repository ships code only. The data and the trained model weights are distributed separately
as a single archive, linked below.

## Obtaining it

**Download: https://gofile.me/7YCnG/JFuqjR9Do**

| | |
|---|---|
| File | `hdac-reproduction-archive.zip` |
| Size | 3,354,658,741 bytes (3.2 GB) |
| SHA-256 | `2dd643a81df49e5cee0229a4a8b90da91fe29f52f92d640adb7f89034c6f910c` |

Check the download before unzipping it:

```bash
sha256sum hdac-reproduction-archive.zip
# expect 2dd643a81df49e5cee0229a4a8b90da91fe29f52f92d640adb7f89034c6f910c
```

If the checksum does not match, or the link does not resolve, contact the authors — see the
Contact section of the README.

## Using it

```bash
unzip hdac-reproduction-archive.zip
scripts/verify_archive.sh hdac-reproduction-archive
cp -r hdac-reproduction-archive/{data,results} /path/to/ml-guide-hdac/
cd /path/to/ml-guide-hdac && scripts/reproduce_all.sh
```

**Verify before you use it.** The archive carries `MANIFEST.sha256`, a checksum for every file.
This matters more than it usually would: the representation caches are *positional*, meaning row 7
of a cache corresponds to row 7 of the matching CSV with nothing inside the file recording that.
A truncated or partial transfer therefore yields confident predictions for the wrong molecules
rather than an error. The pipeline asserts row counts, but the checksums are the real guard.

## Contents

| Path | What it is |
|---|---|
| `data/*.csv` | The splits: train (1811), fit (1629), val (182), test (201), and the pooled source |
| `data/5-fold CV splits/` | The five fixed cross-validation folds |
| `data/virutal_screening_series/` | The 60-compound design series |
| `data/<rep>_fpts/` | Cached representations as `{train,test}_set_fingerprint.npz`, for `<rep>` in ecfp4, rdkit, mordred, padel, base_grover, finetuned_grover |
| `results/final-classification-models/` | The 18 final heads (6 representations × 3 classifiers) |
| `results/final-cv-classifier-result/` | The 90 per-fold heads |
| `results/cv_representation_finetuned_grover/` | Per-fold GROVER encoders and their embeddings |
| `results/finetuned_model_grover/` | GROVER encoder fine-tuned on the full training set |
| `results/pretrained_model_grover/` | The pretrained `grover_large.pt` |
| `results/final-virtual-screening-result/` | Screening embeddings and the published screening output |
| `expected_output/` | Our own reproduction outputs, for file-by-file comparison |

## Two file layouts you may encounter

The cached representations come in two internal forms, and code that assumes one crashes on the
other. `src/hdac/io.py` handles both; use `load_feature_matrix` rather than calling `np.load`
directly.

- GROVER embeddings are true `.npz` archives with the array under the key `fps`.
- The rule-based descriptor caches are a raw `.npy` payload written to a `.npz` filename, which
  `np.load` returns as a plain array.

## What you can and cannot reproduce

**Can:** every published number. Loading the distributed models and scoring them reproduces the
held-out table, the cross-validation table and the screening table. `scripts/reproduce_all.sh`
checks this automatically and fails if anything disagrees.

**Cannot:** the weights themselves, bit for bit. Training depended on library versions, thread
counts and GPU non-determinism that are not pinned. This is why the models are distributed rather
than the recipe alone.

## Licence

The data and trained model weights in this archive are released under
[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
You may share and adapt them, including commercially, provided you give appropriate credit, link
to the licence, and indicate whether changes were made. Please cite the article and the code
deposit, https://doi.org/10.5281/zenodo.21863892.

The analysis code in the repository is separately licensed under the MIT License.

## Third-party components

`results/pretrained_model_grover/grover_large.pt` is the pretrained checkpoint from
[GROVER](https://github.com/tencent-ailab/grover), released by its authors under the MIT License.
It is included so the archive is self-contained; it is not our work, is redistributed under that
licence rather than CC-BY-4.0, and can also be downloaded from that repository.
