# Downloads

The [Google Drive folder](https://drive.google.com/drive/folders/1E0WlKM2xjE1Sibodr_qwCAW4Hl-ZGwfm) contains the prepared data, selected checkpoints, and per-cell results.

Start with the [pretrained-model notebook in Colab](https://colab.research.google.com/github/yyttim/predict-then-select/blob/main/notebooks/pretrained_models.ipynb). It downloads only the packages needed for the chosen marker, training fraction, and candidate pool, verifies SHA-256 hashes, and evaluates all three seeds. The default evaluates every cell in the original test pool without retraining. The same [notebook is saved on Drive](https://drive.google.com/file/d/1xwEXAmoQ2PuRQkmvK0wOl_ncYouC3Oj_/view?usp=drivesdk).

## Data

| File | Contents | Size |
| --- | --- | ---: |
| [main-data.zip](https://drive.google.com/file/d/1dH0RB-eDWQFSCgOa3S8R4WzG_1tTpXL4/view?usp=drivesdk) | 9,437 cells: 6,685 train, 1,394 validation, 1,358 test | 30.1 MB |
| [external-data.zip](https://drive.google.com/file/d/19LHIygsEloioxzvESuipC2hqUWRASam8/view?usp=drivesdk) | 27,074 cells across two external slides | 87.0 MB |

These are the complete prepared cohorts used in the reported experiments, with four-channel 28×28 images, metadata, and manifests. No cells were subsampled for this release. They are subsets of the full 412,941-cell BSCCMNIST source release, available under CC0 from [Dryad](https://doi.org/10.5061/dryad.sxksn038s). The [reproduction guide](reproduce.md) describes preparation from the source files.

## Checkpoints and results

| File | Contents | Size |
| --- | --- | ---: |
| [cd16-full.zip](https://drive.google.com/file/d/1syMMVg-7C_ywbbA5DOA-Jl8k-nx7YpEz/view?usp=drivesdk) | CD16, full training data | 86.2 MB |
| [cd16-half.zip](https://drive.google.com/file/d/17_OZfPeQauzT-7cGrr5htcx-3agxFC6z/view?usp=drivesdk) | CD16, 50% training data | 86.1 MB |
| [cd16-quarter.zip](https://drive.google.com/file/d/1ZDi3oMxxXzCk_pw85RtaT8DgMEk8G5YP/view?usp=drivesdk) | CD16, 25% training data | 86.1 MB |
| [cd45-full.zip](https://drive.google.com/file/d/1sASoRNNAI1-aZzYPSzmSVb1ex4US2BnG/view?usp=drivesdk) | CD45, full training data | 86.2 MB |
| [readouts.zip](https://drive.google.com/file/d/19sRHQQwyCXx_qAmaB9YJBKa9cY8d8PTh/view?usp=drivesdk) | Nine matched marker heads and their frozen encoders | 48.9 MB |
| [evaluation-results.zip](https://drive.google.com/file/d/1GftHW2W_sAFEWbVBoSQm_h7ZETJl4Ve6/view?usp=drivesdk) | Per-cell predictions, metrics, conditional intervals, and model-selection records | 20.7 MB |

The primary checkpoint archives contain every validation-selected state needed for the reported policies and three seeds. Each includes `models/index.json`, which maps policies to checkpoints and records calibration and file hashes. Each checkpoint retains its training normalization. The readout archive supplies the matched frozen-encoder comparison.

To use the files locally, extract the required archives into the repository root. They populate `data/prepared/`, `models/`, and `results/reference/`. The notebook uses these released weights; the training scripts create new runs and protocols.

File IDs, sizes, and SHA-256 hashes are listed in [artifacts.json](../artifacts.json). [SHA256SUMS.txt](../SHA256SUMS.txt) is also available in the Drive folder. The notebook checks downloaded archives and their data and model files before evaluation.
