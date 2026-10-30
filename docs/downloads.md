# Downloads

The [release folder on Google Drive](https://drive.google.com/drive/folders/1A3sDK0kEPocnoLUpg5mFG3833X6mmXEZ) contains the complete prepared cohorts, selected checkpoints, and reference predictions.

Start with the [Colab notebook](https://colab.research.google.com/github/yyttim/predict-then-select/blob/main/notebooks/pretrained_models.ipynb). It downloads and verifies the release, evaluates all three seeds on the complete selected pool, and compares its results with the saved reference. A [copy is also on Drive](https://drive.google.com/file/d/1T8haOk2WQxb-4HvJY0jJgpssHvKUzmG_/view?usp=drivesdk).

| File | Contents | Size |
| --- | --- | ---: |
| [prepared-data.tar.xz](https://drive.google.com/file/d/1NfFsoUeoDD6F-dpK-uySMBHXYRkaF3S-/view?usp=drivesdk) | All 37,254 single-CD16 cells, split metadata and fixed cutoff | 68.1 MB |
| [seed16.zip](https://drive.google.com/file/d/1Z2pqshPMeXlH9G8hxCO0CEb1OPt9v3lG/view?usp=drivesdk) | Selected states for seed 16 | 57.3 MB |
| [seed26.zip](https://drive.google.com/file/d/1KSMtZB6r7EGsCOl5X6Hj3drlhibENgi0/view?usp=drivesdk) | Selected states for seed 26 | 57.3 MB |
| [seed36.zip](https://drive.google.com/file/d/1-O5ZLtcVI9E2qKuMiMhUOjkKwy9VOkS-/view?usp=drivesdk) | Selected states for seed 36 | 57.3 MB |
| [reference-results.zip](https://drive.google.com/file/d/130_1DAII0U6Mi_R0u-Tl-UL-YjHO3QO9/view?usp=drivesdk) | Per-cell predictions, metrics and frozen choices for both stopping rules | 13.2 MB |

Data retain the native four-channel 28×28 BSCCMNIST images with no additional resizing or cell subsampling. The main cohort has 6,616 training cells, 1,411 validation cells and 1,370 test cells; the external pool has 27,857 cells. Other antibody conditions are available from the full [Dryad source release](https://doi.org/10.5061/dryad.sxksn038s).

Each model ZIP contains 12 checkpoints plus shared metadata. Together they cover every selected state in the primary study and the separate predictive-loss stopping check. The notebook uses the primary choices. The archives extract into `data/`, `models/`, and `reference/`; the notebook handles the XZ and ZIP formats with Python's standard library.

For local inference after extracting the data and model archives at the repository root:

```text
python scripts/evaluate_study.py test --data data --study models
python scripts/evaluate_study.py external --data data --study models
```

The saved choices are already fixed. To train a new model bank and select new states, follow the [reproduction guide](reproduce.md).

File IDs, byte sizes, and hashes are recorded in [artifacts.json](../artifacts.json). [SHA256SUMS.txt](../SHA256SUMS.txt) also accompanies the Drive files. The notebook verifies the downloads, prepared data, and checkpoint bindings before inference.
