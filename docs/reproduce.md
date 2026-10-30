# Reproduction

Use Python 3.11 or newer and install a PyTorch build for your GPU. From the repository root:

```text
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Linux/macOS), then install the package:

```text
python -m pip install -e ".[data,dev]" -c requirements-reproduce.txt
python -m pytest -q
```

## Data

Download the two full BSCCMNIST archive chunks from [Dryad](https://doi.org/10.5061/dryad.sxksn038s), version May 6, 2025, and place them in `data/archives/`:

- `BSCCMNIST.tar.gz_chunk00000.bin`
- `BSCCMNIST.tar.gz_chunk00001.bin`

Preparation verifies the official archive hashes, extracts the required metadata and CD16 images, and retains their native four-channel 28×28 resolution.

```text
python scripts/prepare_data.py --source data/raw/BSCCMNIST --archives data/archives --output data/prepared/cd16_selection
```

The prepared arrays contain every eligible single-CD16 cell from batch 0 and batch 1. Batch 0 supplies the grouped training, validation and test partitions; batch 1 is the external pool. The manifest records identities, splits, the fixed marker threshold and file hashes. Existing locked prepared data are not overwritten.

## Training

The declared grid contains 30 runs. Every run uses the full CNN–ViT and the same update budget; the 25% condition is a training-data experiment, not a smaller network.

```text
python scripts/run_study.py --data data/prepared/cd16_selection --output results/study
```

The command saves `study.json`, training histories, checkpoint weights and validation predictions. Completed runs are reused only when their configuration, training source and data match. Use a new output directory after changing the study. Optional `--seed`, `--method` and `--fraction` arguments run a declared portion of the grid. Do not launch two workers for the same run.

## Model selection and evaluation

First select all models using validation data. This step requires the complete grid and writes `selection.json` before any held-out evaluation.

```text
python scripts/evaluate_study.py freeze --data data/prepared/cd16_selection --study results/study
python scripts/evaluate_study.py test --data data/prepared/cd16_selection --study results/study
python scripts/evaluate_study.py external --data data/prepared/cd16_selection --study results/study
python scripts/summarize_study.py --study results/study
python scripts/summarize_study.py --study results/study --pool external
```

Test and external predictions remain in separate directories. The last two commands create the main tables and figures under `results/` and the external report under `results/external/`. Both pools use the original training normalization, threshold and validation choices.

The default is CUDA. Training and evaluation accept `--device cpu`; the full grid is intended for a GPU. The recorded implementation uses FP32 without TF32. Small numerical differences across hardware or library versions can change nearly tied selections.

## Stopping-rule sensitivity

This exploratory check reuses the trained bank and selects states by validation predictive loss. It writes a separate analysis under `results/study/stopping/` and leaves the primary choices intact.

```text
python scripts/stopping_sensitivity.py --study results/study
python scripts/evaluate_study.py test --data data/prepared/cd16_selection --study results/study/stopping
python scripts/evaluate_study.py external --data data/prepared/cd16_selection --study results/study/stopping
python scripts/summarize_study.py --study results/study/stopping --output results/stopping
python scripts/summarize_study.py --study results/study/stopping --pool external --output results/stopping/external
python scripts/compare_stopping.py --study results/study
python scripts/compare_stopping.py --study results/study --pool external
python scripts/error_accounting.py --study results/study
python scripts/plot_decision_examples.py
```

The last two commands attribute loss and utility changes to retained and exchanged cells, then plot the paired external examples. They use saved predictions and tables without fitting new models.

`scripts/verify_results.py --study results/study --pool test` independently reconstructs each reported outcome from saved predictions and checks its model/data bindings. The same command accepts `--pool external` or the stopping-study directory.
