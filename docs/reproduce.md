# Reproducing the study

Run commands from the repository root in an activated Python environment. Python 3.11 or newer is required. Install a CUDA-enabled PyTorch build for the supplied GPU configurations, then install the project with the recorded dependency versions:

```text
python -m pip install -e ".[data,dev]" -c requirements-reproduce.txt
python -m pytest -q
```

The reported experiments used PyTorch 2.7, CUDA 12.8, and an RTX 5090. Numerical results can differ across library versions and hardware. To use a CPU, create separate training configurations with `device` set to `cpu` and add `--device cpu` to the evaluation and readout commands. The full grid is intended for GPU execution.

The repository includes published [tables](../results/tables/) and [figures](../results/figures/). Prepared data, selected checkpoints, and per-cell predictions are available from [Downloads](downloads.md). Use the [pretrained-model notebook](../notebooks/pretrained_models.ipynb) to evaluate those checkpoints. The commands below generate new runs and protocols; keep their output directories separate from the downloaded reference results.

## Prepare the data

To use the exact prepared cohorts, download both data archives from [Downloads](downloads.md) and extract them at the repository root. This supplies every cell used in the study, with the original split and metadata. To rebuild these cohorts from the source release, follow the steps below.

Download the two full **BSCCMNIST** chunks from [Dryad](https://doi.org/10.5061/dryad.sxksn038s), version May 6, 2025. They are `BSCCMNIST.tar.gz_chunk00000.bin` and `BSCCMNIST.tar.gz_chunk00001.bin`, totaling approximately 6.23 GB compressed. On the website, select both files and download the selected-files ZIP. Save it as `./data/bsccmnist.zip`.

```text
python scripts/download_data.py --data-root ./data --from-zip ./data/bsccmnist.zip
python scripts/prepare_data.py --data-root ./data
python scripts/prepare_external_data.py --data-root ./data --main-data ./data/prepared/bsccmnist_cd16_v1
```

The downloader verifies archive checksums and keeps the original chunks. Preparation writes images, metadata, and a split manifest to `./data/prepared/bsccmnist_cd16_v1`; the external slides go to `./data/prepared/bsccmnist_batch1_all_v1`. The full archives must remain available for external-data preparation. API downloading is also supported through the `DRYAD_TOKEN` environment variable:

```text
python scripts/download_data.py --data-root ./data
```

## Initial predictive-objective comparison

This 18-run comparison supplies the frozen encoders and candidate bank used by the readout and validation-budget analyses.

```text
python scripts/train.py --config configs/core.json --data ./data/prepared/bsccmnist_cd16_v1 --output results/runs/core
python scripts/evaluate.py freeze --data ./data/prepared/bsccmnist_cd16_v1 --runs results/runs/core --experiment configs/core.json --protocol results/core_protocol.json
python scripts/validation_stability.py --data ./data/prepared/bsccmnist_cd16_v1 --runs results/runs/core --protocol results/core_protocol.json --output results/validation_stability.json
python scripts/evaluate.py test --data ./data/prepared/bsccmnist_cd16_v1 --runs results/runs/core --protocol results/core_protocol.json --output results/core
```

## Extended objective and training-budget study

The 108-run grid includes the CD16 training budgets, CD45, SPO+, and central PG. It creates a separate set of run records. Freezing checks that the declared grid is complete and selects models from validation predictions before opening either evaluation cohort.

```text
python scripts/train.py --config configs/extended.json --data ./data/prepared/bsccmnist_cd16_v1 --output results/runs/extended
python scripts/extended_study.py freeze --data ./data/prepared/bsccmnist_cd16_v1 --external ./data/prepared/bsccmnist_batch1_all_v1 --runs results/runs/extended --spec configs/extended.json --protocol results/extended_protocol.json
python scripts/extended_study.py evaluate --data ./data/prepared/bsccmnist_cd16_v1 --external ./data/prepared/bsccmnist_batch1_all_v1 --runs results/runs/extended --protocol results/extended_protocol.json --output results/extended
python scripts/extended_study.py uncertainty --data ./data/prepared/bsccmnist_cd16_v1 --external ./data/prepared/bsccmnist_batch1_all_v1 --protocol results/extended_protocol.json --output results/extended --bootstrap 2000
python scripts/verify_extended_results.py --data ./data/prepared/bsccmnist_cd16_v1 --external ./data/prepared/bsccmnist_batch1_all_v1 --runs results/runs/extended --spec configs/extended.json --protocol results/extended_protocol.json --results results/extended
python scripts/summarize_extended_results.py --protocol results/extended_protocol.json --results results/extended
```

Fresh policy tables are written to `results/extended/policy_summary.csv`, `paired_summary.csv`, and `metrics.csv`. The tracked copies in `results/tables/` remain the published results.

## Readout and validation-budget analyses

The readout experiment fits identical marker heads to the fixed epoch-40 encoders from the initial comparison. Its training stage also freezes the head-selection protocol.

```text
python scripts/readout_probe.py train --data ./data/prepared/bsccmnist_cd16_v1 --runs results/runs/core --core-protocol results/core_protocol.json --output results/readout_probe
python scripts/readout_probe.py test --data ./data/prepared/bsccmnist_cd16_v1 --runs results/runs/core --core-protocol results/core_protocol.json --output results/readout_probe
python scripts/validation_budget.py --data ./data/prepared/bsccmnist_cd16_v1 --runs results/runs/core --protocol results/core_protocol.json --output results/validation_budget.json.gz --summary-output results/validation_budget_summary.json --repetitions 500
python scripts/plot_extended_results.py
```

The plotting script reads completed main, readout, and validation-budget results and writes fresh figures to `results/figures_extended/`. It does not fit or choose models. The analytical surrogate example is independent of the microscopy experiments:

```text
python scripts/verify_surrogate_example.py --output results/surrogate_example --plot
```

## Run directories

Each protocol records the data, code, and model versions used for evaluation.
Completed runs are reused only when those records match. After changing an
experiment, use a new output directory and generate a new protocol. Partial
runs and frozen results are protected against replacement.

The published tables come from the original execution. Fresh runs generate
their own manifests and hashes; numerical differences across hardware and
library versions are possible.
