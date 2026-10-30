# Results

These tables and figures summarize the completed experiments. Marker values are released unmixed abundance estimates in arbitrary units. Compare methods within the same marker and candidate pool.

| File in `tables/` | Contents |
| --- | --- |
| `metrics.csv` | Per-run prediction and selection metrics |
| `policy_summary.csv` | Means and standard deviations across paired runs |
| `paired_summary.csv` | Utility differences and conditional 95% field-bootstrap intervals |
| `readout_metrics.csv` | Original scores and matched marker readouts on frozen encoders |
| `readout_comparisons.csv` | Paired readout differences and intervals |
| `validation_budget.csv` | Selection and assessment results across 500 validation splits |
| `validation_comparisons.csv` | Utility-based versus RMSE-based model selection |
| `classifier_continuation.csv` | CD45 classifier validation results through epoch 400 |

The main grid contains 108 runs: three seeds, nine objective settings, and four marker/training-size conditions. Models are selected at a 10% validation budget and evaluated at 5%, 10%, and 20% without retuning. `within_slide_test` is the original held-out-field pool; the two `batch_1` pools are separate external slides.

Field-bootstrap intervals use common draws across paired models and hold their parameters and selected settings fixed. Quantiles from the overlapping validation splits are descriptive ranges, not confidence intervals. Missing classification and type-based metrics on external slides reflect unavailable type labels.

PNG and PDF figures are in `figures/`. Full reruns write checkpoints, predictions, protocols, and verification records to the generated result directories listed in the [reproduction guide](../docs/reproduce.md).
