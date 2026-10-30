# Results

The primary comparison selects models by validation utility at a 10% selection budget. Every selected score is evaluated for both mean CD16 and marker-high purity. `selected_for` records the validation criterion; `primary_score` identifies the intended score for that use.

| Location | Contents |
| --- | --- |
| `tables/` | Main test pool: all 1,370 cells |
| `external/tables/` | External pool: all 27,857 cells |
| `stopping/tables/` | Exploratory predictive-loss stopping, main pool |
| `stopping/external/tables/` | The same stopping check, external pool |
| `figures/` in each report | Performance and paired-effect plots, PNG and PDF |

`metrics.csv` gives per-seed outcomes; `summary.csv` gives means and sample SDs. `paired_comparisons.csv` contains fixed-model field-bootstrap intervals, including the comparison between two joint scores at the same checkpoint. `budget_interactions.csv` compares method differences between the two training budgets. `model_choices.csv` and `candidate_pool.csv` record the selected states and each pool's available marker-high cells.

`head_diagnostics.csv` reports RMSE for trained mean heads and Brier/log loss for trained probability heads. These are checkpoint-level diagnostics even when the other joint head supplies the ranking score. The stopping reports also contain `stopping_differences.csv` and `stopping_exchanges.csv`, comparing predictive-loss selection with the original utility-based choices.

Marker values are relative abundance estimates in arbitrary units. Compare methods within a pool; the external threshold prevalence and purity ceiling differ substantially from the main pool. Intervals condition on the fitted models, choices and fixed cutoff, and are not adjusted for multiple comparisons. Interpretation is in [the results discussion](../docs/results.md).
