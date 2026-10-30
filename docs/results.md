# Results

Regression is a strong baseline for both uses of CD16 predictions. Threshold classification does not establish a purity advantage, while its lower mean-abundance utility is consistent across the two training budgets and evaluation pools. The comparison of joint-model scores and the stopping-rule check show why probability prediction and cell selection need separate evaluation.

The primary study uses 30 runs with matched update budgets. Models are selected on validation data separately for mean abundance and marker-high purity. Results below use the top 10% budget, with three paired training seeds. The main test pool contains all 1,370 held-out cells; the external pool contains all 27,857 cells from the second slide. See the [method](method.md) for the fixed threshold, selection rules, and uncertainty calculation.

## Primary comparison

This table uses full-data models **selected for purity** on the main test pool. The joint model ranks cells with its probability head. Values are mean ± sample SD across seeds; the mean-abundance column evaluates those same selected policies.

| Model | Mean selected CD16 (a.u.) | Marker-high purity (%) |
| --- | ---: | ---: |
| MSE regression | 446.93 ± 5.32 | 50.61 ± 2.23 |
| BCE classification | 428.24 ± 6.42 | 46.96 ± 1.84 |
| Joint training | 432.01 ± 8.45 | 49.39 ± 1.11 |

The full-data BCE-minus-MSE purity difference is −3.65 percentage points, with a conditional 95% field-bootstrap interval of [−7.01, 0.23]. The joint-minus-BCE difference is +2.43 points [−1.22, 5.51]. None of the primary between-method purity intervals excludes zero at either training budget in either pool. These comparisons establish neither a gain from threshold classification nor equivalence between methods. [Main results](../results/tables/summary.csv), [paired comparisons](../results/tables/paired_comparisons.csv).

For models selected for mean abundance, MSE exceeds BCE by 18.69 units with all training fields and 19.61 units with one quarter of them. The corresponding external advantages are 3.93 and 8.00 units. All four conditional intervals exclude zero. Joint training does not establish a consistent improvement over MSE. [External comparisons](../results/external/tables/paired_comparisons.csv).

## Two scores from the same model

The full-data joint checkpoint selected for purity provides a modest external trade-off. Using its probability score instead of its mean score increases purity by **0.167 percentage points** [0.024, 0.297], while reducing mean selected CD16 by **5.45 units** [4.32, 6.57]. The probability score selects 6, 2, and 6 additional marker-high cells in the three runs; mean abundance decreases in all three. The weights are identical within each comparison. This is evidence about the choice of score at that checkpoint, not an advantage of joint training over regression. The corresponding main-pool head comparisons remain inconclusive. [Paired head comparisons](../results/external/tables/paired_comparisons.csv).

Absolute purity cannot be compared directly across these pools. Only 230 external cells exceed the frozen threshold, a prevalence of 0.826%. Selecting 10% of that pool therefore has a hindsight purity ceiling of 8.256%, even with perfect knowledge of every outcome. Acquisition and measurement-scale changes can affect this threshold-defined task. [Candidate-pool statistics](../results/external/tables/candidate_pool.csv).

## Stopping by predictive loss

After the primary evaluation, an exploratory check selected from the same checkpoint bank using validation MSE or log loss. Joint models used the proper loss of the relevant head. There was no retraining, and the primary choices were preserved.

For quarter-data BCE, this changes the selected checkpoint from step 5,200 to step 260:

| Pool | Log loss, utility → predictive stopping | Purity (%), utility → predictive stopping |
| --- | ---: | ---: |
| Main test | 0.782 → 0.278 | 42.09 → 41.12 |
| External | 0.437 → 0.166 | 6.509 → 6.150 |

Better probability loss does not produce better selection here. The main-pool purity change is −0.97 points [−4.94, 2.84]; externally it is −0.359 points [−0.651, −0.072]. [Main stopping differences](../results/stopping/tables/stopping_differences.csv), [external stopping differences](../results/stopping/external/tables/stopping_differences.csv).

Under this alternative rule, joint-minus-BCE purity differences on the main pool are +4.62 points [0.48, 8.27] at the quarter budget and +4.14 points [0.24, 6.91] at the full budget. This pattern does not extend to the external pool. These exploratory comparisons do not replace the inconclusive primary joint-versus-BCE result. [Main](../results/stopping/tables/paired_comparisons.csv) and [external](../results/stopping/external/tables/paired_comparisons.csv) sensitivity tables.

All intervals condition on the fitted models, selected settings, cutoff, and observed fields; they are not adjusted for multiple comparisons. Three training seeds and two physical slides do not establish donor-level generalization or a mechanism involving conditional marker variance.
