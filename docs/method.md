# Method

We study which predictive objective supports cell selection when the desired outcome is either high average marker abundance or a high fraction of cells above a marker threshold. Both tasks use the same label-free images and paired CD16 measurements. The comparison asks whether continuous regression, threshold classification, or their combination uses limited training labels most effectively. The distinction between predictive error and decision quality follows [Elmachtoub and Grigas](https://doi.org/10.1287/mnsc.2020.3922).

## Two selection objectives

For a pool of $N$ cells, a score selects the largest $k=\lceil qN\rceil$ values. With measured abundance $y_i$ and a fixed threshold $\tau$, the two utilities are

$$U_{\mathrm{mean}}(S)=\frac{1}{k}\sum_{i\in S}y_i,\qquad U_{\mathrm{purity}}(S)=\frac{1}{k}\sum_{i\in S}\mathbf{1}\{y_i\ge\tau\}.$$

The primary budget is $q=0.10$. Mean abundance values every increase in marker level. The fraction at or above the threshold is precision at $k$, stored as `purity` in the result tables. Under the usual per-cell conditional prediction model, their optimal scores are $E[Y\mid X]$ and $P(Y\ge\tau\mid X)$, respectively. These can induce different rankings. Matching the predictive target to its intended use follows the decision-theoretic treatment of [Gneiting](https://doi.org/10.1198/jasa.2011.r10138); fixed-fraction selection also connects to [accuracy at the top](https://proceedings.neurips.cc/paper/2012/hash/7fe1f8abaad094e0b5cb1b01d712f708-Abstract.html).

## Cohort and label budgets

We use the batch-0, single-antibody CD16 condition from [BSCCM](https://arxiv.org/abs/2402.06191), distributed as BSCCMNIST. The cohort contains all 9,397 cells with finite targets and valid four-channel $28\times28$ images from one physical slide, without cell-type gating. The inputs are the released directional LED-intensity channels `DPC_Left`, `DPC_Right`, `DPC_Top`, and `DPC_Bottom`.

The target is `CD16_single_antibody_model_unmixed`, stored as `target` in the prepared metadata. It is the released two-spectrum CD16/autofluorescence abundance estimate in arbitrary units. We apply no additional clipping or log transformation; the released estimates already include the authors' preprocessing.

Fields of view are assigned to training, validation, and test partitions in proportions 70/15/15 using seed `20261030`, yielding 6,616, 1,411, and 1,370 cells. Field identity includes its physical slide. A fixed quarter of the training fields is chosen with seed `20261031`; its 1,663 cells are shared across all methods and training seeds and included in full-data training.

The threshold $\tau$ is the 90th percentile of targets in that quarter-training subset, using linear quantile interpolation. It is fixed for every model, budget, and evaluation pool. Thus the smaller training budget does not use extra labels to define its target. "Marker-high" is an operational definition, not a clinical positivity threshold. Image normalization and regression-target standardization use only the training cells available at each budget.

## Models and training

The full CNN-ViT encoder is adapted from [Nazir and Behera](https://doi.org/10.1007/978-3-032-31666-0_41). Independent regression and binary-classification heads share the encoder, with no cross-task gate. The classification target is $z_i=\mathbf{1}\{y_i\ge\tau\}$.

| Method | Training objective | Score used for selection |
| --- | --- | --- |
| MSE | Squared error on standardized CD16 | Regression output for either utility |
| BCE | Unweighted binary cross-entropy | Binary logit for either utility |
| Joint | $L_{\mathrm{MSE}}+\lambda L_{\mathrm{BCE}}$ | Regression output for mean abundance; binary logit for purity |

Joint weights are $\lambda\in\{0.3,1,3\}$. Each objective is averaged over the same minibatch. There is no class weighting or class-balanced resampling. The grid has 30 runs: five objective settings, two training-field budgets, and seeds 16, 26, and 36. Methods are paired by initialization and data order within each budget.

Let $B=\lceil N_{\mathrm{full\ train}}/256\rceil$. Every run uses $200B$ optimizer updates, batch size 256, FP32, AdamW, learning rate $3\times10^{-4}$, and weight decay $10^{-4}$. Warmup lasts $5B$ updates. The learning-rate schedule is shared across budgets. Checkpoints occur at $B$ times 5, 10, 20, 40, 60, 80, 100, 150, and 200. Matching updates gives the smaller dataset more passes through its examples while keeping the optimization budget fixed.

All methods receive the same quantitative assay measurements; the training loss determines whether their magnitudes, threshold events, or both contribute to fitting. The joint model tests whether retaining continuous supervision helps the threshold task. [Rauschenberger and Glaab](https://doi.org/10.1080/02664763.2023.2233057) study this information-retention problem by combining separately fitted linear and logistic predictions. Our shared-encoder joint loss is a different implementation of the broader idea.

## Model choice and evaluation

Model selection is performed separately for each training budget and deployment utility. Each method maximizes its corresponding validation utility at $q=0.10$, averaged across the three training seeds. Joint models select both a weight and a checkpoint. Ties favor the earlier checkpoint, then the smaller weight. Selected models remain separate across seeds; predictions are not ensembled.

Both joint heads are evaluated at every chosen joint checkpoint. This permits a comparison of their ranking scores with the model weights held fixed. Comparisons across separately chosen checkpoints are reported as differences between selected policies, rather than as effects of changing only the readout.

After choices are frozen, every retained score is evaluated for both mean abundance and purity at $q\in\{0.05,0.10,0.20\}$. The 5% and 20% budgets use the models selected at 10%. Selection uses predicted scores only. Binary logits are ranked directly to avoid numerical ties from saturated probabilities; probabilities are computed separately for evaluation. RMSE describes the trained regression head; Brier score and log loss assess the trained probability head. These remain head-specific diagnostics when the other joint head supplies the ranking score.

External evaluation uses all 27,857 single-CD16 cells from one batch-1 slide as a separate candidate pool. It applies the frozen models, training normalization, and marker threshold without external fitting or model selection.

Each pool's marker-high prevalence and hindsight purity ceiling, $\min(n_+,k)/k$, provide context for selection performance. Paired differences use 2,000 field-bootstrap draws shared across models and seeds, with selection recomputed in every draw. These intervals condition on the trained models, selected settings, fixed threshold, and data split. They are per-comparison intervals without multiplicity adjustment. Variation across the three training seeds is reported separately.

## Scope

After the primary evaluation, we added an exploratory stopping-rule check using the same trained model bank. MSE models select checkpoints by validation MSE and BCE models by validation log loss; joint models select their weight and checkpoint by the proper loss of the head used for the intended task. MSE is averaged across seeds before selection, rather than averaging RMSE. This analysis has separate choices and outputs; it does not replace the utility-selected results.

BCE is a proper predictive loss for threshold exceedance, not a direct top-k optimization loss. Differences between learned scores do not establish a mechanism involving conditional variance. Keeping the same numerical threshold on the external slide tests transport of the measurement-scale task; acquisition and fluorescence-processing differences can affect its meaning. These two slides do not establish donor-level generalization.

Full bibliographic details are in [references.bib](references.bib).
