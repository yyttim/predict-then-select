# Method

We study when better marker prediction leads to better cell selection, and whether classification supervision or a decision-focused loss changes that relationship. The downstream task is to retain a fixed fraction of cells with high measured marker abundance, using only label-free images to rank them.

## Data and model

The study uses [BSCCMNIST](https://arxiv.org/abs/2402.06191), with four 28 Ã— 28 directional illumination channels: `DPC_Left`, `DPC_Right`, `DPC_Top`, and `DPC_Bottom`. The inputs are LED-array intensity images. The main cohort contains all 9,437 eligible, uniquely type-labeled cells from the batch-0, all-antibody condition on one physical slide. The field-level split gives 6,685 training, 1,394 validation, and 1,358 test cells.

The three classes are lymphocytes, granulocytes, and monocytes. Their fluorescence gates use CD16 and CD45, so classification provides related coarse supervision, not independent clinical labels. Marker targets are the released processed abundance estimates in arbitrary units. Input normalization and marker standardization use the training partition only.

The encoder follows the released [Nazirâ€“Behera CNNâ€“ViT model](https://doi.org/10.1007/978-3-032-31666-0_41). Independent classification and regression heads share its trainable 256-dimensional representation; cross-task gating is disabled. The adapted model has 1,289,713 parameters. Architecture details and attribution are in [vendor/README.md](../vendor/README.md).

CD16 is studied with 25%, 50%, and 100% of the training fields. Subsets are nested within each replicate and identical across methods. CD45 provides a full-cohort second-marker comparison. Two batch-1 slides, containing 13,675 and 13,399 cells, are separate external candidate pools. Their cell types are unavailable and are never inferred for diagnostic stratification.

## Objectives and selection

The predictive objective is

$$L_\lambda=L_{\mathrm{MSE}}+\lambda L_{\mathrm{CE}},\qquad \lambda\in\{0,0.1,0.3,1,3\}.$$

Pure regression is the principal baseline. A second baseline trains a classifier and scores each cell by its class probabilities weighted by training-set class marker means. This baseline also uses marker labels to estimate those means.

For a pool of size $N$, the policy selects $k=\lceil qN\rceil$ cells with the largest scores. Utility is their mean measured marker abundance. The primary fraction is $q=0.10$; the same frozen models are evaluated at 0.05 and 0.20 without retuning. The ranking rule uses predicted scores; measured markers and true types support held-out evaluation.

Two decision-focused objectives act on training minibatches as candidate pools. Let $F_k(v)$ be the sum of the $k$ largest entries of $v$, and $t_k(v)$ their binary selection indicator. With predictions $s$ and standardized targets $y$, [SPO+](https://doi.org/10.1287/mnsc.2020.3922) becomes

$$L_{\mathrm{SPO+}}=\frac{F_k(2s-y)-(2s-y)^\top t_k(y)}{k}.$$

The [central perturbation-gradient loss](https://doi.org/10.52202/079017-2514) approximates negative selection utility:

$$L_{\mathrm{PG},h}=-\frac{F_k(s+hy)-F_k(s-hy)}{2hk},\qquad h\in\{0.1,1\}.$$

PG uses 40 epochs of SPO+ initialization followed by 160 PG epochs. Both losses use deterministic ties and exact top-k operations. Their population guarantees require assumptions that are not established for this dataset; ordinary mean regression can itself support optimal expected selection under the appropriate information model ([Hu et al.](https://doi.org/10.1287/mnsc.2022.4383)). Minibatch training and a full held-out candidate pool also differ in pool size.

## Training and model choice

The [extended configuration](../configs/extended.json) specifies 108 runs: nine settings, three paired replicates, three CD16 training budgets, and one full-data CD45 comparison. All use the full image model, FP32 training, AdamW, and a 200-epoch horizon. At reduced budgets, replicates vary both the sampled training fields and initialization.

Prediction and SPO+ models retain epochs 20, 40, 60, 80, 100, 150, and 200. PG candidates begin at epoch 60, after its initialization stage. Validation metrics are averaged across the three replicates. Each method selects its checkpoint by primary-budget utility; joint models also select the classification weight, and PG selects $h$. A separate predictive selector minimizes validation RMSE over the same MSE/CE candidate family. Metrics are averaged across separately fitted models.

The initial study also compares task weights at epoch 40 to hold stopping time fixed. A bounded, secondary CD45 classifier continuation retained the original epoch-200 model. Protocols record selected states and hashes before evaluation. Model selection uses validation data ([Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)).

## Diagnostics and scope

Matched regression heads are fitted to frozen CE, MSE, and joint encoders from epoch 40. The three encoders use the same readout architecture and training procedure. A positive affine calibration fitted on validation predictions separates scale and offset error from ranking changes.

Achieved enrichment is split into the contribution of selected type proportions and the improvement over random selection within those same types. A separate hindsight decomposition compares the selected set with the best measured-marker sets under fixed and unrestricted type counts. These are retrospective accounting identities, not causal decompositions or estimates of entirely avoidable error.

Paired field bootstrap intervals rerank every resampled pool and use the same field draws for all compared models and replicates. They condition on the frozen models and choices, excluding retraining, model selection, alternative splits, and donor variation. They are unadjusted per-comparison intervals. The validation-budget analysis uses disjoint selection and assessment fields within each repeated split; its overlapping-split quantiles are not confidence intervals from independent datasets.

The within-slide test was inspected in the initial study, so subsequent experiments on it are exploratory. External choices were frozen before external performance was examined. The two external slides differ in acquisition, population eligibility, and measurement conditions; they are not an isolated signal-strength intervention. Compare methods within each slide, rather than comparing absolute marker utility across slides. Donor-level generalization remains untested.

Bibliographic details are in [references.bib](references.bib). Reported summaries and figures are under [results](../results/).
