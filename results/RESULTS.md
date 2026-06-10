# Results summary

PBMC CITE-seq (Hao 2021), 24,735 cells (sqrt-proportional stratified subsample),
30 `celltype.l2` cell types, shared gene universe = 2,081 (2,000 HVGs + 133
protein-encoding marker genes). All four methods scored on the identical
rank-transformed/z-scored matrix; ground truth = top protein markers per cell
type mapped to genes. Metric: driver-recovery `AUC_rel` (0.5 random, 1.0 perfect).

## Headline

**The marginal/conditional axis dominates; nonlinearity adds nothing.** The two
*marginal* measures (Spearman, MI) recover protein-confirmed markers best; the two
*conditional* measures (partial correlation, Integrated Gradients) are
significantly worse and statistically indistinguishable from each other.
Conditioning on the other genes *suppresses* true markers, because cell-type
markers are co-expressed — once one is in the model, the others add little unique
signal. Added model complexity (MLP + IG) does not help and is the least stable.

## Median AUC_rel across 30 cell types

| Method                      | Axis                  | Median AUC_rel |
|-----------------------------|-----------------------|----------------|
| Spearman correlation        | linear, marginal      | **0.926** |
| Mutual information (KSG)     | nonlinear, marginal   | 0.886 |
| Integrated Gradients (MLP)  | nonlinear, conditional| 0.799 |
| Partial correlation         | linear, conditional   | 0.741 |

## Statistics (cell type = unit of replication, n = 30)

- Friedman omnibus: χ² = 39.8, p = 1.2 × 10⁻⁸ — the four methods differ.
- Holm-corrected pairwise Wilcoxon signed-rank (rank-biserial effect size):
  - Spearman > partial correlation: p = 1.2e-5, r = 0.89
  - Spearman > Integrated Gradients: p = 1.6e-4, r = 0.80
  - Spearman > mutual information: p = 0.015, r = 0.55
  - Mutual information > partial correlation: p = 5.3e-5, r = −0.84
  - Mutual information > Integrated Gradients: p = 2.2e-3, r = 0.68
  - Partial correlation vs Integrated Gradients: p = 0.06 (not significant)

## Reading the axes of the 2×2

- **Linear → nonlinear (marginal):** Spearman vs MI — MI does *not* beat Spearman
  (Spearman is marginally higher). Nonlinearity buys nothing for marker recovery
  on these data.
- **Marginal → conditional (linear):** Spearman vs partial correlation — a large
  drop. Conditioning is the costly move.
- **Both relaxed:** Integrated Gradients (nonlinear + conditional) lands with the
  conditional methods, confirming the conditional axis, not nonlinearity, drives
  the gap.

## Supporting observations

- MLP classifier: 90% held-out accuracy on 30 classes; clean early stopping
  (validation loss minimal at epoch 3, then overfits) — IG is read from the
  best-epoch model.
- Cross-method gene-level correlations are low (partial-corr vs MI Spearman
  ρ ≈ 0.02; MI vs IG ρ ≈ 0.06), i.e. the measures genuinely rank genes
  differently; the protein-derived drivers nonetheless sit at high values for the
  marginal measures.
- A few biologically noisy cell types (Eryth, ILC) score near or below 0.5 for
  some methods — expected given small, ambiguous protein-marker sets.

## MLP hyperparameter tuning (macro-F1 sweep)

A 3-fold CV sweep over tanh-MLP configs, ranked by **macro-F1** (weights every
cell type equally, targeting minority lineages), selected hidden (512, 256),
dropout 0.4, lr 3e-4, weight decay 1e-4 (now the `src.mlp` default).

- Held-out accuracy 0.899 → **0.915**; CV macro-F1 0.895 → 0.904. The per-class
  gains concentrate on the hard minority T-subsets (CD8 Proliferating 0.74→0.81,
  CD4 Proliferating 0.80→0.84, Treg/dnT/CD4 TCM/TEM all up); easy classes barely
  move.
- IG seed-to-seed stability tightened (range 0.028 → 0.010).
- **But IG driver-recovery (AUC_rel) is essentially unchanged (~0.79).** Better
  classification does not translate into better attribution-based marker recovery
  — the bottleneck is the conditional+nonlinear attribution paradigm, not
  classifier quality (consistent with the saliency-stability caution [5]).

## Early recovery (recall@k) and the heterogeneous-category effect

AUC_rel integrates the whole ranking; practitioners inspect only the top genes.
- **Recall@k** (drivers in the top-k), bootstrapped over cell types, shows the
  early regime. At L2 (n=30) it has power; at L1 (n≤8) the bands are wide.
- **IG is strongly cell-type-dependent.** At L1 it is the *best* method for the
  clean abundant lineages (B AUC 0.97 vs Spearman 0.88; CD4 T 0.98 vs 0.95) and
  the worst for heterogeneous grab-bags (DC, `other`, `other T` ≈ 0.60).
- **Sensitivity:** dropping `other`/`other T` raises IG's mean AUC_rel
  0.821 → 0.889 (+0.068) — far more than Spearman (+0.012). The aggregate genuinely
  understates IG on well-defined cell types; this is reported as a transparency
  check, not the headline.

## Caveats

- `AUC_rel` is realised as the Mann–Whitney ROC-AUC of the per-gene score
  discriminating drivers from non-drivers (equivalent to the normalized
  driver-recovery-curve area for |D_c| ≪ N).
- MI is estimated on a fixed 10k-cell subsample for tractability (the other three
  methods use all cells); the same subsample is used for every cell type.
- The protein modality informs both the labels (WNN) and the ground truth, a
  dependence identical across all four methods — claims are comparative only.
