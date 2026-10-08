# Reading Notes: Three Papers on Industrial Visual Anomaly Detection

Verification convention used throughout: every number below is tagged with the exact table/figure/line it came from. Anything I could not confirm against a primary source (paper PDF/HTML or the official repository at a named commit) is marked **not verified**.

---

## [R1] PatchCore — Towards Total Recall in Industrial Anomaly Detection

### 1. Full citation

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler. "Towards Total Recall in Industrial Anomaly Detection." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 14318–14328. arXiv:2106.08265 (v1 15 Jun 2021; v2 5 May 2022), DOI [10.48550/arXiv.2106.08265](https://doi.org/10.48550/arXiv.2106.08265). Paper: <https://arxiv.org/abs/2106.08265> · HTML v2: <https://arxiv.org/html/2106.08265v2>. Code (Apache-2.0): <https://github.com/amazon-science/patchcore-inspection> (the CVPR camera-ready points at the older `amazon-research/patchcore-inspection` URL; both resolve to this repo).

Affiliations as printed in the HTML v2: Roth — University of Tübingen; Pemula, Zepeda, Schölkopf, Brox, Gehler — Amazon AWS. Footnote: "Work done during a research internship at Amazon AWS."

### 2. Problem and core idea

**Problem.** Cold-start (one-class) industrial anomaly detection: fit a model from nominal (defect-free) images only, and localize defects at pixel level, with one system that transfers across many product classes without per-class handcrafting. Prior ImageNet-feature + outlier-model methods (SPADE, PaDiM) are limited by (a) reliance on late, very abstract, ImageNet-biased feature levels and (b) a small number of usable nominal high-level feature representations available at test time.

**Core idea (one sentence).** Build a maximally representative memory bank of *locally aggregated mid-level* patch features over all nominal training images, then compress it with greedy minimax-facility-location coreset subsampling so that nearest-neighbour scoring against it is both fast and near-complete in coverage — a test image is anomalous if any single patch is far from its nearest nominal patch.

### 3. Core method, implementable level

**Notation (paper §3.1).** φ is an ImageNet-pretrained network; φ_{i,j} = φ_j(x_i) is the level-`j` feature map, with `j ∈ {1,2,3,4}` indexing the outputs of the four spatial-resolution blocks of a ResNet-like net (ResNet-50 / WideResNet-50). φ_{i,j}(h,w) ∈ R^{c*} is the c*-dim slice at position (h,w).

**Backbone and feature levels.** Default in the paper (§4.1, Appendix A): **WideResNet-50**, ImageNet-pretrained, features from the final outputs of **blocks 2 and 3** (`layer2`, `layer3`). Only two adjacent hierarchies, explicitly to retain spatial resolution while avoiding excessive ImageNet bias. Input protocol: resize to 256×256, center-crop to **224×224** ("images are resized and center cropped to 256×256 and 224×224, respectively", §4.1). No data augmentation (stated rationale: class-retaining augmentations require prior knowledge).

**Patch embedding extraction (local neighbourhood aggregation).** For an uneven patch/neighbourhood size `p`:

```
N_p^(h,w) = { (a,b) | a ∈ [h-⌊p/2⌋, …, h+⌊p/2⌋],  b ∈ [w-⌊p/2⌋, …, w+⌊p/2⌋] }        (Eq. 1)

φ_{i,j}( N_p^(h,w) ) = f_agg( { φ_{i,j}(a,b) | (a,b) ∈ N_p^(h,w) } )              (Eq. 2)

P_{s,p}(φ_{i,j}) = { φ_{i,j}(N_p^(h,w)) | h,w mod s = 0, h < h*, w < w*, h,w ∈ N } (Eq. 3)
```

- `f_agg` = **adaptive average pooling** over the local neighbourhood. This *keeps the feature-map resolution* (it is a smoothing, not a downsample — receptive field grows, grid size does not).
- `s` = stride, **set to 1** except in the stride ablation (§4.4.2).
- Default `p = 3` (justified in §4.4.1/Fig. 4 as the optimum between locality and global context).
- **Multi-hierarchy fusion:** compute `P_{s,p}(φ_{i,j+1})` and **bilinearly rescale** it so that |P_{s,p}(φ_{i,j+1})| == |P_{s,p}(φ_{i,j})| (i.e. match the higher-resolution grid of the *lowest* hierarchy used), then concatenate/aggregate per position. The paper's phrasing is "aggregating each element with its corresponding patch feature at the lowest hierarchy level used."

**Memory bank.** M = ⋃_{x_i ∈ X_N} P_{s,p}(φ_j(x_i)) (Eq. 4) — the union over all nominal training images of all their patch features. Each element is a d-dim vector; the default dimensionality is 1024.

**Coreset selection (Algorithm 1) — step by step.** Objective (Eq. 5) is minimax facility location:

```
M_C* = argmin_{M_C ⊂ M} max_{m ∈ M} min_{n ∈ M_C} || m - n ||_2
```

Exact solution is NP-hard (paper cites this); use the iterative greedy approximation:

1. `M_C ← {}` (empty coreset).
2. Compute a **random linear projection** ψ: R^d → R^{d*}, d* < d (Johnson–Lindenstrauss, following the Greedy Coreset work). Repository default `dimension_to_project_features_to = 128`.
3. Build the full pairwise distance matrix over the projected features (N×N).
4. Compute the initial "anchor distance" for every point as the **L2 norm of its row of the distance matrix**, i.e. an approximate distance to the set centroid. Repository: `coreset_anchor_distances = torch.norm(distance_matrix, dim=1)` (this initialisation is a code-level approximation of the paper's "start from the point farthest from the mean", and is the step a beginner will misread).
5. Repeat `l = percentage · |M|` times:
   a. `m_i ← argmax_{m ∈ M \ M_C} min_{n ∈ M_C} || ψ(m) − ψ(n) ||_2` — pick the point currently worst covered by the coreset (in the repo: `select_idx = torch.argmax(coreset_anchor_distances)`).
   b. `M_C ← M_C ∪ {m_i}`.
   c. Update coverage: `anchor_distances ← min(anchor_distances, dist(·, m_i))` elementwise.
6. Return the **original (unprojected) feature rows** at the selected indices — projection is only used to *choose* indices. `M ← M_C`.
7. Two repository variants: `GreedyCoresetSampler` (exact, full N×N matrix, `_compute_batchwise_differences` then argmax-loop) and `ApproximateGreedyCoresetSampler` (avoids the N×N matrix by measuring distances only to `number_of_starting_points` random anchors, default 10; less memory, more time).

**Score aggregation — and the crucial distinction.** Two different formulas exist in this paper; keep them separate.

*(i) The plain max-nearest-distance score (Eq. 6).* For test image x^test with patch collection P(x^test):

```
m^{test,*}, m* = argmax_{m^test ∈ P(x^test)} argmin_{m ∈ M} || m^test - m ||_2
s* = || m^{test,*} - m* ||_2                                                  (Eq. 6)
```

That is: per test patch take the distance to its nearest neighbour in M; take the **max over patches** as the image score; the argmax patch also gives the segmentation map.

*(ii) The reweighted PatchCore score (Eq. 7) — the paper's actual image-level score.*

```
s = ( 1 − exp( || m^{test,*} − m* ||_2 ) / Σ_{m ∈ N_b(m*)} exp( || m^{test,*} − m ||_2 ) ) · s*    (Eq. 7)
```

with `N_b(m*)` = the **b nearest patch-features in M to m*** (i.e. the neighbours of the *matched reference patch*, not of the test patch), and b a hyperparameter. The paper states: "We found this re-weighting to be more robust than just the maximum patch distance."

**How to read the reweighting (the part the student must internalise).**
- The bracket is `1 − softmax_over_N_b(m*)( ||m^{test,*} − m||_2 )` evaluated at `m = m*`, i.e. `1 − p` where `p` is the softmax weight of the single nearest neighbour among the b neighbours of `m*`.
- If `m*` sits in a **dense, well-covered** region of M, then `||m^{test,*} − m*||` ≈ the distances to its b neighbours ⇒ `p ≈ 1/b`-ish, small when b is large ⇒ bracket → ~1 ⇒ `s ≈ s*`. The score is *not* discounted.
- If `m*` is an **already-rare nominal occurrence** (a lone nominal patch in feature space), the nearest test-to-bank distance is *small* relative to the distances from `m*` to its other b neighbours, so `p → 1` and `1 − p → 0`. The score is **discounted**, suppressing false positives.
- Note the numerator/denominator asymmetry: the numerator uses the *test* patch's distance to `m*` (small), the denominator sums the *test* patch's distance to the b neighbours (mostly larger). Discount therefore grows with how isolated `m*` is within the bank.

*Important implementation caveat — "not verified" applies to the reweighting being reproduced.* The open-source repository does **not** implement Eq. 7. `NearestNeighbourScorer.predict` computes `anomaly_scores = np.mean(query_distances, axis=-1)` where `query_distances` are the distances to the `n_nearest_neighbours` nearest bank patches (default `--anomaly_scorer_num_nn 5` in the CLI, `1` in the README/sample commands), and `PatchMaker.score` reduces the per-patch tensor with `torch.max` over all remaining dims. So the repo's image score = **mean of the k-th nearest-neighbour distances of the single worst patch**, and the repo's patch score = **max over the k-NN distances**. I found no reweighting term anywhere in `src/patchcore/`. The paper's Eq. 7 should therefore be implemented from the paper, not read off the repo.

**Segmentation.** Re-align patch anomaly scores to their spatial locations, bilinearly upsample to the original input resolution, then Gaussian-smooth (`RescaleSegmentor.convert_to_segmentation`, `ndimage.gaussian_filter(..., sigma=4)`).

### 3b. Why dropping reference patches can *increase* nearest-neighbour distances

Three distinct effects, worth separating because conflating them produces wrong conclusions:

1. **Bank-induced distance inflation (the dominant effect).** For any fixed query, `d_NN(q, M_C) = min_{m ∈ M_C} d(q,m) ≥ min_{m ∈ M} d(q,m) = d_NN(q, M)` whenever `M_C ⊂ M`. Removing patches can therefore only *increase or hold* every NN distance. Per-image max-NN scores are consequently **not comparable across different subsampling percentages** — a PatchCore-1% bank produces systematically larger raw distances than a PatchCore-100% bank on identical input. This is exactly why the paper's reweighting exists: it rescales by local bank density so the *ranking* (which is what AUROC uses) stays stable across bank sizes.
2. **The retention effect, which is where the paper's 30% → 95% number comes from.** §4.4.2: "we recorded the percentage of memory bank samples that are used at test time for non-subsampled and coreset-subsampled memory banks. While initially only less than 30% of memory bank samples are used, coreset subsampling (to 1%) increases this factor to nearly 95%." An unsampled bank is massively redundant: many bank entries are near-duplicates, so a single retained entry absorbs the NN query. A coreset deliberately keeps *diverse* entries, so a much larger fraction of the bank ends up being the actual nearest neighbour for some test patch. Read this number as **utilisation**, not as "distances went up".
3. **Consequence for a beginner.** If you swap `max` for `mean` for `k`, or change the coreset percentage, the absolute score scale changes. Thresholds, calibration and any "score > τ" decision rule must be **re-derived per configuration**. AUROC is invariant to monotone rescaling and therefore hides this — which is why so many reimplementations look fine on AUROC and break on a real production threshold.

Also from §4.4.2 (an honest negative result worth noting): for subsampling intervals between ~50% and ~10%, joint detection+localization performance sometimes *partly increases* relative to non-subsampled PatchCore; and the performance of *no* subsampling is "comparable to a coreset-reduced memory bank that is two orders of magnitudes smaller in size."

### 4. Reported numbers

All PatchCore numbers below are WideResNet-50 / 224×224 unless stated. Metrics: image AUROC (class-average on MVTec AD), pixel AUROC, PRO.

**Table 1 — MVTec AD image-level AUROC (`↑`), §4.2** (baseline column = PatchCore-25% / -10% / -1%):

| Method | AUROC | Error |
|---|---|---|
| SPADE | 85.5 | 14.5 |
| PatchSVDD | 92.1 | 7.9 |
| DifferNet | 94.9 | 5.1 |
| PaDiM | 95.3 | 4.7 |
| MahalanobisAD | 95.8 | 4.2 |
| PaDiM* (task-specific backbone) | 97.9 | 2.1 |
| **PatchCore-25%** | **99.1** | 0.9 |
| **PatchCore-10%** | **99.0** | 1.0 |
| **PatchCore-1%** | **99.0** | 1.0 |

Misclassifications at F1-optimal threshold: 42 / 47 / 49 (and the paper notes "less than 50 images remain misclassified"; 1725 test images total). Narrative numbers from §4.2: PatchCore-25% "solves six of the 15 MVTec datasets"; improvement from PaDiM's 2.1% error to PatchCore-25%'s 0.9% error = 57% error reduction.

**Segmentation, §4.2 narrative** (PatchCore vs PaDiM): pixel AUROC **98.1 vs 97.5**; PRO **93.5 vs 92.1**.

**Table 4 — PatchCore-1% with larger resolution/backbones/ensembles** (this is the headline 99.6%):

| Config | AUROC | pixel AUROC | PRO |
|---|---|---|---|
| DenseNet-201 & ResNeXt-101 & WRN-101 (2+3), imagesize 320 | **99.6** | 98.2 | 94.9 |
| WRN-101 (2+3), imagesize 280 | 99.4 | 98.2 | 94.4 |
| WRN-101 (1+2+3), imagesize 280 | 99.2 | **98.4** | **95.0** |

So "pixel AUROC 98.4%" (the maximum) belongs to WRN-101 (1+2+3) @280 — **not** to the 99.6% row. Get this pairing right when citing.

**Table 5 — Mean inference time per image on MVTec AD** (s), scores shown as (image AUROC, pixel AUROC, PRO). Note these scores are the *timing-table* subset and read slightly differently from Tables 1–3:

| Method | Scores | Time (s) |
|---|---|---|
| PatchCore-100% | (99.1, 98.0, 93.3) | 0.6 |
| PatchCore-10% | (99.0, 98.1, 93.5) | 0.22 |
| PatchCore-1% | (99.0, 98.0, 93.1) | 0.17 |
| PatchCore-100% + IVFPQ | (98.0, 97.9, 93.0) | 0.2 |
| SPADE | (85.3, 96.6, 91.5) | 0.66 |
| PaDiM | (95.4, 97.3, 91.8) | 0.19 |

Inference times "include the forward pass through the backbone" (§4.3) and cover *joint* image- and pixel-level anomaly detection. Hardware: "Nvidia Tesla V4 GPUs" (Appendix A — presumably a typo for V100; reported as written). GPU memory: the README states "the majority of experiments should not exceed 11GB of GPU memory."

**Repository-reported numbers differ slightly from the paper** — `README.md` "Expected performance of pretrained models":

| Model | Mean AUROC | Mean Seg. AUROC | Mean PRO |
|---|---|---|---|
| WR50-baseline | 99.2% | 98.1% | 94.4% |
| Ensemble | 99.6% | 98.2% | 94.9% |

The README also warns: "Due to repository changes (& hardware differences), results may deviate slightly from those reported in the paper." The 99.2% vs the paper's 99.1% is such a deviation; do not present them as the same number.

**Stride ablation (§4.4.2):** memory-bank reduction via striding gives image AUROC 97.6% (s=2) and 96.8% (s=3) — i.e. striding is a worse compression lever than coreset subsampling.

**Low-shot (§4.5 / Table S5):** study over 1 to 50 nominal training images (1 image = 0.4% of total nominal training data; 50 = 21%), compared against reimplementations of SPADE and PaDiM on the same WideResNet-50 backbone. Exact per-point values live in Table S5 and are **not reproduced here** (I did not extract the full table; treat any specific few-shot AUROC as not verified until read off Table S5).

### 5. What a beginner should SKIP on a first read

- **§3.2 / Algorithm 1 in full detail** (the greedy coreset derivation, Johnson–Lindenstrauss justification, the NP-hardness remark, and especially the `ApproximateGreedyCoresetSampler` variant). First read: accept "coreset ≈ a diverse subsample that preserves coverage" and move on. The approximation variant is an engineering optimisation with no conceptual content.
- **§4.4.2's proxy-learning baseline (Eq. 8)** — the learned `L_rec` proxy comparison is a control experiment to justify coreset over alternatives. Skim; it exists to defend a design choice you have already accepted.
- **The `b`-nearest-neighbour reweighting (Eq. 7) on the very first pass.** It is a two-line formula with real subtlety (see §3's reading above) and it is *not* in the reference code, so reading it too early creates a mismatch between what the paper says and what any code you run does. Read Eq. 6 first, get a working max-NN implementation, then come back to Eq. 7 deliberately.
- **The full MVTec per-subdataset tables (S1–S4) and §C.4's 42-error taxonomy.** Fifteen columns of near-99% numbers add no information on a first read; the error taxonomy (Figures S1/S2) is only useful *after* you have your own failure cases.
- **The mSTC / non-industrial benchmark (§4.6)** — out of scope for an industrial plan; it is there to show generality.
- **The IVFPQ approximate-NN row.** It is an orthogonal engineering lever; the paper itself concludes coreset subsampling dominates it on both speed and accuracy.

### 6. Exact repository locations

Repo: <https://github.com/amazon-science/patchcore-inspection> (default branch `main`, tree SHA `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` at the time of checking). `PYTHONPATH=src` is required.

| Step | File | Symbol |
|---|---|---|
| End-to-end training entry point | `bin/run_patchcore.py` | `patch_core()` click group (options `--anomaly_scorer_num_nn`, `--patchsize`, `--patchstride`, `--pretrain_embed_dimension`, `--target_embed_dimension`, `--faiss_on_gpu`), `sampler()` subcommand, `dataset()` subcommand |
| Load/evaluate pretrained model | `bin/load_and_evaluate_patchcore.py` | `patch_core_loader()` |
| Main model class | `src/patchcore/patchcore.py` | `class PatchCore(torch.nn.Module)`; `load()`, `_embed()`, `fit()`, `_fill_memory_bank()`, `predict()`, `_predict()` |
| Patchify + patch-score reduction | `src/patchcore/patchcore.py` | `class PatchMaker`; `patchify()` (the `torch.nn.Unfold` neighbourhood aggregation, `padding = (patchsize-1)//2`), `unpatch_scores()`, `score()` (the `torch.max` reduction over dims) |
| Coreset selection | `src/patchcore/sampler.py` | `class GreedyCoresetSampler` (`run()`, `_reduce_features()` = the random linear projection, `_compute_batchwise_differences()`, `_compute_greedy_coreset_indices()`), `class ApproximateGreedyCoresetSampler`, `class RandomSampler`, `class IdentitySampler`, `class BaseSampler` |
| Nearest-neighbour scoring | `src/patchcore/common.py` | `class NearestNeighbourScorer` (`fit()`, `predict()` — this is where `np.mean(query_distances, axis=-1)` lives), `class FaissNN` (`IndexFlatL2`), `class ApproximateFaissNN` (`IndexIVFPQ`, 512 centroids / 64 sub-quantizers / 8 bits), `class ConcatMerger`, `class AverageMerger` |
| Feature extraction from the backbone | `src/patchcore/common.py` | `class NetworkFeatureAggregator`, `class ForwardHook`, `class LastLayerToExtractReachedException` |
| Local neighbourhood aggregation `f_agg` | `src/patchcore/common.py` | `class MeanMapper` (`F.adaptive_avg_pool1d`), `class Preprocessing`, `class Aggregator` |
| Segmentation map | `src/patchcore/common.py` | `class RescaleSegmentor.convert_to_segmentation()` (bilinear `F.interpolate` to `target_size`, then `ndimage.gaussian_filter(sigma=4)`) |
| Backbone registry | `src/patchcore/backbones.py` | `_BACKBONES` dict (`"wideresnet50"`, `"resnet50"`, …), `load(name)` |
| Dataset (MVTec) | `src/patchcore/datasets/mvtec.py` | `class MVTecDataset` |
| Metrics | `src/patchcore/metrics.py` | AUROC / PRO / F1 helpers |
| Reference sample commands | `sample_training.sh`, `sample_evaluation.sh` | shell, no Python symbols |
| Tests (good entry points for a newcomer) | `test/test_patchcore.py`, `test/test_sampler.py`, `test/test_common.py` | — |

Confirmed defaults from `README.md` / `bin/run_patchcore.py`: `-b wideresnet50 -le layer2 -le layer3`, `--pretrain_embed_dimension 1024`, `--target_embed_dimension 1024`, `--patchsize 3`, `sampler -p 0.1 approx_greedy_coreset`, `--resize 256 --imagesize 224`. `GreedyCoresetSampler.__init__` default `dimension_to_project_features_to=128`; `ApproximateGreedyCoresetSampler.__init__` default `number_of_starting_points=10`.

### 7. Learning outcomes (three)

1. **Be able to state and implement the full PatchCore pipeline from scratch** — mid-level (`layer2`+`layer3`) WideResNet-50 features → `3×3` adaptive-average-pooled locally aware patch features → union memory bank over all nominal images → minimax-facility-location greedy coreset at 1–10% → per-patch nearest-neighbour distance → max over patches — and explain *why* mid-level beats late-level features (retained spatial resolution + less ImageNet class bias) without reciting the paper.
2. **Be able to explain, with the formula, why the max-nearest-distance score and the reweighted PatchCore score are different objects** — that Eq. 7 multiplies Eq. 6 by a density-dependent discount factor computed over the *matched reference patch's* `b` neighbours, that this discount suppresses false positives from isolated nominal patches, and that the reference repository implements the plain non-reweighted variant.
3. **Be able to reason correctly about subsampling's effect on distances and thresholds** — that `M_C ⊂ M` implies `d_NN(q, M_C) ≥ d_NN(q, M)`, that raw scores therefore are not comparable across subsampling percentages, that the paper's "30% → 95%" bank-utilisation figure measures coverage/utilisation rather than distance inflation, and that AUROC's scale-invariance is what hides this until you deploy a fixed threshold.

---

## [R2] AnomalyDINO — Boosting Patch-based Few-shot Anomaly Detection with DINOv2

### 1. Full citation

Simon Damm, Mike Laszkiewicz, Johannes Lederer, Asja Fischer. "AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2025 (accepted as **Oral**). arXiv:2405.14529 (v1 23 May 2024; v3 13 Mar 2025), DOI [10.48550/arXiv.2405.14529](https://doi.org/10.48550/arXiv.2405.14529), CC BY 4.0. Paper: <https://arxiv.org/abs/2405.14529> · HTML v3: <https://arxiv.org/html/2405.14529v3>. Code: <https://github.com/dammsi/AnomalyDINO>.

### 2. Problem and core idea

**Problem.** Few-shot (1/2/4/8/16-shot) industrial anomaly detection. The state of the art at the time used vision-language models (WinCLIP+, APRIL-GAN, AnomalyCLIP, ADP), which bring large compute overhead and — for APRIL-GAN/AnomalyCLIP — require training or meta-learning on a related dataset. Can a *frozen, vision-only* encoder plus the classic patch-level deep-nearest-neighbour paradigm compete, training-free?

**Core idea (one sentence).** Freeze DINOv2, take its patch tokens as features, build the memory bank from a handful of reference images (augmented by rotation and masked by DINOv2's own PCA foreground mask), and score a test image as the **mean of the top 1% of patch-to-memory-bank nearest-neighbour distances** — a tail-value-at-risk statistic that is both sensitivity-preserving and robust to a single spurious patch.

### 3. Core method, implementable level

**Backbone.** DINOv2 ViT, **frozen, never trained** (contrast with R1/R3 — see the explicit note below). Default `dinov2_vits14` (ViT-S, 21×10⁶ params); variants evaluated: ViT-S, ViT-B, ViT-L, ViT-G (up to 1.1×10⁹ params). Patch size **14×14 pixels**; any input resolution that is a multiple of 14 is admissible. Default resolution **448** (smaller edge); 672 also evaluated.

**Preprocessing.**
- `Resize(smaller_edge_size=448, interpolation=BICUBIC, antialias=True)` → `ToTensor()` → `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` (ImageNet statistics). No aspect ratio preservation (a plain `Resize` on the smaller edge).
- Then crop both dimensions down to multiples of the patch size: `cropped_h = h - h % 14`, `cropped_w = w - w % 14`; the token grid is `grid_size = (cropped_h // 14, cropped_w // 14)`.
- **Masking** (DINOv2 used for masking too, so the pipeline stays single-model): take PCA `n_components=1` on the patch features; `mask = first_pc > 10`. Adaptive check ("masking test", Fig. 2): look at the central region (`border = 0.2`, i.e. rows/cols in `[0.2·H, 0.8·H)`); if the kept fraction there is `≤ 0.35` of that region's size, flip the sign (`mask = -first_pc > 10`). Then `cv2.dilate` and `cv2.morphologyEx(..., MORPH_CLOSE)` with a 3×3 kernel. Background patches are discarded from both the bank and the query. Textures (MVTec `Wood`, `Tile`) are deliberately **not** masked; ViT/ImageNet backbones cannot mask at all.
- **Rotation augmentation** ("agnostic" default): rotate each reference image by `[0, 45, 90, 135, 180, 225, 270, 315]` degrees (`cv2.getRotationMatrix2D` + `warpAffine`, `INTER_LINEAR`, `BORDER_DEFAULT`). This enlarges the bank ~8×; the paper measures the nearest-neighbour-search cost increase as negligible on GPU. The "informed" variant uses only rotations known to be admissible for the product.

**Memory bank.** `M := ⋃_{x^(i) ∈ X_ref} { p_j^(i) | f(x^(i)) = (p_1^(i),…,p_n^(i)), j ∈ [n] }` (Eq. 1). Explicitly **no coreset subsampling** — the few-shot regime makes the bank small; the paper instead *enriches* it (augmentation) and *filters* it (masking). Bank size grows linearly in k and by the augmentation factor.

**Patch embedding extraction.** `tokens = model.get_intermediate_layers(image_batch)[0].squeeze()` — DINOv2's `get_intermediate_layers` with the default layer set, i.e. the last transformer block's patch tokens (no CLS token is appended/removed in this path — the DINOv2 helper returns patch tokens only). Tokens are L2-normalised before indexing.

**Distance.** Cosine distance, implemented on L2-normalised vectors via FAISS L2^2 / 2:

```
d_NN(p; M) := min_{p_ref ∈ M} d(p, p_ref)                    (Eq. 2)
d(x, y)    := 1 - <x,y> / (||x|| ||y||)                      (Eq. 3)
```
In code: `faiss.normalize_L2(features_ref)` at build time and `faiss.normalize_L2(features2)` at query time, `faiss.IndexFlatL2`, `knn_neighbors = 1`, then `distances = distances / 2`.

**Score aggregation — exact (the "top-tail average").**

```
s(x_test) := q( { d_NN(p_1; M), …, d_NN(p_n; M) } )          (Eq. 4)
q(D)      := mean( H_{0.01}(D) ),  H_{0.01}(D) = the 1% highest values in D
```

Verbatim from §3.1: "Throughout this paper, we define q as the average distance of the 1% most anomalous patches, i.e. `q(D) := mean(H_0.01(D))` with `H_0.01(D)` containing the 1% highest values in the set `D`." The paper describes q as "an empirical estimate of the tail value at risk for the 99% quantile."

Repository implementation of `q` (`src/post_eval.py`):

```python
def mean_top1p(distances):
    if int(len(distances) * 0.01) == 0:
        return np.max(distances)          # fallback for tiny patch counts
    else:
        return np.mean(sorted(distances.flatten(), reverse=True)[:int(len(distances) * 0.01)])
```

Two operational details the paper text does not spell out: (a) the **discrete-count floor** `int(n_patches * 0.01)` — at 448 px with 14 px patches you get a 32×32 = 1024-token grid, so the top-1% average is over `int(1024*0.01) = 10` patches; (b) if that count rounds to 0 (fewer than 100 patches), the statistic silently degenerates to the **max**. Any reimplementation using `ceil` instead of floor, or averaging over a fixed count, will not match. Masked-out background patches are excluded from the set passed to `mean_top1p`; the helper is called on `output_distances.flatten()`, where masked positions are 0.

**Segmentation.** `dists2map`: `cv2.resize(dists, (W,H), INTER_LINEAR)` then `scipy.ndimage.gaussian_filter(dists, sigma=4)` — Gaussian smoothing σ = 4.0, following PatchCore.

**Retrieval setup, latency measurement.** FAISS `GpuIndexFlatL2` by default (`--faiss_on_cpu` for CPU). Inference timing is measured per test image with `time.time()` around the full extract + search + score path, with `torch.cuda.synchronize()` before stopping the clock (and no warm-up in `run_anomalydino.py`, although the CLI exposes `--warmup_iters` default 25 for benchmarking).

**The frozen-encoder point, stated explicitly.** The DINOv2 encoder is *never trained* and no gradient ever flows through it: `extract_features` is wrapped in `torch.inference_mode()` (and `torch.no_grad()` in the ViT wrapper), and `load_model` calls `model.eval()`. There is no fine-tuning, no meta-learning, no adapter, no learned projection on top. `features_ref` are raw DINOv2 tokens (L2-normalised only). The paper's abstract: "The approach is methodologically simple and training-free and, thus, does not require any additional data for fine-tuning or meta-learning." This is the single sharpest architectural contrast with R1 (which also does not train, but *does* select a subset of features via coreset) and with R3 (which trains a student, an autoencoder, and distils a teacher).

**The few-shot reference-count taxonomy.** k ∈ {1, 2, 4, 8, 16} shots, where "k-shot" = the bank is built from exactly **k reference images**, drawn from `<object>/train/good/`. The reference set is chosen **deterministically, not randomly**: `sorted(os.listdir(ref_folder))[seed*n : (seed+1)*n]` — so repetition `seed=0` uses the first k samples, `seed=1` the second k, `seed=2` the third k. Three repetitions per configuration; mean and std reported. (Note this differs from the common convention of k *per class* vs k *per category* — here MVTec categories are the "objects", so each object gets its own bank built from k images of that object, and MVTec-AD's reported numbers are the mean over its 15 categories. VisA likewise over its 12.) One separate setting exists beyond the taxonomy: the **batched zero-shot** adaptation in Appendix D (using the whole train set as the bank), reported in Table 4.

### 4. Reported numbers

All AnomalyDINO rows are `AnomalyDINO-S` = DINOv2 ViT-S/14, at the resolution stated in parentheses. MVTec-AD and VisA, in %, mean ± std over 3 seeds. All other rows are the baselines *as reported by the AnomalyDINO authors* (Tables 2 and 3); values taken from the original publications except WinCLIP+ (from [19]) and its reimplementation.

**MVTec-AD (Table 2)** — image-level AUROC / F1-max / AP, pixel-level AUROC / F1-max / PRO:

| Setting | Method | img AUROC | img F1max | img AP | px AUROC | px F1max | PRO |
|---|---|---|---|---|---|---|---|
| 1-shot | SPADE† | 81.0 ± 2.0 | 90.3 ± 0.8 | 90.6 ± 0.8 | 91.2 ± 0.4 | 42.4 ± 1.0 | 83.9 ± 0.7 |
| 1-shot | PatchCore† | 83.4 ± 3.0 | 90.5 ± 1.5 | 92.2 ± 1.5 | 92.0 ± 1.0 | 50.4 ± 2.1 | 79.7 ± 2.0 |
| 1-shot | GraphCore | 89.9 | / | / | 95.6 | / | / |
| 1-shot | WinCLIP+ | 93.1 ± 2.0 | 93.7 ± 1.1 | 96.5 ± 0.9 | 95.2 ± 0.5 | 55.9 ± 2.7 | 87.1 ± 1.2 |
| 1-shot | APRIL-GAN | 92.0 ± 0.3 | 92.4 ± 0.2 | 95.8 ± 0.2 | 95.1 ± 0.1 | 54.2 ± 0.0 | 90.6 ± 0.2 |
| 1-shot | **AnomalyDINO-S (448)** | **96.5 ± 0.4** | 96.0 ± 0.2 | 98.1 ± 0.3 | 96.3 ± 0.1 | 57.9 ± 0.8 | 91.7 ± 0.1 |
| 1-shot | **AnomalyDINO-S (672)** | **96.6 ± 0.4** | 95.8 ± 0.5 | 98.2 ± 0.2 | 96.8 ± 0.1 | 60.2 ± 1.1 | 92.7 ± 0.1 |
| 2-shot | AnomalyDINO-S (448 / 672) | 96.7 ± 0.8 / 96.9 ± 0.7 | 96.5 / 96.1 | 98.1 / 98.2 | 96.5 / 97.0 | 58.5 / 61.0 | 92.0 / 93.1 |
| 4-shot | AnomalyDINO-S (448 / 672) | 97.6 ± 0.1 / 97.7 ± 0.2 | 97.0 / 96.6 | 98.4 / 98.7 | 96.7 / 97.2 | 59.2 / 61.8 | 92.4 / 93.4 |
| 8-shot | AnomalyDINO-S (448 / 672) | 98.0 ± 0.1 / 98.2 ± 0.2 | 97.4 / 97.4 | 99.0 / 99.1 | 97.0 / 97.4 | 59.6 / 62.3 | 92.7 / 93.8 |
| 16-shot | AnomalyDINO-S (448 / 672) | **98.3 ± 0.1 / 98.4 ± 0.1** | 97.7 / 97.6 | 99.3 / 99.3 | 97.1 / **97.5 ± 0.0** | 60.0 / **62.7** | 92.9 / **94.0 ± 0.1** |

The headline claim in the abstract — "pushing the one-shot performance on MVTec-AD from an AUROC of 93.1% to 96.6%" — compares **AnomalyDINO-S (672) at 96.6%** against **WinCLIP+ at 93.1%**. The 448 config gives 96.5%. Both pairings matter if the student quotes the sentence.

**VisA (Table 3)** — same metric set. Selected rows (AnomalyDINO-S, 448 / 672):

| Setting | img AUROC (448 / 672) | px AUROC (448 / 672) | PRO (448 / 672) |
|---|---|---|---|
| 1-shot | 85.6 ± 1.5 / 87.4 ± 1.2 | 97.5 ± 0.1 / 97.8 ± 0.1 | 90.7 ± 0.5 / 92.5 ± 0.5 |
| 4-shot | 91.3 ± 0.8 / 92.6 ± 0.9 | 98.0 ± 0.0 / 98.2 ± 0.0 | 92.5 ± 0.2 / 94.1 ± 0.1 |
| 16-shot | 93.8 ± 0.1 / 94.8 ± 0.2 | 98.3 ± 0.0 / 98.5 ± 0.0 | 93.8 ± 0.2 / 95.3 ± 0.2 |

For contrast on VisA: APRIL-GAN 1-shot image AUROC 91.2 ± 0.8 (beats AnomalyDINO at 1-shot and 2-shot; AnomalyDINO takes the lead at 8- and 16-shot); PatchCore† 1-shot 79.9 ± 2.9; SPADE† 1-shot 79.5 ± 4.0; PaDiM† 1-shot 62.8 ± 5.4.

**Batched zero-shot (Table 4)** — image AUROC only, %: WinCLIP 91.8 / 78.1 (MVTec-AD / VisA), AnomalyCLIP 91.5 / 82.1, APRIL-GAN 86.1 / 78.0, ACR 85.8 / /, MuSc **97.8 / 92.8**, AnomalyDINO-S (448) 93.0 / 89.7, AnomalyDINO-S (672) 94.2 / 90.7. MuSc is clearly ahead in this setting; the paper is explicit that this is a *different* setting.

**Latency.** "AnomalyDINO-S takes roughly 60ms to process an image at a resolution of 448" (§4.2 discussion of Figure 3). **Hardware: "All runtimes are measured on a single NVIDIA A40 if not stated otherwise."** Figure 3 is a detection-AUROC-vs-inference-time scatter in the 1-shot setting with a logarithmic time axis; a detailed runtime breakdown is in Appendix C, Table 9 (**not extracted here — specific per-configuration millisecond values beyond the ~60 ms figure are not verified**). The only methods faster than AnomalyDINO are PatchCore and ImageNet-trained ViT backbones — "which however sacrifice performance."

**Memory.** An explicit memory footprint (MB/GB) for AnomalyDINO is **not verified** — I found no memory table in the main paper or the appendices I read. The only related statement is qualitative: augmentations grow `M`, and "with increasing size of M reduction techniques such as coreset subsampling are advisable."

**Ablations (§4.2, Appendix C).**
- **Preprocessing:** adds ~**+2% AUROC** across four AnomalyDINO configurations, at no significant latency cost; the effect grows with resolution. "Agnostic" (always rotate) slightly *outperforms* "informed".
- **Aggregation statistic:** "The mean of the 1% highest distances from test patches to M **improves over** the standard choice (maximum of the upsampled and smoothed patch distances)." I did **not** verify the size of that improvement — Appendix C.2's exact numbers are not extracted. Do not quote a delta.
- **Architecture size (Appendix C.3):** no considerable difference between ViT-S/B/L; the **smallest** model is best on MVTec-AD (single objects, simpler scenes), while larger models do better on VisA (multiple complex objects).
- **Backbone choice (Appendix C.4):** ImageNet-pretrained ViTs "slightly outperform PatchCore" but give weaker features than DINOv2 and are **incompatible with the PCA masking procedure**.

### 5. What a beginner should SKIP on a first read

- **Appendix D (batched zero-shot) and the whole MuSc/ACR comparison.** It is a genuinely different problem setting (the entire train set is the bank), included defensively so reviewers cannot claim AnomalyDINO dodges it. Reading it first will make you mis-file AnomalyDINO as a zero-shot method when it is few-shot.
- **The masking test's adaptive sign flip and the `border=0.2` heuristic** (`compute_background_mask`). This is a small dataset-specific hack tuned to MVTec/VisA object framing. First read: "mask background via thresholded first PCA component; textures are not masked." Come back only if your own dataset has a background-common-mode false-positive problem.
- **The rotation-augmentation ablation and the "agnostic vs informed" discussion (Appendix C.1, Figures 11–13).** It is a per-category, judgement-dependent decision (rotation is an anomaly for some products, invariance-desirable for others). Skim to the takeaway: rotate by default, and note that it can hurt.
- **The full Appendix A per-category tables and Figures 4–7 anomaly maps.** Fifteen categories × six metrics × five shot counts is 450 numbers that all say "we win at k ≥ 4."
- **The ViT-S/B/L/G architecture-size study (C.3).** Interesting but non-actionable; the takeaway is one sentence ("use ViT-S unless your scenes are complex and multi-object").
- **The `demo_AnomalyDINO.ipynb` walkthrough on a first read** — read `src/detection.py` instead; the notebook is a 1.6 MB artefact mostly composed of embedded plots.

### 6. Exact repository locations

Repo: <https://github.com/dammsi/AnomalyDINO> (default branch `main`; note there is no `master` branch — a `master` tree query returns 404).

| Step | File | Symbol |
|---|---|---|
| End-to-end few-shot entry point | `run_anomalydino.py` | `main()`; argparse flags `--model_name` (default `dinov2_vits14`), `--resolution` (default 448), `--knn_metric` (default `L2_normalized`), `--k_neighbors` (default 1), `--shots`, `--num_seeds`, `--preprocess` (default `agnostic`), `--mask_ref_images`, `--warmup_iters` (default 25) |
| Batched zero-shot entry point | `run_anomalydino_batched.py` | `main()` |
| Core detection loop (bank construction, masking, kNN, scoring, timing) | `src/detection.py` | `run_anomaly_detection(...)`; the timing blocks are `start_time`/`time_memorybank` and `torch.cuda.synchronize()` + `inf_time` |
| Backbone wrappers, preprocessing, masking | `src/backbones.py` | `class VisionTransformerWrapper` (base), `class DINOv2Wrapper` (`load_model()`, `prepare_image()` — the `% patch_size` crop, `extract_features()` — `torch.inference_mode()` + `get_intermediate_layers(...)[0]`, `compute_background_mask()` — the PCA + adaptive-sign-flip + dilate/close masking, `get_embedding_visualization()`), `class ViTWrapper`, `get_model()` |
| **Score aggregation `q` (top-1% mean)** | `src/post_eval.py` | `mean_top1p(distances)`; also `max_anomaly_map()`, `eval_segmentation(gt_filenames, prediction_filenames, pro_integration_limit=0.3, ...)` |
| Segmentation map (upsample + Gaussian σ=4) | `src/utils.py` | `dists2map(dists, img_shape)`; also `augment_image(img_ref, augmentation="rotate", angles=[0,45,90,135,180,225,270,315])`, `rotate_image()`, `plot_ref_images()`, `get_dataset_info()` (per-object preprocessing config — "agnostic" default) |
| Plots / PCA + mask visualisation | `src/visualize.py` | — |
| Minimal walkthrough | `demo_AnomalyDINO.ipynb` | — (large; mostly embedded figures) |
| Dependency list | `requirements.txt` | includes `faiss`, `timm`, `scikit-learn` (PCA), `opencv-python` (dilate/MORPH_CLOSE, `INTER_LINEAR` resize), `tifffile` (segmentation maps) |

Exact scoring path to read, in order: `run_anomalydino.py:main` → `src/detection.py:run_anomaly_detection` (bank loop, `faiss.normalize_L2`, `knn_index.search`, `distances / 2`) → `src/post_eval.py:mean_top1p` → `src/utils.py:dists2map` (only if segmentation is needed).

### 7. Learning outcomes (three)

1. **Be able to state precisely what "training-free few-shot" buys and costs.** Concretely: the encoder is frozen at `model.eval()` under `torch.inference_mode()`, no parameters are updated, no additional dataset is touched, and the *only* thing that changes between 1-shot and 16-shot is the size and diversity of `M` — so all performance variation with k is a property of bank coverage, not of learned adaptation. Be able to contrast this with PatchCore (frozen features, but *learned-free yet data-dependent* coreset selection) and EfficientAD (genuinely trained student + autoencoder + distilled teacher).
2. **Be able to write down and implement the exact aggregation**, including its discrete details, and explain the trade-off it encodes: `s = mean(top ⌊0.01·n⌋ patch distances)`, which is not the max (robust to one spurious patch) and not a high percentile of the smoothed map (sensitive enough to catch a small defect), with the `int(...)==0 ⇒ max` degenerate branch, and be able to say why the paper frames it as an empirical tail-value-at-risk estimate at the 99% quantile.
3. **Be able to map the few-shot reference-count taxonomy onto the code and the evaluation protocol** — that k ∈ {1,2,4,8,16} reference images are drawn from `<object>/train/good/` as `sorted(listdir)[seed*k : (seed+1)*k]` (deterministic, not random), that three seeds exist, that the reported numbers are means over MVTec-AD's 15 / VisA's 12 categories, and that the batched zero-shot setting in Appendix D is a different experiment rather than a larger k.

---

## [R3] EfficientAD — Accurate Visual Anomaly Detection at Millisecond-Level Latencies

### 1. Full citation

Kilian Batzner, Lars Heckler, Rebecca König. "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies." *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2024, pp. 128–138 (accepted as **Oral**). arXiv:2303.14535 (v1 25 Mar 2023; v3 8 Feb 2024). Journal reference as printed on arXiv: "Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 128-138." Paper: <https://arxiv.org/abs/2303.14535> · HTML v3: <https://arxiv.org/html/2303.14535v3>. Official code: <https://github.com/nelson1425/EfficientAD> (no repository link is printed in the paper; I located this repo by the author name in its URL/README — treat the provenance of this URL as *strong but not paper-attested*). Alternative, actively maintained implementation with the same method: <https://github.com/open-edge-platform/anomalib> at `src/anomalib/models/image/efficient_ad/`.

### 2. Problem and core idea

**Problem.** Existing accurate methods (PatchCore, FastFlow, DSR, AST, GCAD) are too slow for real-time inspection; the standard student–teacher (S–T) approach needs expensive feature extractors (the S–T baseline uses convolutions on large feature maps, exceeding 4.5 TFLOPs), and no efficient method handles *logical* anomalies (invalid combinations of normal local features, e.g. a wrong object ordering, an extra cable) without a separate global model.

**Core idea (one sentence).** Distil a deep pretrained classifier into a tiny 4-convolution *patch description network* (PDN) with a fixed 33×33 receptive field, train a same-architecture student against it using a *hard feature loss* that backpropagates only through the worst-matching feature elements plus an ImageNet penalty, and add a small autoencoder whose reconstruction the student also predicts — yielding a **learned** student–teacher detector for structural anomalies plus a **learned** global detector for logical anomalies at ~2 ms latency.

### 3. Core method, implementable level

**PDN (feature extractor, §3.1).** Four convolutional layers, fully convolutional, applied to the whole image in one pass. Each output neuron has a receptive field of exactly **33×33 pixels**, so each output feature vector describes one 33×33 patch — hence "patch description network." Strided **average-pooling** after Conv-1 and Conv-2 provides early downsampling (the paper's explicit contrast with prior S–T nets: parameter count is a misleading proxy, because those nets do no downsampling and are therefore slow despite 1.6–2.7 M params). A 256×256 image yields features in **< 800 µs on an NVIDIA RTX A6000**.

**PDN-S architecture (Table 6; teacher output 384×64×64 for a 3×256×256 input).** Columns: stride, kernel, #kernels, padding, activation.

| Layer | Stride | Kernel | Kernels | Padding | Activation |
|---|---|---|---|---|---|
| Conv-1 | 1×1 | 4×4 | 256 | 3 | ReLU |
| AvgPool-1 | 2×2 | 2×2 | 256 | 1 | — |
| Conv-2 | 1×1 | 4×4 | 512 | 3 | ReLU |
| AvgPool-2 | 2×2 | 2×2 | 512 | 1 | — |
| Conv-3 | 1×1 | 1×1 | 512 | 0 | ReLU |
| Conv-4 | 1×1 | 3×3 | **512** (teacher) / **768** (student) | 1 | ReLU |
| Conv-5 | 1×1 | 4×4 | 384 | 0 | ReLU |
| Conv-6 | 1×1 | 1×1 | 384 | 0 | — |

Note the paper's Table 6 caption: "The student network has the same architecture, but **768 kernels instead of 384 in the Conv-4 layer**" — while the surrounding prose and Appendix A.1 say the student maps to `R^{768×64×64}` and its first 384 channels are used for the ST branch. The "768" in the Conv-4 row and the prose are consistent with each other; the caption's wording ("instead of 384 in the Conv-4 layer") is what you get when you read the table naively. **For EfficientAD-M (Table 7)** the teacher uses a different, encoder-style PDN (I extracted only the first rows: EncConv-1 2×2/4×4/32/pad1/ReLU, EncConv-2 2×2…); the paper states M "double[s] the number of kernels in the hidden convolutional layers of the teacher and the student" and inserts 1×1 convolutions after the second pooling layer and after the last convolutional layer. **I am not reproducing Table 7 in full — treat the precise M architecture as not verified beyond those statements.**

**Distillation (teacher training).** The PDN is trained on **ImageNet** by minimising the MSE between its output and features from a **WideResNet-101** (the same pretrained features PatchCore uses, for controlled comparison). During distillation, inputs are resized to **512×512** for the pretrained extractor and to **256×256** for the PDN. Pretrained weights ship in the repo as `models/teacher_small.pth` (10.8 MB) and `models/teacher_medium.pth` (32.1 MB).

**Teacher channel normalisation (Appendix A.1, Algorithm 1, lines 6–13).** Before student training, compute per-channel mean `μ_c ∈ R^384` and std `σ_c ∈ R^384` of the teacher's outputs over the training set (`teacher_normalization` in the repo). At training and inference the teacher output is normalised:

```
Ŷ_c = (Y'_c − μ_c) · σ_c^{-1}
```

**Hard feature loss (the central novelty, §3.2).** With teacher `T(I) ∈ R^{C×W×H}` and student `S(I) ∈ R^{C×W×H}`:

```
D_{c,w,h} = ( T(I)_{c,w,h} − S(I)_{c,w,h} )²
d_hard    = the p_hard-quantile of the elements of D
L_hard    = mean of all D_{c,w,h} ≥ d_hard
```

`p_hard = 0.999` in all experiments (Table 3 ablation: `p_hard` = 0 → AUROC 94.9 = "original S–T loss"; 0.99 → 95.7; **0.999 → 96.0**; 0.9999 → 95.8; 0.99999 → 95.7, all EfficientAD-M on the three collections). Rationale: more training data lets the student generalise its imitation to anomalies; restricting backprop to the patches the student currently mimics worst (Online-Hard-Example-Mining-like) prevents that. The paper notes p_hard=0.999 corresponds to using "on average, ten percent of the values in each of the three dimensions of D."

**Pretraining penalty (§3.2).** Sample a random ImageNet image `P` each step:

```
L_ST = L_hard + (CWH)^{-1} Σ_c || S(P)_c ||_F²
```

i.e. the student's output on out-of-distribution images is driven to zero, hindering generalisation beyond the normal images.

**Autoencoder branch (§3.3).** A standard convolutional autoencoder `A` (strided convs in the encoder, bilinear upsampling in the decoder), **bottleneck of 64 latent dimensions**, trained to predict the *teacher's* output:

```
L_AE = (CWH)^{-1} Σ_c || T(I)_c − A(I)_c ||_F²
```

The student's output channels are doubled (768 = 384 + 384); the extra 384 channels `S'(I)` are trained to predict the autoencoder output:

```
L_STAE = (CWH)^{-1} Σ_c || A(I)_c − S'(I)_c ||_F²
```

Total training loss = `L_AE + L_ST + L_STAE`. Why this indirection: the autoencoder's reconstruction is *also* wrong on normal images (fine-grained patterns, background grids), so `T − A` alone produces false positives; the student learns the autoencoder's systematic reconstruction errors on normal data but cannot learn its errors on anomalies (never seen).

**Anomaly maps and image score (§3.2–§3.3).**
- Local map (student–teacher): `M^{ST}_{w,h} = C^{-1} Σ_c D^{ST}_{c,w,h}` — squared difference averaged across channels.
- Global map (autoencoder–student): `M^{AE}_{w,h} = C^{-1} Σ_c D^{STAE}_{c,w,h}`.
- Both maps are resized to the input image size by **bilinear interpolation**.
- **Combined map = average of the local and global maps**; **image-level anomaly score = the maximum value of the combined map.**

**Map normalisation (§3.4).** The two maps must be on comparable scales before averaging, otherwise noise in one drowns the other. Using held-out **validation** images (unseen images from the training set):
1. Collect all pixel anomaly scores across validation images, separately per map type.
2. For each set, compute two quantiles `q_a` (at p = a) and `q_b` (at p = b).
3. Fit the linear transform mapping `q_a → 0` and `q_b → 0.1`.
4. Apply that transform to the map at test time.

Table 3 ablations (EfficientAD-M, mean AUROC over the three collections): varying `a ∈ {0.5, 0.8, 0.9, 0.95, 0.98, 0.99}` gives 95.9/95.9/96.0/95.9/95.9/95.8; varying `b ∈ {0.95, 0.98, 0.99, 0.995, 0.998, 0.999}` gives 95.8/95.9/96.0/96.0/95.9/95.9. Defaults are highlighted in bold in the paper's Table 3 — **the exact default `(a, b)` pair is not verified from the text I extracted**; the repo's `map_normalization()` is the authoritative source. Note the paper's remark that the *destination values* (0 and 0.1) cannot affect AUROC, only the rank-invariant threshold-free metrics — AUROC depends on ranking alone; 0 and 0.1 are chosen purely for a 0–1 colour scale.

**Training hyperparameters (Appendix A.1, Algorithm 1).** Adam, lr `10^{-4}`, weight decay `10^{-5}`, **70 000 iterations**. Training images 3×256×256; validation images 3×256×256. Autoencoder trained on *augmented* inputs: at each step pick one of {brightness, contrast, saturation} with coefficient λ ~ U(0.8, 1.2). EfficientAD-M swaps in Table 7's architecture.

**Inference.** Resize input to **256×256**, compute maps, resize the map back to the original image size by bilinear interpolation. All evaluated methods run in **float16** during inference ("Switching from float32 to float16 for the inference of EfficientAD does not change the anomaly detection results for the 32 scenarios"). In latency-critical settings, class-specific rules are avoided — training/validation contain no anomalous images.

**LEARNED student–teacher detector vs RETRIEVAL bank — the distinction to draw.** These are two fundamentally different mechanisms and EfficientAD is deliberately on the *learned* side:

| | EfficientAD (R3) | PatchCore (R1) | AnomalyDINO (R2) |
|---|---|---|---|
| What determines the anomaly score | **Parameters learned during training** on nominal images (student weights, autoencoder weights) | **Distances to stored feature vectors** from the training set (no parameters fit at test-relevant time) | **Distances to stored feature vectors** (no training at all) |
| Reference representation | none — the "reference" is compressed into the student's weights | explicit memory bank `M_C` of `l = percentage·|M|` feature vectors | explicit memory bank `M` of all reference patch tokens |
| Direction of the comparison | teacher output vs. **predicted** teacher output | test patch vs. **stored** nominal patch | test patch vs. **stored** nominal patch |
| Growth with training-set size | constant at inference (fixed parameter count) | **linear** in training images × patches until the coreset caps it | **linear** in k and in the augmentation factor |
| Failure mode | student over-generalises to anomalies (mitigated by `L_hard` and the ImageNet penalty) | bank too sparse → missed defect; bank too dense → slow | bank too sparse (few-shot) → false positives from background |
| Test-time memory | fixed 100 MB (S) / 161 MB (M), independent of dataset size | scales with the bank; the paper notes PatchCore-Ens "contains 8 million values" | scales with bank size; no coreset in the repo |

Concretely for the student: PatchCore's `NearestNeighbourScorer` holds an explicit FAISS index and *retrieves* neighbours; EfficientAD's `EfficientAdModel` holds no index at all — it *predicts* the teacher's features and measures the residual. EfficientAD's "reference" is 8 M parameters (S) rather than a vector database.

**Latency measurement protocol (Appendix E, "Timing Methodology") — reproduce this exactly if you compare against it.**
- **Latency** = inference runtime to generate the anomaly detection result for **a single image** (batch size **1**). **Throughput** = images per second with **batch size 16**, computed as `16000 / (sum of the runtimes of 1000 forward passes at batch size 16)`.
- All methods implemented in PyTorch; **all executed on a GPU**, including PatchCore's nearest-neighbour search (all methods run faster on GPU than CPU in their setup).
- **Clock start:** the transfer of the test image from CPU to GPU. **Clock stop:** when the anomaly detection result — an anomaly map for every method — is **available on the CPU**. So the measurement is an end-to-end host-to-host path, not a kernel-only time.
- **Exclusions and fixed settings for fairness:** remove unnecessary parts such as loss computation during inference; **float16 for all networks**; for methods using hidden-layer features, exclude layers not needed (classification heads etc.).
- **EfficientAD specifics:** timed **without padding** in the PDN (disabling padding speeds the PDN forward pass by **80 µs** and does not impair detection — and the paper's headline results are reported for this no-padding setting). 1000 warm-up forward passes, then report the **mean of the following 1000 forward passes**.
- **FLOPs:** `with torch.profiler.profile(with_flops=True)` (PyTorch 1.12.0). **GPU memory:** peak reserved memory via `torch.cuda.memory_stats()['reserved_bytes.all.peak']`. Both measured for single-image processing, mean over 1000 passes.
- **Hardware:** main table on **NVIDIA RTX A6000**; per-GPU latency in Table 17 also covers RTX A5000, Tesla V100, RTX 3080, RTX 2080 Ti. Claim: "The ranking of methods is the same on each GPU, except for two cases in which DSR is slightly faster than FastFlow."
- **PatchCore baseline caveat:** because PatchCore's parameter/FLOP/memory footprint depends on training data, the authors benchmark it on VisA's **"cashew"** scenario (450 training images, closest to the 439-image average of the 32 scenarios). They report feature-extraction and kNN costs separately, and could not measure kNN FLOPs/memory with the library PatchCore uses. They also **disable PatchCore's center-76.6% cropping**, because 99.9% of MVTec AD defects fall inside that crop and relying on it implies test-set knowledge.
- The paper's broader methodological caution: parameter count and FLOPs are unreliable proxies for latency ("the number of FLOPs of S–T is more than 2000% higher than that of AST, but the latency is only 42% higher").

### 4. Reported numbers

Dataset protocol: 32 anomaly-detection scenarios across **MVTec AD, VisA, MVTec LOCO**. Image-level AU-ROC for detection; **AU-PRO up to a false-positive rate of 30%** for localization (as recommended by [7]); **AU-sPRO** for MVTec LOCO logical anomalies. Per-collection means are computed per scenario then averaged; the three-collection average is the average of the three collection means (which weights logical anomalies ~1/6 and structural ~5/6). EfficientAD numbers are means of **five runs**.

**Table 1 — detection AU-ROC, segmentation AU-PRO, latency, throughput** (all on NVIDIA RTX A6000):

| Method | Detect. AU-ROC | Segment. AU-PRO | Latency (ms) | Throughput (img/s) |
|---|---|---|---|---|
| GCAD | 85.4 | 88.0 | 11 | 121 |
| SimpleNet | 87.9 | 74.4 | 12 | 194 |
| S–T | 88.4 | 89.7 | 75 | 16 |
| FastFlow | 90.0 | 86.5 | 17 | 120 |
| DSR | 90.8 | 78.6 | 17 | 104 |
| PatchCore | 91.1 | 80.9 | 32 | 76 |
| PatchCore Ens | 92.1 | 80.7 | 148 | 13 |
| AST | 92.4 | 77.2 | 53 | 41 |
| **EfficientAD-S** | **95.4 (± 0.06)** | **92.5 (± 0.05)** | **2.2 (± 0.01)** | **614 (± 2)** |
| **EfficientAD-M** | **96.0 (± 0.09)** | **93.3 (± 0.04)** | **4.5 (± 0.01)** | **269 (± 1)** |

**Table 16 (Appendix E) — same rows extended with params / FLOPs / GPU memory:**

| Method | Latency (ms) | Throughput | Params (×10⁶) | FLOPs (×10⁹) | GPU mem (MB) |
|---|---|---|---|---|---|
| GCAD | 11 | 121 | 65 | 416 | 555 |
| SimpleNet | 12 | 194 | 73 | 38 | 508 |
| S–T | 75 | 16 | 26 | 4468 | 1077 |
| FastFlow | 17 | 120 | 92 | 85 | 404 |
| DSR | 17 | 104 | 40 | 267 | 314 |
| PatchCore | 32 | 76 | 83 + 3 | 41 + kNN | 637 + kNN |
| PatchCore Ens | 148 | 13 | 150 + 8 | 159 + kNN | 1335 + kNN |
| AST | 53 | 41 | 154 | 199 | 618 |
| **EfficientAD-S** | **2.2** | **614** | **8 (± 0)** | **76 (± 0)** | **100 (± 0)** |
| **EfficientAD-M** | **4.5** | **269** | **21 (± 0)** | **235 (± 0)** | **161 (± 0)** |

The abstract's "latency of two milliseconds and a throughput of six hundred images per second" = the EfficientAD-S row (2.2 ms, 614 img/s).

**Table 2 — per-collection detection AU-ROC (%), plus MVTec LOCO logical/structural split:**

| Method | MVTec AD | MVTec LOCO | VisA | Mean | LOCO Logical | LOCO Structural |
|---|---|---|---|---|---|---|
| GCAD | 89.1 | 83.3 | 83.7 | 85.4 | 83.9 | 82.7 |
| SimpleNet | 98.2 | 77.6 | 87.9 | 87.9 | 71.5 | 83.7 |
| S–T | 93.2 | 77.4 | 94.6 | 88.4 | 66.5 | 88.3 |
| FastFlow | 96.9 | 79.2 | 93.9 | 90.0 | 75.5 | 82.9 |
| DSR | 98.1 | 82.6 | 91.8 | 90.8 | 75.0 | 90.2 |
| PatchCore | 98.7 | 80.3 | 94.3 | 91.1 | 75.8 | 84.8 |
| PatchCore Ens | 99.3 | 79.4 | 97.7 | 92.1 | 71.0 | 87.7 |
| AST | 98.9 | 83.4 | 94.9 | 92.4 | 79.7 | 87.1 |
| **EfficientAD-S** | **98.8** | **90.0** | **97.5** | **95.4** | **85.8** | **94.1** |
| **EfficientAD-M** | **99.1** | **90.7** | **98.1** | **96.0** | **86.8** | **94.7** |

Reading guide: EfficientAD is essentially at parity with PatchCore/PatchCore-Ens on **MVTec AD** (98.8 / 99.1 vs 98.7 / 99.3) — its advantage is overwhelmingly on **MVTec LOCO** (90.0 / 90.7 vs 80.3 / 79.4), i.e. the logical-anomaly branch, and it is competitive on VisA. All while being ~15–60× lower latency than PatchCore variants. Note also the paper's own framing caution: "Performing method development solely on MVTec AD (MAD) becomes prone to overfitting design choices to the few remaining misclassified test images." EfficientAD with early stopping enabled would reach **99.8%** image AU-ROC on MVTec AD — the authors deliberately **disable** early stopping (and SimpleNet's test-set model selection), calling it an overestimate requiring a validation set of anomalous images.

**Table 17 — latency (ms) per GPU:**

| Method | RTX A6000 | RTX A5000 | Tesla V100 | RTX 3080 | RTX 2080 Ti |
|---|---|---|---|---|---|
| EfficientAD-S | 2.2 | 2.5 | 3.9 | 3.8 | 4.5 |
| EfficientAD-M | 4.5 | 5.3 | 6.3 | 7.0 | 7.6 |
| GCAD | 10.7 | 11.7 | 12.9 | 13.7 | 18.0 |
| SimpleNet | 12.0 | 13.3 | 19.2 | 18.1 | 21.9 |
| FastFlow | 16.5 | 17.1 | 26.1 | 27.5 | 31.0 |
| DSR | 17.2 | 18.0 | 24.8 | 24.6 | 34.5 |
| PatchCore | 32.0 | 31.5 | 47.1 | 41.1 | 53.2 |
| AST | 53.1 | 53.4 | 75.6 | 82.3 | 87.1 |
| S–T | 74.7 | 81.0 | 82.2 | 99.6 | 121.7 |
| PatchCore Ens | 147.6 | 145.0 | 229.2 | 189.0 | 216.9 |

**Official-repo MVTec AD numbers** (from the authors' own result files, `results/mvtec_ad_small.json` / `results/mvtec_ad_medium.json` — these are per-15-category means, not the 3-collection means above):

| Model | Mean image AU-ROC | Mean AU-PRO |
|---|---|---|
| EfficientAD-S | 0.9903 (99.03%) | 0.9328 (93.28%) |
| EfficientAD-M | 0.9909 (99.09%) | 0.9374 (93.74%) |

Per-category EfficientAD-S (`classification_au_roc` / `au_pro`), from `results/mvtec_ad_small.json`: bottle 0.9992/0.9519, cable 0.9352/0.8823, capsule 0.9844/0.9760, carpet 0.9944/0.9183, grid 0.9975/0.8932, hazelnut 0.9943/0.9484, leather 1.0000/0.9822, metal_nut 0.9951/0.9362, pill 0.9850/0.9703, screw 0.9787/0.9558, tile 0.9982/0.8815, toothbrush 1.0000/0.9456, transistor 0.9992/0.9027, wood 0.9965/0.9043, zipper 0.9963/0.9436. These align closely with the paper's MVTec-AD column (98.8 / 99.1) and are the numbers to reproduce if you run the repo.

**Ablation (Table 4, EfficientAD-M, mean over the three collections):** PDN alone → 93.2 AU-ROC at 2.2 ms; with map normalisation → **94.0** (+0.8). The hard feature loss alone adds **+1.0 AU-ROC**; the pretraining penalty adds more. The paper's framing: the map normalisation, hard feature loss, and pretraining penalty together "keep the computational requirements of EfficientAD low, while creating a substantial margin w.r.t. the anomaly detection performance." The full Table 4/Table 5 rows beyond these are **not reproduced here (not verified)** — the HTML I extracted shows the table truncated at the "with map normalization +0.8" row.

**Pixel-level AU-ROC, AU-PR:** the paper says "Appendix D provides the results for additional anomaly detection metrics, such as the area under the precision-recall curve and the pixel-wise AU-ROC" — Appendix D.2 "Anomaly Localization" exists but I did not extract its tables, so **pixel-AUROC numbers for EfficientAD are not verified** in these notes. The paper's headline localization metric is AU-PRO (92.5 S / 93.3 M), not pixel-AUROC.

### 5. What a beginner should SKIP on a first read

- **The autoencoder / logical-anomaly branch (§3.3) and the whole MVTec LOCO story.** This is a second, largely independent method bolted on. If you are reading EfficientAD as a fast *structural* detector — which for a PatchCore/AnomalyDINO comparison is the point — read §3.1 + §3.2 + the local map in §3.2, and treat the global branch as a separate later topic. Note the honest dependency: EfficientAD's headline *average* advantage over PatchCore comes mostly from LOCO, so if your plan is MVTec-AD-only, the interesting comparison is 98.8/99.1 vs 98.7/99.3 at 2.2/4.5 ms vs 32 ms — a **latency** win, not an accuracy win.
- **Table 7 (EfficientAD-M architecture) on a first read.** It is a duplicated, differently-shaped PDN. Read EfficientAD-S, get it working, then decide whether you need M.
- **The quantile-normalisation derivation (§3.4) and Table 3's `a`/`b` sweep.** The sensitivity is provably flat (95.8–96.0 across the whole sweep), and the paper itself notes AUROC cannot see the destination values. First read: "map both maps into a common scale using validation-set quantiles; if the anomaly shows in only one map, this is what lets it survive the average." The specific `a`, `b` you should copy from `map_normalization()` in the repo, not from the table.
- **Appendix E's FLOPs and parameter columns, and the "Interpretability of Efficiency Metrics" discussion.** Valuable context for a *research plan* (it is the paper's argument for why you must measure latency rather than parameter count) but skippable when you are learning the method. If anything, read just the two-sentence S–T example — 4.5 TFLOPs but only 42% higher latency than AST.
- **Appendix C (robustness to the distillation backbone) and the per-scenario Appendix D tables** — 32 scenarios × several methods; they support the average, they do not change the method.
- **The `SimpleNet`/`PatchCore` re-benchmarking minutiae** (disabling cropping, disabling early stopping, disabling test-set model selection). Important for the paper's *fairness argument* and worth a paragraph in a research plan's "experimental protocol" section — but skip on a first read of the method itself. Do **not** skip the conclusion it draws: any method that selects its stopping point on test data (SimpleNet, FastFlow with early stopping enabled) will look better than it is.
- **Table 17's five-GPU latency matrix** on a first read — one number (2.2 ms on A6000) is enough until you know your deployment hardware. Come back to it to calibrate expectations for an older GPU (2.2 → 4.5 ms on a 2080 Ti, i.e. ~2×).

### 6. Exact repository locations

**Official repo: <https://github.com/nelson1425/EfficientAD>** (default branch `main`). Flat layout, no package structure — every file is top-level.

| Step | File | Symbol |
|---|---|---|
| Training + test driver (CLI) | `efficientad.py` | `get_argparse()`, `main()`, `train_transform(image)` |
| Inference per test set (the local/global maps + score) | `efficientad.py` | `test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std, ...)` |
| Single-image inference path (this is the latency-critical function) | `efficientad.py` | `predict(image, teacher, student, autoencoder, teacher_mean, teacher_std, ...)` |
| Quantile-based map normalisation (§3.4) | `efficientad.py` | `map_normalization(validation_loader, teacher, student, autoencoder, ...)` |
| Teacher per-channel mean/std (§3.2, Appendix A.1 lines 6–13) | `efficientad.py` | `teacher_normalization(teacher, train_loader)` |
| Autoencoder definition (64-dim bottleneck) | `common.py` | `get_autoencoder(out_channels=384)` |
| PDN-S (Table 6) | `common.py` | `get_pdn_small(out_channels=384, padding=False)` |
| PDN-M (Table 7) | `common.py` | `get_pdn_medium(out_channels=384, padding=False)` |
| Dataloaders / batching | `common.py` | `class ImageFolderWithoutTarget`, `class ImageFolderWithPath`, `InfiniteDataloader(loader)` |
| **PDN distillation on ImageNet** (teacher pretraining) | `pretraining.py` | `main()`, `train_transform(image)`, `feature_normalization(extractor, train_loader, steps=10000)` |
| The WideResNet-101 feature extractor used as distillation target | `pretraining.py` | `class FeatureExtractor(torch.nn.Module)` (plus copies of PatchCore's `class PatchMaker`, `class Preprocessing`, `class MeanMapper`, `class Aggregator`, `class NetworkFeatureAggregator`, `class ForwardHook`, `class LastLayerToExtractReachedException` — the distillation pipeline is literally adapted from PatchCore's code, which is a useful cross-reference between R1 and R3) |
| Pretrained teacher weights | `models/teacher_small.pth` (10.8 MB), `models/teacher_medium.pth` (32.1 MB) | — |
| Reproduction entry point | `benchmark.py` | `main()` |
| Reference results | `results/mvtec_ad_small.json`, `results/mvtec_ad_medium.json` | keys `mean_classification_au_roc`, `mean_au_pro`; per-category `classification_au_roc`, `au_pro`, ROC curve arrays |

**Alternative implementation (actively maintained, production-oriented): <https://github.com/open-edge-platform/anomalib> at `src/anomalib/models/image/efficient_ad/`.**

| Step | File | Symbol |
|---|---|---|
| Lightning module (training loop, optimiser, validation) | `lightning_model.py` | `class EfficientAd` |
| Model + losses + maps | `torch_model.py` | `class EfficientAdModel`, `class SmallPatchDescriptionNetwork`, `class MediumPatchDescriptionNetwork`, `class Encoder`, `class Decoder`, `class AutoEncoder`, `class EfficientAdModelSize` (enum) |
| Hard feature loss + pretraining penalty + AE losses | `torch_model.py` | `EfficientAdModel.compute_losses()` — contains the authoritative lines `d_hard = torch.quantile(distance_st, 0.999)`, `loss_hard = torch.mean(distance_st[distance_st >= d_hard])`, `loss_penalty = torch.mean(student_output_penalty**2)`, `loss_st = loss_hard + loss_penalty` |
| Local (`map_st`) and global (`map_stae`) maps + quantile normalisation | `torch_model.py` | `EfficientAdModel.compute_maps()` (note it returns the **two maps separately** and a separate caller averages them, and `_normalize_teacher()`), `EfficientAdModel.get_maps()`, `EfficientAdModel.compute_student_teacher_distance()` |
| Config (image size, quantiles, model size) | `examples/configs/model/efficient_ad.yaml` | — |
| Docs | `docs/source/markdown/guides/reference/models/image/efficient_ad.md` | — |

The anomalib `compute_losses` is the cleanest single place to read the hard feature loss and the ImageNet penalty in isolation — it matches Eq. `L_ST` above line for line, and it is the function to read if the paper's Algorithm 1 markup is hard to follow.

### 7. Learning outcomes (three)

1. **Be able to distinguish a learned student–teacher detector from a retrieval/memory-bank method, and say what each implies operationally.** Concretely: EfficientAD stores no reference vectors — the training set is compressed into 8 M (S) / 21 M (M) student weights, so test-time memory is constant at 100 / 161 MB regardless of dataset size and each image costs 2.2 / 4.5 ms on an A6000; PatchCore and AnomalyDINO store explicit feature banks whose size grows with the training set (PatchCore's Ens bank alone holds 8 M values per the paper's Appendix E) and pay a nearest-neighbour search at inference (32 ms for PatchCore in the same measurement). Be able to name the failure mode each incurs: retrieval banks fail by being too sparse (missed defect) or too dense (slow), student–teacher fails by the student generalising its imitation to anomalies — which is exactly what `L_hard` and the ImageNet penalty exist to prevent.
2. **Be able to reproduce the latency/throughput protocol exactly and to argue why parameter count and FLOPs are not substitutes for it.** That means: single-image latency at batch size 1, throughput at batch size 16 as `16000 / Σ r_i` over 1000 passes at that batch size, 1000 warm-up passes first, float16 for all networks, clock from CPU→GPU image transfer to the anomaly map being back on the CPU, EfficientAD specifically timed without PDN padding (worth 80 µs), and the paper's counterexamples — S–T's 4.5 × 10¹² FLOPs with a latency only 42% above AST's, and FastFlow's ~2.5× parameter count of S–T but 4.4× lower latency and 6.5× higher throughput. Be able to explain the underlying reason: FLOPs ignore how well the work parallelises, and parameter count ignores how many times each parameter is applied at a given feature-map resolution.
3. **Be able to read the headline claim with the right scope and the right caveats.** "95.4% AU-ROC at 2.2 ms" is the mean over MVTec AD + VisA + MVTec LOCO with EfficientAD-S on an RTX A6000; on **MVTec AD alone** it is 98.8 (S) / 99.1 (M) against PatchCore's 98.7 / PatchCore-Ens' 99.3 — i.e. the accuracy advantage essentially disappears on MVTec AD and the win there is latency, while the 3-collection lead is driven by MVTec LOCO's logical anomalies (90.0 / 90.7 vs 80.3 / 79.4). Be able to state that the 99.8% MVTec-AD figure quoted in the paper requires early stopping (test-set-derived stopping) and is **deliberately not** the reported result, and that the reported PatchCore baseline has center-cropping disabled for the same fairness reason — so a naive head-to-head table using each author's best self-reported number would be comparing incompatible protocols.

---

## Cross-cutting notes for the research plan

**Numbers whose exact pairing is easy to get wrong — double-check before citing:**
- PatchCore's 99.6% image AUROC and its 98.4% pixel AUROC come from **different rows of Table 4** (99.6 is DenseNet-201+ResNeXt-101+WRN-101 @320; 98.4 is WRN-101 (1+2+3) @280). The README's single WR50 baseline is 99.2/98.1/94.4, and the paper's WR50 @224 numbers are 99.1/98.1(PatchCore-10%)/93.5.
- AnomalyDINO's abstract "93.1% → 96.6%" is WinCLIP+ → AnomalyDINO-S **@672**; the @448 1-shot number is 96.5%.
- EfficientAD's "2 ms / 600 img/s" is EfficientAD-**S** on an **A6000**; its 96.0 AU-ROC is EfficientAD-**M** at 4.5 ms. The abstract's cachet comes from combining the S latency with the M-level accuracy claim across tables.

**Explicit "not verified" list (do not present these as facts):**
- PatchCore's exact per-point low-shot AUROC values (Table S5 not extracted).
- PatchCore's hardware string — the paper says "Nvidia Tesla V4 GPUs", which is likely a typo for V100; reported as written.
- AnomalyDINO's per-configuration latency beyond "roughly 60 ms @448 on an NVIDIA A40" (Appendix C Table 9 not extracted), and **any** AnomalyDINO memory footprint figure (none found).
- The size of AnomalyDINO's improvement from the top-1% mean over the max-based score (Appendix C.2 not extracted) — the *direction* is stated in §4.2, the *magnitude* is not.
- EfficientAD-M's full PDN architecture (Table 7 beyond the first rows), the exact default `(a, b)` quantile pair for map normalisation, and all EfficientAD pixel-level AUROC / AU-PR values (Appendix D not extracted).
- The provenance of `github.com/nelson1425/EfficientAD` as the paper's own repository: the paper prints no code URL; the repo name/author match and its files mirror the paper's algorithm exactly, but I found no line in the paper asserting it.

**Highest-leverage cross-paper engineering fact for a practical plan:** the three methods partition cleanly by *where the reference knowledge lives* — in a coreset-subsampled vector index (R1), in the full few-shot token bank (R2), or in student weights (R3). That single distinction predicts their scaling behaviour in memory (constant for R3; linear in training data for R1/R2), their failure modes (bank sparsity for R1/R2; student over-generalisation for R3), and their latency profile (dominated by NN search for R1/R2; by the fixed PDN forward pass for R3).
