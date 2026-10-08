# Verified Reading Notes: Industrial Visual Anomaly Detection and Metric Nearest-Neighbour Search

**Verification status.** Every number below was read off a primary source (arXiv full text, arXiv LaTeX source, the publisher's landing metadata via Crossref, the MVTec dataset page, or the authors' own page). Anything I could not check is marked **not verified**. Where the preprint and the published version could differ, this is flagged. Backing files used: arXiv HTML `2503.21622v1`, the arXiv LaTeX source of `2503.21622` (downloaded from `https://arxiv.org/e-print/2503.21622`), arXiv HTML `2608.18585v1`, `https://hunch.net/~jl/projects/cover_tree/paper/paper.pdf` (extended version, 13 pp.), `https://hunch.net/~jl/projects/cover_tree/icml_final/final-icml.pdf` (ICML camera-ready), the ICML 2023 PDF `https://proceedings.mlr.press/v202/elkin23a/elkin23a.pdf`, and arXiv `2208.09447`.

---

## [R4] MVTec AD 2

### Full citation

Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger. "The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection." *International Journal of Computer Vision* **134**(4), 2026. DOI [10.1007/s11263-026-02743-0](https://doi.org/10.1007/s11263-026-02743-0). Preprint: [arXiv:2503.21622](https://arxiv.org/abs/2503.21622) (v1, submitted 27 March 2025; cs.CV). Dataset page: [mvtec.com/research-teaching/datasets/mvtec-ad-2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2). Evaluation server: [benchmark.mvtec.com](https://benchmark.mvtec.com/).

Crossref confirms the publisher record: title, journal *International Journal of Computer Vision*, vol. 134, issue 4, published 2026-03-09, authors as listed ([Crossref API](https://api.crossref.org/works/10.1007/s11263-026-02743-0)).

> **Caveat on version.** `https://link.springer.com/content/pdf/10.1007/s11263-026-02743-0.pdf` returned an HTML block page (paywall), so **the published IJCV text was not read**. All technical detail below comes from the arXiv v1 preprint (full text plus its LaTeX source) and the official dataset page. The arXiv v1 is marked "paper under review", so wording in the IJCV version **may differ — not verified** which specific numbers changed. A second, smaller discrepancy is noted inline below.

### Problem

Performance on MVTec AD and VisA has saturated in segmentation AU-PRO, with state-of-the-art models separated by under one percentage point — too little to discriminate methods given the stochasticity of ML results. Existing benchmarks also miss industrially relevant regimes: transparent and overlapping objects, dark-field and back light illumination, high normal-class variability, and very small defects. They also lack real (non-synthetic) test-time distribution shift.

### Core idea

Eight new industrial anomaly-detection scenarios, 8,004 high-resolution images total (2.6–5 MP), split into defect-free train/validation plus a three-way test split, one part public with pixel-precise ground truth and two parts private (image data public, ground truth only on an evaluation server). Every scene is captured under **at least four different lighting conditions**, and lighting change is used as a *controlled* distribution shift between train and test. The private ground truth enforces the unsupervised setting by preventing test-set tuning.

### Precise technical content

**(a) The exact eight categories** (paper names; on-disk / figure-file names from the arXiv LaTeX source are given in parentheses, and they match anomalib's `CATEGORIES` tuple):

| # | Category | On-disk name |
|---|---|---|
| 1 | Can | `can` |
| 2 | Fabric | `fabric` |
| 3 | Fruit Jelly | `fruit_jelly` |
| 4 | Rice | `rice` |
| 5 | Sheet Metal | `sheet_metal` |
| 6 | Vial | `vial` |
| 7 | Wall Plugs | `wallplugs` |
| 8 | Walnuts | `walnuts` |

Scenario design: bulk goods with overlapping/occluded, uncontrolled placement → **Wall Plugs, Walnuts, Rice**; textured with high normal variability → **Fabric, Sheet Metal**; reflective metal → **Sheet Metal, Can**; transparent → **Vial, Fruit Jelly**. Sheet Metal uses directed **dark-field** front light; Vial and Fruit Jelly use **back light**. Sheet Metal, Vial and Wall Plugs are **single-channel gray-value**, not RGB. Image resolutions 2.6–5 MP with varying aspect ratios.

**(b) Images per split** (Table IV of the paper; counts are `normal/anomalous` in the "without/with anomalies" column):

| Object | # Train | # Val | TEST_pub | TEST_priv | TEST_priv,mix | Test total | Image size (W×H) |
|---|---|---|---|---|---|---|---|
| Can | 412 | 46 | 162 (72/90) | 321 (145/176) | 321 (145/176) | 804 (362/442) | 2232 × 1024 |
| Fabric | 387 | 43 | 156 (66/90) | 314 (133/181) | 314 (133/181) | 784 (332/452) | 2448 × 2048 |
| Fruit Jelly | 263 | 37 | 80 (20/60) | 255 (71/184) | 255 (71/184) | 590 (162/428) | 2100 × 1520 |
| Rice | 313 | 35 | 132 (42/90) | 277 (96/181) | 277 (96/181) | 686 (234/452) | 2448 × 2048 |
| Sheet Metal | 137 | 19 | 114 (24/90) | 142 (36/106) | 142 (36/106) | 398 (96/302) | 4224 × 1056 |
| Vial | 291 | 41 | 140 (35/105) | 276 (78/198) | 276 (78/198) | 692 (191/501) | 1400 × 1900 |
| Wall Plugs | 293 | 33 | 150 (60/90) | 232 (96/136) | 232 (96/136) | 614 (252/362) | 2448 × 2048 |
| Walnuts | 432 | 48 | 150 (60/90) | 228 (93/135) | 228 (93/135) | 606 (246/360) | 2448 × 2048 |
| **Σ** | **2,528** | **302** | **1,084** | **2,045** | **2,045** | **5,174** | — |

Unit of every parenthetical is `normal/anomalous` images. The final "Test total" column is that object's test images summed over all three test parts (`TEST_pub + TEST_priv + TEST_priv,mix`), and its parenthetical is the corresponding normal/anomalous total — e.g. for Can, 162 + 321 + 321 = 804, and 72 + 145 + 145 = 362. The Σ row is my own column-wise sum of the paper's Table IV: TEST_pub = 1,084; TEST_priv = 2,045; TEST_priv,mix = 2,045; test total = 5,174. The overall total is 2,528 + 302 + 5,174 = **8,004**, matching the paper's stated "a total of 8,004 high-resolution images" (the abstract's "more than 8000" is the rounded statement). Note **TEST_priv and TEST_priv,mix have identical counts** — see (c).

**(c) Split roles, exactly.** The paper: *"MVTec AD 2 comprises a training, validation, and test set. In line with the unsupervised setting, the training and validation sets contain only non-anomalous data."*

| Split | Content | Ground truth | Permitted use (verbatim intent) |
|---|---|---|---|
| **TRAIN** | defect-free only; **regular lighting only** ("since illumination changes in practice will occur mainly during test time") | none needed | Train the model on anomaly-free training data. |
| **VALIDATION** | defect-free only | none needed | "Use defect-free validation images to derive hyperparameters such as training duration or thresholds for binary segmentation." This is the *only* sanctioned place to fix a segmentation threshold. |
| **TEST_pub** (public test) | "a small number of normal and anomalous images with their corresponding segmentation ground truth **for all lighting conditions**" | **pixel-precise, released** | "facilitating local testing and an initial performance estimation." Explicitly **optional**: "Optionally, use TEST_pub for initial performance estimation." |
| **TEST_priv** (private test) | "images with the **same lighting conditions as the training set**" | image data public; **ground truth private**, scoring only via the server | Leaderboard submission; measures performance without distribution shift. |
| **TEST_priv,mix** (private mixed test) | "**depicts the same scenes as TEST_priv** but includes both seen and unseen lighting conditions, **randomly selected for each test image**, encompassing both normal and anomalous data" | image data public; **ground truth private**, scoring only via the server | Leaderboard submission; isolates robustness to the illumination shift. |

**Which categories are public test vs private test.** This is the point most easily got wrong: **there is no category-level public/private partition**. All eight categories (Can, Fabric, Fruit Jelly, Rice, Sheet Metal, Vial, Wall Plugs, Walnuts) appear in **all three** test parts. What is partitioned is the *images within each category*, and for TEST_priv / TEST_priv,mix only the *ground truth* is withheld. Concretely, for every one of the eight categories: TEST_pub is public with masks; TEST_priv and TEST_priv,mix have public images but non-public masks. Do **not** write "categories X, Y are private test categories" — that is wrong for this dataset.

**A documented discrepancy between sources.** The official dataset page says *"The test data is split into **two** parts"* — the first with pixel-precise annotations, the second with non-public ground truth. The paper says *"The test data is divided into **three** parts."* Both are reconcilable if the page's "parts" means public-vs-private while the paper counts `pub` / `priv` / `priv,mix`; on the server directory structure the private side is split into two directories, `private` and `private_mixed`. For a written plan, quote the **paper's three-way wording** and note the page's two-part phrasing.

**(d) Illumination-change scenarios, precisely.** Verbatim from the paper: *"For every object, we provide images that are brighter or darker than the images seen during training by adjusting the exposure time accordingly."* Further, object-specific shifts:

- dark-field illumination → **two lighting setups with different illumination angles**;
- reflective objects → **additional LED spot lights**, creating reflections;
- bulk goods → **additional spot lights**, producing **uneven light distribution** within the scene;
- **slight changes in colour temperature** (Rice, Walnuts);
- **"For all object categories, at least four different lighting conditions were used."**
- **"When varying the lighting condition, all other parameters such as the position of the object or the camera parameters (apart from the exposure time) were kept constant."**

Where the shifts live: *"For inducing lighting condition changes in the test data, i.e., TEST_pub and TEST_priv,mix, we varied the exposure time and installed additional light sources used in combination with the regular lighting."* So the lighting-changed images are in **TEST_pub and TEST_priv,mix**, and **not** in TEST_priv.

**(e) The low-FPR localisation evaluation protocol.** Localisation is scored by **area under the per-region-overlap (PRO) curve** at pixel level, AU-PRO. Per-region overlap weights each ground-truth anomaly region equally (unlike pixel AU-ROC), because "industrial applications require the detection of every single defect regardless of its size". AU-PRO integrates the PRO curve over the false-positive rate:

> "Typically, the PRO curve is integrated up to a false positive rate (FPR) of 0.3 (AU-PRO_0.30). However, because of the utmost importance of an accurate anomaly segmentation ... we decrease the tolerance for false positive pixels and establish an **integration limit for the PRO curve of 0.05** for MVTec AD 2 (**AU-PRO_0.05**)."

The stated justification is quantitative: for a defect of 10 pixels in a 5 MP image, an FPR integration limit of even 5 % corresponds to a wrongly segmented area "more than 25,000 times larger than the defect itself". AU-PRO_0.30 is still reported "to highlight the difficulty of our dataset even with respect to the more tolerant evaluation setting". **AU-PRO_0.05 is the headline metric of the benchmark.**

**Threshold-dependent metrics also exist** and matter for a threshold study: the segmentation threshold is *"the average of the per-pixel anomaly values of the defect-free validation images plus three times their standard deviation"*, i.e. a mean + 3σ rule calibrated on defect-free validation data only. Submitted systems may **also** submit a segmentation threshold alongside the continuous anomaly maps, so the server can score threshold-dependent metrics too.

**(f) Verified headline numbers.** Best model **58.7 % AU-PRO_0.30** on MVTec AD 2 (at 256 × 256), versus > 90 % AU-PRO_0.30 on MVTec AD. **All methods achieve less than 31 % AU-PRO_0.05 on TEST_priv.** EfficientAD is best in the unchanged-lighting setting at **30.8 % AU-PRO_0.05 on TEST_priv**. Robustness spread between the two private sets: **RD degrades only 1.4 pp** from TEST_priv to TEST_priv,mix, whereas **MSFlow drops 12.4 pp (24.3 % → 11.9 %)**. Threshold-independent and threshold-dependent metrics **can disagree**: *"PatchCore performs significantly worse than all other methods in terms of F1 score, yet achieves similar results in terms of AU-PRO."* Increasing input resolution up to half the original width/height roughly **doubles** PatchCore's and MSFlow's AU-PRO_0.05 on Rice, at >10× inference-time and memory cost; PatchCore then takes ~2 s/image, and SimpleNet exceeds available memory at the largest size (measured on a single NVIDIA RTX 2080 Ti).

**(g) Licence terms.** The data is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Licence (CC BY-NC-SA 4.0)**. The dataset page adds the operative restriction: *"In particular, it is not allowed to use the dataset for commercial purposes. If you are unsure whether or not your application violates the non-commercial use clause of the license, please contact us."* Access requires filling in a download form. This is a **non-commercial-only** licence — relevant if the compression study has any industrial partner.

**(h) The official loader / evaluation utilities — and their names.** What is officially documented:

- The dataset page offers a bundle labelled **"Code Utils for Development"** (download link *"Download code utils"*, gated by the form) containing, in the page's words: *"A PyTorch dataset class for MVTec AD 2 ... which can be easily integrated into a PyTorch dataloader and be used to store anomaly images in the correct structure for evaluation"*; *"a script that checks a submission for correctness and compresses it for you"*; and *"snippets to measure runtime and memory footprint of a method."*
- The paper states: *"code snippets for measuring inference runtime and memory consumption can be downloaded along with **general evaluation code** and a **PyTorch data loader** for easy integration of MVTec AD 2."*
- Submission layout (official, from the paper): TIFF anomaly maps under
  `/mvtec_ad_2/{object_name}/{private,private_mixed}/anomaly_images/test/{good,bad}/{image_name}.tiff`
  followed by local structure checks, automatic zipping, upload, and server-side scoring.

**Not verified: the official class, module, function or script file names.** These are the strong candidates and my checks:
- No public official repository: `https://api.github.com/orgs/mvtec` returns **HTTP 404** (no `mvtec` GitHub organisation).
- No official PyPI package: `https://pypi.org/pypi/mvtec-ad2/json` → 404; `https://pypi.org/pypi/mvtecad2/json` → 404.
- The download is form-gated, so the archive's internal file names could not be inspected.
- The evaluation-server front end at `https://benchmark.mvtec.com/` is a JavaScript application; the served HTML contains only the string `"MAD 2 Benchmark"` and no script or module names, so the checker's name could not be recovered from it.

**Third-party names (not official MVTec naming).** [anomalib](https://github.com/open-edge-platform/anomalib) (Intel / Open Edge Platform, Apache-2.0) provides `MVTecAD2` (a Lightning `AnomalibDataModule`, source `src/anomalib/data/datamodules/image/mvtecad2.py`) and `MVTecAD2Dataset` (`src/anomalib/data/datasets/image/mvtecad2.py`), with `class TestType(str, Enum)` equal to `PUBLIC = "public"`, `PRIVATE = "private"`, `PRIVATE_MIXED = "private_mixed"`, and directory names `test_public/`, `test_private/`, `test_private_mixed/`. Its `CATEGORIES` tuple is `("can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts")` — note `wallplugs` without an underscore, versus the paper's "Wall Plugs". Docs: [anomalib MVTecAD2 datamodule](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/data/datamodules/image/mvtecad2.html). **Cite these as anomalib's names, never as MVTec's.**

### What to skip on a first read

Skip the per-object acquisition-hardware tables (cameras, light models, exposure times), the full qualitative anomaly-map atlas in the appendix, the per-object AU-PRO_0.30 tables (Tables IX–XI), and the detailed runtime/memory scaling study — those matter only once you have chosen an image resolution and a hardware budget. Read Section III-B (dataset design), the AU-PRO paragraph in Section IV-B, Section V (evaluation server), and Table IV.

### Learning outcomes

1. Reproduce, from memory, the three-way test split of MVTec AD 2 and state which part may be used for what — including that **only validation data may set a threshold** and that TEST_pub is *optional*.
2. Explain why AU-PRO with an FPR integration limit of **0.05**, not 0.30, is the benchmark's headline metric, and justify it with the 10-pixel-defect-in-5-MP argument.
3. Correct the common misconception that some *categories* are private test categories: the public/private distinction is at the level of ground truth and images, and all eight categories appear in all three test parts.

---

## [R6] SPARC

### Full citation

Seokhee Han, Seungjun Chu, Mateusz Nowak, Peter Chin. "SPARC: Subspace Position-Aware Robust Few-Shot Calibration for Distribution-Shifted Industrial Anomaly Detection." [arXiv:2608.18585](https://arxiv.org/abs/2608.18585) [cs.CV], v1 submitted **19 August 2026**; HTML: [arxiv.org/html/2608.18585v1](https://arxiv.org/html/2608.18585v1). Licence: [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/). DOI [10.48550/arXiv.2608.18585](https://doi.org/10.48550/arXiv.2608.18585).

> **The arXiv ID resolves.** I explicitly checked `https://arxiv.org/abs/2608.18585`: **HTTP 200**, with a matching title, the four listed authors, submission history `[v1] Wed, 19 Aug 2026 06:25:35 UTC`, subject cs.CV, and an accessible HTML full text at `https://arxiv.org/html/2608.18585v1`. So the plan's premise "if that arXiv ID does not resolve" is **falsified** — it does resolve, and everything below is read from the abstract page plus the v1 HTML full text (6,133 extracted lines, including appendices A–J). The paper carries no venue string and no journal reference on arXiv, so **peer-review status is not verified**; there is no "accepted at X" note.

### Problem

Vision-based industrial anomaly detectors are calibrated on one distribution and deployed on another that differs in **illumination, fixture placement, or sensor characteristics**, which sharply degrades an otherwise accurate detector. Adapting to the incoming production lot is the natural response, but labelled anomalies are scarce. The paper therefore targets calibration from **only a handful of verified-normal images** available before scoring the rest of the lot. Existing fixes are rejected because they require backpropagation, detector-specific tuning, or commitments about feature directions that a few samples cannot justify.

### Core idea

SPARC **intercepts patch features between the encoder and the detector** and removes a **closed-form, spatially indexed estimate of deployment-time nuisance** by **per-cell subspace projection**. It needs only **k ≤ 8** verified-normal images, uses the **algebraic saturation rank r = k − 1**, operates on the **encoder's native patch grid**, and requires **no gradients and no weight updates**. It is designed to compose with memory-bank, density, prototype, and mutual detectors.

### Precise technical content

**Nuisance model (Eq. 1).** For latent feature `f_{i,j}^{(ℓ)} ∈ ℝ^d` at spatial cell `(i,j)` in deployment image `ℓ`:

```
f_{i,j}^{(ℓ)} = f_{i,j}^{train} + η_{i,j}^{(ℓ)} + ε_{i,j}^{(ℓ)},     η_{i,j}^{(ℓ)} ∈ S_{i,j}
```

`f^{train}_{i,j}` is the cell's canonical training-distribution feature (constant across ℓ); `η` is an image-dependent nuisance **constrained to a cell-specific low-dimensional subspace** `S_{i,j} ⊂ ℝ^d`; `ε` is residual variation. Any component **shared across all k images merges into `f^{train}`** and is therefore left in place — which is why the estimator uses **centered** features. The paper is candid about the identification limit: *"The centered calibration matrix cannot separate deployment nuisance from intrinsic variation among normal images. SPARC therefore treats the full subspace spanned by the observed per-cell disagreement as removable deployment nuisance."*

**Estimator (Eqs. 2–7).** Stack the k calibration features per cell into `X_{i,j} ∈ ℝ^{k×d}`; form the per-cell mean `μ̄_{i,j}` and the centered matrix `X̃_{i,j} = X_{i,j} − 1_k μ̄_{i,j}^T`. Let `q = rank(X̃_{i,j})`; because centered rows sum to zero, `q ≤ min(k−1, d)` (Eq. 4), with equality when `d ≥ k−1` and the calibration features are in **general affine position**. Take the reduced SVD `X̃ = U Σ W^T`, `W ∈ ℝ^{d×q}`, `W^T W = I_q`. For a requested rank `r`, set `s = min(r, q)` and `V_{i,j} = [w_{i,j,1} … w_{i,j,s}]` (Eq. 6). Then:

```
f_{i,j}^{corr} = f_{i,j}^{test} − V_{i,j} V_{i,j}^T ( f_{i,j}^{test} − μ̄_{i,j} )        (Eq. 7)
```

i.e. the corrected feature **retains the calibration mean plus the component of the test deviation orthogonal to the estimated subspace**. `V V^T` is the orthogonal projector onto the retained empirical disagreement subspace `Ŝ_{i,j}^{(r)} = span(V_{i,j})`.

**"Position-aware" means per-cell, on the encoder's own grid.** SPARC estimates **one subspace for each cell of the encoder's native patch grid `H′ × W′`**. Verbatim: *"It introduces no separate spatial grid and requires no grid-size tuning."* The motivation is explicit: *"an illumination change may shift edge features differently from interior features, so a single global subspace need not fit all locations."* There is no positional encoding, no learned position embedding, and no continuous coordinate regression — "position-aware" is discharged entirely by **indexing the subspace by grid cell**.

**Rank choice and its justification.** Since `q ≤ k−1`, setting `r = k−1` gives `s = q` and retains every nonzero disagreement direction. Verbatim: *"This choice requires no held-out validation."* The stated property: Eq. 7 **modifies only an at-most-(k−1)-dimensional component and leaves every orthogonal direction unchanged**. *"Under isotropic centered deviations, the expected removed-energy fraction is s/d. This average-case identity does not guarantee anomaly preservation because real feature directions need not be isotropic."* Ablations show *"the shift-prone gains generally increase toward this value, although the exact empirical optimum varies by detector and metric."*

**Cost.** Fitting the projector costs **O(H′W′k²d)**; applying it costs **O(H′W′kd)** per image; neither uses gradients. Reference-based detectors additionally incur a **one-time, detector-specific cost** to transform cached training features and rebuild their references.

**What it changes about the reference detector — and what it does not.**

| | Detail (verbatim-backed) |
|---|---|
| **Does NOT change** | No backpropagation, no weight updates, no retraining; the encoder is untouched; *"the host detector applies its original scoring rule to the corrected feature"*; no held-out validation needed to pick r; no detector-specific tuning. |
| **Does change — reference-based detectors** (memory-bank / density / prototype) | *"To avoid comparing corrected test features with an uncorrected reference, SPARC transforms **both** the cached training and test features. The reference is rebuilt once per deployment lot from the transformed cached features using the detector's original procedure."* This needs access to cached training features but no gradients. |
| **Does change — reference-free (mutual) detectors** (MuSc) | MuSc scores each test patch by mutual comparison against the pool of other test images, so SPARC *"correct[s] only the test side ... so that the mutual comparison operates on corrected features throughout. With no reference side, symmetric application does not apply."* |
| **Does change — CLS-decoupled image-level detectors** (AnomalyCLIP) | SPARC is applied **only to patch features**, so **Image AUROC is unchanged** because the image score comes from the encoder's CLS token. But the pixel score uses those patch features, so AU-PRO can still move. Concretely, on MVTec AD 2 **AU-PRO_0.05** at k = 8 (Table 2), AnomalyCLIP V goes 25.2 → 25.5 (**+0.3 pp**) and AnomalyCLIP M goes 25.2 → 25.5 (**+0.4 pp**) — small but non-zero, confirming the patch path is live even when the image path is untouched. For scale, the same MVTec AD 2 AU-PRO_0.05 table gives SubspaceAD 35.4 → 45.6 (**+10.2 pp**), PaDiM 11.3 → 17.1 (+5.8 pp), MuSc 26.6 → 29.9 (+3.2 pp), AnomalyDINO 37.0 → 38.8 (+1.8 pp), PatchCore 19.0 → 20.4 (+1.3 pp), SPADE 18.4 → 19.5 (+1.1 pp), WinCLIP 17.1 → 17.7 (+0.7 pp). |

**Data roles and equal-calibration controls.** This is the part a threshold/compression study must copy carefully.

- **Calibration pool.** For each dataset–category pair and each `k ∈ {2, 4, 8}`, *"we uniformly sample k verified-normal images from the **test-split normal pool** and use the remainder as the held-out evaluation set."* The stated reason: *"to model verified-normal images from the incoming deployment lot, which is the distribution SPARC is intended to calibrate."* A **calibration-source control** using training-pool normals instead is reported in Appendix G.
- **Paired protocol.** *"Both base (without SPARC) and +S (with SPARC) are scored on the same evaluation set, so every reported Δ is a paired comparison."* **Five random seeds**, mean score reported. The host detector, its source training, the encoder, and the held-out evaluation set are all held fixed; **only the calibration correction is toggled**.
- **Equal-calibration / information-matched controls.** All get *"the same seed-specific k = 8 calibration draws and evaluation subsets"*:
  - **correction-mode ablation** — `base` (no correction), `mean-only` (per-cell centroid subtraction), `SPARC (global subspace)` (a single rank-(k−1) subspace pooled over all cells), `SPARC (per-cell)` (the method);
  - **CORAL few-shot** (global second-order statistic alignment);
  - **calibration-only BN** and **source-mixed BN** (batch-norm statistic adaptation);
  - **full-normal transductive diagnostics** (CORAL full-normal, BN full-normal) which use *all* ground-truth normal deployment images including normals in the evaluation subset — explicitly flagged as *"nondeployable information-budget controls"* that test whether few-shot statistic estimation is the limiting factor;
  - **query-conditioned few-shot baselines** FastRecon and FastRef. These are **not** information-matched, and the paper says so: *"Because the reference-based SPARC hosts also retain their training-derived normal references, these absolute scores do not constitute an information-matched comparison."*
- **A caveat the plan must carry.** Verbatim: *"Because the baseline does not use the calibration images, the base-versus-SPARC gap also reflects the additional deployment information available to SPARC. The matched-budget comparisons in Section 4.3 isolate the effect of the correction design by giving each method the same k normal images."* So the headline pooled Δ is **not** an information-matched number; the matched-budget number is.

**Datasets, detectors, metrics, statistics.**

- Benchmarks, **28 dataset–category pairs**: **shift-prone** — MVTec AD 2 (**8 categories**) and AeBAD-S (**4 subdomains**, each treated as a category); **shift-free** — VisA (**12 categories**) and RAD (**4 categories**), called shift-free *"because their splits contain no engineered shift."*
- **Nine off-the-shelf detector configurations** in three integration families: (a) reference-based — **SubspaceAD, AnomalyDINO, PaDiM, PatchCore, SPADE, WinCLIP**; (b) reference-free mutual — **MuSc**; (c) CLS-decoupled image-level — **AnomalyCLIP** with **two prompt sets, AnomalyCLIP V (VisA prompts) and AnomalyCLIP M (MVTec AD prompts)**. Hence **seven** detectors have image scores that depend on corrected patch features (9 − 2 CLS-decoupled).
- **Metrics.** Image AUROC and **AU-PRO_τ**, the area under the per-region-overlap curve up to an FPR limit τ. *"AU-PRO is computed with the standard Anomalib implementation at native ground-truth resolution, without any method-specific tuning."* **τ = 0.3 on all four benchmarks, plus τ = 0.05 on MVTec AD 2.** Scores on a 0–100 scale; Δ in percentage points (pp).
- **Significance.** On the shift-prone benchmarks, a **one-sided Wilcoxon signed-rank test (SPARC > base)** on **seed-averaged per-category Δs** as paired samples, with **Holm correction applied jointly across the seven detectors and two metrics**.
- **Implementation.** PyTorch; runs distributed across three compute nodes with **16 GPUs total**, one GPU per run.

**Concrete verified numbers** (all at k = 8 unless stated; Δs are paired SPARC-minus-base, computed before rounding):

- **Headline pooled, shift-prone (MVTec AD 2 + AeBAD-S), across the seven patch-feature-image-score detectors: +13.8 pp Image AUROC and +3.5 pp AU-PRO_0.3.**
- After Holm correction: **Image AUROC gains are significant for all seven detectors; five show significant AU-PRO_0.3 gains.** All seven also improve in the detector-level mean on each reported MVTec AD 2 metric.
- **On shift-free VisA and RAD, the changes are smaller and mixed across detectors.**
- **Matched-budget feature-level controls** (macro-averaged over categories and over the shared WRN-50 detectors PaDiM / PatchCore / SPADE, mean ± sample SD across five seeds):

| Variant | Δ Image AUROC (shift-prone) | Δ AU-PRO_0.3 (shift-prone) | Δ AU-PRO_0.05 (MVTec AD 2) |
|---|---|---|---|
| CORAL few-shot | +1.8 ± 0.4 | −2.8 ± 0.1 | −2.5 ± 0.1 |
| Calibration-only BN | −0.5 ± 0.4 | +2.1 ± 0.2 | +4.4 ± 0.1 |
| Source-mixed BN | +1.5 ± 0.2 | +4.0 ± 0.1 | +6.1 ± 0.2 |
| CORAL full-normal † | +3.0 ± 0.4 | −2.7 ± 0.0 | −2.6 ± 0.0 |
| BN full-normal † | +0.4 ± 0.1 | +2.3 ± 0.0 | +4.7 ± 0.0 |
| **SPARC (ours)** | **+12.9 ± 1.4** | **+3.4 ± 0.8** | **+2.7 ± 1.6** |
| *shift-free:* SPARC (ours) | +1.0 ± 0.2 | −0.4 ± 0.1 | — |
| *shift-free:* CORAL few-shot | +2.5 ± 0.5 | −3.8 ± 0.1 | — |
| *shift-free:* Source-mixed BN | +0.2 ± 0.1 | +1.7 ± 0.0 | — |

  († = nondeployable transductive control.) Note the honest tension: **source-mixed BN beats SPARC on localisation** (+6.1 vs +2.7 pp AU-PRO_0.05; +4.0 vs +3.4 pp AU-PRO_0.3) while SPARC dominates on Image AUROC.
- **Correction-mode ablation**, pooled over all 28 dataset–category pairs at k = 8: *"Per-cell SPARC (our method) is the only mode with consistent pooled gains on all three metrics, whereas per-cell mean subtraction alone degrades every metric and a single global subspace is essentially neutral."*
- **Calibration size.** Across k ∈ {2, 4, 8}: *"gains generally increase on the shift-prone benchmarks, while changes on the shift-free benchmarks stay small and lack a consistent trend with k."*
- **Contamination.** Replacing up to four of the eight calibration normals with same-category anomalies: the pooled Image AUROC gain is **+4.4 pp at 3/8 and +1.3 pp at 4/8** (still positive through three contaminated samples); *"pooled AU-PRO_0.3 degrades earlier and reaches −3.5 pp at 4/8, indicating greater sensitivity of localization to calibration contamination."*
- **Backbone.** Across **eight PatchCore backbones**, Image AUROC and AU-PRO_0.3 improve on MVTec AD 2 **for every backbone**, and on AeBAD-S on average, *"although the AeBAD-S gains vary in sign across backbones; changes on the shift-free benchmarks remain small on average."*
- **CORAL detail.** *"CORAL aligns global second-order statistics, and its matched-budget and full-normal conditions produce nearly identical per-detector changes while often degrading localization, with the largest drops on PaDiM and PatchCore. This stability across calibration budgets indicates that few-shot estimation alone does not explain CORAL's behavior."*

### Official code repository: none

**Confirmed: no official repository exists.** Verbatim from Section 4.1.5: **"Code and configurations will be released upon acceptance."** Corroborating checks: the arXiv abstract page lists **no** code/DOI code link; a GitHub repository search for `SPARC anomaly detection calibration` returns **total_count = 0**. The paper also notes that **FastRef** *"does not have an official reference implementation at the time of our experiments"* — they reimplemented the Euclidean prototype-refinement variant of its Algorithm 1 themselves. So the plan's statement that "none was confirmed" is **correct**; the sharper and better-supported statement is: **no repository exists as of v1; the authors promise release only upon acceptance.**

### What to skip on a first read

Skip Appendix A (theoretical analysis: algebraic saturation proof, removed energy under isotropy, computational cost), Appendices C–H (comparison details, per-detector tables, robustness/sensitivity, cross-scene transfer, VisA input-geometry sensitivity), and Appendix J (extra qualitative maps). The essential reading is Sections 3.1–3.2 (model + estimator), 3.3 (the three integration families, which is where "what it does not change" lives), 4.1.3–4.1.4 (calibration protocol + metrics), and 4.3 (the matched-budget controls). Sections 4.4.1 and 4.5.2 carry the two most decision-relevant ablations.

### Learning outcomes

1. Write down Eq. 7 and explain why the correction is *idempotent in spirit* — it removes the component of the test deviation from the calibration mean that lies in the retained subspace, and leaves every orthogonal direction untouched.
2. Explain why the rank saturates at **r = k − 1** and why that requires no held-out validation, then explain the paper's own caveat that the isotropic expected-removed-energy identity `s/d` does **not** guarantee anomaly preservation.
3. Distinguish the **unmatched** base-vs-SPARC gap (+13.8 / +3.5 pp pooled) from the **matched-budget** number (+12.9 ± 1.4 / +3.4 ± 0.8 pp), and state which one can be attributed to the correction *design* rather than to the extra calibration images.

---

## [R7] Beygelzimer, Kakade & Langford (2006) and [R10] Elkin & Kurlin (2023)

### Full citations

**[R7]** Alina Beygelzimer, Sham Kakade, John Langford. "Cover Trees for Nearest Neighbor." In *Proceedings of the 23rd International Conference on Machine Learning (ICML '06)*, pp. **97–104**, 2006. DOI [10.1145/1143844.1143857](https://doi.org/10.1145/1143844.1143857). Author page: [hunch.net/~jl/projects/cover_tree/cover_tree.html](https://hunch.net/~jl/projects/cover_tree/cover_tree.html).
Crossref confirms: title `Cover trees for nearest neighbor`, container `Proceedings of the 23rd international conference on Machine learning - ICML '06`, page `97-104`, year 2006, authors Beygelzimer / Kakade / Langford ([Crossref API](https://api.crossref.org/works/10.1145/1143844.1143857)). The ACM DOI itself returned **HTTP 403** to a scripted request (ACM blocks non-browser clients); the DOI is nonetheless valid per Crossref metadata.
Supplemental primary sources, all live at the author page: **extended version with experimental results** [paper/paper.pdf](https://hunch.net/~jl/projects/cover_tree/paper/paper.pdf) (13 pp — this is where the invariants are stated verbatim); **ICML camera-ready** [icml_final/final-icml.pdf](https://hunch.net/~jl/projects/cover_tree/icml_final/final-icml.pdf); **addendum** [paper/addendum/comparison.ps](https://hunch.net/~jl/projects/cover_tree/paper/addendum/comparison.ps); **code v4** `cover_tree.tar.gz` (LGPL/GPL), **templated v2** `cover_tree_2.tar.gz`, `sparse_data.tar.gz`, `datasets.tar.gz`, `faq.html`. Author-page errata: *"Thomas Kollar found a small bug in the insert algorithm description. This doesn't appear in the code because the code uses a batch insert implementation."* and *"v4 fixes a subtle bug that Ryan Curtin found and has a more robust zero-distance check."*

**[R10]** Yury Elkin, Vitaliy Kurlin. "A new near-linear time algorithm for k-nearest neighbor search using a compressed cover tree." In *Proceedings of the 40th International Conference on Machine Learning*, **PMLR 202:9267–9311**, 2023. [proceedings.mlr.press/v202/elkin23a.html](https://proceedings.mlr.press/v202/elkin23a.html); PDF [elkin23a.pdf](https://proceedings.mlr.press/v202/elkin23a/elkin23a.pdf).
Supporting counterexample paper: Yury Elkin, Vitaliy Kurlin. "Counterexamples expose gaps in the proof of time complexity for cover trees introduced in 2006." Accepted in peer-reviewed *Proceedings of TopoInVis 2022* (IEEE Workshop on Topological Data Analysis and Visualization). [arXiv:2208.09447](https://arxiv.org/abs/2208.09447), DOI [10.48550/arXiv.2208.09447](https://doi.org/10.48550/arXiv.2208.09447), cs.CG.

### Problem

Nearest-neighbour search in a **general metric space** (no coordinates, no vector operations — only a black-box distance `d`). The goal is exact NN/k-NN with sub-linear query time and linear space, without assuming Euclidean structure.

### Core idea

A **cover tree**: a levelled tree over the data set whose level sets are simultaneously a *covering* of the level above and a *separated* set, with geometrically decreasing radii `2^i`. The tree turns one distance computation `d(p, c)` into a certified **lower bound on the distance from the query to every reference in a whole subtree**, which is what permits branch pruning without ever touching the individual points.

### The cover tree invariants, stated precisely

Verbatim from the extended version (§2.1), which is the definitive statement:

> "Let `C_i` denote the set of nodes at level `i`. A cover tree `T` on a data set `S` obeys the following invariants for all `i`:
> **(1) (nesting)** `C_i ⊆ C_{i−1}`.
> **(2) (covering tree)** For every `p ∈ C_{i−1}`, there exists a `q ∈ C_i` satisfying `d(p,q) ≤ 2^i`, and exactly one such `q` is a parent of `p`.
> **(3) (separation)** For all `p,q ∈ C_i`, `d(p,q) > 2^i`.
> These invariants are essentially the same as used in navigating nets [KL04a], except for (2) where we require only one parent of a node rather than all possible parents."

Reading the three invariants:

- **Nesting** `C_i ⊆ C_{i−1}`: level sets are nested, and each level is a *superset* of the next (deeper) one. Equivalently, every node has a self-child at every lower level; the level sets are monotonically shrinking as `i` decreases. `C_∞ = {root}` and `C_{−∞} = S`.
- **Covering** `∀p ∈ C_{i−1} ∃q ∈ C_i : d(p,q) ≤ 2^i`, with exactly one such `q` designated the parent: each node is "covered" by a node one level up within radius `2^i`.
- **Separation** `∀p ≠ q ∈ C_i : d(p,q) > 2^i`: points sharing a level are strictly more than `2^i` apart, so level `i` has bounded local density.

Note the **constant convention**: the 2006 statement uses covering radius `2^i` and separation `2^i`. Elkin & Kurlin restate the tree with shifted constants: cover condition `d(q,p) ≤ 2^{l(q)+1}` with `l(q) < l(p)`, and separation `d_min(C_i) > 2^i` for `C_i = {p : l(p) ≥ i}`. Always check which convention a paper uses before copying an exponent.

### How the triangle inequality prunes a branch

The exact algorithm, verbatim from the extended version (Algorithm 1; `d(p,Q) := min_{q∈Q} d(p,q)`):

```
Find-Nearest(cover tree T, query point p)
1. set Q_1 = C_1                  (C_1 is the root level of T)
2. for i from 1 down to −1
   (a) set Q = { Children(q) : q ∈ Q_i }
   (b) form next cover set  Q_{i−1} = { q ∈ Q : d(p,q) ≤ d(p,Q) + 2^i }
3. return argmin_{q ∈ Q_{−1}} d(p,q)
```

The `1` and `−1` here are the paper's own shorthand for `+∞` and `−∞`: verbatim, *"it is easier to think of the tree as having an infinite number of levels (with `C_∞` containing only the root of the tree, and with `C_{−∞} = S`)."* So read the loop as descending from the root level down to the single-point level, one `2^i` scale at a time.

Correctness (Theorem 2 in the ICML camera-ready; Theorem 2.2 in the extended version), verbatim:

> "For any `q` in `C_{i−1}` the distance between `q` and any descendant `q'` is bounded by `d(q,q') ≤ Σ_{j=−∞}^{i−1} 2^j = 2^i`. Consequently, step 2(b) can never throw out a grandparent of the nearest neighbor of `p`."

So the mechanism is exactly this. Because *every descendant of a node `q ∈ C_{i−1}` lies inside the ball `B(q, 2^i)`*, and because the triangle inequality gives, for any `s ∈ B(q, 2^i)`,

```
d(p,s) ≥ d(p,q) − d(q,s) ≥ d(p,q) − 2^i ,
```

a candidate `q` that satisfies `d(p,q) − 2^i > d(p,Q)` — equivalently `d(p,q) > d(p,Q) + 2^i`, the complement of step 2(b) — has **no descendant that can beat the current best**. The whole subtree under `q` is discarded after computing **one** distance, `d(p,q)`.

### The exact reason a ball containing references gives a LOWER bound

Suppose all references in some group are known to lie in a ball `B(c, R)` — the covering invariant is precisely a promise of this form, with `c = q` and `R = 2^i`. Then for the query `p` and *any* point `s ∈ B(c, R)`:

```
d(p,s) ≥ d(p,c) − d(c,s)        (triangle inequality on (p, c, s))
       ≥ d(p,c) − R             (since d(c,s) ≤ R)
```

Therefore

```
d(p, S ∩ B(c,R))  ≥  d(p,c) − R  ,
```

and since this holds for every group, also `d(p,S) ≥ d(p,c) − R`.

Three things to be precise about:

1. **It is a lower bound, not an upper bound.** The same ball gives the *upper* bound `d(p,c) + R`, but only the lower bound is useful for pruning: if the *smallest possible* distance to the ball already exceeds the current best, nothing inside can improve the answer. Knowing set membership never lets you conclude that a point is *close*; it only lets you conclude that a point cannot be *closer than* `d(p,c) − R`.
2. **The triangle inequality can never certify an exact distance from a ball**, so pruning is inherently **conservative**: the search may keep candidates that turn out to be far (extra work), but it can **never** discard the true nearest neighbour. That conservativeness is exactly why cover-tree search returns the *exact* NN rather than an approximation, and it is the property that makes the data structure safe to use as a ground-truth oracle in a benchmark.
3. **The accounting matters.** Computing `d(p,c)` costs one distance evaluation; the same ball may contain `|B(c,R) ∩ S|` references. Pruning replaces that many evaluations with one, and the separation invariant is what keeps `|B(c,R) ∩ S|` bounded.

### Worked 2-D triangle-inequality examples (checkable by hand)

**Example A — the lower bound, and why it is the bound you want.** Query `p = (0,0)`. A subtree is known (by the covering invariant) to be contained in the ball centred at `c = (10,0)` with radius `R = 3`. Then for any `s` in that subtree, `d(p,c) = 10`, so `d(p,s) ≥ 10 − 3 = 7`. Suppose the best distance found so far is `d(p,Q) = 6`. Since `7 > 6`, **discard the entire subtree**. Sanity checks: the point of the ball *closest* to `p` is `(7,0)`, with `d = 7` — the bound is attained, so it is tight and safe. The point of the ball *farthest* from `p` is `(13,0)`, with `d = 13`; the ball's own distance range to `p` is `[7, 13]`, and only the lower end justifies pruning.

**Example B — the descendant radius from the covering invariant.** Level scales are `2^i`. Take a node `q ∈ C_2`, i.e. `i − 1 = 2` so `i = 3`, placed at `q = (0,0)`. The correctness lemma gives, for any descendant `s`, `d(q,s) ≤ Σ_{j=−∞}^{2} 2^j = 4 + 2 + 1 + ½ + … = 8 = 2^3`. Let the query be `p = (20,0)` and suppose the current best is `d(p,Q) = 11`. The closest a descendant could possibly be is `d(p,q) − 8 = 20 − 8 = 12 > 11`, so prune. Verify the tightest case by hand: the descendant nearest to `p` lies on the segment from `q` toward `p`, i.e. at `(8,0)`, giving `d = 12 > 11`. ✓ Safe. Had the current best been `13`, the bound `12 < 13` would not justify pruning, and the subtree must be kept — the search is allowed to do extra work but never to be wrong.

**Example C — the actual step 2(b) test.** Query `p = (0,0)`, level `i = 3` so `2^i = 8`, candidate node `q = (10,0)` with `d(p,q) = 10`. The rule keeps `q` iff `d(p,q) ≤ d(p,Q) + 2^i`, i.e. iff `10 ≤ d(p,Q) + 8`.
- If `d(p,Q) = 3`: threshold `= 11`, and `10 ≤ 11` → **keep** `q`, expand its children.
- If `d(p,Q) = 1`: threshold `= 9`, and `10 > 9` → **discard** the whole subtree under `q`.
Safety of the discard: every descendant `s` satisfies `d(q,s) ≤ 2^3 = 8`, so `d(p,s) ≥ 10 − 8 = 2 > 1 = d(p,Q)`. Nothing inside can beat the incumbent. Note the slack: what makes the discard valid is `2 > 1`, and that inequality is exactly the triangle inequality plus the covering promise.

### Complexity claims and the known gaps in the original proofs

**What the 2006 paper actually claims** (verified from the ICML camera-ready text and the extended version):

| Quantity | ICML 2006 |
|---|---|
| Space | `O(n)`, **regardless of the metric's structure** (Theorem 1, "Space bound") |
| Construction | `O(c^6 n log n)` |
| Insert / remove | `O(c^6 log n)` (Theorem 6) |
| Query (single NN, exact) | **`O(c^12 log n)`** — Theorem 5, "*If the data set `S ∪ {p}` has expansion constant `c`, the nearest neighbor of `p` can be found in time `O(c^12 log n)`*" |
| Query nearest neighbour of all `O(n)` points | `O(c^16 n)` |
| Experimental speedups over brute force | "*varying between 1 and 2000*" (extended version abstract); "*one and several orders of magnitude*" (ICML abstract) |

The author page summarises this as: *"The running time of a nearest neighbor query is only `O(log(n))` given a fixed intrinsic dimensionality. (like KR2002 and KL04)"* and *"The space usage and query time are `O(n)` under no assumptions."*

**Two critical caveats, both in the primary source.**

1. Everything except the `O(n)` space bound is **conditional on the expansion constant `c`** (Karger–Ruhl `[KR02]`), where `c` is *"a measure of the intrinsic dimensionality"*. The `O(log n)` claim is therefore **not unconditional** — it is `O(c^12 log n)`, and `c` can itself grow with `n`.
2. Verbatim from the ICML paper: *"It is important to note that the algorithms here (as in [KL04a] but not in [KR02]) work **without knowledge of the structure**; only the analysis is done with respect to the assumptions."* The algorithm never estimates `c`. So `c` is an artefact of the *analysis*, not a run-time quantity — a user cannot check the precondition of the theorem.

**The gaps.**

- **Curtin (2015), §5.3.** Ryan R. Curtin, *Improving dual-tree algorithms*, PhD thesis, Georgia Institute of Technology, 2015. Per Elkin & Kurlin verbatim: *"In 2015, Curtin (2015, Section 5.3) pointed out that the proof of Beygelzimer et al. (2006a, Theorem 5) contains a crucial gap."* (The thesis itself I did **not** fetch — **not verified** beyond this quotation; Elkin & Kurlin are the source for the claim.)
- **The nature of the gap**, verbatim from [arXiv:2208.09447](https://arxiv.org/abs/2208.09447):
  > "Section 5.3 of Curtin's PhD (2015) pointed out that the proof of this result was wrong. **The key step in the original proof attempted to show that the number of iterations can be estimated by multiplying the length of the longest root-to-leaf path in a cover tree by a constant factor. However, this estimate can miss many potential nodes in several branches of a cover tree, that should be considered during the neighbor search.** The same argument was unfortunately repeated in several subsequent papers using cover trees from 2006."
  > "This paper explicitly constructs challenging datasets that provide counterexamples to the past proofs of time complexity for **the cover tree construction**, the **k-nearest neighbor search presented at ICML 2006**, and the **dual-tree search algorithm published in NIPS 2009**."
- **Named counterexamples** (Elkin & Kurlin 2022a, as cited by the ICML 2023 paper): **Counterexample 4.2** — cover tree construction, *"shows that the past proof is incorrect"*; **Counterexample 5.2** — the `k = 1` ICML 2006 nearest-neighbour query bound; **Counterexample 6.5** — Ram et al. (NIPS 2009) dual-tree search. The 2022 paper's arXiv admin note records *"substantial text overlap with arXiv:2205.10194; text overlap with arXiv:2111.15478"*, which is why the 2023 paper cites it as `2022a` in a family of related manuscripts.
- **Why the flaw was so persistent:** it was not an arithmetic slip but a structural mismatch between what the algorithm traverses and what the proof counted. The original tree is defined **implicitly** (infinitely many repetitions of each point down infinite branches). Since an implicit tree is formally infinite, the authors used a different, **explicit** representation, quoting (verbatim from Beygelzimer et al. §2, reproduced by Elkin & Kurlin): *"The explicit representation of the tree coalesces all nodes in which the only child is a self-child."* In that explicit tree *"any given point `p` can appear in many different nodes simultaneously"* and *"levels of this explicit cover tree ... are not the same as the nesting sets `C_i`"* — so counting the longest root-to-leaf path does not bound the number of nodes the search must visit.

**How Elkin & Kurlin address it.**

They abandon the original tree and define a **compressed cover tree on the reference set `R` only** (Definition 2.1; `T(R)` has **vertex set `R`**, a root `r ∈ R`, and a level function `l : R → ℤ`):

- **(2.1a) Root condition:** `l(r) ≥ 1 + max_{p ∈ R \ {r}} l(p)`.
- **(2.1b) Cover condition:** for every node `q ∈ R \ {r}` we select a **unique parent `p`** and a level `l(q)` such that `d(q,p) ≤ 2^{l(q)+1}` and `l(q) < l(p)`; that parent has a **single link** to `q`.
- **(2.1c) Separation condition:** for `i ∈ ℤ`, the cover set `C_i = {p ∈ R : l(p) ≥ i}` has `d_min(C_i) > 2^i`.

So **each point appears exactly once** — as against the implicit tree (infinite repetitions) and the explicit tree (points in many nodes simultaneously). To recover what the explicit tree gave for free, they introduce the new **distinctive descendant set** `S_i(p, T(R))` (Definition 2.8). This is the direct fix for the "missed branches" step: because each point occurs once and the level sets `C_i` are honestly defined, counting no longer over- or under-counts branches.

**Their results** (verified from the PMLR abstract and PDF):

- Construct a compressed cover tree in **`O(n log n)`** — with hidden factors `c_m(R)^{O(1)}` and `log(Λ(R))` where `Λ` is the aspect ratio (Theorem 3.6 / Corollary 3.10); space `O(n)`.
- Find **all** k-nearest neighbours of **all** points of `Q` in time **`O(m (k + log n) log k)`**, *"with a hidden dimensionality factor depending on point distributions of the sets `R, Q` but not on their sizes."*
- One query point: `O(c(R ∪ {q})^{O(1)} · log k · (log|R| + k))` (Theorem 4.9); and `O(log k · ( c_m(R)^{O(1)} log|Δ| + |B(q, O(d_k(q,R)))| ))` (Corollary 4.7).
- New constant: the **minimized expansion constant** `c_m(R) = inf_{0<ε<inf R} inf_{A ⊆ ℝ^n} sup_{p ∈ A, t>ε} |B(p,2t) ∩ A| / |B(p,t) ∩ A|`, with the new upper bound `c_m(R) ≤ 2^n` in a normed vector space (Theorem C.15), proved by a volume argument. The compressed tree's build bound is expressed in `c_m(R)`, which is at most `c(R)`-like and avoids the original's `c^6`-style exponents.
- **Stated limitation:** reaching *purely* linear `O(c(R)^{O(1)}|R|)` with no other hidden parameters would need a compressed cover tree on **both** `Q` and `R`; the authors say this *"will require significantly more effort to understand if `O(c(R)^{O(1)}|R|)` is achievable"*. They also note their results *"justify that the MLpack implementations of the k-nearest neighbors search now have proved theoretical guarantees for a near-linear time complexity"*, where MLpack implements a version of an **explicit** cover tree.

**Cross-reference table.** In the ICML 2023 paper, Tables 2–4 list the "Cover tree, Beygelzimer et al. (2006a)" rows with the explicit note *"Elkin & Kurlin (2022a, Counterexample 4.2 / 5.2) shows that the past proof is incorrect"*, alongside the corresponding compressed-cover-tree rows whose proofs are *"Lemma B.1 / Corollary 3.10 / Theorem 4.9"*.

### Why an empirical guarantee differs from an algorithmic guarantee

This distinction is the single most important thing to carry out of this resource, and the primary sources themselves make it cleanly:

- An **algorithmic (worst-case) guarantee** is a statement quantified over an *entire class of inputs*, with the cost written as an explicit function of input parameters. Here: for **every** metric space whose `S ∪ {p}` has expansion constant `c`, the query terminates within `O(c^12 log n)` distance evaluations. Nothing about the particular dataset matters; the bound holds even for inputs you have never seen, including adversarially chosen ones.
- An **empirical guarantee** — "cover trees give speedups between 1 and 2000× over brute force on natural machine-learning datasets" — is a statement about **a finite sample of inputs, implementations, and hardware**. (The number and identity of the datasets is **not verified** from a listing; the paper's own phrasing is "natural machine learning datasets".) It establishes that a speedup *occurred*, and even that it occurred repeatedly and robustly. It cannot establish that a speedup *must* occur, and it cannot rule out an input family on which the tree degenerates to brute force.
- **The gap between them is not hypothetical here — it was historical.** For roughly sixteen years (2006 → Elkin & Kurlin 2022) the *only* support for the near-linear query claim was (i) a proof that contained a crucial gap and (ii) strong empirical speedups. The empirical results were real; the guarantee was not. This is exactly the failure mode an empirical result cannot detect: fast timings on natural datasets are perfectly consistent with a broken asymptotic proof, because broken proofs are wrong about the hard instances, not the easy ones.
- **Conditionality is not the same as a guarantee either.** Even the corrected claim is *"near-linear with a hidden dimensionality factor depending on point distributions of `R, Q` but not on their sizes"*. The bound can be vacuous for a distribution with a large expansion constant. And the algorithm *cannot check this precondition at run time*, because it never estimates `c`. So the theorem is a statement about the input, not a certificate the algorithm can produce.
- **Practical consequence for a compression/threshold study.** "Cover trees give `O(log n)` queries, so a compressed memory bank is cheap" is a **conditional, distribution-dependent** claim, and the parameter it is conditional on (intrinsic dimensionality / expansion constant) is precisely the thing that deep-feature embeddings are famous for violating. Report *measured* query cost and pruning statistics on your own reference-set embedding, and treat any asymptotic claim as a hypothesis to be tested, not a licence to skip measurement. Also note the sibling phenomenon in R4/R6: `O(n)` **space** is the one unconditional claim in the whole resource set.

### What to skip on a first read

For [R7]: skip the dynamic insert/remove algorithms (Algorithms 2–3, Theorems 3–4 and 6), the approximate-NN variant, the `c^16 n` batch-bound derivation, and the benchmarking plots. Read §2.1 (invariants), §2.2.1 (Algorithm 1 + Theorem 2), and the theorem statements plus the ICML comparison table.
For [R10]: skip Appendix C (the volume argument for `c_m(R) ≤ 2^n`), Appendices B/D (algebra), and the full k-NN algorithm. Read Definition 2.1 (compressed tree), Definition 2.8 (distinctive descendant sets), Theorem 3.6 / Corollary 3.10 (construction), Theorem 4.9 / Corollary 4.7 (search), and the related-work passage that names Curtin §5.3 and the counterexamples.
For [R10]'s gap discussion: read the two-page abstract and introduction of [arXiv:2208.09447](https://arxiv.org/abs/2208.09447); the counterexample constructions themselves are technical.

### Learning outcomes

1. State the three cover-tree invariants precisely (nesting `C_i ⊆ C_{i−1}`; covering `∃q ∈ C_i` with `d(p,q) ≤ 2^i`, exactly one parent; separation `d(p,q) > 2^i` for distinct `p,q ∈ C_i`), and explain why each is needed: nesting for the level structure, covering for the `2^i` descendant radius, separation for bounded local density and hence the work bound.
2. Derive, in one line of algebra, that a ball `B(c,R)` containing references yields `d(p, S ∩ B(c,R)) ≥ d(p,c) − R`; explain why this is a *lower* bound and why only a lower bound permits safe pruning; and verify the pruning decision by hand on the Example C numbers.
3. Explain that the original `O(log n)` query-time claim was a **proof with a gap** (the "longest root-to-leaf path times a constant" step, which miscounts branches) rather than a merely loose bound, name the two authorities that exposed it (Curtin 2015 §5.3; Elkin & Kurlin TopoInVis 2022), and articulate why empirical speedups could not have detected it.

---

## Why these matter for a compression-versus-alarm-threshold study

The claim to be careful about is: *"our compressed memory bank preserves detection quality, so the deployed alarm threshold is safe."* Image/pixel AUROC does not license that conclusion. Here is why, in plain language.

**1. AUROC is a ranking statistic; a deployed alarm is a single operating point.** AUROC integrates the true-positive rate over the *entire* false-positive range. It answers "on average over all possible thresholds, how well does the score rank anomalies above normals?" The deployed system answers a completely different question: "at the one threshold I actually set, how many defects do I catch, and how often do I cry wolf?" Those are one point on the ROC curve versus the area under it. Two models with **identical AUROC** can have wildly different TPR at FPR = 0.1 %, and 0.1 % is the regime industrial inspection lives in — a false alarm rate of even 1 % on a high-throughput line is unusable. Compression that leaves AUROC unchanged can still destroy the operating point that matters, and AUROC will never show it.

**2. Compression does not merely rescale the score — it reshapes the score distribution, so the threshold drifts.** The usual defence is "compression only changes the scale, and AUROC is invariant to monotone transforms." That defence fails because the transform is *not* monotone. Nearest-neighbour distances are order statistics of the reference set: remove reference points and the distance to the *k*-th nearest normal changes in a way that depends on **where** the removed points sat relative to the query, not by a constant factor. Aggregation rules on top (PatchCore's re-weighting, softmax-over-distances, max-over-patches) are non-linear and can reorder scores. The practical consequence: the score value that corresponds to "1 % of normal images alarm" moves after compression, so a threshold calibrated before compression is **the wrong threshold after** compression — and the AUROC number is silent about how far it moved.

**3. AUROC hides threshold calibration, which is where the deployment risk actually lives.** R4 documents the standard industrial recipe: the segmentation threshold is *"the average of the per-pixel anomaly values of the defect-free validation images plus three times their standard deviation"* — a mean + 3σ rule fitted to defect-free data. This is a *statistical* object, and compression changes the statistics it is computed from: fewer references generally shrinks distances, which moves both the mean and the standard deviation of the null score distribution, and the 3σ margin can shrink or grow. Reporting AUROC on anomaly-containing test data says nothing about whether that null-calibrated threshold still attains its target false-alarm rate. A threshold study must report the **null score distribution** (histograms, quantiles, mean, σ) for normal images before and after compression, plus the resulting threshold and its **stable** FPR. Note also that R4 goes out of its way to let submitters *supply a segmentation threshold*, precisely because threshold derivation is an open research problem — the benchmark itself treats the threshold as a first-class deliverable, not a detail.

**4. AU-PRO at a low integration limit narrows the gap but does not close it.** AU-PRO_0.05 (R4's headline metric; also reported by R6 on MVTec AD 2) restricts the integration to FPR ∈ [0, 0.05], which is much closer to the deployed regime than AU-PRO_0.30, and per-region weighting stops large defects from dominating — both genuinely better than pixel AUROC. But AU-PRO is still an **integral over a range of thresholds**, not the TPR at one threshold, and it is still a ranking metric. Choosing τ = 0.05 makes it a better proxy; it does not make it a measurement of the deployed binary decision.

**5. The primary sources contain direct evidence that threshold-free and thresholded metrics disagree.** R4 reports that *"PatchCore performs significantly worse than all other methods in terms of F1 score, yet achieves similar results in terms of AU-PRO."* That is a documented case where the threshold-independent ranking metric and the thresholded decision metric rank the same model differently. R4 also documents that robustness *rankings* can change between lighting regimes (TEST_priv → TEST_priv,mix: RD loses 1.4 pp, MSFlow loses 12.4 pp). Both findings transfer directly to a compression study: a compressed bank could preserve AUROC while changing F1 or changing which regime it degrades in.

**6. The reference results are themselves not a clean "compression" ablation, which is a warning about experimental design.** R6 is the closest analogue in this resource set — it removes/reprojects part of the reference-feature geometry using k ≤ 8 normal images — and it is scrupulous about the confounding: *"Because the baseline does not use the calibration images, the base-versus-SPARC gap also reflects the additional deployment information available to SPARC."* Its headline +13.8 pp Image AUROC / +3.5 pp AU-PRO_0.3 is the **unmatched** number; the matched-budget numbers are +12.9 ± 1.4 / +3.4 ± 0.8 pp. And its own error analysis shows localisation is more fragile than detection: with 4 of 8 calibration images contaminated, Image AUROC is still +1.3 pp but pooled AU-PRO_0.3 turns **negative (−3.5 pp)**. So even a method whose *detection* metric is robust can lose its *localisation* operating point. A compression study should expect, and therefore measure, the same asymmetry.

**7. What a defensible threshold study must report.** Given the above, AUROC alone is insufficient; the minimum set is:

- **Fix a small set of target false-alarm rates** (e.g. FPR ∈ {0.1 %, 1 %, 5 %}) on **held-out defect-free** data, and report **TPR (recall) at each fixed FPR**, before and after compression. This is the deployed quantity.
- **Threshold-transfer test.** Calibrate the threshold on the uncompressed model using defect-free validation data only (the R4-sanctioned procedure), apply it **unchanged** to the compressed model, and report the realised FPR (alarm-rate drift) and the resulting TPR loss. This isolates "does the alarm threshold still mean the same thing?" from "is the model still accurate?".
- **Report the null score distribution itself** — mean, σ, and quantiles of the normal-image score — before and after compression, since all threshold rules (mean + 3σ, empirical quantile) are functionals of it.
- **Report the compression ratio jointly with the measured query cost and pruning statistics**, not with an asymptotic claim alone: per Section 3, tree-based NN claims are conditional on the expansion constant / intrinsic dimensionality of *your* feature distribution, and the algorithm cannot verify that precondition. A compression ratio bought with a query cost blow-up is not a win.
- **At pixel level, report AU-PRO at a low integration limit (τ = 0.05)** rather than AU-ROC or AU-PRO_0.30 — this is R4's own stated reason for lowering the limit — and additionally report the thresholded F1 on binarised maps, since R4 documents that these two can disagree.
- **State the pairing.** Follow R6's protocol: same evaluation subset, paired before/after comparisons, multiple seeds with mean and spread, and an explicit note on whether the comparison is information-matched (same calibration/compression budget) or not.

Two claims are supported verbatim by the primary sources above: the AU-PRO integration-limit definition and its 0.05 value (R4), and the F1-versus-AU-PRO disagreement and the cross-lighting-regime ranking change (R4). The threshold-drift and null-distribution arguments (points 1–3 and 7) are standard ROC/order-statistic reasoning that I have derived here; they are **not** quoted from these papers and should be presented in the plan as the study's own methodological argument rather than as a literature finding.

### Learning outcomes

1. Explain, in one sentence each, why (a) identical AUROC does not imply identical TPR at a fixed FPR, and (b) nearest-neighbour compression is not a monotone rescaling of the anomaly score, so a pre-compression threshold is invalidated.
2. Design the minimum experimental protocol — TPR at fixed low FPR, a threshold-transfer/FPR-drift test, the null score distribution, and measured (not asymptotic) query cost — that would establish whether a compressed memory bank preserves a deployed binary alarm threshold.
