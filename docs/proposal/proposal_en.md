# Research Proposal

**When Does Memory Compression Change Industrial Alarms?**

**Decision-Preserving Screening for Edge Anomaly Detection**

Wu Yuantai　·　November 2026　·　10 weeks

---

## Abstract

Industrial visual anomaly detection often relies on a normal-patch memory bank with nearest-neighbour retrieval. The system stores every normal patch seen during training as a feature vector. For a new image, each patch is matched against the bank, and the distance to its nearest neighbour becomes an anomaly score. This design is accurate. However, the bank is large, and edge devices can neither hold it comfortably nor search it quickly. Therefore, compression is usually applied.

The difficulty is that compression may quietly change the binary alarm decision at a fixed threshold. Ranking metrics such as AUROC cannot reveal this change, because AUROC is insensitive to monotone rescaling. What matters on a production line is not the ranking; it is the single threshold that separates pass from fail.

This project asks one concrete question. Given a frozen encoder, a frozen preprocessing pipeline, a frozen bank version and a frozen threshold, does compression change the alarm decision? If it does, can a conservative interval separate "certain normal", "certain alarm" and "must query the full bank", without ever turning "unknown" into "normal"?

The method claims no new mathematics and reuses the triangle inequality. The edge stores only a small number of observed centres together with conservative coverage radii, and it derives an interval `[L, U]` around the true score. If the whole interval lies on one side of the threshold, the decision is made locally. If the interval straddles the threshold, the query is escalated to the full bank. If the bank is unavailable or the version does not match, the system returns "unresolved". The contract is decision preservation, not higher accuracy. A false alarm that is faithfully preserved remains a false alarm.

Evaluation uses public industrial datasets. The study measures decision disagreement, real defect recall, normal false-alarm rate, interval width, serialised bytes and end-to-end latency, with paired confidence intervals over five seeds. The expected outcome is not a better detector. It is a reproducible boundary that states how far the bank can be compressed before alarms stop matching, and where that boundary fails.

---

## 1. Introduction

### 1.1 Background

The core difficulty of industrial inspection is simple. Good products look alike, while defects come in endless forms. Collecting every defect type in advance is not realistic. Therefore, the usual approach works the other way round: learn what normal looks like, then treat anything far from normal as anomalous. This is one-class or unsupervised anomaly detection.

The strongest recent approach is memory-bank retrieval. Every normal patch is encoded into a feature vector and stored. At test time, each patch is matched against the bank, and the maximum nearest-neighbour distance becomes the image score. PatchCore [1] is the reference work in this line. It compresses the bank to between 1% and 25% with greedy coreset selection, and accuracy barely drops. AnomalyDINO [2] goes further and shows that a frozen DINOv2 encoder together with a handful of reference images already works well, with no training at all.

Yet there is a hard engineering constraint. The bank grows linearly with the training set. A category with 200 images and 676 patches per image holds roughly 135,000 vectors of 1,536 dimensions, which is several hundred megabytes. That is uncomfortable on an edge box. Consequently, compression is almost unavoidable.

### 1.2 Problem Statement

When the bank shrinks, the nearest-neighbour distance of each test patch usually grows, because fewer candidates remain. This change looks monotone and therefore harmless, yet it is not the whole story. The image score takes a maximum over patches, which is a nonlinear aggregation. Moreover, different images have different score distributions. As a result, compression does not simply shift all scores by a constant; it moves some images more than others. An image that sat just below the threshold before compression may cross above it and become a new false alarm, and the opposite case may arise as well, in which a true defect is silently missed.

Nevertheless, AUROC cannot see any of this. The reason is straightforward. AUROC integrates performance over all thresholds, whereas a production line uses exactly one. Two models with identical AUROC can behave very differently at a one percent false-alarm rate. Industrial datasets such as MSC-AD [7] make the same point, since industrial practice cares about performance at low false-alarm rates.

Therefore the project narrows the question to a single sentence. Given a frozen detector, does memory-bank compression change the binary alarm decision at a fixed threshold? If it does, can an auditable interval keep that change under control?

### 1.3 Hypotheses

Two hypotheses are proposed, and both can be falsified. The gap hypothesis (H1) states that at a useful storage reduction, for example ten percent, compression produces non-negligible alarm disagreement, reaching at least two percentage points on at least two categories. The mechanism hypothesis (H2) states that conservative intervals built from observed centres and coverage radii can decide a substantial share of images locally, at least twenty percent, without changing any decision.

Both hypotheses may fail. If ordinary compression followed by recalibration already matches the useful trade-off, the gap does not exist and the work should become an empirical evaluation instead. If the intervals are so wide that nearly every image must escalate, the mechanism adds no value.

### 1.4 Scope

The project targets decision preservation, not detection accuracy. Errors made by the reference detector are preserved faithfully, and compression does not repair them. Test-set defect labels are never used to set the threshold or select parameters. In addition, no physical edge device is currently available, so latency results come from workstation measurement only.

---

## 2. Literature Review

### 2.1 Memory-Bank Industrial Anomaly Detection

PatchCore [1] established the paradigm. It extracts mid-level features from an ImageNet-pretrained WideResNet-50, aggregates local neighbourhoods into patch features, unions all patches from the training set into a memory bank, scores tests by nearest-neighbour distance, and subsamples the bank with a greedy coreset. One of its key observations is that coreset subsampling to one percent raises the share of bank samples actually used at test time from below thirty percent to nearly ninety-five percent, which shows that the uncompressed bank is highly redundant.

AnomalyDINO [2] moves to the opposite extreme. It trains nothing, freezes DINOv2, builds the bank from one, four or sixteen reference images, and scores an image by the mean of its top one percent of patch distances. It shows that frozen features with a small bank are already strong in the few-shot regime.

On the industrial side, the memory-bank idea has been adapted repeatedly. Wang et al. [5] combine it with contrastive learning for few-shot catheter defect detection, and Jia and Chen [6] use memory-aided denoising reconstruction for wind turbine blade damage. Among Chinese-language work, Zhang and Liu [9] apply coupled hyperspheres with deviation reduction to industrial image anomaly detection, while Zhao and Zhang [10] focus on fusing semantic and detail cues.

Overall, the memory-bank approach is mature, and compression is standard practice. However, the works above report ranking metrics such as AUROC and AP, and none of them reports whether the binary decision at a fixed threshold survives compression, which is the only thing a production line actually cares about.

### 2.2 Retrieval Acceleration

Since retrieval is the bottleneck, cost reduction takes two forms: a smaller bank or a faster search. Li et al. [4] take the second route. Their cascade patch retrieval narrows candidates from coarse to fine and brings detection and localisation below one millisecond while keeping accuracy, which shows that staged screening is practical in engineering terms.

A distinction matters here. Most acceleration work optimises for unchanged accuracy, whereas this project optimises for unchanged decisions. The first is a ranking property and the second is a decision property. The two look identical under AUROC and behave completely differently on a production line.

### 2.3 Edge Deployment

Shallari et al. [8] discuss deploying vision anomaly detection from edge to fog nodes and identify latency and bandwidth as the main constraints. EfficientAD [3] answers the same question from the model side, replacing bank retrieval with a small student-teacher network that reaches good accuracy at millisecond latency.

A small learned model and a small retrieval bank are nevertheless different things. In the first, reference knowledge is compressed into network weights, so test-time memory is constant. In the second, memory grows with the training set. This project studies the second, because its failure mode is easier to observe, in that changing the bank changes the scores, and because it admits an auditable guarantee.

### 2.4 Research Gap

Putting the strands together, the gap becomes visible. Existing work reports ranking metrics, yet rarely reports decision agreement at a fixed threshold. Compression work reports accuracy loss, yet rarely reports whether the threshold still means the same thing. Acceleration work reports latency, yet rarely treats decision preservation as an auditable contract. To put it plainly, most of the field compares scores, and very few ask whether the score is still trustworthy near the threshold. That is the position this project fills.

---

## 3. Method

### 3.1 Notation

Let `M` denote the frozen full memory bank with `N` normal patch vectors of dimension `D`. Let `q` denote a query patch, and let `d` denote Euclidean distance. The reference detector is defined as

```
s_M(q) = min_{m ∈ M} d(q, m)          patch score
S_M(x) = max_{q ∈ x} s_M(q)           image score, maximum over patches
```

The decision rule is uniform: alarm if and only if `S_M(x) > t`, with equality treated as normal. The threshold `t` is fixed on separate calibration normal images at a nominal five percent image false-alarm rate.

### 3.2 Compression and Coverage Radii

A greedy coreset selects `K` observed centres `c_j` from `M`. Every centre has to be an actual bank vector, not a computed mean, otherwise the upper bound below does not hold. All bank vectors are assigned to their nearest centre, forming `K` cells, and each cell receives a coverage radius

```
r_j = max_{m ∈ C_j} d(m, c_j)
```

which is simply the distance from the centre to the farthest point in its cell. Hence the edge stores only `K` centres together with `K` radii.

### 3.3 The Interval

The interval needs no new mathematics and relies only on the triangle inequality. The upper bound is `U(q) = min_j d(q, c_j)`, the distance from the query to its nearest centre. The lower bound is `L(q) = min_j max(0, d(q, c_j) − r_j)`.

The upper bound holds because every centre is itself a bank vector, and minimising over the subset of centres can only give a larger value than minimising over the whole bank. The lower bound follows from the same inequality. In plain terms, a detour is never shorter. The centre `c` lies at distance `d` from `q`, and every point in the cell lies within `r` of `c`. Therefore every point in the cell lies at least `d − r` away from `q`, and taking the minimum over cells gives `L`.

Consequently `L(q) ≤ s_M(q) ≤ U(q)`. Because the maximum aggregation is monotone, the interval transfers to the image level directly.

### 3.4 Four-Branch Decision

The decision has four cases. If the upper bound `U_x` does not exceed the threshold `t`, the true score cannot exceed it either, so the image is declared normal locally without querying the bank. If the lower bound `L_x` is strictly greater than `t`, the true score certainly exceeds it, so the image is declared an alarm locally.

The strict inequality matters. The reference rule alarms only when the score is strictly greater than the threshold, so when `L_x` equals `t` exactly the reference answer may well be normal. Writing `>=` would manufacture a disagreement out of nothing, and no error message would ever appear.

When neither condition holds, the interval straddles the threshold, and the query must be escalated to the full bank, whose answer is adopted. A fourth case remains, and it deserves separate attention. If the full bank is unavailable, if the version does not match, or if the query times out, the rule requires the system to return "unresolved" and never to return "normal". Treating unknown as normal means a missed defect, which is the worst failure mode on a production line.

### 3.5 Numerical Reliability

Floating-point error can break the interval, so distances are computed by direct subtraction rather than the expansion `|a|² + |b|² − 2ab`, because the expansion loses most of its significant digits in high dimensions. In one measurement, the expansion made thirty-one of 676 patches report `L > s`, meaning the lower bound exceeded the true value, which would fabricate a verified alarm that was never verified at all. Radii are rounded upwards and lower bounds downwards, so that the interval stays conservative. With this guard in place, the result is empirically decision-preserving rather than formally certified.

### 3.6 Version Control

A claim of decision preservation is meaningless without a version. Each artefact therefore carries a fingerprint over the encoder, the preprocessing pipeline, a hash of the bank contents and the threshold. If any of these disagree, the system returns "unresolved" rather than guessing.

---

## 4. Experimental Design

### 4.1 Data and Role Split

The main experiments use MSC-AD [7] for casting surface defects and three categories from MVTec AD (bottle, tile and screw), prepared according to the protocol of PatchCore [1]. The three categories cover objects, textures and small defects respectively.

Normal images are split into three disjoint roles. The reference-fit role is used only for bank construction and takes sixty percent, the development role is used only for development decisions and takes twenty percent, and the final-calibration role is used only to set the threshold and takes the remaining twenty percent.

| Role | Use | Share |
|---|---|---|
| reference-fit | bank construction only | 60% |
| development | development decisions only | 20% |
| final-calibration | threshold setting only | 20% |

No image may appear in two roles. Few-shot variants with one, four or sixteen reference images draw only from reference-fit, and images in the other two roles are counted separately.

### 4.2 Baselines and Ablations

The full bank serves as the reference detector and the gold standard. An equal-byte random bank represents the simplest form of cost reduction, while an equal-byte greedy bank corresponds to coreset subsampling in the style of PatchCore [1]. Compression followed by an independent recalibration tests whether simple recalibration already solves the problem; if it does, the value of this project is reduced. Finally, a comparison of a global radius against per-cell radii forms the central ablation of the method itself.

### 4.3 Metrics

| Category | Metric |
|---|---|
| Decision fidelity | disagreement rate against the reference, split into extra alarms and missed alarms |
| Real performance | defect recall and normal false-alarm rate at the nominal five percent point |
| Mechanism diagnostics | interval width distribution, share of `L = 0`, escalation rate, unresolved rate |
| Resources | bank bytes, mean and p95 latency, construction time |
| Statistics | paired bootstrap intervals grouped by original image, five seeds |

### 4.4 Preliminary Result

The complete pipeline already runs on a frozen WideResNet-50. The result does not fully support H2.

| Quantity | Measured |
|---|---|
| True score range `s_M` | 0.6637 – 1.2635 |
| Mean interval width | 1.1267 |
| Share of patches with `L = 0` | 73.1% |
| Violations `L > s_M` | 0 |
| Violations `s_M > U` | 0 |

Two facts hold at the same time. The interval is correct, with zero violations, yet it is wider than the signal itself. A mean width of 1.12 already exceeds the entire range of true scores, which means that local decisions can cover only part of the input distribution. This must be re-examined at realistic bank sizes in the main experiments.


---

## 5. Timeline

| Week | Task | Deliverable |
|---|---|---|
| 1 | Environment lock, repository skeleton, interval implementation | reproducible environment and unit tests |
| 2–3 | Reproduce PatchCore and AnomalyDINO baselines | reproduction table within a two-point AUROC tolerance |
| 3–4 | Verify the gap (H1) at 1/5/10/25% compression | disagreement curve and counterexamples |
| 5 | Minimal method with single-level screening | interval proof and adversarial tests |
| 6–7 | Full system, ablations, numerical reliability | version contract and cost table |
| 8 | Main experiments on MSC-AD and three MVTec AD categories | main tables and figures |
| 9 | Failure analysis and robustness | failure gallery |
| 10 | Writing and archiving | draft paper and reproduction checklist |

---

## 6. Expected Outcomes

The expected outcomes are an auditable evaluation protocol that separates compression-induced decision disagreement from detection accuracy, a working screening system with explicit versioning and an unresolved state released as code, an empirical boundary stating at which compression ratio, feature dimension and lighting condition alarms still match, and a negative result with concrete evidence if the mechanism does not hold.


---

## 7. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| The gap does not exist (H1 falsified) | Switch to an empirical evaluation and report the negative result |
| Intervals too wide, nearly everything escalates (H2 falsified) | Add level refinement, or downgrade the claim to a cheap certificate on the normal side |
| Simple recalibration already solves the problem | Treat recalibration as the strongest baseline |
| No edge device available | Report workstation measurements only and label them as simulation |
| Dataset access restricted | Use public datasets and never use unauthorised data |

---

## References

> All references are retrievable on IEEE Xplore or CNKI and were published between 2021 and 2026.

[1] Roth K, Pemula L, Zepeda J, Schölkopf B, Brox T, Gehler P. Towards Total Recall in Industrial Anomaly Detection[C]//2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). New Orleans: IEEE, 2022: 14298–14308. DOI: 10.1109/CVPR52688.2022.01392.

[2] Damm S, Laszkiewicz M, Lederer J, Fischer A. AnomalyDINO: Boosting Patch-based Few-Shot Anomaly Detection with DINOv2[C]//2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). Tucson: IEEE, 2025: 1319–1329. DOI: 10.1109/WACV61041.2025.00136.

[3] Batzner K, Heckler L, König R. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies[C]//2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). Waikoloa: IEEE, 2024: 127–137. DOI: 10.1109/WACV57701.2024.00020.

[4] Li H, Hu J, Li B, Chen H, Zheng Y, Shen C. Target Before Shooting: Accurate Anomaly Detection and Localization Under One Millisecond via Cascade Patch Retrieval[J]. IEEE Transactions on Image Processing, 2024, 33: 5606–5621. DOI: 10.1109/TIP.2024.3448263.

[5] Wang Y, Khan W A, Chung S H. Few-Shot Defect Detection of Catheter Products via Enlarged Scale Feature Pyramid and Contrastive Proposal Memory Bank[J]. IEEE Transactions on Industrial Informatics, 2024, 20(11): 13036–13046. DOI: 10.1109/TII.2024.3431069.

[6] Jia X, Chen X. Unsupervised Wind Turbine Blade Damage Detection With Memory-Aided Denoising Reconstruction[J]. IEEE Transactions on Industrial Informatics, 2025, 21(1): 762–770. DOI: 10.1109/TII.2024.3459612.

[7] Zhao Q, Wang Y, Wang B, Lin J, Yan S, Song W, Liotta A, Yu J, Gao S, Zhang W. MSC-AD: A Multiscene Unsupervised Anomaly Detection Dataset for Small Defect Detection of Casting Surface[J]. IEEE Transactions on Industrial Informatics, 2024, 20(4): 6041–6052. DOI: 10.1109/TII.2023.3341259.

[8] Shallari I, Mousavirad S J, O'Nils M. Edge-to-Fog Deployment of Vision Anomaly Detection for Industrial Automation[C]//2026 IEEE International Symposium on Measurement in Networking and Communications (MNC). 2026: 1–6. DOI: 10.1109/MNC69143.2026.11680699.

[9] Zhang S, Liu M. 基于偏差减少耦合超球面的工业图像异常检测 (Industrial image anomaly detection based on deviation-reduced coupled hyperspheres)[J]. Laser & Optoelectronics Progress, 2025, 62(12): 1215006. DOI: 10.3788/LOP242084.

[10] Zhao M, Zhang E. 语义—细节协同融合的细粒度工业品缺陷检测 (Fine-grained industrial product defect detection via semantic-detail collaborative fusion)[J]. Journal of Image and Graphics, 2026. DOI: 10.11834/jig.260260.

---

## Appendix: Items to Verify Before Submission

Three items need checking before submission. Crossref returns only romanised author names for [10] (Zhao Mingming, Zhang Erhu), so the Chinese characters should be confirmed on CNKI. The value `1215006` for [9] is that journal's article number rather than a page range, so the required citation style should be confirmed on CNKI. The venue location and page range for [8] should be taken from the IEEE Xplore record.

The remaining eight entries were checked item by item against Crossref, including authors, venue, volume, issue, pages and DOI, and can be used as they stand.
