# Phase 1 阅读笔记 / Reading Notes (3 pages)

**作者 / Author:** Wu Yuantai (CAPOOCATXP) · **日期 / Date:** 2026-10-08
**对应 / Corresponds to:** Research Plan Part A5 (Essential Reading Path), Part A12 Phase 1

详细逐篇笔记（含每个数字的出处、表格编号、代码路径）在：
- `docs/notes/reading-notes-patchcore-anomalydino-efficientad.md` (R1, R2, R3)
- `docs/notes/reading-notes-industrial-AD-and-NN-search.md` (R4, R6, R7, R10)

本文件是**三页综合**，只保留能直接改变本项目做法的内容。

---

## 0. 一句话总结六篇文献 / The six resources in one line each

| # | 资源 | 与本项目的关系 |
|---|---|---|
| R7 | Cover Trees (ICML 2006) | 提供 `L ≤ s ≤ U` 的度量几何基础。**它已经存在，所以本项目不能声称新定理。** 但它的 `O(log n)` 查询界有证明缺陷。 |
| R1 | PatchCore (CVPR 2022) | 复现基线；同时是"压缩改变阈值"这一缺口的来源。 |
| R2 | AnomalyDINO (WACV 2025) | 第二个基线；给了 1/4/16-shot 的参考数量分类法与精确的聚合规则。 |
| R3 | EfficientAD (WACV 2024) | 延迟对照；说明"小模型"与"小检索库"是两件事。 |
| R4 | MVTec AD 2 (IJCV 2026) | 主基准；提供真实光照偏移与低 FPR 评测协议。 |
| R6 | SPARC (arXiv 2026) | 最接近的当代工作；说明"更好检测"与"与冻结检测器一致"是两回事。 |

---

## 1. 为什么 `L(q) ≤ s_M(q) ≤ U(q)` / Why the interval holds

**上界。** 每个观测中心 `c_j` **本身就是一个参考向量**（`c_j ∈ M`）。
在子集 `{c_j}` 上取最小值，只会比在整个 `M` 上取最小值**更难得到小值**，
所以 `min_j d(q,c_j) ≥ min_{m∈M} d(q,m) = s_M(q)`。这就是 `U`。
**前提是 `c_j ∈ M`**——如果不是，`U` 可能低于 `s_M`，判 NORMAL 就会漏报。
本仓库 `build_coverage_index` 对每个中心做逐字节检查。

**下界。** 对任意 `j` 和单元 `C_j` 内的任意 `m`，三角不等式给出
`d(m,c_j) ≤ d(m,q) + d(q,c_j)`，整理得 `d(q,m) ≥ d(q,c_j) − d(m,c_j) ≥ d(q,c_j) − r_j`。
距离还非负，所以 `d(q,m) ≥ max(0, d(q,c_j) − r_j)`；对 `m` 取最小值即得 `L(q)`。
这就是 `L`。**前提是 `r_j ≥ max_{m∈C_j} d(m,c_j)`**，即半径必须覆盖整个单元，
且**只能遍历非空单元**（空单元的项 `d(q,c_j)` 其实是上界，放进 min 会让证明失效）。

**一个必须记住的推论。** `U` 说的是"银行里有个点离查询不超过这么远"，
`L` 说的是"银行里所有点都离查询至少这么远"。**只有下界能用来剪枝**，
因为只有"最乐观的情况也不可能更近"才允许丢弃。

---

## 2. 为什么"保持"不等于"正确" / Preservation is not correctness

这是整个项目的核心区分，也是我读了 R1 之后才真正理解的：

- **Reference decision** = `S_M(x) > t`，由冻结的完整银行做出。
- **Screened decision** = 只用中心与半径做出的判决（必要时升级）。
- **Preservation** = 两者相等。

如果参考检测器对某张正常图误报，preservation 只会**忠实地保留这个误报**。
所以任何报告必须同时给出两件事：
(i) 与参考的一致率，(ii) 真实缺陷召回率与正常图误报率。
只报 (i) 是有欺骗性的。

**R1 里有个具体例子说明为什么这不只是哲学问题。** PatchCore 论文 §4.4.2 报告
"30% → 95%" 的银行利用率：未压缩银行中不到 30% 的样本在测试时被用到，
coreset 压缩到 1% 后这个比例升到近 95%。这**不是**距离变大了，
而是压缩保留了**多样性**，所以有更大比例的参考点真正成为某个测试 patch 的最近邻。
把这两个现象混为一谈会导致错误的结论。

**R6 (SPARC) 说明了另一半。** SPARC 在 MVTec AD 2 + AeBAD-S 上把 Image AUROC
提高 13.8 个百分点（pooled），但作者自己指出：baseline 没有使用校准图像，
所以这个差距**同时包含了"多出来的部署信息"**。matched-budget 数字是 +12.9±1.4。
更重要的是它的失败模式：8 张校准图中 4 张被污染时，Image AUROC 仍 +1.3，
但 AU-PRO_0.3 变成 **−3.5**。**检测指标稳健不代表定位指标稳健。**

---

## 3. 为什么测试缺陷不能用来调阈值 / Why test defects cannot tune thresholds

R4 (MVTec AD 2) 把这件事制度化了：TRAIN 与 VALIDATION **只含正常图**；
VALIDATION 是**唯一**被许可用来定阈值的地方
（原文：用无缺陷验证图推导"training duration or thresholds for binary segmentation"）；
TEST_pub 是**可选**的初始估计；TEST_priv / TEST_priv,mix 的 ground truth 是私有的。
数据集设计者把这条规则做成基础设施，说明它容易被违反。

**R2 (AnomalyDINO) 提供了一个更微妙的反面教材。** 它的 few-shot 参考图选择是
**确定性**的：`sorted(listdir)[seed*k : (seed+1)*k]`，不是随机抽样。
这意味着"参考图"这个变量在不同 seed 之间是系统性的，不是噪声。
如果我把参考图选择当作随机重复来算不确定性，区间会算错。

**本项目的做法**（已写入 `docs/protocol/`）：正常训练图必须切成三个互斥角色
`reference_fit / development / final_calibration`，few-shot 参考只从
`reference_fit` 抽，另外两个角色的图**单独计数**。
"few-shot"绝不意味着总共只用了那几张图。

---

## 4. 三个会直接改变实现的技术细节 / Three details that change the code

**(a) 官方 PatchCore 仓库没有实现论文的加权分数。** 论文 Eq. 7 的 reweighting 是
`s = (1 − exp(d*)/Σ_{m∈N_b(m*)} exp(d(m))) · s*`，其中 `N_b(m*)` 是**参考侧匹配点
m\*** 的 b 个近邻，**不是**测试 patch 的近邻。而在 `fcaa92f` 的
`src/patchcore/` 中没有任何 reweighting 代码：
`NearestNeighbourScorer.predict` 是 `np.mean(query_distances, axis=-1)`，
`PatchMaker.score` 是 `torch.max`。
**后果：** "复现 PatchCore"和"实现 max-score 参考检测器"是两件不同的事，
输出必须分开命名。计划的 Part A8 也正是这么要求的。

**(b) AnomalyDINO 的聚合有一个会静默退化的细节。**
`q(D) = mean(前 int(0.01·len(D)) 个最大距离)`；若该计数下取整为 0，
**静默退化为 `np.max`**。448 分辨率 + 14 像素 patch 得到 32×32 = 1024 个 token，
所以取前 `int(10.24) = 10` 个。用 `ceil` 或固定数量都无法复现。

**(c) Cover Tree 的 `O(log n)` 有历史性的证明缺陷。** 2006 年 ICML 论文的
Theorem 5 声称 `O(c^12 log n)`；Curtin 2015 §5.3 指出证明有**关键缺陷**，
Elkin & Kurlin (TopoInVis 2022, arXiv:2208.09447) 构造了具名反例
（4.2 构造、5.2 ICML'06 k-NN、6.5 Ram et al. NIPS 2009），
并在 ICML 2023 用压缩 cover tree 修复。
**缺陷的根源**：原证明试图用"最长根到叶路径 × 常数"来估计迭代次数，
但在显式表示中"一个点可以同时出现在许多节点里"，所以路径长度**不能**界定访问节点数。
**教训：** 经验加速（1–2000×）与有缺陷的渐近证明**完全可以共存**，
因为坏证明错在难例上，而难例不出现在自然数据集里。
即使修复后的界也**条件于分布相关的膨胀常数**，而算法本身**无法检查**这个前提。
所以"cover tree 给出 `O(log n)`，压缩银行因此便宜"是一个**条件性、分布相关**的说法，
必须用**实测**的查询成本与剪枝统计来支持。

---

## 5. 五个未解决的问题 / Five unresolved questions

这些问题我**没有**答案，需要教授指路。它们已同步到
`docs/notes/open_questions.md`。

**Q1. 阈值的分位数约定。** 计划 A7 要求报告"exact empirical quantile convention"。
在只有 42 张校准正常图（bottle 的 `final_calibration`）时，名义 5% 目标意味着
约 2 张图在阈值之上。不同的插值约定（`lower`/`higher`/`linear`/`midpoint`）
在这么小的样本上会给出**不同**的阈值，进而改变所有不一致计数。
我应该在预注册里固定哪一个？论文里通常不写，这是不是需要我自己声明一个并做敏感性分析？

**Q2. 压缩银行的阈值来源。** 计划要求两件事：(i) 冻结 `t` 做决策保持测试，
(ii) 单独重新校准每个压缩检测器看能否消除问题。
但"重新校准"本身需要在 `final_calibration` 上取分位数——这与 (i) 用的是同一批图。
这是否构成"用同一批校准数据做两个实验"的泄漏？需不需要再切出第四个角色？

**Q3. 区间宽度与维度的关系。** Phase 0 用**留出查询**观测到：`K=68` 个中心覆盖
`N=676` 个 1536 维 patch 时，**73.1% 的 patch 下界为 0**（`L` 最大 0.7118），
平均区间宽度 **1.13**，而真实分数的整个跨度只有 **0.60**——区间比信号本身还宽。
这意味着 ALARM 侧在多数 patch 上没有判别力。
这是我的银行太小（单图 676 个 patch）造成的假象，还是高维特征的本质？
一个真实的 MVTec 银行有 `N ≈ 10^5`，`K` 也会更大，宽度会改善还是恶化？
这直接决定 C2/C3 是否还成立，我认为这是**第一个应该测的东西**。

（更正记录：首版测量用银行自身的 patch 当查询，`s_M` 恒为 0，
所以得到的 "100% 恒为 0" 是同义反复。已改为留出查询。）

**Q4. 精确 screening 的公平基线。** Phase 0 实测：`N=676, K=68` 时
screening 扫描 18.8 ms，**比**完整银行扫描 8.5 ms **更慢**。
原因是完整银行用一次 BLAS matmul 排序、只对 4 个候选做精确精化，
而 screening 必须对**每个**中心算精确距离（因为 `L` 在所有单元上取 min）。
只有 `N/K` 很大时前者才划算。那么：
(a) 公平的比较应该是"同样精确"的实现（即也给完整银行逐点算距离），
还是"各自最快的实现"？(b) 计划 A10 要求的 break-even 分析，
应该在哪一层做——patch 数、银行大小，还是字节数？

**Q5. `U` 的上界质量。** `U(q) = min_j d(q,c_j)` 用的是"中心本身在银行里"这一点。
但 `L` 几乎总是 0，说明**真正松的是 `L`，不是 `U`**。
`L` 松的原因是 `r_j` 是单元内最大距离，被离中心最远的那个点支配。
问题：既然 `U` 相对紧、`L` 很松，那么"判 ALARM"这一侧还有用吗？
如果 `L` 在几乎所有 patch 上都为 0，规则就退化成"`U ≤ t` 则正常，否则升级"，
那么贡献就只剩"用 `K` 次距离判定一部分正常图"，与"用全局半径"没有区别。
这是否说明 **C2 的价值主要在于 `U` 而不是区间**？如果是，
计划 A10 里 "per-cell radii vs single global radius" 的消融是否还有意义？

---

## 6. 工作向量例子 / Worked vector example

见 `docs/notes/worked_example_bounds.md`，可执行版本
`scripts/02_worked_bound_example.py`。用 6 个参考向量、2 个观测中心，
逐行算出 `L`、`s_M`、`U`，并覆盖了：
- 本地判 NORMAL（`U ≤ t`）
- 本地判 ALARM（`L > t`）
- **阈值相等（`L = t`）必须升级而非报警**——写成 `>=` 会静默制造不一致
- 升级后判 ALARM
- 解析器缺失 / 版本不匹配 → `UNRESOLVED`

---

## 7. 阅读后我改变的一个做法 / One change I made after reading

**原计划：** 以为 `L` 会因为半径松散而偏小，所以主要风险在"升级太多"。

**改变：** 读了 R7 和 R4 之后，我意识到真正的风险**不是**升级太多，
而是**区间过宽使 screening 退化**——Phase 0 实测区间宽度 1.13 与真实分数的
整个跨度 0.60 同量级，且 73% 的 patch 下界为 0，已经证实了这一点。
（首版我误把"银行自己查自己"得到的 100% 当作证据，那是同义反复，已更正。）
同时 R1 的 30%→95% 利用率数字说明
压缩的收益可能来自**覆盖多样性**而非"距离变大"，
这意味着我的"压缩改变报警"假设的机制描述可能不准确：
真正被改变的可能是**分数分布的形状**（因此阈值位置漂移），
而不是"距离整体变小"。这需要 Phase 3 用分数分布而不是只用不一致率来验证。

这句话如果要写进周报的"假设变更"，就是：
**假设从"压缩使距离整体变小从而改变阈值"改为"压缩使分数分布形状改变从而改变阈值位置"，
后者才是需要在 Phase 3 测量的对象。**
