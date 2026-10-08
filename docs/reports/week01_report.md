# Week 1 报告 / Weekly Report

**提交人 / From:** Wu Yuantai (CAPOOCATXP) · **日期 / Date:** 2026-10-08
**阶段 / Phase:** Phase 0 (环境) + Phase 1 (学习) · **对应 / Refs:** A12, A13, A14

---

## Done — 完成的任务与工时

| # | 任务 | 对应要求 | 工时 |
|---|---|---|---|
| 1 | Phase 0 环境：定位并**解决**一个真实的安装阻塞（macOS 签名）；锁定 torch 2.14.1 / Python 3.12.14；下载并 SHA-256 校验 WideResNet-50 权重 | A12 P0 | 3.5 h |
| 2 | 钉住两个官方仓库到具体 commit（PatchCore `fcaa92f`、AnomalyDINO `b9d1c26`），记录在 `env/repos.lock.yaml` | A6 | 0.5 h |
| 3 | 硬件/数据清单，含明确的 UNKNOWN 登记（无边缘设备、无功耗计、无 CUDA） | A14 #1 | 1.0 h |
| 4 | 真实 WideResNet-50 前向（MPS）→ 676×1536 patch 特征；精确距离自检 vs float64 暴力 | A14 #2 | 2.0 h |
| 5 | 十个 2-D 人造向量的距离检查（数据未到时的**教授指定替代方案**） | A14 #2 | 0.5 h |
| 6 | 六向量 / 两中心的 `L`/`s_M`/`U` 手算例子，含阈值 tie 与升级 case | A14 #3 | 2.0 h |
| 7 | 三个试点类别的 reference-fit / development / final-calibration 清单 + 互斥性校验 | A14 #4 | 1.5 h |
| 8 | 核心库：区间、screening 四分支、版本 ID、精确距离、银行构造 | A8 | 4.0 h |
| 9 | 66 个单元测试，含针对真实故障的回归测试 | A9 | 2.5 h |
| 10 | 六篇文献笔记（3 页综合 + 两篇详细版，全部对过原始来源） | A5/P1 | 5.0 h |
| 11 | 数值契约、范围与声明边界、仓库骨架与复现脚本 | A9/A13 | 2.5 h |

**合计约 25 h。**

---

## Evidence — 主要证据

### 主表：区间自检（真实骨干特征，`D=1536`，**留出查询**）

| 指标 | 数值 |
|---|---|
| 银行 patch 数 × 维度 | 676 × 1536（layer2+layer3） |
| 查询来源 | **另一张图**的 patch（seed+1，非银行自身） |
| `L > s_M` 违规数 | **0** |
| `s_M > U` 违规数 | **0** |
| 真实分数 `s_M` 范围 | 0.6637 – 1.2635（均值 0.9183） |
| `L` 均值 / 最大 | 0.0333 / **0.7118** |
| **`L = 0` 的 patch 比例** | **73.1 %** |
| `L > 0` 的 patch 比例 | **26.9 %** |
| 平均区间宽度 | 1.1267 |
| 平均半径 `r_j` / 平均「到最近中心距离」 | 1.1595 / 1.1599 |
| float32 vs float64 暴力距离最大绝对差 | 0.0 |
| 分块 vs 非分块（逐位相等） | 是 |
| 同一输入两次前向（逐位相等） | 是 |

### ⚠ 一处已更正的方法学错误 / A corrected methodological error

**第一版 Phase 0 脚本用「银行自己的 patch」当查询**（`queries = bank = feats`），
于是真实分数 `s_M` 恒等于 0，而 `L ≤ s_M` 又强制 `L ≥ 0`，
所以「`L` 全部为 0」是**同义反复**，不是发现。

改成用**另一张图**的 patch 当查询后，结论变为：
`L = 0` 的比例从 100% 降到 73.1%，`L` 最大值从 0 变成 0.7118。

脚本已修正为同时报告两个对照，并把退化情形显式标注为 `DEGENERATE`：
`results/logs/phase0_env_check.json` 中的 `interval_sanity_self_query`
与 `interval_sanity_held_out`。这一条我留在报告里，因为它说明
「用银行自己验证银行」这类错误很容易发生且不易被察觉。

### 配置（可复现）

```
encoder          wide_resnet50_2:IMAGENET1K_V1:layer2+layer3
preprocessing    resize256-center224-ps3-st1-norm(ImageNet)
weights sha256   95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8
device           Apple M4 GPU via MPS (mps_available=True, cuda_available=False)
repo commits     patchcore fcaa92f124fb1ad74a7acf56726decd4b27cbcad
                 anomalydino b9d1c2648e3a5247437d4d953d907a8f3d994457
data split       PROVISIONAL_NO_DATA (MVTec 未下载；清单为占位符)
queries          synthetic image seed+1（与银行图不同）
```

### 产物路径

| 产物 | 路径 |
|---|---|
| 图：查询图 / 精确分数图 / 下界图 / 上界图 | `results/figures/phase0_patch_score_map.png` |
| 环境与计时日志（含两种区间对照） | `results/logs/phase0_env_check.json` |
| 距离检查日志 | `results/logs/vector_distance_check.json` |
| 手算区间日志 | `results/logs/worked_bound_example.json` |
| 角色清单（3 类别 + few-shot） | `docs/manifests/` |
| 教学图与说明 | `docs/tutorial/` |

**主图：**

![Phase 0 patch score map, exact score, lower bound, upper bound](../../results/figures/phase0_patch_score_map.png)

（左起：留出查询图；精确 `s_M(q)`；下界 `L(q)`；上界 `U(q)`。
第三张图**大部分为黑**——73% 的 patch 下界为 0，但**不是全部**。）

---

## 最重要的负面结果 / Most important negative result

**下界 `L(q)` 在 73.1% 的留出 patch 上等于 0，且区间宽度（1.13）
比真实分数的整个取值范围（0.60）还大。**

这不是 bug，是几何事实：`L(q) = min_j max(0, d(q,c_j) − r_j)`，
而 `r_j` 是单元内**最远**点到中心的距离，被离群点支配。
实测平均半径 `r_j = 1.1595`，平均「查询到最近中心的距离」`d = 1.1599`——
两者几乎相等，所以减法经常落在 0 附近。

**后果**：对 73% 的 patch，screening 规则

```
if U_x <= t:  NORMAL
elif L_x > t: ALARM        <-- 这一支只在 27% 的 patch 上有机会触发
else:         escalate
```

在**图像级聚合（取最大）**下尤其受限：图像分数取所有 patch 的最大值，
只要有一个 patch 的 `L > t` 就能本地判 ALARM；但如果全部 patch 的 `L = 0`，
该图必然升级。所以它**不是**完全退化，但适用范围被显著压缩。

**待回答的关键问题**：区间宽度 1.13 vs 信号范围 0.60 —— 这个比例是
单图银行（N=676）造成的，还是高维特征的固有性质？如果是前者，
真实 MVTec 银行（N ≈ 10^5）会改善；如果是后者，C2 的前提不成立。
这个问题已列为下周第一个任务（见 Q3）。

### 第二个负面结果 / Second negative result

**在 `N=676, K=68` 时，screening 扫描（18.8 ms）比完整银行扫描（8.5 ms）更慢。**

原因已定位且不是实现瑕疵：完整银行用一次 BLAS matmul 排序 + 只对 4 个候选做
精确精化；而 `L` 要在所有单元上取 min，所以 screening **必须**对每个中心算
精确距离。理论 FLOP 比只有 `N/K ≈ 9.9`，不足以补偿。
（详见 `docs/notes/open_questions.md` Q4。）

### 第三个发现（架构层面）

**官方 PatchCore 仓库根本没有实现论文 Eq. 7 的加权分数。**
`NearestNeighbourScorer.predict` 是 `np.mean(query_distances, axis=-1)`，
`PatchMaker.score` 是 `torch.max`。已在 commit `fcaa92f` 的 `src/patchcore/` 中确认。
所以"复现 PatchCore"与"实现 max-score 参考检测器"必须分开命名（Part A8 也正是此意）。

---

## Blockers — 阻塞与我已尝试的修复

| # | 阻塞 | 尝试过的修复 | 需要教授支持 |
|---|---|---|---|
| 1 | **MVTec AD / AD 2 未下载**（表单注册 + 非商业许可接受） | 已按 A14 #2 的指定替代方案完成 2-D 距离检查；清单做成 `PROVISIONAL_NO_DATA` 并带 provenance 标记 | 能否提供数据集，或确认由我自行注册下载 |
| 2 | **无边缘设备**（A6 要求 acquire） | 无；已把清单中该项标为 UNKNOWN，并把所有结论限制在工作站 | 是否有可借用的设备；否则确认只做工作站测量 + 成本模拟 |
| 3 | **无功耗计** | 无 | 确认本项目完全不报告能耗 |
| 4 | **区间宽度与信号范围同量级**（1.13 vs 0.60），73% 的 patch 下界为 0 | 已定位为几何事实而非 bug；已提出便宜的合成规模实验（Q3）来判定它是否随 `N`/`K` 改善 | 是否同意在 Phase 3 之前先做这个判定实验（结果可能要求修改 C2） |
| 5 | **我自己的一处方法学错误已更正**：首版 Phase 0 用银行自身当查询，导致 `L=0` 是同义反复 | 已改为留出查询，脚本同时报告两个对照并把退化情形标为 `DEGENERATE`；周报保留此记录 | 无需支持，但请在审阅时确认更正后的数字口径 |
| 5 | **16 GB 统一内存 / 无 CUDA** 与 A6 假设的 8–12 GB 独立显存 GPU 不一致 | 已记录；MPS 可用且实测前向正常 | 主网格（A10）在此机器上的可行性判断，或确认缩减网格并记录 |

---

## Next — 下周三个具体任务

| # | 任务 | 预期产出 |
|---|---|---|
| 1 | **先做区间宽度的合成规模实验**（`D ∈ {32,256,1536}` × `N ∈ {10^3,10^4,10^5}` × `K/N ∈ {1%,10%}`），记录 `L=0` 比例与平均宽度 | 一张判定图 + 一句话结论：C2 的前提是否成立。**这个结果决定后面所有工作的方向。** |
| 2 | 拿到 MVTec AD 后：重跑 `scripts/03_build_manifests.py` 覆盖占位清单；实现特征缓存管线；核对 bottle 的 209 张正常图计数 | 真实清单（`provenance=REAL`）+ 缓存特征 + 计数核对记录 |
| 3 | Phase 2 复现准备：为 AnomalyDINO 建**独立 venv**（它锁定 `numpy==1.26`，与本项目 `numpy 2.5.3` 冲突）；复现 PatchCore bottle 单类 | 两个可运行命令 + 与已发表结果的对比表（预先声明 2 点 AUROC 容差） |

---

## 假设变更（一句话）/ One-sentence hypothesis change

> **原假设：** 压缩使近邻距离整体变大，从而使阈值附近的判决发生改变。
>
> **修改为：** 压缩更可能改变**分数分布的形状**（因此阈值应处的位置漂移），
> 而不是简单地整体平移距离；而机制上的主要风险不是"不一致率太小"，
> 而是**区间太宽**——实测区间宽度（1.13）与真实分数的整个取值范围（0.60）
> 同量级，且 73% 的 patch 下界为 0。Phase 3 必须同时测量分数分布的形状
> 和 `L > t` 的实际触发率，而不只是不一致率。

依据：R1 §4.4.2 的 "30% → 95%" 银行利用率说明压缩的效应来自**覆盖多样性**而非
"距离变大"；同时本阶段 Phase 0 实测区间宽度与信号范围同量级、
下界在多数 patch 上无信息。两条证据指向同一个方向。

---

## 一句话状态 / One-line status

**GO with a caveat**：环境与骨架已就绪并可复现。
下界**并非**在全部 patch 上失效（73% 为 0，27% 给出正信息），
但区间宽度与信号范围同量级，因此 C2 的机制前提**尚未被支持**。
建议下周先做任务 1 的判定实验（`D`/`N`/`K` 规模扫描），再决定是否继续 C2。
