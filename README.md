# master-research

**Compression 何时改变工业报警：决策保持的边缘筛选及其代价**

*When Memory Compression Changes Industrial Alarms: Decision-Preserving Edge
Screening and Its Cost under Appearance Shift*

Wu Yuantai (CAPOOCATXP) · 十周研究计划的执行仓库 · Week 1 状态

---

## 这个仓库在做什么 / What this repository is

固定一个**冻结的**参考异常检测器（完整正常 patch 记忆库 + 固定阈值 `t`），
在边缘侧只保留**一小部分观测到的参考 patch（中心 `c_j`）+ 保守覆盖半径 `r_j`**，
用度量界

```
L(q) = min_j max(0, d(q, c_j) − r_j)   ≤   s_M(q) = min_{m∈M} d(q, m)   ≤   U(q) = min_j d(q, c_j)
```

对每张图的分数区间 `[L_x, U_x]` 做判定：

1. `U_x ≤ t` → **normal**（本地判定，不查询完整银行）
2. `L_x > t` → **alarm**（本地判定，不查询完整银行）
3. 否则 → **升级**到版本匹配的完整银行，采用它的判决
4. 解析器不可用 / 版本不匹配 / 超时 → **unresolved**，**绝不**静默当作 normal

研究问题是：**这个过程能否保持参考检测器的二元判决**，同时减少存储与查询成本，
并且在光照变化下仍然成立。

**本项目不声称新的数学定理。** 三角不等式与度量球是 Cover Tree（ICML 2006）的
既有技术。潜在贡献是**可复现的可靠性—资源实证结论及其适用边界**。

---

## 快速开始 / Quick start

```bash
# 1. 环境（macOS / Apple Silicon；见 env/setup_macos_arm64.sh 头部说明）
bash env/setup_macos_arm64.sh

# 2. 跑全部检查
.venv/bin/python scripts/01_vector_distance_check.py    # 十个 2-D 向量的距离自检
.venv/bin/python scripts/02_worked_bound_example.py     # 六向量手算区间例子
.venv/bin/python scripts/03_build_manifests.py          # 数据角色清单 + 互斥校验
.venv/bin/python scripts/00_env_check.py                # 真实骨干 + 精确距离 + 计时
.venv/bin/python -m pytest -q                           # 66 个单元测试
```

所有脚本都会把 JSON 日志写到 `results/logs/`，并把 `PASS`/`FAIL` 打到 stdout。

**第一次接触这个方向？** 先看 `docs/tutorial/how_it_works.md` ——
它用 7 张图从零解释整个项目在做什么、代码怎么组织、以及我踩过的坑。

---

## 目录结构 / Layout

```
master-research/
├── README.md                    本文件
├── conftest.py                  让 pytest 不依赖 editable 安装（见 env/ §2b）
├── pyproject.toml               可安装包 (src layout)
├── requirements/                base.txt / dev.txt（不锁版本，锁版本见 env/）
├── env/                         Phase 0 交付物
│   ├── hardware_inventory.md      硬件与数据清单 + 显式 UNKNOWN 登记
│   ├── environment_lock.md        解析出的版本 + 两个安装阻塞与解决
│   ├── environment_lock_freeze.txt  完整 pip freeze
│   ├── repos.lock.yaml           钉住的官方仓库 commit 与权重 sha256
│   └── setup_macos_arm64.sh      可复现安装脚本
├── src/master_research/         核心库
│   ├── distances.py               精确欧氏距离（分块 + 直接相减精化）
│   ├── bounds.py                  区间 L/U、覆盖索引、数值护栏
│   ├── bank.py                    银行构造（贪心 coreset）、字节核算
│   ├── screening.py               参考检测器 + 四分支 screening + UNRESOLVED
│   ├── versioning.py              版本 ID（编码器/预处理/银行/阈值）
│   └── features.py                冻结骨干与预处理（WideResNet-50 layer2+3）
├── scripts/                     可执行交付物（编号 = 运行顺序）
│   ├── 00_env_check.py            Phase 0：环境 + 前向 + 距离检查 + 计时 + 图
│   ├── 01_vector_distance_check.py  十个 2-D 向量（数据未到时的指定替代方案）
│   ├── 02_worked_bound_example.py   六向量两中心 L/s/U 手算
│   └── 03_build_manifests.py        三类别角色清单 + 互斥性校验
├── tests/                       66 个单元测试（含针对真实故障的回归测试）
├── docs/
│   ├── tutorial/                  ★ 新手图解：how_it_works.md + 7 张图
│   │   ├── how_it_works.md        从零解释整个项目在干什么
│   │   ├── make_figures.py        生成这 7 张图的脚本
│   │   └── figures/               图（第 6、7 张用真实骨干特征算的）
│   ├── plan/scope_and_claims.md   什么算证据、什么不许声称（预注册）
│   ├── protocol/numerics_contract.md  数值契约（精度、护栏、相等情形、空单元）
│   ├── notes/                    阅读笔记、手算例子、五个未解决问题
│   ├── manifests/                三个类别的角色清单（当前为占位）
│   └── reports/week01_report.md  Week 1 一页报告
├── data/        (gitignored)     数据集，非商业许可，不得再分发
├── results/     (gitignored)     图 / 日志 / 表
└── third_party/ (gitignored)     钉住的官方仓库，由 setup 脚本重建
```

---

## 重要约束 / Hard constraints

这些不是风格偏好，违反其中任何一条都会使实验失效。

| 约束 | 出处 |
|---|---|
| 只允许**欧氏**距离。余弦**不是**度量，区间证明不成立。 | `numerics_contract.md` §1 |
| 区间**必须**用数值护栏加宽；结果只能标 "empirically decision-preserving"。 | A9 |
| `L_x > t` 必须**严格**；等号为 NORMAL（与参考判据一致）。 | `worked_example_bounds.md` §2 Q4 |
| 空单元**必须**被排除，否则 `L` 可能超过 `s_M`，证书不成立。 | `numerics_contract.md` §6 |
| 中心**必须**属于银行，否则 `U` 不是上界。 | `numerics_contract.md` §7 |
| 测试缺陷标签/掩码**不得**用于阈值、预算或层级选择。 | A7 / A10 |
| 正常图必须切为互斥的 reference-fit / development / final-calibration。 | A7 |
| few-shot 参考**只能**取自 reference-fit；其他角色的图**另计**。 | A7 |
| 未解析**绝不**降级为 normal。 | A8 |
| 量化、剪枝、近似检索**不进入**第一版保证。 | A8 |

---

## 当前状态与已知负面结果 / Status and known negative results

**Week 1 完成了 Phase 0 + Phase 1。** 两个真实故障已定位并修复，两个负面结果
已如实记录，不隐藏：

1. **安装阻塞 ①（已解决）** — DSH 运行时 Python 带 hardened runtime
   (`flags=0x10000(runtime)`, TeamID `NAN929V4UM`)，拒绝加载 PyTorch 的
   ad-hoc 签名 dylib。解决方法是在同目录复制 Python 并 ad-hoc 重签名。
   详见 `env/environment_lock.md` §2。

1b. **安装阻塞 ②（已解决）** — 该运行时 Python 的 `site.py` 会**静默跳过**
   带 macOS `UF_HIDDEN` 标志的 `.pth` 文件（实测 `st_flags=0x8040`）。
   结果是 `pip install -e .` 报告成功、文件也在、但 `import master_research`
   报 `ModuleNotFoundError`。修法是 `chflags nohidden`（已写进 setup 脚本），
   并加 `conftest.py` 让测试不依赖 editable 安装。
   详见 `env/environment_lock.md` §2b。

2. **下界 bug（已修复）** — 第一版 `bounds._dist_to_centers` 用了
   `|a|²+|b|²−2ab` 展开式，在 1536 维特征上灾难性抵消，
   导致 **676 个 patch 中有 31 个出现 `L > s_M`**（无效下界，会制造假证书）。
   改为直接相减后违规数为 0。回归测试在 `tests/test_bounds.py`。

3. **区间宽度与信号范围同量级。** 留出查询下，`L = 0` 的 patch 占 **73.1%**
   （不是全部），`L` 最大 0.7118；平均区间宽度 **1.1267**，而真实分数的整个
   取值范围只有 **0.6637–1.2635（跨度 0.60）**。所以区间比信号本身还宽，
   ALARM 侧在多数 patch 上无法本地判定。详见 `docs/reports/week01_report.md`。

4. **已更正的方法学错误。** 第一版 Phase 0 脚本用**银行自身的 patch** 当查询，
   于是 `s_M` 恒为 0，`L = 0` 由 `L ≤ s_M` 强制成立 —— 那是同义反复，不是发现。
   改为用另一张图的 patch 当查询后才得到上面第 3 条的真实数字。
   脚本现在同时输出 `interval_sanity_self_query`（标注为 `DEGENERATE`）
   与 `interval_sanity_held_out` 两个对照。

5. **负面结果 2：小 `N/K` 时 screening 比完整银行慢**（18.0 ms vs 8.2 ms）。
   原因是完整银行用 BLAS 排序 + 少量精化，而 `L` 要求对每个中心算精确距离。

6. **架构发现：官方 PatchCore 仓库没有实现论文 Eq. 7 的加权分数。**
   `NearestNeighbourScorer.predict` 是 `np.mean(...)`，
   `PatchMaker.score` 是 `torch.max`。所以"复现 PatchCore"与
   "实现 max-score 参考检测器"必须分开命名。

**尚未开始（按教授 A14 明确禁止）：** YOLO、剪枝、扩散生成、多模态融合。

---

## 数据许可 / Data licence

MVTec AD 与 MVTec AD 2 均为**非商业**许可（AD 2 为 CC BY-NC-SA 4.0），
**不得再分发**。本仓库的 `data/raw/` 与 `data/cache/` 已在 `.gitignore` 中，
`results/` 中的图与日志同样默认忽略。
本仓库内**没有**任何数据集图像。

---

## 引用与本仓库的关系 / Relationship to the cited work

本仓库**包含**对以下工作的**阅读笔记**，但**不包含**它们的代码或数据：

- R1 PatchCore (CVPR 2022) — 官方仓库钉在 `fcaa92f`，**未运行**
- R2 AnomalyDINO (WACV 2025) — 官方仓库钉在 `b9d1c26`，**未运行**
- R3 EfficientAD (WACV 2024) · R4 MVTec AD 2 (IJCV 2026) · R5 MH-PatchCore
- R6 SPARC (arXiv 2026，**无官方代码**) · R7 Cover Trees (ICML 2006) · R10 (ICML 2023)

`third_party/` 中的两个仓库按各自的许可使用，本仓库不修改它们，
只通过 `env/repos.lock.yaml` 记录 commit。

---

## 复现声明的诚实边界 / Honest limits of reproduction

读本仓库的任何数字之前，请先读 `docs/plan/scope_and_claims.md` §3。简言之：

- 所有结果都是 **empirically** decision-preserving，**不是** formal certificate。
- 所有数据角色清单当前是 `PROVISIONAL_NO_DATA`（MVTec 未下载）。
- 所有延迟数字来自 **Apple M4 / MPS / 16 GB 统一内存**，**无 CUDA、无边缘设备**。
- **没有**任何 AUROC / 缺陷召回率数字——那些需要真实数据，属 Phase 2–3。
- **没有**能耗数字（无功耗计）。
