# 范围与声明边界 / Scope and Claim Boundaries

**日期 / Date:** 2026-10-08
**目的 / Purpose:** 在任何结果产生之前写下"什么算证据、什么不算"。
这是预注册 (preregistration) 的一部分，防止事后挑选解释。

---

## 1. 研究问题（照抄计划，不修改）/ The question, verbatim

> **Provisional Research Title:** *When Memory Compression Changes Industrial
> Alarms: Decision-Preserving Edge Screening and Its Cost under Appearance Shift.*
>
> **One-Sentence Research Question:** Can a small normal-reference bank with
> conservative distance bounds preserve a frozen full-bank anomaly detector's
> binary decisions while reducing storage and expensive full-bank queries,
> including under lighting changes?

**可证伪性 / Falsifiability.** 这个问题的答案可以是"不能"。两种否证结果都成立：

- **F1:** 压缩银行在有用存储比下的报警不一致率低于 2 个百分点 → 缺口不存在，
  项目应转为严谨评测研究 (Part A11 primary fallback)。
- **F2:** 区间过宽导致升级率接近 100%，screening 无收益 → 机制无效，
  即使不一致存在也无法被利用。

两种情况都必须如实报告。**"假设被否证"是有效的阶段产出**
(Part A12: "Stopping a hypothesis is a valid phase outcome")。

---

## 2. 允许的声明 / Claims that are in scope

| 编号 | 声明 | 需要的证据 |
|---|---|---|
| C1 | 一套可审计的协议，把**银行压缩引起的二元决策不一致**与**检测器准确率**分开度量；固定阈值、等数据访问。 | 冻结阈值 t 下的逐图配对不一致计数 + 完整原始分数清单 |
| C2 | 基于既有度量界的紧凑 screening/升级实现，带显式版本与数值检查。**不声称新的最近邻定理。** | 区间证明 + 逐位单元测试 + 版本不匹配演示 |
| C3 | 跨内存预算、参考数量与光照变化的**实测**可靠性/资源边界，并与精确检索树、快速独立检测器比较。 | 至少两个数据集、5 seed、配对的 95% 区间、含全部状态的字节数 |

## 3. 明确不允许的声明 / Claims that are out of scope

| 禁止声明 | 理由 |
|---|---|
| "我们提出了一个新的最近邻上界/下界定理" | 三角不等式 + 度量球是 Cover Tree 的既有技术 (R7, 2006)。C2 明确不声称这个。 |
| "形式化验证的决策保持" | 未做区间算术，只有经验一致 (见 `numerics_contract.md` §4)。只能说 empirically decision-preserving。 |
| "已在油气场站部署验证" | 公开制造业图像只是代理数据。计划 A12/A14 明确禁止此类泛化声称。 |
| "AUROC 提升说明压缩更安全" | AUROC 无视阈值处的行为；这正是本项目要检验的缺口。 |
| "边缘设备上更快" | 本机无边缘设备，只有 M4 + MPS 工作站测量。 |
| "能耗降低 X%" | 无功耗计。 |
| "复现了 PatchCore 的 99.x% AUROC" | 需要 Phase 2 的实际运行；本阶段只搭了骨架和一个明确命名的 max-score 检测器。 |
| "加权 PatchCore 分数也被区间覆盖" | 需要单独推导；Part A8 明确 excluded。 |

---

## 4. 命名纪律 / Naming discipline

计划反复强调不要把不同的东西混在一起。本项目强制以下命名：

| 名称 | 含义 | 不是什么 |
|---|---|---|
| **Reference detector / full-bank detector** | 冻结银行 M + 冻结阈值 t 的精确 `S_M(x) = max_q min_m d(q,m)` | 不是 PatchCore 复现 |
| **PatchCore-style max-distance detector** | 同上，用于避免与 PatchCore 混淆的显式名字 | 不是官方 PatchCore 分数 |
| **Canonical PatchCore reproduction** | 运行 `third_party/patchcore-inspection` 的官方 CLI | Phase 2 才做 |
| **Decision disagreement** | 压缩银行判决 ≠ 参考判决的比率 | 不是准确率下降 |
| **Preservation** | screening 判决 == 参考判决 | 不是"判决正确" |

**最关键的一句 / The single most important sentence.** 保持完整模型的报警
**不代表**保持真实缺陷正确率，也**不保证**照明变化下的低误报。
preservation 不能修复参考模型自身的错误——一个被忠实保留的误报仍是误报。
所有报告必须同时给出 (i) 与参考的一致性 和 (ii) 真实缺陷召回率 / 正常误报率。

---

## 5. 数据角色规则 / Data-role rules (hard constraints)

1. 正常训练图像必须划分为**三个互不重叠**的角色：
   `reference_fit`（只用于建银行）、`development`（只用于开发期决策）、
   `final_calibration`（只用于定阈值 t）。
2. **测试集缺陷标签与掩码不得用于**阈值、银行预算或层级选择。
   一旦使用，实验即失效，必须重跑。
3. few-shot 的 1/4/16 参考图**只能**从 `reference_fit` 中抽；
   development 与 calibration 的正常图**另外计数**。
   "few-shot references" 绝不意味着总共只用了那些图。
4. 同一原图的多个增广版本**不是**独立样本；bootstrap 必须按原图分组。
5. 若数据不足以支撑三个互斥角色，就**减少调参**并报告该约束，
   **不得**制造虚假的独立性。

当前这三个规则由 `scripts/03_build_manifests.py` 的 `validate()` 强制检查
（在占位清单上已通过；下载真实数据后必须重跑）。

---

## 6. 阈值约定 / Threshold convention

- 主操作点：名义**图像误报率 5%**，在 `final_calibration` 正常图上取经验分位数。
- 次要：10%。1% 仅在校准/测试正常图数量足够时才报告
  （Part A7: "Small calibration sets cannot support a precise 1% false-alarm claim"）。
- 必须报告**样本数**与**确切的分位数约定**（插值方法、取等号方向）。
- 判决规则：`alarm ⟺ score > t`，取等号为 NORMAL。这一点在 screening 与
  calibration 两侧必须一致，否则会出现系统性的一图偏差。

**当前状态：尚未定阈值。** 需要真实数据，属 Phase 3。

---

## 7. 阶段门 / Gates

| 门 | 问题 | 通过条件 |
|---|---|---|
| Gate 1 | 经验缺口存在吗？ | pilot 目标：在有用的存储降幅下，**至少两个类别**的不一致 ≥ 2 个百分点 |
| Gate 2 | 复现是否可信？ | 与已发表/仓库结果在**预先声明**的 2 点 AUROC 容差内，或差异可归因于显式设置变化 |
| Gate 3 | 最小信号存在吗？ | 无未解释的区间违规；在有用银行上限下 ≥20% 输入本地判定 |
| Gate 4 | 完整系统有用吗？ | 相对最强可比基线、在等数据/资源访问下的实测优势 |
| Gate 5 | 新颖性还成立吗？ | 重新检索最接近工作后 C1–C3 仍未被覆盖 |

每个门报告必须包含：门号与问题；冻结的方法/数据/版本 ID；全部 seed 与类别及
不确定性；**最强反例**；等资源下的基线比较；GO/MODIFY/STOP 建议。

模板见 `docs/reports/gates/gate_template.md`。

---

## 8. 本阶段（Week 1）的自我限制 / Week-1 self-restriction

教授 A14 原文："Do not begin YOLO pruning, diffusion generation or multimodal
collection before the pilot question is tested."

本仓库因此**不包含**：YOLO、剪枝、扩散生成、多模态融合、任何训练循环、
任何测试集评估、任何 AUROC 数字。
