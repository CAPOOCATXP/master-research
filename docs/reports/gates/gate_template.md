# 阶段门模板 / Gate Report Template

**用法 / How to use:** 复制本文件为 `docs/reports/gates/gate{N}_{name}.md`，
填写每一个字段。**不得留空**；没有的数据写 `NOT MEASURED` 并说明原因。

A13 要求每个门报告额外包含：门号与问题、冻结的方法/数据/版本 ID、
全部 seed 与类别及不确定性、**最强反例**、等数据/资源访问下的基线比较、
GO/MODIFY/STOP 建议与下周工作。

---

## 门号与问题 / Gate number and question

- **Gate:** N
- **Question:**
- **Date:**
- **Author:**

## 1. 冻结的版本 / Frozen versions

| 项 | 值 |
|---|---|
| encoder id | |
| preprocessing id | |
| bank id (sha256) | |
| bank size N | |
| n centers K | |
| threshold t | |
| threshold policy + quantile convention | |
| calibration sample count | |
| code commit (this repo) | |
| patchcore commit | |
| anomalydino commit | |
| dataset version / split | |

## 2. 数据角色 / Data roles

| 角色 | 图像数 | 来源 | 是否被其他实验用过 |
|---|---|---|---|
| reference_fit | | | |
| development | | | |
| final_calibration | | | |
| (test, 仅在协议冻结后) | | | |

**泄漏声明 / Leakage declaration.** 明确写出本门的任何决定是否用到了测试缺陷标签
或掩码。如果用到，实验作废并重跑。

## 3. 全部 seed 与类别 / All seeds and categories

| seed | 类别 | 指标 | 值 | 备注 |
|---|---|---|---|---|
| | | | | |

**不确定性 / Uncertainty.**

- 配对 95% 区间（bootstrap，按原图分组，≥1000 次重采样）:
- 正常图误报率的 Wilson/binomial 区间:
- 参考选择造成的方差（≥5 seed）:

**未跑的配置 / Excluded configurations.** 列出并说明为何排除。

## 4. 结果 / Results

| 指标 | 完整银行（参考） | 压缩银行 | 筛选系统 |
|---|---|---|---|
| 图像不一致率（vs 参考） | — | | |
| 额外报警 / 漏掉的参考报警 | — | | |
| 真实缺陷召回率 | | | |
| 正常图误报率 | | | |
| 升级率 | — | — | |
| 未解析率 | — | — | |
| 银行字节 | | | |
| 总延迟 p50 / p95 | | | |

**分母声明 / Denominators.** 本地判定图与全体输入图的指标必须**分开**报告。

## 5. 最强反例 / Strongest counterexample

**必填。** 描述一个具体的、使结论最受威胁的样本。
如果找不到反例，说明为什么找不到（例如"区间违规为 0，且穷举了所有测试图"）。

- 样本 ID:
- 现象:
- 为什么它威胁结论:
- 是否已解释:

## 6. 基线比较（等数据/资源访问）/ Baseline comparison at equal access

| 基线 | 数据访问 | 内存字节 | 延迟 | 关键指标 |
|---|---|---|---|---|
| (a) 精确完整银行 | | | | |
| (b) 等字节随机银行 | | | | |
| (b') 等字节贪心压缩银行 | | | | |
| (c) 压缩 + 独立重校准 | | | | |
| (d) 基于 margin 的启发式升级 | | | | |
| (e) 全局半径 screening | | | | |
| (f) 精确 ball/cover tree | | | | |
| (g) AnomalyDINO / EfficientAD-S | | | | |

**必须说明**每个基线是否使用了**相同的**校准图预算与相同的参考图数量。
不等预算的比较必须标注为 unmatched，并单列 matched-budget 数字。

## 7. 建议 / Recommendation

- [ ] **GO** — 继续当前方向
- [ ] **MODIFY** — 修改问题或方法，说明改什么
- [ ] **STOP** — 否证，转入 A11 的 fallback 方向

**理由（≤5 句）:**

## 8. 下周三个任务 / Three tasks for next week

1.
2.
3.

## 9. 声明检查 / Claim checklist

- [ ] 没有声称新的最近邻定理
- [ ] 使用 "empirically decision-preserving" 而非 "formally certified"
- [ ] 没有把 preservation 等同于 correctness
- [ ] 没有用 AUROC 推断阈值行为
- [ ] 没有在没有实测设备的情况下声称部署性能
- [ ] 没有在没有功耗计的情况下报告能耗
- [ ] 没有把公开制造业数据的结果说成油气现场验证
