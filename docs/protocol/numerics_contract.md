# 数值契约 / Numerics Contract

**冻结日期 / Frozen:** 2026-10-08
**作用域 / Scope:** 所有区间、所有 screening 决策、所有"决策保持"声明。

这份文件固定本项目里"数值上允许做什么"。任何偏离都必须在此处修改并给出理由，
而不是在某个脚本里悄悄发生。

---

## 1. 允许的距离 / Permitted distance

**只允许欧氏距离 (Euclidean)，在归一化后的向量上使用原始尺度。**

理由是硬的：区间证明依赖三角不等式，而三角不等式要求度量。
余弦距离**不是**度量，`cos(q,m) ≤ cos(q,c) + cos(c,m)` 不成立。
如果将来要用余弦，必须先做一个显式审计过的转换，并把转换误差计入区间宽度。
Part A8 已经写明这一点，本文件只是把它变成可执行的约束。

本项目当前**不做**任何 L2 归一化到单位球——因为归一化会改变 `s_M` 的取值，
从而改变参考检测器的阈值位置。归一化如果要做，必须在编码器版本 ID 里体现。

## 2. 精度 / Precision

| 项 | 规定 |
|---|---|
| 存储 | `float32` (`np.float32`)，C 连续 |
| 累加 | 允许在单个 kernel 内部使用更高精度，但**输出**必须回落到 float32 |
| 区间 | 一律用 `NumericalGuard` 加宽，见 §4 |
| 明确排除 | 量化 (int8/fp16 存储)、骨干剪枝、近似检索 (FAISS IVF/HNSW) |

最后一行是 Part A8 的原文要求："Quantization/backbone pruning and approximate
search are excluded from the first guarantee."

## 3. 距离计算的实现约束 / Implementation constraints

### 3.1 展开式禁止用于区间 / The expansion form is banned in bounds

`|a-b|² = |a|² + |b|² − 2a·b` 在 float32 下会灾难性抵消。在 1536 维骨干特征上
（幅值 ~1，`|a|² ≈ 1536`），两个大项相减后有效位数所剩无几。

**已记录的故障 (2026-10-08, Phase 0).** 最初 `bounds._dist_to_centers` 使用展开式。
结果是 `d(q,c_j)` 偏大 → `L(q)` 偏大 → **676 个 patch 中有 31 个出现 `L(q) > s_M(q)`**，
即下界不再是下界。这会产生"已验证的报警"这种假证书。改为直接相减后违规数为 0。

**规则：**
- 用于**排序**（找候选）时可以用展开式，因为它只影响顺序，且有 refine 兜底。
- 用于**区间的任何数值**必须来自直接相减 `‖q − c_j‖`。
- `min_distances` 对前 `refine_candidates`（默认 4）个候选做直接相减再取最小，
  因为展开式可能把两个几乎等距的参考点**排序颠倒**；只 refine 第一名会**高估**
  `s_M`。实测确认这个 refine 确实改变了结果
  (`max_abs_diff_unrefined_vs_refined = 6.18e-3`)，不是防御性冗余。

### 3.2 分块只是内存手段 / Chunking is a memory device only

`chunk_bytes` 不得改变任何结果。测试 `test_chunking_is_numerically_inert`
逐位比较默认分块与 1024 字节分块。

## 4. 数值护栏 / The numerical guard

```python
NumericalGuard(rel = 8 * eps_f32 ≈ 4.77e-07, abs_ = 1e-30)
```

应用方式：

| 量 | 方向 | 理由 |
|---|---|---|
| `r_j` (半径) | 向上取整 | `r_j` 偏大 → `L` 偏小 → 安全 |
| `L(q)` | 向下取整 | 下界只能低不能高 |
| `U(q)` | 向上取整 | 上界只能高不能低 |

**这仍然不是形式化证书 / Still not a formal certificate.** 8 ulp 是一个务实的
允许量，**不是**对分块 kernel 的证明误差界。Part A9 明确要求：
"Without conservative verified numerical bounds, label results 'empirically
decision-preserving', not 'formally certified'."

因此本项目的所有结果一律标注为 **empirically decision-preserving**。
要做形式化证书，需要区间算术 (interval arithmetic) 或经审计的误差界，两者都未做。

## 5. 边界的相等情形 / Equality at the threshold

参考判决定义为 `alarm ⟺ S_M(x) > t`。因此：

- `L_x > t` 是**严格**不等式才可判 ALARM。写成 `L_x >= t` 会在 `L_x == t` 时
  对一个参考判决为 NORMAL 的图报 ALARM，静默地制造一次不一致。
- `U_x <= t` 判 NORMAL，这里取等号是安全的：`S_M(x) ≤ U_x ≤ t` 意味着
  参考判决必为 NORMAL。

这两条已由 `tests/test_screening.py::test_lower_equal_to_threshold_escalates_not_alarm`
和 `scripts/02_worked_bound_example.py` 的手算例子钉住。

## 6. 空单元 / Empty cells

`L(q) = min_j max(0, d(q,c_j) − r_j)` 中的 `j` **只能遍历非空单元**。

反例说明为什么：若单元 `C_j` 为空而给 `r_j = 0`，该项变成 `d(q,c_j)`。
但 `c_j ∈ M`，所以 `d(q,c_j) ≥ s_M(q)`，这是一个**上界**而非下界。
把它放进 min 会让 `L` 有可能超过 `s_M`，证明失效。

`build_coverage_index` 因此**拒绝**构造含空单元的索引并抛出异常，
而不是悄悄使用半径 0。当中心取自银行本身时单元必然非空
（每个中心到自己是 0 距离），所以这只是一个防御性检查。

## 7. 中心必须属于银行 / Centers must be elements of the bank

`U(q) = min_j d(q,c_j)` 之所以是上界，唯一原因是 `{c_j} ⊆ M`。
若某个 `c_j ∉ M`，`U` 可能小于 `s_M`，判 NORMAL 就会漏报。
`build_coverage_index` 对每个中心做逐字节匹配检查并抛异常。

## 8. 不得静默降级 / No silent downgrade

`screen_image` 的第 4 分支：解析器不可用、版本不匹配、超时，
一律返回 **`UNRESOLVED`**，绝不返回 NORMAL。
Part A8 原文："never silently convert unresolved to normal"。
三个 UNRESOLVED 子类型都在 `ScreenOutcome` 中分别记录，便于失败分析。

## 9. 已知的未解决问题 / Known open numerical issues

1. `NumericalGuard` 的 8 ulp 没有推导，只有经验依据。是否有更紧且仍安全的界，
   未研究。
2. MPS 后端上的 float16 行为完全未测。
3. `np.einsum` 在 MPS/CPU 上的归约顺序是否固定，未审计；这影响跨机器逐位可复现性。
   当前所有测试都在同一台机器上通过。
4. 若将来 GPU FAISS 或树结构进入对照，必须证明"精确性"而非假定，见 §2 排除项。
