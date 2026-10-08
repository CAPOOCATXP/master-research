# 硬件与数据清单 / Hardware and Data Inventory

**记录日期 / Recorded:** 2026-10-08 (Asia/Shanghai)
**记录者 / Recorded by:** Wu Yuantai (CAPOOCATXP)
**依据 / Authority:** Research Plan Part A12 Phase 0 and Part A14 Immediate Assignment #1

> 教授要求："Record hardware/RAM/disk ... Leave unavailable resources explicitly
> marked unknown." 本文件严格区分 **实测 (measured)**、**厂商声明 (vendor-stated)**
> 与 **未知 (UNKNOWN)**。任何未实测项都不得在论文中作为部署证据。

---

## 1. 主机 / Host machine

| 项目 Item | 值 Value | 状态 Status |
|---|---|---|
| 机型 Model | Apple M4, 内置 (Mac, `arm64`) | measured |
| 操作系统 OS | macOS 26.6.2, build 25G83 | measured |
| CPU | Apple M4, 10 逻辑核 (`hw.ncpu = 10`) | measured |
| 统一内存 Unified RAM | 16 GB (`hw.memsize`) | measured |
| 磁盘可用 Free disk | 145.2 GB on `/` (次测量于 Phase 0 运行时) | measured |
| 独显 Discrete GPU | **无 / none** | measured |
| 集成 GPU Integrated GPU | Apple M4 8-core GPU, Metal 4 | measured |
| CUDA | **不可用 / unavailable** | measured (`torch.cuda.is_available() == False`) |
| MPS | **可用 / available** | measured (`torch.backends.mps.is_available() == True`) |
| Python | 3.12.14 (venv) | measured |
| torch | 2.14.1 (macOS arm64 wheel) | measured |
| torchvision | 0.29.1 | measured |

**说明 / Note.** 这是 Apple Silicon 统一内存架构：16 GB 由 CPU 与 GPU 共享，不存在
"8–12 GB VRAM" 这一独立显存池。计划 A6 假设的是"一块 8–12 GB VRAM 的实验室 GPU"。
本机与该假设**不一致**，必须在任何资源结论中显式声明：

- 本机**不能**用于论证 edge-device 延迟 (A6 明确排除 CPU-only 试点的延迟外推)。
- 本机可以用于：正确性、决策保持、区间宽度、银行字节数、以及
  host-side 相对耗时比较。
- 任何 latency/throughput 数字都必须标注 "Apple M4 MPS, 16 GB unified memory"，
  且不得与论文中 A6000/V100 的数字直接并列。

---

## 2. 计算能力分级 / Compute tier

| 能力 Capability | 本机 This machine | 计划假设 Plan assumption | 差距 Gap |
|---|---|---|---|
| 加速器 Accelerator | Apple M4 GPU via MPS | NVIDIA 8–12 GB VRAM | 架构不同，不可外推 |
| FP32 训练 Training | 未测 UNKNOWN | 假设可用 | — |
| FP16/BF16 推断 Inference | 未测 UNKNOWN | 计划 A10 要求 float16 复现 EfficientAD | 待测 |
| 边缘设备 Edge device | **无 / none** | "Acquire a representative available device" | **阻塞项 BLOCKER** |

**阻塞 / Blocker.** A6 与 A11 明确：没有真实设备时，结论必须限制在 workstation
资源测量与成本模拟，并且不得声称部署性能。当前**没有任何物理边缘设备**，
因此本项目在拿到设备之前只做模拟与工作站测量。

---

## 3. 电力与其它 / Power and others

| 项目 Item | 值 Value | 状态 Status |
|---|---|---|
| 功耗计 Power meter | 无 | **UNKNOWN / 缺失** |
| 网络链路测量 Network link measurement | 未做 | **UNKNOWN** |
| 网络延迟模拟 Network delay replay | 未做 | 计划 A10 要求 0/20/100 ms + outage |

计划 A10 写明 "Report energy only if a real meter or defensible device measurement
is available"。当前无功耗计，因此 **本项目的任何阶段都不得报告能耗数字**。

---

## 4. 数据 / Datasets

| 数据集 Dataset | 是否需要注册 Registration | 下载是否成功 Downloaded | 许可 License | 状态 Status |
|---|---|---|---|---|
| MVTec AD (R1) | 是 (表单) | **否 / NO** | 非商业 Non-commercial | **阻塞 BLOCKER** |
| MVTec AD 2 (R4) | 是 (表单) | **否 / NO** | CC BY-NC-SA 4.0, 仅非商业 | **阻塞 BLOCKER** |
| VisA (经 R2 官方链接) | 未确认 | **否 / NO** | 未确认 UNKNOWN | 未知 |

**MVTec AD 阻塞的后果 / Consequence.** 计划 A7 要求先做 MVTec AD 的
bottle/tile/screw 试点。当前数据未下载，因此：
- 教授已在 A14 #2 中预设了此情况，并给出了替代方案：
  "If data access is pending, run tensor-distance checks on ten artificial 2-D vectors."
- 本仓库已按该替代方案执行：`scripts/01_vector_distance_check.py`（十个 2-D 向量）
  与 `scripts/00_env_check.py`（真实 WideResNet-50 前向 + 精确距离检查 + 计时）。
- 角色清单 (`docs/manifests/`) 目前是 **PROVISIONAL_NO_DATA**：使用数据集公开
  的训练集图像数量，文件名是占位符。下载后必须重跑
  `scripts/03_build_manifests.py` 覆盖。

**这不是可以含糊过去的事 / This must not be blurred.** 所有由占位清单产生的数字都
不能作为复现结果。清单文件中每一行都带有 `provenance=PROVISIONAL_NO_DATA`。

---

## 5. 软件许可 / Software licences

| 组件 Component | 许可 License | 备注 |
|---|---|---|
| patchcore-inspection (R1) | Apache-2.0 | 见 `third_party/patchcore-inspection/LICENSE` |
| AnomalyDINO (R2) | 见仓库 LICENSE | 未逐条审阅 UNKNOWN |
| 本项目代码 | 学术用途 | 见 `pyproject.toml` |
| MVTec AD / AD 2 | 非商业 | **不得再分发**；`data/raw/` 已在 `.gitignore` |

---

## 6. 未知项汇总 / Explicit UNKNOWN register

按教授要求，以下项目保持 "unknown" 而**不是**填默认值：

1. 独立显存容量 — 本机无独立显存，等价项为共享 16 GB。
2. 边缘设备型号与规格 — 完全未知。
3. MPS 上的 float16 推断行为与精度 — 未测。
4. MPS 上的长时间稳定性 / 热降频 — 未测。
5. VisA 可获取性与许可 — 未确认。
6. 网络链路特性（带宽、RTT、丢包）— 未测。
7. 功耗 — 无测量手段。
8. 训练侧吞吐（本阶段未训练任何模型）。
