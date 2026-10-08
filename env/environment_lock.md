# 环境锁定 / Environment Lock

**记录日期 / Recorded:** 2026-10-08
**对应 / Corresponds to:** Research Plan Part A12 Phase 0
**目标 / Purpose:** 让另一次运行能解析出**同一环境**、同一划分与有限分数。

---

## 1. 实测版本 / Measured resolved versions

完整 `pip freeze` 见同目录 `environment_lock_freeze.txt`。关键项：

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.14 | venv，非系统 Python |
| torch | 2.14.1 | macOS arm64 wheel，MPS 后端 |
| torchvision | 0.29.1 | |
| numpy | 2.5.3 | |
| scipy | 1.18.1 | |
| scikit-learn | 1.9.1 | |
| pandas | 3.0.6 | |
| matplotlib | 3.11.2 | |
| Pillow | 12.3.0 | |
| pytest | 9.1.1 | 仅测试用 |
| faiss-cpu | **未安装 NOT INSTALLED** | 见 §4 |
| tqdm | 4.70.1 | |

平台：`macOS-26.6.2-arm64-arm-64bit`，设备解析为 `mps`。

---

## 2. 安装阻塞与解决 / Installation blocker and its resolution

**这是本阶段最重要的环境发现，教授要求报告 "installation blockers"。**

### 症状 Symptom

用 DSH 运行时自带的 Python 创建 venv 并 `pip install torch` 之后，`import torch`
立即失败：

```
OSError: dlopen(.../torch/lib/libtorch_global_deps.dylib, 0x000A):
  code signature in '...libtorch_global_deps.dylib' not valid for use in process:
  mapping process and mapped file (non-platform) have different Team IDs
```

### 根因 Root cause

macOS 的 **library validation**。该运行时 Python 的签名是：

```
Authority=Developer ID Application: Hangzhou DeepSeek Artificial Intelligence Co., Ltd (NAN929V4UM)
TeamIdentifier=NAN929V4UM
CodeDirectory flags=0x10000(runtime)      <-- hardened runtime 已开启
```

PyTorch 官方 wheel 中的 dylib 是 **ad-hoc 签名**（无 Team ID）。开启 hardened
runtime 的进程只允许加载同一 Team ID 的库，因此加载被内核拒绝。
这不是 PyTorch 的问题，也不是依赖冲突，纯粹是签名策略问题。

### 解决 Resolution（可复现）

在**同一目录**内复制运行时 Python 并 ad-hoc 重新签名。必须在同一目录，因为该
二进制使用 `LC_RPATH = @executable_path/../lib`：

```bash
BIN=/Users/zhangege/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin
cp "$BIN/python3.12" "$BIN/python3.12-mps"
codesign --force --sign - "$BIN/python3.12-mps"     # adhoc, 无 hardened runtime
"$BIN/python3.12-mps" -m venv .venv
```

重新签名后 `flags=0x2(adhoc)`, `TeamIdentifier=not set`，库校验关闭，torch 正常导入。

### 代价与风险 Cost and risk

- 这是**绕过**签名策略，不是修复它。属于本机工作环境适配，不进入研究结论。
- 换机器、升级 DSH 运行时、或改用 python.org / Homebrew 的 Python 时，此步骤
  可能不再需要。`env/setup_macos_arm64.sh` 会自动检测并只在需要时执行。
- 如果将来要报告 "在标准环境下可复现"，应在没有 hardened runtime 的 Python
  上重跑一次 `scripts/00_env_check.py` 作为交叉验证。**当前尚未做此交叉验证。**

---

## 3. 精确复现步骤 / Exact reproduction

见 `env/setup_macos_arm64.sh`。等价手工步骤：

```bash
cd master-research
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install numpy scipy scikit-learn Pillow matplotlib pandas pyyaml tqdm
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
.venv/bin/python scripts/00_env_check.py
```

---

## 4. 明确未锁定项 / Explicitly not locked

| 项 | 原因 |
|---|---|
| `faiss-cpu` | 官方 PatchCore 与 AnomalyDINO 都依赖它，但本阶段全部使用精确分块
距离，不需要近似检索。macOS arm64 的 wheel 可用性**未验证**，而且把它装进来会
让"精确 vs 近似"的对照变得含糊。列在 `requirements/base.txt` 中但注释掉。 |
| `pretrainedmodels` | PatchCore `requirements.txt` 里用于 WideResNet 之外的骨干；
本项目复现只用 torchvision 的 `wide_resnet50_2`，未安装。 |
| DINOv2 权重 | AnomalyDINO 复现属于 Phase 2，本阶段未下载。 |
| EfficientAD / anomalib | 计划 A10 的对照方法，属于 Phase 7。 |
| GPU FAISS | 本机无 CUDA，不适用。 |

**依赖冲突预警 / Dependency conflict warning.** AnomalyDINO 的
`requirements.txt` 锁定 `numpy==1.26` 并声明 "numpy 2.0 not compatible at the
moment"。本项目的 venv 是 numpy 2.5.3。Phase 2 复现 AnomalyDINO 时**必须**
为它单独建一个 venv，不能与本项目共用。这是一个已知的、计划内的工作量，
现在记下来以免届时误判为 bug。

---

## 5. 有限性与确定性检查 / Finiteness and determinism checks

Phase 0 脚本 `scripts/00_env_check.py` 每次运行都验证：

| 检查 | 结果 |
|---|---|
| 特征全部有限 (no NaN/Inf) | PASS |
| 同一输入两次前向逐位相同 | PASS |
| 分块距离 == 非分块距离（逐位） | PASS |
| float32 距离 == float64 暴力距离 | PASS (max abs diff 0.0) |
| 区间违规数 `L > s_M` | PASS (0) |
| 区间违规数 `s_M > U` | PASS (0) |

最后一项曾**失败 31/676**，原因与修复记录在
`src/master_research/bounds.py::_dist_to_centers` 的 docstring 与
`docs/protocol/numerics_contract.md`。这是本阶段除签名问题外第二个真实故障。
