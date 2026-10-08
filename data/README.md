# data/

**这个目录里没有任何数据集图像，也不应该有。** 它只存放说明与（被 gitignore 的）
本地数据。

## 许可 / Licences

| 数据集 | 许可 | 可否再分发 |
|---|---|---|
| MVTec AD | 非商业 (non-commercial) | **否** |
| MVTec AD 2 | CC BY-NC-SA 4.0, 仅非商业 | **否** |
| VisA | 未确认 UNKNOWN | 未确认 |

两个 MVTec 数据集都要求填写表单并接受许可后才能下载。
**本仓库不包含、也不再分发任何数据集图像。**
`data/raw/`、`data/cache/`、`data/downloads/` 全部在 `.gitignore` 中。

## 期望的目录结构 / Expected layout

`scripts/03_build_manifests.py` 会按以下任一结构查找正常训练图：

```
data/raw/mvtec_ad/<category>/train/good/*.png
```

或

```
data/raw/mvtec_ad/mvtec/<category>/train/good/*.png
```

MVTec AD 2 的结构（供 Phase 3 使用，来自计划 A7 与官方提交布局）：

```
data/raw/mvtec_ad2/<object_name>/
    train/good/           # 仅正常，仅常规光照
    validation/good/      # 仅正常；唯一被许可定阈值的地方
    test_public/          # 正常 + 异常，ground truth 公开，覆盖全部光照
    test_private/         # 图像公开，ground truth 私有（服务器评分）
    test_private_mixed/   # 同上，混合已知/未知光照
```

**注意：** MVTec AD 2 的公开/私有之分**不是**按类别分的。
全部 8 个类别都出现在三个 test 部分中；私有的是 `test_private` 与
`test_private_mixed` 的 **ground truth**，不是类别。

## 下载后必须做的事 / After downloading

```bash
.venv/bin/python scripts/03_build_manifests.py --mvtec-root data/raw/mvtec_ad
```

这会把 `docs/manifests/` 下的占位清单（`PROVISIONAL_NO_DATA`）
替换为真实清单（`REAL`），并重新校验三个角色的互斥性与计数。

如果计数与公开的训练集计数不一致，脚本会**直接报错退出**，
而不是继续。这时应检查下载是否完整，而不是修改计数。

## 缓存 / Caching

提取到的特征缓存放 `data/cache/`，同样被 gitignore。
缓存必须带编码器 ID 与预处理 ID，否则不得复用
（见 `docs/protocol/numerics_contract.md` 与 `src/master_research/versioning.py`）。
