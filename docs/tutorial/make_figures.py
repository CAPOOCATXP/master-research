#!/usr/bin/env python
"""Generate the tutorial figures for docs/tutorial/how_it_works.md.

Every figure is produced by this script so the pictures cannot drift away from
the numbers. Figures 6 and 7 are computed from the *real* Phase-0 pipeline
(frozen WideResNet-50 features), not from a toy stand-in.

Usage:
    .venv/bin/python docs/tutorial/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

C_BANK = "#9aa7bd"
C_CENTER = "#d1495b"
C_QUERY = "#1b6ca8"
C_OK = "#2a9d8f"
C_WARN = "#e76f51"
C_ESC = "#8e7dbe"


def box(ax, x, y, w, h, text, fc, ec="black", fs=10, tc="black"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, color=tc)


def arrow(ax, x1, y1, x2, y2, color="#333333", style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
            linewidth=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )


# ---------------------------------------------------------------------------
# Figure 1 — the whole pipeline
# ---------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ax.text(6.75, 6.25, "整个系统做了什么：从一张图到一个报警决定",
            ha="center", fontsize=15, fontweight="bold")

    # Row 1: offline (build once)
    ax.text(0.25, 5.35, "① 离线：建库（只做一次）", fontsize=11.5, fontweight="bold", color="#1b4965")
    box(ax, 0.25, 4.25, 1.7, 0.85, "正常图片\n（只有好的）", "#e8f1f8", fs=9.5)
    box(ax, 2.25, 4.25, 1.9, 0.85, "冻结的 CNN\n编码器\n（不训练）", "#dbe9f4", fs=9.5)
    box(ax, 4.45, 4.25, 2.0, 0.85, "每张图切成\n676 个 patch\n每个 1536 维", "#cfe3f0", fs=9.5)
    box(ax, 6.75, 4.25, 2.0, 0.85, "完整记忆库 M\nN 个向量\n(存全部 patch)", "#bcd7ea", fs=9.5)

    arrow(ax, 1.95, 4.67, 2.25, 4.67)
    arrow(ax, 4.15, 4.67, 4.45, 4.67)
    arrow(ax, 6.45, 4.67, 6.75, 4.67)

    # Compression step
    box(ax, 9.05, 4.25, 2.1, 0.85, "压缩：贪心选 K 个\n中心 + 算覆盖半径",
        "#f6d9dd", fs=9.5, ec="#d1495b")
    arrow(ax, 8.75, 4.67, 9.05, 4.67, color="#d1495b")
    box(ax, 11.45, 4.25, 1.8, 0.85, "边缘只存\nK 个中心 + K 个半径",
        "#f6d9dd", fs=9.5, ec="#d1495b")
    arrow(ax, 11.15, 4.67, 11.45, 4.67, color="#d1495b")

    ax.text(6.75, 3.85, "存储：N × 1536 个浮点  →  K × 1536 + K 个浮点    （K 远小于 N）",
            ha="center", fontsize=10.5, style="italic", color="#444444")

    # Row 2: online (per image)
    ax.text(0.25, 3.35, "② 在线：每来一张新图（要快、要省）", fontsize=11.5, fontweight="bold", color="#1b4965")
    box(ax, 0.25, 2.25, 1.7, 0.85, "新图片\n（可能有缺陷）", "#e8f8f5", fs=9.5)
    box(ax, 2.25, 2.25, 1.9, 0.85, "同一个\n冻结编码器", "#d9f0eb", fs=9.5)
    box(ax, 4.45, 2.25, 2.0, 0.85, "676 个\n查询 patch", "#c9ebe3", fs=9.5)
    box(ax, 6.75, 2.25, 2.0, 0.85, "只跟 K 个中心\n算距离\n→ 区间 [L, U]",
        "#c9ebe3", fs=9.5)
    arrow(ax, 1.95, 2.67, 2.25, 2.67, color="#2a9d8f")
    arrow(ax, 4.15, 2.67, 4.45, 2.67, color="#2a9d8f")
    arrow(ax, 6.45, 2.67, 6.75, 2.67, color="#2a9d8f")

    box(ax, 9.35, 2.25, 1.5, 0.85, "判决\n(4 个分支)", "#fdf0d5", fs=9.5, ec="#c98b00")
    arrow(ax, 8.75, 2.67, 9.35, 2.67, color="#c98b00")

    box(ax, 11.15, 3.05, 2.1, 0.6, "正常 / 报警\n（本地判定，便宜）", "#f6e7c1", fs=9)
    box(ax, 11.15, 2.25, 2.1, 0.6, "升级：查完整库\n（贵，但少）", "#f6e7c1", fs=9)
    arrow(ax, 10.85, 2.78, 11.15, 3.35, color="#c98b00")
    arrow(ax, 10.85, 2.55, 11.15, 2.55, color="#c98b00")

    # Row 3: the key contract
    box(ax, 0.25, 0.55, 13.0, 1.15,
        "整件事的核心保证：   L(q) ≤ s_M(q) ≤ U(q)\n"
        "s_M = 跟完整库的真实最近距离（参考检测器用的量）    L = 下界（保证不会高估）    U = 上界（保证不会低估）\n"
        "所以：如果 U 都已经 ≤ 阈值，那真实分数一定也 ≤ 阈值 → 可以放心判「正常」，不用查完整库",
        "#fff4e6", fs=10.5, ec="#c98b00")

    fig.savefig(OUT / "fig1_pipeline.png")
    plt.close(fig)
    print("fig1_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2 — what compression actually is
# ---------------------------------------------------------------------------
def fig2_compression():
    rng = np.random.default_rng(3)
    # a curved, anisotropic "manifold" of normal patches, like real features
    t = rng.uniform(0, 1, 400)
    bank = np.stack([t * 10 + rng.normal(0, 0.18, 400),
                     3.2 * np.sin(t * 3.4) + rng.normal(0, 0.22, 400)], axis=1)

    k = 12
    idx = np.sort(rng.choice(len(bank), k, replace=False))
    centers = bank[idx]
    d = np.linalg.norm(bank[:, None, :] - centers[None, :, :], axis=2)
    assign = d.argmin(axis=1)
    radii = np.array([np.linalg.norm(bank[assign == j] - centers[j], axis=1).max() for j in range(k)])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    ax.scatter(bank[:, 0], bank[:, 1], s=22, c=C_BANK, label=f"完整记忆库 M（N={len(bank)} 个 patch）")
    ax.set_title("压缩前：存下所有正常 patch", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("特征维度 1")
    ax.set_ylabel("特征维度 2")
    ax.text(0.5, -0.26, "存储 = N × D 个浮点数\nN 越大越占地方，查一次也越慢",
            transform=ax.transAxes, fontsize=10.5, ha="center", va="top", color="#444444")

    ax = axes[1]
    ax.scatter(bank[:, 0], bank[:, 1], s=12, c=C_BANK, alpha=0.5)
    for j in range(k):
        ax.add_patch(Circle(centers[j], radii[j], facecolor="#f6d9dd", edgecolor="#d1495b",
                            alpha=0.35, linewidth=1.1, zorder=1))
    ax.scatter(centers[:, 0], centers[:, 1], s=130, marker="*", c=C_CENTER,
               edgecolor="white", linewidth=0.8, zorder=3, label=f"观测中心 c_j（K={k} 个，都是真实 patch）")
    ax.set_title("压缩后：只存 K 个中心 + K 个覆盖半径", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("特征维度 1")
    ax.text(0.5, -0.26,
            "存储 = K×D + K 个浮点数\n每个粉圈是「这个单元里所有点都在这个半径内」的保证\n半径 = 单元内离中心最远那个点的距离",
            transform=ax.transAxes, fontsize=10.5, ha="center", va="top", color="#444444")

    fig.suptitle("压缩 = 用「少数代表点 + 覆盖范围」替代「全部点」", fontsize=15, fontweight="bold", y=1.04)
    fig.savefig(OUT / "fig2_compression.png")
    plt.close(fig)
    print("fig2_compression.png")


# ---------------------------------------------------------------------------
# Figure 3 — L <= s <= U and what the three decisions look like
# ---------------------------------------------------------------------------
def fig3_interval_decisions():
    t = 3.0
    rows = [
        ("图 A：明显正常", 0.0, 1.42, 1.41, "U ≤ t  →  本地判「正常」", C_OK),
        ("图 B：明显异常", 18.0, 20.0, 18.0, "L > t  →  本地判「报警」", C_WARN),
        ("图 C：拿不准", 1.5, 3.5, 3.5, "区间跨过阈值  →  必须升级查完整库", C_ESC),
        ("图 D：正好卡在阈值", 3.0, 5.0, 3.0, "L = t  →  也要升级（等号不算报警）", C_ESC),
    ]

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.axvline(t, color="black", linewidth=2.2, linestyle="--", zorder=1)
    ax.text(t, 4.62, "  阈值 t = 3\n  （分数超过它就报警）", fontsize=11, va="top", fontweight="bold")

    for i, (name, L, U, S, verdict, color) in enumerate(rows):
        y = 3.5 - i * 1.05
        ax.plot([L, U], [y, y], color=color, linewidth=9, solid_capstyle="round",
                alpha=0.42, zorder=2)
        ax.plot([L, L], [y - 0.2, y + 0.2], color=color, linewidth=2.4, zorder=4)
        ax.plot([U, U], [y - 0.2, y + 0.2], color=color, linewidth=2.4, zorder=4)
        ax.plot([S], [y], marker="o", markersize=11, color=color,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        ax.text(-0.9, y, name, fontsize=11.5, ha="right", va="center", fontweight="bold")
        ax.text(21.2, y, verdict, fontsize=11, va="center", color=color, fontweight="bold")

    ax.plot([], [], color="grey", linewidth=9, alpha=0.42, label="区间 [L, U]（我们知道的分数范围）")
    ax.plot([], [], marker="o", color="grey", markersize=9, linestyle="none",
            label="s：真实分数（只有查完整库才知道）")
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)

    ax.set_xlim(-4.5, 30)
    ax.set_ylim(0.2, 5.2)
    ax.set_yticks([])
    ax.set_xlabel("异常分数（离正常记忆库的最近距离，越大越异常）", fontsize=12)
    ax.set_title("区间落在阈值的哪一侧，决定了要不要花那次昂贵的查询",
                 fontsize=15, fontweight="bold", pad=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(OUT / "fig3_interval_decisions.png")
    plt.close(fig)
    print("fig3_interval_decisions.png")


# ---------------------------------------------------------------------------
# Figure 4 — the four-branch decision rule
# ---------------------------------------------------------------------------
def fig4_decision_flow():
    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.text(6.5, 7.9, "判决规则：只有 4 种结局，没有第 5 种",
            ha="center", fontsize=15.5, fontweight="bold")

    box(ax, 4.6, 6.9, 3.8, 0.62, "来了一张新图\n算出区间 [L_x, U_x]", "#eef2f7", fs=10.5)

    box(ax, 4.3, 5.75, 4.4, 0.62, "①  U_x ≤ t  ？", "#dff3ef", fs=11.5, ec=C_OK)
    arrow(ax, 6.5, 6.9, 6.5, 6.37)
    box(ax, 9.4, 5.75, 3.3, 0.62, "判「正常」\n本地判定 · 不查库", "#dff3ef", fs=10.5, ec=C_OK)
    arrow(ax, 8.7, 6.06, 9.4, 6.06, color=C_OK)
    ax.text(9.05, 6.2, "是", fontsize=10.5, color=C_OK, fontweight="bold")

    box(ax, 4.3, 4.6, 4.4, 0.62, "②  L_x > t  ？（严格大于）", "#fdeee9", fs=11.5, ec=C_WARN)
    arrow(ax, 6.5, 5.75, 6.5, 5.22, color="#888888")
    ax.text(6.25, 5.48, "否", fontsize=10.5, color="#666666", ha="right")
    box(ax, 9.4, 4.6, 3.3, 0.62, "判「报警」\n本地判定 · 不查库", "#fdeee9", fs=10.5, ec=C_WARN)
    arrow(ax, 8.7, 4.91, 9.4, 4.91, color=C_WARN)
    ax.text(9.05, 5.05, "是", fontsize=10.5, color=C_WARN, fontweight="bold")

    box(ax, 4.3, 3.35, 4.4, 0.72, "③  升级：查完整库 M\n（版本必须匹配）", "#f3effa", fs=11.5, ec=C_ESC)
    arrow(ax, 6.5, 4.6, 6.5, 4.07, color="#888888")
    ax.text(6.25, 4.33, "否", fontsize=10.5, color="#666666", ha="right")
    box(ax, 9.4, 3.35, 3.3, 0.72, "用完整库的判决\n（贵，但少）", "#f3effa", fs=10.5, ec=C_ESC)
    arrow(ax, 8.7, 3.71, 9.4, 3.71, color=C_ESC)
    ax.text(9.05, 3.85, "查到了", fontsize=10, color=C_ESC)

    box(ax, 4.3, 1.85, 4.4, 0.78,
        "④  查不到？\n解析器坏了 / 版本对不上 / 超时", "#ffe9e9", fs=11, ec="#c0392b")
    arrow(ax, 6.5, 3.35, 6.5, 2.63, color="#c0392b")
    ax.text(6.2, 2.99, "失败", fontsize=10, color="#c0392b", ha="right")

    box(ax, 1.0, 1.85, 2.9, 0.78, "返回「未解决」\nUNRESOLVED", "#ffe9e9", fs=11.5, ec="#c0392b", tc="#8b1a1a")
    arrow(ax, 4.3, 2.24, 3.9, 2.24, color="#c0392b")

    box(ax, 9.4, 1.7, 3.3, 1.05,
        "注意：绝对不返回「正常」\n\n把「不知道」当成「正常」\n＝漏检，是最危险的错误",
        "#fff0f0", fs=10, ec="#c0392b")

    fig.savefig(OUT / "fig4_decision_flow.png")
    plt.close(fig)
    print("fig4_decision_flow.png")


# ---------------------------------------------------------------------------
# Figure 5 — the hand-checked 6-vector example
# ---------------------------------------------------------------------------
def fig5_worked_example():
    M = np.array([[0, 0], [2, 0], [0, 2], [10, 0], [12, 0], [10, 2]], dtype=float)
    C = np.array([[0, 0], [10, 0]], dtype=float)
    r = 2.0
    t = 3.0
    queries = [(np.array([1, 1.0]), 0.0, np.sqrt(2), np.sqrt(2), "正常"),
               (np.array([11, 0.0]), 0.0, 1.0, 1.0, "正常"),
               (np.array([30, 0.0]), 18.0, 20.0, 18.0, "报警"),
               (np.array([5, 0.0]), 3.0, 5.0, 3.0, "升级→正常"),
               (np.array([6.5, 0.0]), 1.5, 3.5, 3.5, "升级→报警")]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    for j, c in enumerate(C):
        ax.add_patch(Circle(c, r, facecolor="#f6d9dd", edgecolor="#d1495b",
                            alpha=0.4, linewidth=1.4, zorder=1))
        ax.annotate(f"$c_{j+1}$", c, textcoords="offset points", xytext=(8, 8),
                    fontsize=13, color=C_CENTER, fontweight="bold")
    for i, m in enumerate(M):
        ax.scatter(m[0], m[1], s=95, c=C_CENTER, edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(f"$m_{i+1}$", m, textcoords="offset points", xytext=(6, -14), fontsize=11)
    for q, L, U, S, verdict in queries:
        if q[0] > 20:
            continue
        ax.scatter(q[0], q[1], s=130, marker="s", c=C_QUERY, edgecolor="white",
                   linewidth=1.2, zorder=4)
        ax.annotate(f"({q[0]:g},{q[1]:g})\n{verdict}", q, textcoords="offset points",
                    xytext=(8, 8), fontsize=9.5, color=C_QUERY)

    ax.set_xlim(-2.5, 14.5)
    ax.set_ylim(-3.2, 5.2)
    ax.set_aspect("equal")
    ax.set_xlabel("特征维度 1")
    ax.set_ylabel("特征维度 2")
    ax.set_title("几何：6 个参考向量、2 个中心、每个半径 = 2", fontsize=12.5)

    ax = axes[1]
    ax.axvline(t, color="black", linewidth=2.2, linestyle="--")
    ax.text(t, 4.55, " t = 3", fontsize=11, fontweight="bold", va="top")
    for i, (q, L, U, S, verdict) in enumerate(queries):
        y = 4.0 - i * 1.0
        col = C_OK if verdict == "正常" else (C_WARN if verdict == "报警" else C_ESC)
        ax.plot([L, U], [y, y], color=col, linewidth=9, alpha=0.42, solid_capstyle="round")
        ax.plot([S], [y], marker="o", markersize=11, color=col,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        ax.text(-1.2, y, f"({q[0]:g},{q[1]:g})", fontsize=11, ha="right", va="center")
        ax.text(25.5, y, verdict, fontsize=11, va="center", color=col, fontweight="bold")
    ax.set_xlim(-7, 32)
    ax.set_ylim(0.2, 5.0)
    ax.set_yticks([])
    ax.set_xlabel("分数")
    ax.set_title("同一个例子的区间与判决", fontsize=12.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("手算例子：(5,0) 恰好落在 L = t，必须升级 —— 写成 ≥ 就会误报",
                 fontsize=14.5, fontweight="bold", y=1.03)
    fig.savefig(OUT / "fig5_worked_example.png")
    plt.close(fig)
    print("fig5_worked_example.png")


# ---------------------------------------------------------------------------
# Figures 6 & 7 — real data: why L collapses to zero
# ---------------------------------------------------------------------------
def synthetic_image(size: int = 224, seed: int = 0):
    """Same deterministic stand-in image used by scripts/00_env_check.py.

    Re-defined here (rather than imported) because the script lives in a
    numbered module that is not importable by name.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 90 + 70 * (xx / size) + 30 * (yy / size)
    blob = 120 * np.exp(-(((xx - size * 0.62) ** 2 + (yy - size * 0.38) ** 2) / (2 * (size * 0.11) ** 2)))
    noise = rng.normal(0, 3.0, size=(size, size)).astype(np.float32)
    gray = np.clip(base + blob + noise, 0, 255).astype(np.uint8)
    rgb = np.stack(
        [gray,
         np.clip(gray * 0.92, 0, 255).astype(np.uint8),
         np.clip(gray * 1.05, 0, 255).astype(np.uint8)],
        -1,
    )
    return Image.fromarray(rgb, mode="RGB")


def compute_real_phase0():
    """Build a bank from image A and take held-out queries from image B.

    The held-out part is essential. Screening a bank with its *own* patches makes
    the exact score identically zero, so ``L = 0`` is forced by ``L <= s_M`` and
    measures nothing. An earlier version of scripts/00_env_check.py made exactly
    that mistake; the self-query numbers are returned here too, labelled, so the
    degeneracy is visible rather than hidden.
    """
    from master_research import build_coverage_index, greedy_coreset, min_distances, patch_bounds, subset_size
    from master_research.features import (
        WR50_LAYER23, describe_device, extract_patch_embeddings,
        image_to_tensor, load_backbone,
    )

    import torch

    device, _ = describe_device("auto")
    model, _ = load_backbone(WR50_LAYER23)
    model = model.to(device)

    def feats_of(img):
        batch = image_to_tensor(img, WR50_LAYER23.resize, WR50_LAYER23.image_size,
                                WR50_LAYER23.normalise_mean, WR50_LAYER23.normalise_std).to(device)
        with torch.no_grad():
            return extract_patch_embeddings(model, batch, WR50_LAYER23)

    img_a = synthetic_image(WR50_LAYER23.image_size, 0)
    img_b = synthetic_image(WR50_LAYER23.image_size, 1)
    bank = feats_of(img_a)
    queries = feats_of(img_b)

    k = subset_size(bank.shape[0], 0.10)
    centers = greedy_coreset(bank, k, seed=0)
    index = build_coverage_index(bank, centers)

    held_exact = min_distances(queries, bank)
    held_pb = patch_bounds(queries, index, exact_scores=held_exact)

    self_exact = min_distances(bank, bank)
    self_pb = patch_bounds(bank, index, exact_scores=self_exact)

    return {
        "bank": bank, "queries": queries, "centers": centers, "index": index,
        "held_exact": held_exact, "held_pb": held_pb,
        "self_exact": self_exact, "self_pb": self_pb,
        "img_a": img_a, "img_b": img_b,
    }


def fig6_why_zero(d):
    rng = np.random.default_rng(11)
    t = rng.uniform(0, 1, 300)
    bank = np.stack([t * 12, 2.6 * np.sin(t * 3.0) + rng.normal(0, 0.15, 300)], axis=1)
    centers = bank[np.array([20, 150, 280])]
    dist = np.linalg.norm(bank[:, None, :] - centers[None, :, :], axis=2)
    assign = dist.argmin(axis=1)
    radii = np.array([np.linalg.norm(bank[assign == j] - centers[j], axis=1).max() for j in range(3)])
    q = np.array([5.0, 2.6 * np.sin((5.0 / 12) * 3.0) + 0.55])

    idx = d["index"]
    mean_r = float(idx.radii.mean())
    mean_d = float(np.linalg.norm(
        d["queries"][:, None, :] - d["centers"][None, :, :], axis=2).min(axis=1).mean())
    zero_frac = float((d["held_pb"].lower == 0).mean())

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(bank[:, 0], bank[:, 1], s=16, c=C_BANK, alpha=0.75)
    for j in range(3):
        ax.add_patch(Circle(centers[j], radii[j], facecolor="#f6d9dd",
                            edgecolor="#d1495b", alpha=0.30, linewidth=1.3))
        ax.scatter(*centers[j], s=150, marker="*", c=C_CENTER,
                   edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(*q, s=150, marker="s", c=C_QUERY, edgecolor="white", linewidth=1.3, zorder=5)
    jn = int(np.argmin(np.linalg.norm(centers - q, axis=1)))
    ax.plot([q[0], centers[jn, 0]], [q[1], centers[jn, 1]], color=C_QUERY, linewidth=2)
    ax.annotate("查询 q", q, textcoords="offset points", xytext=(6, 10), fontsize=11,
                color=C_QUERY, fontweight="bold")
    ax.annotate(f"$d(q,c)$={np.linalg.norm(centers[jn]-q):.1f}\n$r$={radii[jn]:.1f}"
                "\n相减为负 → L=0",
                (q + centers[jn]) / 2, textcoords="offset points", xytext=(-10, -34),
                fontsize=9.5, color=C_QUERY)
    ax.set_aspect("equal")
    ax.set_xlabel("特征维度 1")
    ax.set_ylabel("特征维度 2")
    ax.set_title("几何原因：\n半径被单元里最远的点撑大", fontsize=12)

    ax = fig.add_subplot(gs[0, 1])
    labels = ["平均半径\n$r_j$", "平均「到最近中心\n的距离」 $d(q,c)$"]
    vals = [mean_r, mean_d]
    bars = ax.bar(labels, vals, color=[C_CENTER, C_QUERY], alpha=0.85, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("距离")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("真实数据：半径比查询距离还大\n所以相减经常为负", fontsize=12)
    ax.text(0.5, 0.72,
            f"$r$ > $d$  →  $L=0$\n\n实测：{zero_frac * 100:.0f}% 的 patch 落在这一侧",
            transform=ax.transAxes, ha="center", fontsize=11.5, color="#8b1a1a",
            fontweight="bold")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0.0, 1.0, "L(q) = 对所有单元 j 取最小  of  max(0,  d(q,c_j) − r_j )",
            fontsize=11.5, fontweight="bold", va="top")
    ax.text(0.0, 0.87,
            "翻译成人话：\n\n"
            "  对每个单元先问「这个圈里的点，离 q 至少多远？」\n"
            "  答案 =  q 到圈心的距离 − 圈的半径\n"
            "  如果 q 就在圈里，减出来是负的，截断成 0。\n\n"
            "只要有一个圈把 q 包住，L 就等于 0。",
            fontsize=10.5, va="top")
    ax.text(0.0, 0.48,
            "我在这里犯过一个错误，值得记下来：\n\n"
            "  第一版 Phase 0 脚本用「银行自己的 patch」当查询，\n"
            "  于是真实分数 s 恒等于 0，L 必然也全是 0。\n"
            "  那不是发现，是同义反复。\n\n"
            "  改成用另一张图的 patch 当查询后：\n"
            f"    L = 0 的比例从 100% 变成 {zero_frac * 100:.0f}%\n"
            f"    L 的最大值从 0 变成 {float(d['held_pb'].lower.max()):.2f}",
            fontsize=10.5, va="top", color="#8b1a1a")
    ax.text(0.0, 0.04,
            f"结论（修正后）：下界不是恒为 0，而是对 {(1 - zero_frac) * 100:.0f}% 的 patch 给出正信息。\n"
            "但它仍然使大多数 patch 无法在本地被判定为「报警」。",
            fontsize=10.5, va="top", fontweight="bold", color="#1b4965")

    fig.suptitle("下界为什么会塌成 0 —— 以及我第一次测错在哪", fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(OUT / "fig6_why_L_is_zero.png")
    plt.close(fig)
    print("fig6_why_L_is_zero.png")


def fig7_real_data(d):
    feats, index = d["bank"], d["index"]
    ex, pb = d["held_exact"], d["held_pb"]
    width = pb.upper - pb.lower

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

    ax = axes[0]
    ax.hist(pb.lower, bins=40, color=C_WARN, alpha=0.85)
    ax.set_title("下界 L 的分布（留出查询）", fontsize=12.5)
    ax.set_xlabel("L(q)")
    ax.set_ylabel("patch 数量")
    ax.text(0.62, 0.72,
            f"L = 0 的占 {(pb.lower == 0).mean() * 100:.0f}%\n"
            f"L > 0 的占 {(pb.lower > 0).mean() * 100:.0f}%\n"
            f"L 最大 {pb.lower.max():.2f}",
            transform=ax.transAxes, fontsize=11, color="#8b1a1a", fontweight="bold")

    ax = axes[1]
    o = np.argsort(ex)
    ax.plot(ex[o], pb.upper[o], ".", ms=4, color=C_QUERY, label="上界 U(q)")
    ax.plot(ex[o], pb.lower[o], ".", ms=4, color=C_WARN, label="下界 L(q)")
    lim = [0, max(ex.max(), pb.upper.max()) * 1.05]
    ax.plot(lim, lim, color="black", linewidth=1.2, linestyle="--", label="y = $s_M(q)$（真实分数）")
    ax.fill_between(ex[o], pb.lower[o], pb.upper[o], color=C_ESC, alpha=0.15)
    ax.set_xlabel("真实分数 $s_M(q)$")
    ax.set_ylabel("区间端点")
    ax.set_title("区间确实夹住了真实分数", fontsize=12.5)
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[2]
    ax.hist(width, bins=40, color=C_ESC, alpha=0.85)
    ax.axvline(width.mean(), color="black", linestyle="--", label=f"平均 {width.mean():.2f}")
    ax.set_xlabel("区间宽度  U − L")
    ax.set_ylabel("patch 数量")
    ax.set_title("区间宽度分布", fontsize=12.5)
    ax.legend(fontsize=9.5)
    ax.text(0.5, 0.62,
            f"平均宽度 {width.mean():.2f}\n"
            f"而真实分数的范围只有\n"
            f"{ex.min():.2f} ~ {ex.max():.2f}\n\n"
            "→ 区间比信号本身还宽",
            transform=ax.transAxes, fontsize=10.5, color="#444444", ha="center")

    fig.suptitle(
        f"真实的 Phase 0 数据（留出查询）：WideResNet-50 layer2+3，"
        f"{feats.shape[0]} patch × {feats.shape[1]} 维，K={index.n_centers} 中心",
        fontsize=13, fontweight="bold", y=1.06)
    fig.savefig(OUT / "fig7_real_data.png")
    plt.close(fig)
    print("fig7_real_data.png")

    summary = {
        "bank_patches": int(feats.shape[0]),
        "query_patches": int(d["queries"].shape[0]),
        "dim": int(feats.shape[1]),
        "n_centers": int(index.n_centers),
        "held_out": {
            "exact_mean": float(ex.mean()), "exact_min": float(ex.min()), "exact_max": float(ex.max()),
            "L_mean": float(pb.lower.mean()), "L_max": float(pb.lower.max()),
            "L_zero_fraction": float((pb.lower == 0).mean()),
            "U_mean": float(pb.upper.mean()),
            "mean_width": float(width.mean()),
            "violations_L_gt_S": int((pb.lower > ex).sum()),
            "violations_S_gt_U": int((ex > pb.upper).sum()),
        },
        "self_query_control": {
            "exact_all_zero": bool((d["self_exact"] == 0).all()),
            "L_zero_fraction": float((d["self_pb"].lower == 0).mean()),
            "note": "degenerate: L=0 is forced here, not discovered",
        },
        "geometry": {
            "mean_radius": float(index.radii.mean()),
            "mean_nearest_center_distance": float(np.linalg.norm(
                d["queries"][:, None, :] - d["centers"][None, :, :], axis=2).min(axis=1).mean()),
        },
    }
    (OUT / "real_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    fig1_pipeline()
    fig2_compression()
    fig3_interval_decisions()
    fig4_decision_flow()
    fig5_worked_example()

    d = compute_real_phase0()
    fig6_why_zero(d)
    s = fig7_real_data(d)
    print("\n" + json.dumps(s, indent=2, ensure_ascii=False))
    print("\nall figures written to", OUT)


if __name__ == "__main__":
    main()
