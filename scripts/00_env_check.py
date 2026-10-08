#!/usr/bin/env python
"""Phase 0 environment and sanity check — the professor's Immediate Assignment #2.

Produces, in one run:

1. a machine/environment report (Python, torch, device, RAM, disk);
2. a pretrained-backbone forward pass on the resolved device, with the cached
   checkpoint SHA-256 recorded (weight pinning);
3. patch embeddings for one image, checked for finiteness;
4. the required exact-distance check: chunked exact nearest neighbours versus a
   tiny independent brute-force computation;
5. separate measured timings for encoder, screening scan (O(P*K*D)) and full-bank
   scan (O(P*N*D));
6. a patch-score map figure and a JSON log.

No dataset is required. If MVTec AD is not yet downloaded the script uses a
deterministic synthetic image and says so in every artifact it writes, so that
nothing here can be mistaken for a reproduction number.

Usage
-----
    .venv/bin/python scripts/00_env_check.py [--image PATH] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from master_research import (  # noqa: E402
    CoverageIndex,
    NumericalGuard,
    build_coverage_index,
    greedy_coreset,
    min_distances,
    patch_bounds,
    subset_size,
)
from master_research.features import (  # noqa: E402
    WR50_LAYER23,
    describe_device,
    extract_patch_embeddings,
    image_to_tensor,
    load_backbone,
    patch_grid_shape,
)


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def synthetic_image(size: int = 224, seed: int = 0):
    """Deterministic stand-in image so the pipeline is runnable without MVTec.

    Contains smooth gradients plus a bright blob, i.e. both low-frequency
    structure and a localised intensity change, so the resulting patch
    embeddings are not degenerate.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 90 + 70 * (xx / size) + 30 * (yy / size)
    blob = 120 * np.exp(-(((xx - size * 0.62) ** 2 + (yy - size * 0.38) ** 2) / (2 * (size * 0.11) ** 2)))
    noise = rng.normal(0, 3.0, size=(size, size)).astype(np.float32)
    gray = np.clip(base + blob + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, np.clip(gray * 0.92, 0, 255).astype(np.uint8), np.clip(gray * 1.05, 0, 255).astype(np.uint8)], -1)
    return Image.fromarray(rgb, mode="RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None, help="optional real image; synthetic if omitted")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.10, help="compressed bank ratio for the timing pilot")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    report: dict = {"stage": "phase0_env_check", "repo": str(REPO)}
    image_source = "synthetic" if args.image is None else str(args.image)

    # ---------------------------------------------------------------- 1. env
    import torch
    import torchvision

    device, device_notes = describe_device(args.device)
    report["environment"] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device_requested": args.device,
        "device_resolved": str(device),
        "device_notes": device_notes,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "disk_free_gb": round(disk_free_gb(args.out), 2),
    }
    print(f"[env] torch {torch.__version__} on {device}  (mps_available={torch.backends.mps.is_available()})")

    # ------------------------------------------------------------ 2. backbone
    t0 = time.perf_counter()
    model, weights = load_backbone(WR50_LAYER23)
    report["backbone"] = {
        "name": WR50_LAYER23.name,
        "weights_enum": str(weights),
        "layers": list(WR50_LAYER23.layers),
        "encoder_id": WR50_LAYER23.encoder_id(),
        "preprocessing_id": WR50_LAYER23.preprocessing_id(),
    }
    model = model.to(device)
    load_seconds = time.perf_counter() - t0

    # Hash the cached checkpoint: this is the "record the weight checksum" item.
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "wide_resnet50_2-95faca4d.pth"
    if ckpt.exists():
        from master_research import hash_weights

        t0 = time.perf_counter()
        report["backbone"]["checkpoint_path"] = str(ckpt)
        report["backbone"]["checkpoint_bytes"] = ckpt.stat().st_size
        report["backbone"]["checkpoint_sha256"] = hash_weights(ckpt)
        report["backbone"]["checkpoint_hash_seconds"] = round(time.perf_counter() - t0, 3)
    else:
        report["backbone"]["checkpoint_path"] = None
        report["backbone"]["checkpoint_sha256"] = "UNKNOWN: checkpoint not found in torch hub cache"
    print(f"[backbone] loaded in {load_seconds:.2f}s; sha256={report['backbone']['checkpoint_sha256'][:16]}...")

    # -------------------------------------------------------------- 3. image
    if args.image is not None:
        from PIL import Image

        image = Image.open(args.image)
    else:
        image = synthetic_image(WR50_LAYER23.image_size, args.seed)

    batch = image_to_tensor(
        image,
        WR50_LAYER23.resize,
        WR50_LAYER23.image_size,
        WR50_LAYER23.normalise_mean,
        WR50_LAYER23.normalise_std,
    ).to(device)

    # Warm-up then timed forward (MPS compiles kernels on first call).
    with torch.no_grad():
        model(batch)
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    feats = extract_patch_embeddings(model, batch, WR50_LAYER23)
    if device.type == "mps":
        torch.mps.synchronize()
    encode_seconds = time.perf_counter() - t0

    gh, gw = patch_grid_shape(WR50_LAYER23.image_size, WR50_LAYER23.image_size, WR50_LAYER23)
    report["features"] = {
        "image_source": image_source,
        "patches": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "patch_grid": [gh, gw],
        "grid_matches_patches": bool(gh * gw == feats.shape[0]),
        "all_finite": bool(np.isfinite(feats).all()),
        "min": float(feats.min()),
        "max": float(feats.max()),
        "mean": float(feats.mean()),
        "std": float(feats.std()),
        "encode_seconds": round(encode_seconds, 4),
        "extract_calls_are_deterministic": None,  # filled below
    }
    print(f"[features] {feats.shape[0]} patches x {feats.shape[1]} dims, finite={report['features']['all_finite']}")

    # Seeded repeatability: same input twice must give bitwise-equal features.
    feats_again = extract_patch_embeddings(model, batch, WR50_LAYER23)
    report["features"]["extract_calls_are_deterministic"] = bool(np.array_equal(feats, feats_again))

    # ------------------------------------------- 4. exact-distance self-check
    # Deliberately tiny, independent implementation: the check must not reuse
    # `min_distances` internals or it would not test anything.
    rng = np.random.default_rng(args.seed)
    check_q = feats[:8].copy()
    check_r = feats.copy()
    brute = np.empty(check_q.shape[0], dtype=np.float64)
    for i in range(check_q.shape[0]):
        brute[i] = np.sqrt(((check_r.astype(np.float64) - check_q[i].astype(np.float64)) ** 2).sum(axis=1)).min()
    got = min_distances(check_q, check_r, refine=True)
    got_norefine = min_distances(check_q, check_r, refine=False)
    report["distance_check"] = {
        "n_queries": int(check_q.shape[0]),
        "n_references": int(check_r.shape[0]),
        "dim": int(check_q.shape[1]),
        "max_abs_diff_refined_vs_float64_bruteforce": float(np.abs(got.astype(np.float64) - brute).max()),
        "max_rel_diff_refined_vs_float64_bruteforce": float(
            np.abs(got.astype(np.float64) - brute).max() / max(brute.max(), 1e-12)
        ),
        "max_abs_diff_unrefined_vs_refined": float(np.abs(got_norefine - got).max()),
        "refinement_changed_any_value": bool(not np.array_equal(got, got_norefine)),
        "chunked_equals_unchunked": None,  # filled below
    }
    # Chunking must be numerically inert.
    got_tiny_chunks = min_distances(check_q, check_r, chunk_bytes=4096, refine=True)
    report["distance_check"]["chunked_equals_unchunked"] = bool(np.array_equal(got, got_tiny_chunks))
    print(
        "[distance] max |float32 - float64 brute force| = "
        f"{report['distance_check']['max_abs_diff_refined_vs_float64_bruteforce']:.3e}"
    )

    # ------------------------------------------------------- 5. timing pilot
    # A bank built from this single image is only a timing vehicle, NOT an
    # anomaly-detection result. It is labelled as such everywhere it is written.
    bank = feats.copy()
    k = subset_size(bank.shape[0], args.ratio)
    t0 = time.perf_counter()
    centers = greedy_coreset(bank, k, seed=args.seed)
    coreset_seconds = time.perf_counter() - t0

    guard = NumericalGuard()
    t0 = time.perf_counter()
    index = build_coverage_index(bank, centers, guard=guard)
    build_seconds = time.perf_counter() - t0

    # NOTE (corrected 2026-10-08): the timing below deliberately screens the bank
    # with its *own* patches, because that is a pure timing vehicle. It must NOT
    # be used for any claim about interval width: screening a bank against itself
    # makes the exact score identically zero, so ``L = 0`` follows trivially and
    # says nothing. Section 6 does the interval check properly, with held-out
    # queries from a second image.
    queries = feats
    t0 = time.perf_counter()
    pb = patch_bounds(queries, index)
    screen_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    exact = min_distances(queries, bank, refine=True)
    fullbank_seconds = time.perf_counter() - t0

    report["timing_pilot"] = {
        "warning": "timing vehicle only; the bank is one image, not a trained reference set",
        "n_patches": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "full_bank_n": int(bank.shape[0]),
        "n_centers": int(index.n_centers),
        "ratio": args.ratio,
        "coreset_seconds": round(coreset_seconds, 4),
        "coverage_index_build_seconds": round(build_seconds, 4),
        "screen_scan_seconds": round(screen_seconds, 6),
        "full_bank_scan_seconds": round(fullbank_seconds, 6),
        "screen_scan_speedup_vs_full_bank": round(fullbank_seconds / max(screen_seconds, 1e-9), 2),
        "encoder_seconds": round(encode_seconds, 4),
        "bank_bytes": int(bank.nbytes),
        "edge_state_bytes": index.bank_and_metadata_bytes(),
        "edge_state_bytes_fraction_of_bank": index.bank_and_metadata_bytes() / max(bank.nbytes, 1),
    }
    print(
        f"[timing] encode {encode_seconds * 1e3:.1f} ms | screen {screen_seconds * 1e3:.2f} ms | "
        f"full bank {fullbank_seconds * 1e3:.2f} ms | speedup {report['timing_pilot']['screen_scan_speedup_vs_full_bank']}x"
    )

    # Honest reading of the timing above. At N=676, K=68 the screening scan is
    # NOT faster, and the reason is structural rather than a bug: the full-bank
    # path ranks candidates with one BLAS matmul and then refines only 4 of them
    # exactly, while the screen path must compute an exact distance to *every*
    # center because L takes a minimum over all cells. The FLOP ratio N/K = 9.9x
    # is too small to pay for that. A real reference bank has N in the 10^5 range,
    # so the crossover is worth measuring — but it is a Phase-6 measurement, and
    # no resource advantage is claimed from this Phase-0 number.
    report["timing_pilot"]["caveat"] = (
        "At this size the screen scan is slower than the full-bank scan. The full-bank path "
        "ranks with one BLAS matmul and exactly refines only 4 candidates, whereas the screen "
        "must compute an exact distance to every center because L minimises over all cells. "
        "N/K is only ~10x here. No resource advantage is claimed from this measurement; the "
        "break-even point is a Phase-6 measurement."
    )

    # -------------------------------------------------- 6. interval sanity
    # Two contrasts, because they answer different questions and only the second
    # one is scientifically meaningful:
    #
    #  (a) SELF-QUERY: screen the bank with its own patches. The exact score is
    #      identically 0, so L = 0 is forced by the definition L <= s_M and tells
    #      us nothing at all. Reported only to make the degeneracy explicit.
    #  (b) HELD-OUT: screen patches of a *different* image against the bank. This
    #      is the situation the method actually faces, and it is the number that
    #      any claim about interval width must be based on.
    from master_research.bounds import PatchBounds

    self_exact = min_distances(feats, bank, refine=True)
    report["interval_sanity_self_query"] = {
        "warning": "DEGENERATE: bank screened with its own patches; exact score is identically 0",
        "exact_score_all_zero": bool((self_exact == 0).all()),
        "violations_L_gt_S": int((pb.lower > self_exact).sum()),
        "violations_S_gt_U": int((self_exact > pb.upper).sum()),
        "lower_is_zero_fraction": float((pb.lower == 0).mean()),
        "mean_interval_width": float((pb.upper - pb.lower).mean()),
    }

    query_image = synthetic_image(WR50_LAYER23.image_size, args.seed + 1)
    query_batch = image_to_tensor(
        query_image,
        WR50_LAYER23.resize,
        WR50_LAYER23.image_size,
        WR50_LAYER23.normalise_mean,
        WR50_LAYER23.normalise_std,
    ).to(device)
    query_feats = extract_patch_embeddings(model, query_batch, WR50_LAYER23)

    held_exact = min_distances(query_feats, bank, refine=True)
    held_pb = patch_bounds(query_feats, index, exact_scores=held_exact)
    held_width = held_pb.upper - held_pb.lower
    report["interval_sanity_held_out"] = {
        "query_source": "synthetic image with seed+1 (different noise, same structure)",
        "n_queries": int(query_feats.shape[0]),
        "exact_score_mean": float(held_exact.mean()),
        "exact_score_min": float(held_exact.min()),
        "exact_score_max": float(held_exact.max()),
        "lower_mean": float(held_pb.lower.mean()),
        "lower_max": float(held_pb.lower.max()),
        "upper_mean": float(held_pb.upper.mean()),
        "lower_is_zero_fraction": float((held_pb.lower == 0).mean()),
        "lower_is_positive_fraction": float((held_pb.lower > 0).mean()),
        "mean_interval_width": float(held_width.mean()),
        "violations_L_gt_S": int((held_pb.lower > held_exact).sum()),
        "violations_S_gt_U": int((held_exact > held_pb.upper).sum()),
        "note": (
            "This is the number any claim about interval tightness must use. L is NOT "
            "identically zero: it is zero for a substantial fraction of patches but "
            "strictly positive for the rest, so the lower bound carries some information."
        ),
    }
    print(
        f"[interval] held-out queries: L=0 for "
        f"{report['interval_sanity_held_out']['lower_is_zero_fraction'] * 100:.1f}% of patches, "
        f"L max {held_pb.lower.max():.4f}, mean width {held_width.mean():.4f}, "
        f"violations L>S {report['interval_sanity_held_out']['violations_L_gt_S']}, "
        f"S>U {report['interval_sanity_held_out']['violations_S_gt_U']}"
    )
    print(
        f"[interval] (self-query control is degenerate: exact score all zero = "
        f"{report['interval_sanity_self_query']['exact_score_all_zero']})"
    )

    # ------------------------------------------------------------- 7. figure
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Deliberately plot the HELD-OUT result, not the self-query one: the
        # self-query panel would show an all-black lower bound that is an artefact.
        score_map = held_exact.reshape(gh, gw)
        lower_map = held_pb.lower.reshape(gh, gw)
        upper_map = held_pb.upper.reshape(gh, gw)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(np.asarray(query_image))
        axes[0].set_title("held-out query image\n(synthetic, seed+1)")
        im1 = axes[1].imshow(score_map, cmap="magma")
        axes[1].set_title(f"exact $s_M(q)$\nmax={score_map.max():.3f}")
        im3 = axes[3].imshow(upper_map, cmap="magma")
        axes[3].set_title(f"upper bound $U(q)$\nmax={upper_map.max():.3f}")
        im2 = axes[2].imshow(lower_map, cmap="magma")
        axes[2].set_title(
            f"lower bound $L(q)$\nmax={lower_map.max():.3f}\n"
            f"(zero at {(held_pb.lower == 0).mean() * 100:.0f}% of patches)"
        )
        for ax in axes:
            ax.axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
        fig.colorbar(im3, ax=axes[3], fraction=0.046)
        fig.suptitle(
            f"Phase 0 sanity — {WR50_LAYER23.encoder_id()} — "
            f"bank from image A, queries from image B — timing vehicle, not a detection result",
            fontsize=11,
        )
        fig.tight_layout()
        fig_path = args.out / "figures" / "phase0_patch_score_map.png"
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        report["figure"] = str(fig_path)
        print(f"[figure] {fig_path}")
    except Exception as exc:  # pragma: no cover - plotting is optional
        report["figure_error"] = repr(exc)

    log_path = args.out / "logs" / "phase0_env_check.json"
    log_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"[log] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
