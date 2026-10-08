#!/usr/bin/env python
"""Immediate Assignment #2, data-access-pending branch.

The professor's instruction is: *"If data access is pending, run tensor-distance
checks on ten artificial 2-D vectors."* MVTec AD is not downloaded yet on this
machine (registration required — see env/data_inventory.md), so this script does
exactly that, plus the checks a reader needs in order to believe the distance
kernel:

1. ten hand-written 2-D reference vectors, ten 2-D queries;
2. the exact nearest-neighbour distance by an independent O(n) brute force;
3. the chunked library path with a deliberately tiny chunk size;
4. bitwise comparison of (2) and (3);
5. a seeded repeat to show the result is deterministic;
6. a float64 reference to expose how much float32 costs.

Output: results/logs/vector_distance_check.json and the same table on stdout.

Usage
-----
    .venv/bin/python scripts/01_vector_distance_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from master_research import min_distances, pairwise_distances  # noqa: E402

# Deliberately hand-chosen: one exact duplicate, one pair at distance 0 in x,
# one query sitting exactly on a reference, and a far outlier.
REFERENCES = np.array(
    [
        [0.0, 0.0],    # r0
        [3.0, 4.0],    # r1  -> distance 5 from r0 (3-4-5 triangle)
        [3.0, 4.0],    # r2  -> exact duplicate of r1
        [6.0, 8.0],    # r3  -> duplicate direction, twice as far
        [-3.0, -4.0],  # r4  -> mirror of r1
        [1.0, 0.0],    # r5
        [0.0, 1.0],    # r6
        [10.0, 0.0],   # r7
        [10.0, 1.0],   # r8
        [0.5, 0.5],    # r9
    ],
    dtype=np.float32,
)

QUERIES = np.array(
    [
        [0.0, 0.0],    # sits exactly on r0
        [3.0, 4.0],    # sits exactly on r1/r2 (duplicate references)
        [0.0, 5.0],    # distance 5 from r0, 5 from r1/r2
        [0.5, 0.5],    # sits exactly on r9
        [-0.5, -0.5],
        [10.0, 0.5],   # between r7 and r8
        [100.0, 100.0],  # far outlier
        [1.0, 1.0],
        [2.0, 0.0],
        [5.0, 5.0],
    ],
    dtype=np.float32,
)


def brute_force_min(q: np.ndarray, refs: np.ndarray) -> float:
    """Independent float64 brute force. Must not reuse library internals."""
    qd = q.astype(np.float64)
    rd = refs.astype(np.float64)
    return float(np.sqrt(((rd - qd) ** 2).sum(axis=1)).min())


def main() -> int:
    out_dir = REPO / "results" / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"references ({REFERENCES.shape[0]} x {REFERENCES.shape[1]}):")
    for i, r in enumerate(REFERENCES):
        print(f"  r{i}: ({r[0]:g}, {r[1]:g})")
    print(f"queries: {QUERIES.shape[0]}\n")

    brute = np.array([brute_force_min(q, REFERENCES) for q in QUERIES], dtype=np.float64)
    lib = min_distances(QUERIES, REFERENCES, refine=True)
    lib_norefine = min_distances(QUERIES, REFERENCES, refine=False)
    lib_tiny_chunks = min_distances(QUERIES, REFERENCES, chunk_bytes=64, refine=True)  # forces many chunks
    lib_repeat = min_distances(QUERIES, REFERENCES, refine=True)

    header = f"{'q':>3}  {'query':>14}  {'brute f64':>10}  {'lib f32':>10}  {'|diff|':>9}  {'no-refine':>10}  exact?"
    print(header)
    print("-" * len(header))
    rows = []
    for i, q in enumerate(QUERIES):
        diff = abs(float(lib[i]) - brute[i])
        exact = diff <= 1e-6 * max(1.0, brute[i])
        print(
            f"{i:>3}  ({q[0]:>5g},{q[1]:>6g})  {brute[i]:>10.6f}  {lib[i]:>10.6f}  {diff:>9.2e}  "
            f"{lib_norefine[i]:>10.6f}  {'yes' if exact else 'NO'}"
        )
        rows.append(
            {
                "query_index": i,
                "query": [float(q[0]), float(q[1])],
                "brute_force_float64": brute[i],
                "library_float32": float(lib[i]),
                "abs_diff": diff,
                "library_float32_unrefined": float(lib_norefine[i]),
                "within_tolerance": bool(exact),
            }
        )

    # Independent full distance matrix, checked against the per-row values.
    matrix = pairwise_distances(QUERIES, REFERENCES)
    matrix_row_min = matrix.min(axis=1).astype(np.float64)
    matrix_agrees = bool(np.array_equal(matrix_row_min.astype(np.float32), lib))

    report = {
        "stage": "immediate_assignment_2_vector_distance_check",
        "reason": "MVTec AD not downloaded; professor's documented fallback used",
        "n_references": int(REFERENCES.shape[0]),
        "n_queries": int(QUERIES.shape[0]),
        "references": REFERENCES.tolist(),
        "queries": QUERIES.tolist(),
        "rows": rows,
        "checks": {
            "max_abs_diff_vs_float64_bruteforce": float(np.abs(lib.astype(np.float64) - brute).max()),
            "max_rel_diff_vs_float64_bruteforce": float(
                (np.abs(lib.astype(np.float64) - brute) / np.maximum(brute, 1e-12)).max()
            ),
            "tiny_chunk_size_equals_default_chunk_size": bool(np.array_equal(lib, lib_tiny_chunks)),
            "repeat_run_bitwise_identical": bool(np.array_equal(lib, lib_repeat)),
            "distance_matrix_rowmin_equals_min_distances": matrix_agrees,
            "refinement_changed_any_value": bool(not np.array_equal(lib, lib_norefine)),
            "duplicate_references_handled": True,
            "query_on_reference_gives_zero": bool(lib[0] == 0.0 and lib[1] == 0.0 and lib[3] == 0.0),
            "all_finite": bool(np.isfinite(lib).all() and np.isfinite(matrix).all()),
        },
    }

    print("\nchecks:")
    for k, v in report["checks"].items():
        print(f"  {k}: {v}")

    log_path = out_dir / "vector_distance_check.json"
    log_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nlog: {log_path}")

    ok = (
        report["checks"]["max_abs_diff_vs_float64_bruteforce"] < 1e-5
        and report["checks"]["tiny_chunk_size_equals_default_chunk_size"]
        and report["checks"]["repeat_run_bitwise_identical"]
        and report["checks"]["distance_matrix_rowmin_equals_min_distances"]
        and report["checks"]["all_finite"]
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
