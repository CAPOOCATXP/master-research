#!/usr/bin/env python
"""Immediate Assignment #3 — the worked six-vector bound example.

The professor asked for exactly this:

    "Work out L(q), s_M(q), U(q) for a six-vector reference set with two observed
     centers; include a tie at threshold and one case requiring escalation.
     Explain the triangle inequality in plain language."

Everything below is hand-checkable arithmetic on a 2-D example. The script prints
a table and writes docs/notes/worked_example_bounds.md so the numbers and the
prose cannot drift apart.

The example is built so that each branch of the screening rule fires at least
once, including the two cases a reader is most likely to get wrong:

* ``L(q) == t`` exactly, where the reference decision is NORMAL because the rule
  is ``alarm iff S_M(x) > t``. Using ``L >= t`` here would report ALARM and
  silently disagree with the reference detector.
* a query where the interval straddles ``t`` and the *only* correct behaviour is
  to escalate.

Usage
-----
    .venv/bin/python scripts/02_worked_bound_example.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from master_research import (  # noqa: E402
    FullBankDetector,
    NumericalGuard,
    build_coverage_index,
    make_version_id,
    min_distances,
    patch_bounds,
    screen_image,
)

# --------------------------------------------------------------------------
# The example, in hand-checkable coordinates.
# --------------------------------------------------------------------------
M = np.array(
    [
        [0.0, 0.0],    # m1
        [2.0, 0.0],    # m2
        [0.0, 2.0],    # m3
        [10.0, 0.0],   # m4
        [12.0, 0.0],   # m5
        [10.0, 2.0],   # m6
    ],
    dtype=np.float32,
)
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)  # c1 = m1, c2 = m4
THRESHOLD = 3.0

QUERY_NOTES = [
    (np.array([1.0, 1.0], dtype=np.float32), "inside the first cell; interval far below t"),
    (np.array([11.0, 0.0], dtype=np.float32), "inside the second cell; sits between m4 and m5"),
    (np.array([30.0, 0.0], dtype=np.float32), "far outside every ball; interval entirely above t"),
    (np.array([5.0, 0.0], dtype=np.float32), "cell boundary; L(q) == t exactly -> tie"),
    (np.array([6.5, 0.0], dtype=np.float32), "straddles t -> must escalate; reference says ALARM"),
]


def main() -> int:
    guard = NumericalGuard()
    index = build_coverage_index(M, CENTERS, guard=guard)
    version = make_version_id(
        encoder="worked-example/identity",
        preprocessing="none",
        bank=M,
        threshold=THRESHOLD,
        threshold_policy="empirical 5% image false-alarm target (illustrative here)",
    )
    detector = FullBankDetector(bank=M, threshold=THRESHOLD, version=version, guard=guard)

    print("reference bank M (6 vectors) and 2 observed centers\n")
    print("  index   m_j        cell   d(m_j, c_j)")
    assignment = np.argmin(_dist(M, CENTERS), axis=1)
    for j in range(M.shape[0]):
        c = CENTERS[assignment[j]]
        d = float(np.linalg.norm(M[j] - c))
        print(f"  m{j + 1}      ({M[j, 0]:>4g},{M[j, 1]:>4g})   C{assignment[j] + 1}     {d:>6.3f}")

    print("\nradii (rounded UP by the guard, so L stays a valid lower bound):")
    for j in range(index.n_centers):
        raw = float(np.linalg.norm(M[assignment == j] - CENTERS[j], axis=1).max())
        print(f"  C{j + 1}: exact max = {raw:.6f}   stored r{j + 1} = {index.radii[j]:.6f}   cell size = {index.cell_sizes[j]}")

    print(f"\nthreshold t = {THRESHOLD:g}   (alarm iff S_M(x) > t; equality is NORMAL)\n")

    header = (
        f"{'query':>14}  {'U(q)':>9}  {'L(q)':>9}  {'s_M(q)':>9}  {'L<=s<=U':>8}  "
        f"{'reference':>9}  {'screened':>9}  {'escalated':>9}  note"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for q, note in QUERY_NOTES:
        qb = patch_bounds(q.reshape(1, -1), index, exact_scores=detector.patch_scores(q.reshape(1, -1)))
        lower, upper = float(qb.lower[0]), float(qb.upper[0])
        exact = float(qb.exact[0])
        reference = detector.decide(exact)

        result = screen_image(
            lower,
            upper,
            THRESHOLD,
            resolver=detector,
            request_version=version,
            queried_patches=q.reshape(1, -1),
        )
        holds = lower <= exact <= upper
        print(
            f"({q[0]:>5g},{q[1]:>6g})  {upper:>9.6f}  {lower:>9.6f}  {exact:>9.6f}  "
            f"{str(holds):>8}  {reference.value:>9}  {result.decision.value:>9}  "
            f"{str(result.was_escalated):>9}  {note}"
        )
        rows.append(
            {
                "query": [float(q[0]), float(q[1])],
                "note": note,
                "upper_U": upper,
                "lower_L": lower,
                "exact_S": exact,
                "interval_contains_exact": bool(holds),
                "reference_decision": reference.value,
                "screened_decision": result.decision.value,
                "outcome": result.outcome.value,
                "escalated": bool(result.was_escalated),
                "agrees_with_reference": bool(result.decision == reference),
            }
        )

    # ---------------- the tie, examined in detail ----------------
    tie_q = np.array([[5.0, 0.0]], dtype=np.float32)
    qb = patch_bounds(tie_q, index, exact_scores=detector.patch_scores(tie_q))
    L, U, S = float(qb.lower[0]), float(qb.upper[0]), float(qb.exact[0])
    print("\n--- why L(q) > t must be strict ---")
    print(f"  S_M(q) = {S:.6f}, t = {THRESHOLD:g}  -> reference decision = {detector.decide(S).value} (equality is NOT an alarm)")
    print(f"  L(q)   = {L:.6f}.  A rule using `L >= t` would certify ALARM without querying the bank")
    print("  and would therefore report alarm for an image the reference detector calls normal.")
    print("  The rule is therefore `L > t`, and this image escalates.")

    # ---------------- unresolved branches ----------------
    print("\n--- the fourth branch: no silent conversion to normal ---")
    ambiguous_q = np.array([[6.5, 0.0]], dtype=np.float32)
    qb2 = patch_bounds(ambiguous_q, index)
    unresolved = screen_image(float(qb2.lower[0]), float(qb2.upper[0]), THRESHOLD, resolver=None)
    print(f"  ambiguous interval + no resolver        -> {unresolved.decision.value} ({unresolved.outcome.value})")

    wrong_version = make_version_id(
        encoder="worked-example/identity",
        preprocessing="none",
        bank=M,
        threshold=THRESHOLD + 0.5,  # a different frozen threshold
        threshold_policy="different",
    )
    mismatched = screen_image(
        float(qb2.lower[0]),
        float(qb2.upper[0]),
        THRESHOLD,
        resolver=detector,
        request_version=wrong_version,
        queried_patches=ambiguous_q,
    )
    print(f"  ambiguous interval + version mismatch   -> {mismatched.decision.value} ({mismatched.outcome.value})")
    print(f"    mismatching fields: {wrong_version.mismatch_fields(detector.version)}")

    # ---------------- what breaks if the guard is removed ----------------
    print("\n--- effect of the numerical guard ---")
    tight = build_coverage_index(M, CENTERS, guard=NumericalGuard(rel=0.0, abs_=0.0))
    print(f"  radii with guard    : {np.round(index.radii, 8).tolist()}")
    print(f"  radii without guard : {np.round(tight.radii, 8).tolist()}")
    print("  On this well-conditioned integer example the guard changes nothing; it exists")
    print("  because on float32 embeddings the radius is a max over ~10^3-10^5 terms and a")
    print("  down-rounding of r_j would make L too large, i.e. an unsafe certificate.")

    report = {
        "stage": "immediate_assignment_3_worked_bounds",
        "reference_bank": M.tolist(),
        "centers": CENTERS.tolist(),
        "radii_exact": [float(np.linalg.norm(M[assignment == j] - CENTERS[j], axis=1).max()) for j in range(len(CENTERS))],
        "radii_stored_after_guard": index.radii.tolist(),
        "cell_sizes": index.cell_sizes.tolist(),
        "threshold": THRESHOLD,
        "version_id": version.digest(),
        "guard": guard.as_dict(),
        "rows": rows,
        "tie_case": {"query": [5.0, 0.0], "L": L, "U": U, "S": S, "strict_rule_required": True},
        "unresolved_cases": {
            "no_resolver": unresolved.as_row(),
            "version_mismatch": mismatched.as_row(),
            "mismatch_fields": wrong_version.mismatch_fields(detector.version),
        },
        "all_intervals_contain_exact": bool(all(r["interval_contains_exact"] for r in rows)),
        "all_decisions_agree_with_reference": bool(all(r["agrees_with_reference"] for r in rows)),
    }

    out_json = REPO / "results" / "logs" / "worked_bound_example.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\njson: {out_json}")
    print(
        "RESULT:",
        "PASS" if report["all_intervals_contain_exact"] and report["all_decisions_agree_with_reference"] else "FAIL",
    )
    return 0


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


if __name__ == "__main__":
    raise SystemExit(main())
