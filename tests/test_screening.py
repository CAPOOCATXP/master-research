"""Tests for the screening rule, including every UNRESOLVED branch.

The rule has four branches and the plan requires all four to be exercised:
local normal, local alarm, escalate-and-resolve, and unresolved. The equality
case is tested explicitly because it is the one that a ``>=`` typo silently
breaks.
"""

from __future__ import annotations

import numpy as np
import pytest

from master_research.bounds import build_coverage_index, patch_bounds
from master_research.distances import min_distances
from master_research.screening import (
    Decision,
    FullBankDetector,
    ScreenOutcome,
    screen_image,
    summarise_screening,
)
from master_research.versioning import make_version_id


def make_detector(n_bank=120, n_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    bank = (rng.standard_normal((n_bank, n_dim)) * 3).astype(np.float32)
    version = make_version_id(
        encoder="test/encoder",
        preprocessing="test/pre",
        bank=bank,
        threshold=2.5,
        threshold_policy="test",
    )
    return FullBankDetector(bank=bank, threshold=2.5, version=version)


def test_upper_at_or_below_threshold_is_local_normal():
    r = screen_image(lower=1.0, upper=2.5, threshold=2.5)
    assert r.decision is Decision.NORMAL
    assert r.outcome is ScreenOutcome.CERTIFIED_LOCAL
    assert not r.was_escalated


def test_lower_strictly_above_threshold_is_local_alarm():
    r = screen_image(lower=2.5001, upper=9.0, threshold=2.5)
    assert r.decision is Decision.ALARM
    assert r.outcome is ScreenOutcome.CERTIFIED_LOCAL
    assert not r.was_escalated


def test_lower_equal_to_threshold_escalates_not_alarm():
    """The equality case that a `>=` typo would get wrong.

    The reference decision is ``alarm iff S_M(x) > t``, so ``L == t`` must NOT be
    certified as an alarm even though the interval is entirely at or above t.
    """
    detector = make_detector()
    r = screen_image(lower=2.5, upper=9.0, threshold=2.5, resolver=detector, request_version=detector.version, queried_patches=detector.bank[:1])
    assert r.outcome is ScreenOutcome.ESCALATED_RESOLVED


def test_ambiguous_interval_without_resolver_is_unresolved_never_normal():
    r = screen_image(lower=1.0, upper=9.0, threshold=2.5, resolver=None)
    assert r.decision is Decision.UNRESOLVED
    assert r.outcome is ScreenOutcome.UNRESOLVED_NO_RESOLVER


def test_version_mismatch_is_unresolved():
    detector = make_detector(seed=1)
    other = make_version_id(
        encoder="test/encoder",
        preprocessing="test/pre",
        bank_id="deadbeef" * 8,
        threshold=2.5,
    )
    r = screen_image(
        lower=1.0,
        upper=9.0,
        threshold=2.5,
        resolver=detector,
        request_version=other,
        queried_patches=detector.bank[:1],
    )
    assert r.decision is Decision.UNRESOLVED
    assert r.outcome is ScreenOutcome.UNRESOLVED_VERSION_MISMATCH
    assert "bank_id" in r.note


def test_escalation_requires_patches():
    detector = make_detector(seed=2)
    with pytest.raises(ValueError, match="queried_patches"):
        screen_image(lower=1.0, upper=9.0, threshold=2.5, resolver=detector, request_version=detector.version)


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError, match="invalid interval"):
        screen_image(lower=5.0, upper=1.0, threshold=2.5)


def test_fullbank_detector_uses_strict_inequality():
    detector = make_detector(seed=3)
    assert detector.decide(detector.threshold) is Decision.NORMAL
    assert detector.decide(detector.threshold + 1e-6) is Decision.ALARM
    assert detector.decide(detector.threshold - 1e-6) is Decision.NORMAL


def _run_screen(bank, centers, images, threshold_policy="median"):
    """Screen a list of images against a bank/center set and return (results, detector)."""
    index = build_coverage_index(bank, centers)
    exact_per_image = [min_distances(img, bank) for img in images]
    pool = np.concatenate(exact_per_image)
    threshold = float(np.median(pool)) if threshold_policy == "median" else float(threshold_policy)
    detector = FullBankDetector(
        bank=bank,
        threshold=threshold,
        version=make_version_id("e", "p", bank=bank, threshold=threshold, threshold_policy=threshold_policy),
    )
    results, references = [], []
    for img, exact in zip(images, exact_per_image):
        pb = patch_bounds(img, index)
        res = screen_image(
            float(pb.lower.max()),
            float(pb.upper.max()),
            threshold,
            resolver=detector,
            request_version=detector.version,
            queried_patches=img,
        )
        references.append(detector.decide(float(exact.max())))
        results.append(res)
    return results, references, detector


def test_screened_decisions_match_the_reference_detector():
    """End-to-end contract: with a resolver available, no decision may disagree.

    This is the property the whole project is about. Note what is deliberately
    *not* asserted here: that many images are certified locally. With 25 observed
    centers covering 400 bank vectors in 48 dimensions the intervals are wide and
    almost everything escalates. That is the failure mode the plan predicts, and
    it is measured explicitly in the next two tests rather than hidden.
    """
    rng = np.random.default_rng(42)
    bank = (rng.standard_normal((400, 48)) * 4).astype(np.float32)
    centers = bank[np.sort(rng.choice(400, 25, replace=False))]
    images = [bank[:10].copy()] + [(rng.standard_normal((10, 48)) * 4).astype(np.float32) for _ in range(12)]

    results, references, _ = _run_screen(bank, centers, images)

    assert all(r.decision is ref for r, ref in zip(results, references))
    stats = summarise_screening(results)
    assert stats["unresolved_rate"] == 0.0
    assert stats["resolved_fraction"] == 1.0


def test_sparse_centers_force_escalation():
    """The predicted failure boundary: wide radii make the screen uninformative.

    With K=25 centers for N=400 bank vectors, L is almost always 0 and U almost
    always exceeds a threshold placed in the middle of the score distribution, so
    the screen buys nothing. This is the "wide bounds cause near-universal
    escalation" risk from Part A11, not a bug.
    """
    rng = np.random.default_rng(7)
    bank = (rng.standard_normal((400, 48)) * 4).astype(np.float32)
    centers = bank[np.sort(rng.choice(400, 25, replace=False))]
    images = [(rng.standard_normal((10, 48)) * 4).astype(np.float32) for _ in range(20)]

    results, references, _ = _run_screen(bank, centers, images)
    stats = summarise_screening(results)

    assert all(r.decision is ref for r, ref in zip(results, references))
    assert stats["escalation_rate"] >= 0.9, f"expected near-universal escalation, got {stats}"
    assert stats["locally_certified_fraction"] <= 0.1


def test_dense_centers_certify_every_image_locally():
    """The opposite boundary: when every bank vector is its own center, L == U == s_M.

    With ``centers = bank``, each cell is a single vector, every radius is 0, and
    the interval collapses onto the exact score. Every decision is then certified
    locally with no full-bank query. This pins down the two ends of the
    tightness-versus-cost trade-off that the research question is about.
    """
    rng = np.random.default_rng(11)
    bank = (rng.standard_normal((60, 16)) * 3).astype(np.float32)
    images = [(rng.standard_normal((5, 16)) * 3).astype(np.float32) for _ in range(8)]

    results, references, _ = _run_screen(bank, bank.copy(), images)
    stats = summarise_screening(results)

    assert all(r.decision is ref for r, ref in zip(results, references))
    assert stats["locally_certified_fraction"] == 1.0
    assert stats["escalation_rate"] == 0.0


def test_summarise_handles_empty_input():
    assert summarise_screening([]) == {"n": 0}


def test_screen_result_row_is_serialisable():
    r = screen_image(lower=0.0, upper=1.0, threshold=2.5)
    row = r.as_row()
    assert set(row) == {"decision", "outcome", "lower", "upper", "threshold", "exact_score", "note"}
    assert row["decision"] == "normal"
