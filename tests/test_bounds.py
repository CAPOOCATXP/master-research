"""Tests for the interval bounds.

The single most important property in the whole project is asserted here:

    L(q) <= s_M(q) <= U(q)

and it is asserted in the regime where a naive implementation silently fails
(high-dimensional features, large magnitudes). See the docstring of
``master_research.bounds._dist_to_centers`` for the recorded failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from master_research.bounds import (
    CoverageIndex,
    NumericalGuard,
    aggregate_max,
    build_coverage_index,
    image_bounds,
    patch_bounds,
)
from master_research.distances import min_distances


def make_problem(n_bank=200, n_dim=64, k=8, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    bank = (rng.standard_normal((n_bank, n_dim)) * scale).astype(np.float32)
    idx = np.sort(rng.choice(n_bank, size=k, replace=False))
    centers = bank[idx]
    queries = (rng.standard_normal((37, n_dim)) * scale).astype(np.float32)
    return bank, centers, queries


def test_interval_contains_exact_score_on_random_data():
    bank, centers, queries = make_problem(seed=1)
    index = build_coverage_index(bank, centers)
    exact = min_distances(queries, bank)
    pb = patch_bounds(queries, index, exact_scores=exact)
    assert pb.violations() == 0


@pytest.mark.parametrize("scale", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("n_dim", [2, 64, 1536])
def test_interval_contains_exact_score_across_scales_and_dimensions(scale, n_dim):
    """Regression test for the invalid-lower-bound bug.

    With the original expansion-form center distances this failed at n_dim=1536
    with scale>=1.0 (31 of 676 patches had L > s_M).
    """
    bank, centers, queries = make_problem(n_bank=150, n_dim=n_dim, k=10, seed=2, scale=scale)
    index = build_coverage_index(bank, centers)
    exact = min_distances(queries, bank)
    pb = patch_bounds(queries, index, exact_scores=exact)
    assert pb.violations() == 0, f"scale={scale} dim={n_dim}: {pb.violations()} invalid patches"


def test_interval_contains_exact_score_with_backbone_like_features():
    """Reproduces the Phase-0 regime: 1536-dim features around +0.5 mean, std 1.5."""
    rng = np.random.default_rng(99)
    bank = (rng.standard_normal((676, 1536)) * 1.5 + 0.5).astype(np.float32)
    centers = bank[np.sort(rng.choice(676, 68, replace=False))]
    index = build_coverage_index(bank, centers)
    exact = min_distances(bank, bank)
    pb = patch_bounds(bank, index, exact_scores=exact)
    assert pb.violations() == 0


def test_upper_bound_is_never_below_exact_even_without_guard():
    bank, centers, queries = make_problem(seed=5)
    index = build_coverage_index(bank, centers, guard=NumericalGuard(rel=0.0, abs_=0.0))
    exact = min_distances(queries, bank)
    pb = patch_bounds(queries, index, exact_scores=exact)
    assert np.all(pb.upper >= exact)


def test_guard_widens_the_interval():
    bank, centers, queries = make_problem(seed=6)
    tight = build_coverage_index(bank, centers, guard=NumericalGuard(rel=0.0, abs_=0.0))
    wide = build_coverage_index(bank, centers, guard=NumericalGuard(rel=1e-3, abs_=0.0))
    pb_tight = patch_bounds(queries, tight)
    pb_wide = patch_bounds(queries, wide)
    assert np.all(pb_wide.upper >= pb_tight.upper)
    assert np.all(pb_wide.lower <= pb_tight.lower)
    assert (pb_wide.upper - pb_wide.lower).mean() >= (pb_tight.upper - pb_tight.lower).mean()


def test_radii_are_conservative_for_their_cells():
    bank, centers, queries = make_problem(seed=7)
    index = build_coverage_index(bank, centers)
    for j in range(index.n_centers):
        d = np.linalg.norm(bank - centers[j], axis=1)
        # every bank vector within r_j of c_j must belong to cell j
        within = np.flatnonzero(d <= index.radii[j] + 1e-5)
        assert within.size >= index.cell_sizes[j]
    assert index.empty_cell_count == 0


def test_center_not_in_bank_is_rejected():
    bank = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    centres = np.array([[0.5, 0.5]], dtype=np.float32)  # not a bank element
    with pytest.raises(ValueError, match="not an element of the bank"):
        build_coverage_index(bank, centres)


def test_duplicate_bank_vectors_are_allowed():
    bank = np.array([[0.0, 0.0], [0.0, 0.0], [4.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    centres = np.array([[0.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    index = build_coverage_index(bank, centres)
    assert index.cell_sizes.tolist() == [2, 2]
    assert index.empty_cell_count == 0


def test_cell_sizes_sum_to_bank_size():
    bank, centers, _ = make_problem(n_bank=137, k=9, seed=8)
    index = build_coverage_index(bank, centers)
    assert int(index.cell_sizes.sum()) == 137
    assert index.bank_and_metadata_bytes() > 0


def test_empty_cell_raises_rather_than_silently_using_radius_zero():
    """An empty cell must never contribute a term to the min in L.

    If it did, that term would be d(q, c_j) for a center in M, which is an upper
    bound, not a lower bound -- the certificate would be unsound. The index
    therefore refuses to be built at all.
    """
    bank = np.array([[0.0, 0.0], [10.0, 0.0], [10.1, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        # Force an empty cell by claiming a non-contiguous assignment manually.
        idx = CoverageIndex(
            centers=np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
            radii=np.array([0.0, 0.0], dtype=np.float32),
            cell_sizes=np.array([3, 0], dtype=np.int64),
        )
        patch_bounds(np.array([[1.0, 1.0]], dtype=np.float32), idx)


def test_image_bounds_are_monotone_aggregates():
    bank, centers, _ = make_problem(n_bank=180, k=7, seed=9)
    index = build_coverage_index(bank, centers)
    queries = bank[:25]
    exact = min_distances(queries, bank).reshape(5, 5)
    pb = patch_bounds(queries, index)
    lx, ux, sx = image_bounds(pb.lower.reshape(5, 5), pb.upper.reshape(5, 5), exact)
    assert np.all(lx <= sx)
    assert np.all(sx <= ux)


def test_aggregate_max_rejects_non_2d():
    with pytest.raises(ValueError):
        aggregate_max(np.zeros(5, dtype=np.float32))


def test_patch_bounds_rejects_dimension_mismatch():
    bank, centers, queries = make_problem(n_dim=16, seed=10)
    index = build_coverage_index(bank, centers)
    with pytest.raises(ValueError):
        patch_bounds(np.zeros((3, 17), dtype=np.float32), index)


def test_patch_bounds_rejects_wrong_length_exact_scores():
    bank, centers, queries = make_problem(seed=11)
    index = build_coverage_index(bank, centers)
    with pytest.raises(ValueError):
        patch_bounds(queries, index, exact_scores=np.zeros(3, dtype=np.float32))
