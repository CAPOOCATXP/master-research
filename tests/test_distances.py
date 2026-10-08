"""Tests for the exact distance kernel.

The kernel is the foundation of every bound in this repository, so the tests
here are deliberately adversarial rather than illustrative: duplicates, exact
zero distances, high dynamic range, and chunk sizes small enough to force many
chunks.
"""

from __future__ import annotations

import numpy as np
import pytest

from master_research.distances import (
    chunk_rows,
    float32_epsilon,
    min_distances,
    pairwise_distances,
)


def brute_force_min(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Independent float64 brute force, written differently on purpose."""
    out = np.empty(queries.shape[0], dtype=np.float64)
    for i, q in enumerate(queries.astype(np.float64)):
        out[i] = np.linalg.norm(references.astype(np.float64) - q, axis=1).min()
    return out


def test_matches_brute_force_on_small_integers():
    refs = np.array([[0, 0], [3, 4], [3, 4], [6, 8], [-3, -4], [1, 0], [0, 1], [10, 0], [10, 1], [0.5, 0.5]])
    queries = np.array([[0, 0], [3, 4], [0, 5], [0.5, 0.5], [100, 100], [2, 0]])
    got = min_distances(queries, refs)
    expected = brute_force_min(queries, refs)
    assert np.allclose(got.astype(np.float64), expected, atol=1e-6)


def test_query_on_reference_yields_exact_zero():
    refs = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    got = min_distances(np.array([[5.0, 5.0]], dtype=np.float32), refs)
    assert got[0] == 0.0


def test_chunking_is_numerically_inert():
    rng = np.random.default_rng(0)
    refs = rng.standard_normal((500, 64)).astype(np.float32) * 5
    queries = rng.standard_normal((40, 64)).astype(np.float32) * 5
    default = min_distances(queries, refs)
    tiny = min_distances(queries, refs, chunk_bytes=1024)
    assert np.array_equal(default, tiny)


def test_repeated_calls_are_bitwise_identical():
    rng = np.random.default_rng(1)
    refs = rng.standard_normal((200, 32)).astype(np.float32)
    queries = rng.standard_normal((20, 32)).astype(np.float32)
    assert np.array_equal(min_distances(queries, refs), min_distances(queries, refs))


def test_high_dynamic_range_matches_float64():
    """Cancellation stress: the expansion form loses digits here.

    This is the regime where the first implementation of the interval bounds
    produced invalid lower bounds. The kernel must still agree with float64.
    """
    rng = np.random.default_rng(7)
    refs = (rng.standard_normal((400, 1536)) * 1.5 + 0.5).astype(np.float32)
    queries = (rng.standard_normal((50, 1536)) * 1.5 + 0.5).astype(np.float32)
    got = min_distances(queries, refs)
    expected = brute_force_min(queries, refs)
    rel = np.abs(got.astype(np.float64) - expected) / np.maximum(expected, 1e-9)
    assert rel.max() < 1e-5, f"max relative error {rel.max():.3e}"


def test_more_refine_candidates_never_increases_the_result():
    """Refining more candidates can only find a smaller (more correct) minimum."""
    rng = np.random.default_rng(3)
    refs = rng.standard_normal((300, 512)).astype(np.float32) * 10
    queries = rng.standard_normal((30, 512)).astype(np.float32) * 10
    previous = None
    for k in (1, 2, 4, 8):
        got = min_distances(queries, refs, refine_candidates=k)
        if previous is not None:
            assert np.all(got <= previous + 1e-6), "refining more candidates returned a larger minimum"
        previous = got


def test_duplicate_references_are_fine():
    rng = np.random.default_rng(4)
    base = rng.standard_normal((10, 16)).astype(np.float32)
    refs = np.vstack([base, base])  # every reference duplicated
    queries = rng.standard_normal((5, 16)).astype(np.float32)
    assert np.array_equal(min_distances(queries, refs), min_distances(queries, base))


def test_single_reference():
    refs = np.array([[1.0, 2.0]], dtype=np.float32)
    got = min_distances(np.array([[4.0, 6.0]], dtype=np.float32), refs)
    assert got[0] == pytest.approx(5.0, abs=1e-6)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dimension mismatch"):
        min_distances(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 4), dtype=np.float32))


def test_empty_references_raises():
    with pytest.raises(ValueError, match="empty"):
        min_distances(np.zeros((2, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))


def test_non_finite_input_raises():
    bad = np.array([[0.0, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        min_distances(bad, np.zeros((1, 2), dtype=np.float32))


def test_1d_input_is_promoted_to_a_single_row():
    refs = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    got = min_distances(np.array([0.5, 0.0], dtype=np.float32), refs)
    assert got.shape == (1,)


def test_distance_matrix_rowmin_agrees_with_min_distances():
    rng = np.random.default_rng(11)
    refs = rng.standard_normal((60, 8)).astype(np.float32)
    queries = rng.standard_normal((15, 8)).astype(np.float32)
    matrix = pairwise_distances(queries, refs)
    got = min_distances(queries, refs)
    assert np.allclose(matrix.min(axis=1), got, atol=1e-6)


def test_chunk_rows_is_at_least_one():
    assert chunk_rows(10**9, 10**4, chunk_bytes=1) == 1
    assert float32_epsilon() == float(np.finfo(np.float32).eps)
