"""Tests for bank construction, sizing and version identifiers."""

from __future__ import annotations

import numpy as np
import pytest

from master_research.bank import Bank, greedy_coreset, random_subset, serialize_bank, subset_size
from master_research.versioning import hash_bank, make_version_id


def test_subset_size_is_never_zero_and_never_exceeds_the_bank():
    assert subset_size(1000, 0.01) == 10
    assert subset_size(1000, 0.25) == 250
    assert subset_size(1000, 1.0) == 1000
    assert subset_size(10, 0.01) == 2  # floor of 2 references
    assert subset_size(1, 1.0) == 1


def test_subset_size_rejects_invalid_ratios():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            subset_size(100, bad)


def test_greedy_coreset_returns_observed_centers_only():
    """Every selected center must be an element of the bank (c_j in M)."""
    rng = np.random.default_rng(0)
    bank = rng.standard_normal((300, 24)).astype(np.float32)
    centers = greedy_coreset(bank, 30, seed=0)
    assert centers.shape == (30, 24)
    view = {row.tobytes() for row in bank}
    assert all(row.tobytes() in view for row in centers)


def test_greedy_coreset_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(1)
    bank = rng.standard_normal((200, 16)).astype(np.float32)
    a = greedy_coreset(bank, 15, seed=3)
    b = greedy_coreset(bank, 15, seed=3)
    assert np.array_equal(a, b)


def test_greedy_coreset_different_seeds_differ():
    rng = np.random.default_rng(2)
    bank = rng.standard_normal((200, 16)).astype(np.float32)
    assert not np.array_equal(greedy_coreset(bank, 15, seed=3), greedy_coreset(bank, 15, seed=4))


def test_greedy_coreset_improves_coverage_over_random():
    """The greedy selector should not be worse than random at covering the bank.

    Measured by the max distance from any bank vector to the selected set, which
    is exactly the quantity the minimax facility-location objective minimises.
    """
    rng = np.random.default_rng(5)
    bank = rng.standard_normal((400, 32)).astype(np.float32)
    k = 40

    def coverage(sel):
        return float(np.linalg.norm(bank[:, None, :] - sel[None, :, :], axis=2).min(axis=1).max())

    greedy = coverage(greedy_coreset(bank, k, seed=0))
    random_cov = np.mean([coverage(random_subset(bank, k, seed=s)) for s in range(5)])
    assert greedy <= random_cov


def test_greedy_coreset_k_ge_n_returns_the_whole_bank():
    bank = np.arange(20, dtype=np.float32).reshape(10, 2)
    assert np.array_equal(greedy_coreset(bank, 10, seed=0), bank)


def test_greedy_coreset_rejects_k_below_one():
    bank = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        greedy_coreset(bank, 0, seed=0)


def test_random_subset_is_deterministic_and_unique():
    rng = np.random.default_rng(6)
    bank = rng.standard_normal((100, 8)).astype(np.float32)
    a = random_subset(bank, 10, seed=2)
    b = random_subset(bank, 10, seed=2)
    assert np.array_equal(a, b)
    assert a.shape == (10, 8)
    assert len({row.tobytes() for row in a}) == 10  # sampling without replacement


def test_bank_byte_accounting_matches_float32_size():
    bank = np.zeros((50, 32), dtype=np.float32)
    b = Bank(features=bank, ratio=0.5, selector="test", seed=0)
    assert b.nbytes == 50 * 32 * 4
    assert b.summary()["n"] == 50


def test_hash_bank_is_content_sensitive_and_order_sensitive():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    b = a.copy()
    assert hash_bank(a) == hash_bank(b)
    b[0, 0] = 1.5
    assert hash_bank(a) != hash_bank(b), "bank identity must depend on content"
    c = np.ascontiguousarray(a[::-1])
    assert hash_bank(a) != hash_bank(c), "bank identity must depend on row order"


def test_serialize_bank_is_canonical():
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    assert serialize_bank(x) == serialize_bank(x.copy(order="C"))


def test_version_id_detects_a_threshold_change():
    bank = np.zeros((10, 4), dtype=np.float32)
    a = make_version_id("enc", "pre", bank=bank, threshold=1.0, threshold_policy="p")
    b = make_version_id("enc", "pre", bank=bank, threshold=1.1, threshold_policy="p")
    assert not a.matches(b)
    assert a.mismatch_fields(b) == ["threshold"]


def test_version_id_detects_a_bank_change():
    a = make_version_id("enc", "pre", bank=np.zeros((10, 4), dtype=np.float32), threshold=1.0)
    b = make_version_id("enc", "pre", bank=np.ones((10, 4), dtype=np.float32), threshold=1.0)
    assert not a.matches(b)
    assert "bank_id" in a.mismatch_fields(b)


def test_version_id_digest_is_stable_and_short():
    bank = np.zeros((10, 4), dtype=np.float32)
    a = make_version_id("enc", "pre", bank=bank, threshold=1.0)
    b = make_version_id("enc", "pre", bank=bank, threshold=1.0)
    assert a.digest() == b.digest()
    assert len(a.digest()) == 16


def test_make_version_id_requires_a_bank_or_an_id():
    with pytest.raises(ValueError):
        make_version_id("enc", "pre")


def test_hash_weights_accepts_bytes_and_a_path(tmp_path):
    from master_research.versioning import hash_weights

    payload = b"pretend weights"
    p = tmp_path / "w.bin"
    p.write_bytes(payload)
    assert hash_weights(p) == hash_weights(payload)
