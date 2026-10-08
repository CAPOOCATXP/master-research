"""Reference-bank construction and memory accounting.

A "bank" here is simply the ``(N, D)`` float32 matrix of observed normal patch
embeddings that defines the reference detector. Everything this module produces
is a *subset selection* over that matrix: no training, no adaptation, no learned
projection. That restriction is deliberate — the research question is about what
compression does to an already-fixed detector, so the selector must not be able
to change the feature space.

The greedy selector below is the standard farthest-point / k-center greedy used
by PatchCore-style coreset selection. It is re-implemented rather than imported
so that the selection is auditable and seed-reproducible; it is **not** claimed
to be a numerical reproduction of PatchCore's `approx_greedy_coreset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distances import as_features, min_distances


@dataclass
class Bank:
    """A frozen set of reference patch embeddings plus its provenance."""

    features: np.ndarray
    ratio: float
    selector: str
    seed: int
    source_bank_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.features = as_features(self.features, "bank.features")

    @property
    def n(self) -> int:
        return int(self.features.shape[0])

    @property
    def d(self) -> int:
        return int(self.features.shape[1])

    @property
    def nbytes(self) -> int:
        """Serialized bytes of the bank vectors alone (no index, no radii)."""
        return int(self.features.nbytes)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "d": self.d,
            "ratio": self.ratio,
            "selector": self.selector,
            "seed": self.seed,
            "bank_bytes": self.nbytes,
            "source_bank_id": self.source_bank_id,
        }


def subset_size(n_full: int, ratio: float) -> int:
    """Cardinality of a compressed bank at a given ratio of the full bank.

    Uses ``ceil`` so that a ratio is never silently rounded down to zero
    references, and prefers at least 2 references so that a greedy selector has
    something to choose between. The result is clamped to ``n_full``: a tiny bank
    must never be asked for more rows than it has, which is why the clamp is
    applied *after* the floor rather than before.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if n_full < 1:
        raise ValueError(f"n_full must be >= 1, got {n_full}")
    wanted = max(2, int(np.ceil(n_full * ratio)))
    return int(min(n_full, wanted))


def random_subset(bank: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Uniform random subset of ``k`` rows. The plan's baseline (b)."""
    arr = as_features(bank, "bank")
    if k >= arr.shape[0]:
        return arr.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=k, replace=False)
    idx.sort()
    return arr[idx]


def greedy_coreset(bank: np.ndarray, k: int, seed: int, chunk_bytes: int = 64 * 1024 * 1024) -> np.ndarray:
    """Farthest-point greedy selection of ``k`` observed centers from ``bank``.

    Algorithm
    ---------
    1. Start from one seed row chosen deterministically by ``seed``.
    2. Maintain ``nearest[i] = min distance from row i to the selected set``.
    3. Repeatedly add ``argmax_i nearest[i]`` and update ``nearest`` in place.

    Cost is ``O(k * N * D)`` and every selected center is by construction an
    element of the bank, which is exactly the precondition ``c_j in M`` that
    :func:`master_research.bounds.build_coverage_index` enforces.

    Ties in ``argmax`` are broken by the lowest index, which makes the selection
    reproducible for a fixed seed and input order.
    """
    arr = as_features(bank, "bank")
    n = arr.shape[0]
    if k >= n:
        return arr.copy()
    if k < 1:
        raise ValueError("k must be >= 1")

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    selected = np.empty(k, dtype=np.int64)
    selected[0] = first

    nearest = min_distances(arr, arr[first : first + 1], chunk_bytes=chunk_bytes)
    for step in range(1, k):
        nxt = int(np.argmax(nearest))
        selected[step] = nxt
        d_new = min_distances(arr, arr[nxt : nxt + 1], chunk_bytes=chunk_bytes)
        np.minimum(nearest, d_new, out=nearest)
    return arr[selected]


def serialize_bank(bank: np.ndarray) -> bytes:
    """Canonical byte serialization used for bank identity hashing."""
    return np.ascontiguousarray(as_features(bank, "bank"), dtype=np.float32).tobytes(order="C")
