"""Conservative distance bounds for a fixed reference bank.

This module implements exactly the construction described in Part A8 of the
research plan, and nothing more:

    s_M(q) = min_{m in M} d(q, m)          the exact full-bank (reference) score
    L(q)   = min_j max(0, d(q, c_j) - r_j) a LOWER bound on s_M(q)
    U(q)   = min_j d(q, c_j)               an UPPER bound on s_M(q)

with observed centers ``c_j in M``, cells ``C_j`` the nearest-center partition of
``M``, and conservative coverage radii ``r_j >= max_{m in C_j} d(m, c_j)``.

Correctness argument (standard metric argument, **not** a new theorem):

* ``U``: every center is itself a bank vector, so ``min_j d(q, c_j)`` minimises
  ``d(q, m)`` over the subset ``{c_j}`` of ``M``; minimising over a subset can
  only be worse (larger) than minimising over all of ``M``. Hence ``s_M <= U``.
* ``L``: for any ``j`` and any ``m in C_j`` the triangle inequality gives
  ``d(m, c_j) <= d(m, q) + d(q, c_j)``, i.e.
  ``d(q, m) >= d(q, c_j) - d(m, c_j) >= d(q, c_j) - r_j``. Distances are also
  non-negative, so ``d(q, m) >= max(0, d(q, c_j) - r_j)``; minimising the right
  side over all cells gives a value no larger than ``s_M(q)``. Hence ``L <= s_M``.

Two consequences that the tests pin down and that a reader should not lose:

1. ``L <= s_M <= U`` is a statement in **exact arithmetic**. Float32 evaluation
   can break it in either direction unless the interval is deliberately widened,
   which is what :class:`NumericalGuard` does. With the guard applied, the
   bounds are *conservative*; the resulting agreement with the full bank is an
   *empirical* property, not a formal floating-point certificate.
2. This only bounds a score of the form ``min`` over a reference bank. Any
   reweighted or rescaled PatchCore score, and any cosine-distance variant, needs
   a separate derivation. See docs/protocol/numerics_contract.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .distances import as_features, min_distances

# --------------------------------------------------------------------------
# Numerical guard
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalGuard:
    """Deliberate widening applied to every reported bound.

    The guard is a *choice*, not a derivation. It is documented, configurable and
    reported alongside every result so that a reader can tell how much of an
    observed interval width is arithmetic slack rather than genuine geometric
    uncertainty.

    ``rel`` is a relative term applied as ``x * (1 +/- rel)`` and ``abs_`` an
    additive floor for values near zero. The default of 8 float32 ulps is a
    pragmatic allowance for a handful of roundings in the chunked distance
    kernel; it is **not** a proven error bound for that kernel.
    """

    rel: float = 8.0 * float(np.finfo(np.float32).eps)
    abs_: float = 1e-30

    def lower(self, x: np.ndarray) -> np.ndarray:
        """Round a value that must not be overestimated, downwards."""
        return (x.astype(np.float32) * np.float32(1.0 - self.rel)) - np.float32(self.abs_)

    def upper(self, x: np.ndarray) -> np.ndarray:
        """Round a value that must not be underestimated, upwards."""
        return (x.astype(np.float32) * np.float32(1.0 + self.rel)) + np.float32(self.abs_)

    def as_dict(self) -> dict:
        return {"rel": self.rel, "abs": self.abs_, "rel_ulps_f32": self.rel / float(np.finfo(np.float32).eps)}


# --------------------------------------------------------------------------
# Coverage index
# --------------------------------------------------------------------------


@dataclass
class CoverageIndex:
    """Observed centers plus conservative per-cell coverage radii.

    Attributes
    ----------
    centers:
        ``(K, D)`` float32. Every center **must** be an element of the full bank
        (``c_j in M``); :func:`build_coverage_index` enforces this.
    radii:
        ``(K,)`` float32, already rounded upwards, with ``r_j >= max_{m in C_j}
        d(m, c_j)``.
    cell_sizes:
        ``(K,)`` int64, number of bank vectors assigned to each cell. Cells are
        never empty when centers are drawn from the bank, but the field is
        recorded and asserted because the plan requires an explicit empty-cell
        check.
    guard:
        the :class:`NumericalGuard` used to widen the radii.
    """

    centers: np.ndarray
    radii: np.ndarray
    cell_sizes: np.ndarray
    guard: NumericalGuard = field(default_factory=NumericalGuard)
    n_bank: int = 0
    n_dim: int = 0

    def __post_init__(self) -> None:
        self.centers = as_features(self.centers, "centers")
        self.radii = np.asarray(self.radii, dtype=np.float32).reshape(-1)
        self.cell_sizes = np.asarray(self.cell_sizes, dtype=np.int64).reshape(-1)
        if not (len(self.centers) == len(self.radii) == len(self.cell_sizes)):
            raise ValueError("centers, radii and cell_sizes must have the same length")
        if len(self.centers) == 0:
            raise ValueError("coverage index has no centers")
        if not np.isfinite(self.radii).all():
            raise ValueError("non-finite radius")

    @property
    def n_centers(self) -> int:
        return int(self.centers.shape[0])

    @property
    def nonempty(self) -> np.ndarray:
        """Boolean mask of cells that actually contain at least one bank vector."""
        return self.cell_sizes > 0

    @property
    def empty_cell_count(self) -> int:
        return int((~self.nonempty).sum())

    def bank_and_metadata_bytes(self) -> int:
        """Serialized bytes of the edge state: centers + radii + cell sizes.

        Encoder / runtime memory is deliberately **excluded** and must be
        reported separately (Part A6). Metadata is counted here, not ignored.
        """
        return int(self.centers.nbytes + self.radii.nbytes + self.cell_sizes.nbytes)


def build_coverage_index(
    bank: np.ndarray,
    centers: np.ndarray,
    guard: Optional[NumericalGuard] = None,
) -> CoverageIndex:
    """Partition ``bank`` into nearest-center cells and compute safe radii.

    Steps, in the order the plan states them:

    1. Assign **every** bank vector to its nearest center (no vector is dropped).
    2. Set ``r_j = max_{m in C_j} d(m, c_j)``.
    3. Round ``r_j`` upwards using the guard, so ``L`` stays a valid lower bound.

    A center that is not an element of the bank raises ``ValueError``: ``c_j in M``
    is required for ``U`` to be an upper bound at all.

    Raises
    ------
    ValueError
        If a center is not present in the bank, or if a cell ends up empty.
    """
    guard = guard or NumericalGuard()
    bank_arr = as_features(bank, "bank")
    centers_arr = as_features(centers, "centers")
    if bank_arr.shape[1] != centers_arr.shape[1]:
        raise ValueError("bank and centers must share the feature dimension")

    # Step 1: full assignment. `min_distances` gives value + we need the argmin,
    # so compute the distance matrix to centers explicitly (K is small).
    dist_to_centers = _dist_to_centers(bank_arr, centers_arr)  # (N, K)
    assignment = np.argmin(dist_to_centers, axis=1)

    # Enforce c_j in M. Nearest-center assignment never assigns a bank vector to
    # a center that is not closer than any other, so a center absent from the
    # bank would silently produce radius 0 and a WRONG (too tight) lower bound.
    _assert_centers_present(bank_arr, centers_arr)

    k = centers_arr.shape[0]
    cell_sizes = np.bincount(assignment, minlength=k).astype(np.int64)
    radii = np.zeros(k, dtype=np.float32)
    for j in range(k):
        members = np.flatnonzero(assignment == j)
        if members.size == 0:
            raise ValueError(
                f"cell {j} is empty. The bound L=min_j max(0, d(q,c_j)-r_j) is only valid "
                "over cells that contain bank vectors; an empty cell must be dropped, not "
                "given radius 0."
            )
        cell = bank_arr[members]
        diff = cell - centers_arr[j]
        member_dists = np.sqrt(np.einsum("ij,ij->i", diff, diff), dtype=np.float32)
        radii[j] = float(member_dists.max())

    radii = guard.upper(radii).astype(np.float32)
    return CoverageIndex(
        centers=centers_arr,
        radii=radii,
        cell_sizes=cell_sizes,
        guard=guard,
        n_bank=int(bank_arr.shape[0]),
        n_dim=int(bank_arr.shape[1]),
    )


def _dist_to_centers(x: np.ndarray, centers: np.ndarray, block_bytes: int = 32 * 1024 * 1024) -> np.ndarray:
    """``(N, K)`` query-to-center distances in float32, by **direct subtraction**.

    This deliberately does *not* use the ``|a|^2 + |b|^2 - 2ab`` expansion.

    Recorded failure (2026-10, Phase 0): the expansion form was used here first.
    On 1536-dimensional backbone features the two squared-norm terms are orders of
    magnitude larger than their difference, so the subtraction loses most of its
    significant digits. The computed ``d(q, c_j)`` then came out slightly *too
    large*, ``L(q) = min_j max(0, d(q,c_j) - r_j)`` came out too large with it,
    and 31 of 676 patches reported ``L(q) > s_M(q)`` — an invalid lower bound, and
    precisely the kind of silent failure that would have manufactured a fake
    "certified alarm". After switching to direct subtraction the count is 0.

    Direct subtraction is ``O(N*K*D)``, the same as the expansion form, so nothing
    is given up by being careful. Centers are processed in blocks so that the
    broadcast difference tensor stays inside a fixed memory budget instead of
    growing with ``K``; blocking changes no value, it only shortens the loop.
    """
    if x.shape[1] != centers.shape[1]:
        raise ValueError("dimension mismatch")
    n, k, d = x.shape[0], centers.shape[0], x.shape[1]
    out = np.empty((n, k), dtype=np.float32)
    per_center_bytes = max(1, n * d * 4)
    block = max(1, min(k, int(block_bytes // per_center_bytes)))
    for start in range(0, k, block):
        stop = min(start + block, k)
        diff = x[:, None, :] - centers[None, start:stop, :]  # (n, b, d)
        out[:, start:stop] = np.sqrt(np.einsum("nbd,nbd->nb", diff, diff), dtype=np.float32)
    return out


def _assert_centers_present(bank: np.ndarray, centers: np.ndarray) -> None:
    """Verify every center occurs in the bank (exact float32 match)."""
    view = {row.tobytes() for row in np.ascontiguousarray(bank, dtype=np.float32)}
    for j, row in enumerate(np.ascontiguousarray(centers, dtype=np.float32)):
        if row.tobytes() not in view:
            raise ValueError(
                f"center {j} is not an element of the bank. U(q)=min_j d(q,c_j) is only an "
                "upper bound on s_M(q) if every center is itself in M."
            )


# --------------------------------------------------------------------------
# Patch-level bounds
# --------------------------------------------------------------------------


@dataclass
class PatchBounds:
    """Patch-level intervals for one query image."""

    lower: np.ndarray  # L(q), conservative (may be too small)
    upper: np.ndarray  # U(q), conservative (may be too large)
    exact: Optional[np.ndarray] = None  # s_M(q), only when the full bank was queried

    def __len__(self) -> int:
        return int(self.lower.shape[0])

    def violations(self, atol: float = 0.0) -> int:
        """Count patches where the interval fails to contain the exact score."""
        if self.exact is None:
            raise ValueError("no exact scores available to check against")
        bad = (self.lower > self.exact + atol) | (self.exact > self.upper + atol)
        return int(bad.sum())


def patch_bounds(
    queries: np.ndarray,
    index: CoverageIndex,
    exact_scores: Optional[np.ndarray] = None,
) -> PatchBounds:
    """Compute conservative ``(L(q), U(q))`` for every query patch.

    Cost is ``O(P * K * D)``: only the ``K`` centers are touched, never the ``N``
    bank vectors. This is the entire resource argument for screening.
    """
    q = as_features(queries, "queries")
    if q.shape[1] != index.centers.shape[1]:
        raise ValueError("query dimension does not match the coverage index")
    if index.empty_cell_count:
        raise ValueError("coverage index contains empty cells; rebuild it")

    d_centers = _dist_to_centers(q, index.centers)  # (P, K)

    # U(q) = min_j d(q, c_j), rounded UP so it cannot fall below s_M(q).
    u = index.guard.upper(d_centers.min(axis=1)).astype(np.float32)

    # L(q) = min_j max(0, d(q, c_j) - r_j), rounded DOWN.
    slack = d_centers - index.radii[None, :]
    np.maximum(slack, 0.0, out=slack)
    l = index.guard.lower(slack.min(axis=1)).astype(np.float32)
    np.maximum(l, 0.0, out=l)

    exact = None
    if exact_scores is not None:
        exact = np.asarray(exact_scores, dtype=np.float32).reshape(-1)
        if exact.shape[0] != q.shape[0]:
            raise ValueError("exact_scores length does not match queries")
    return PatchBounds(lower=l, upper=u, exact=exact)


# --------------------------------------------------------------------------
# Image-level aggregation
# --------------------------------------------------------------------------


def aggregate_max(values: np.ndarray) -> np.ndarray:
    """Image score = maximum patch score.

    The plan's headline aggregation. ``max`` is coordinatewise monotone, so an
    interval on each patch yields an interval on the image score.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("expected a (n_images, n_patches) array")
    return arr.max(axis=1)


def image_bounds(
    lower_patches: np.ndarray,
    upper_patches: np.ndarray,
    exact_patches: Optional[np.ndarray] = None,
):
    """Aggregate patch intervals to image intervals under the ``max`` rule.

    Returns ``(L_x, U_x)`` and, when provided, ``S_M(x)``. Because ``max`` is
    monotone: ``max_q L(q) <= max_q s_M(q) <= max_q U(q)``.
    """
    lx = aggregate_max(lower_patches)
    ux = aggregate_max(upper_patches)
    sx = aggregate_max(exact_patches) if exact_patches is not None else None
    return lx, ux, sx
