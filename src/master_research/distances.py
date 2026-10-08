"""Exact Euclidean distance primitives used by every other module.

Design rules (frozen for Week 1, see docs/protocol/numerics_contract.md):

* All arithmetic is ``float32``. Every routine has an explicit ``dtype`` so that
  a caller cannot silently get float64 and believe it got the frozen contract.
* Distances are computed **exactly** (brute force, chunked for memory), never
  approximately. No FAISS, no tree, no early exit, in the Week-1 code path.
* Chunking is a memory device only. It must not change the result: the test
  suite asserts that the chunked path is bitwise identical to the brute-force
  path for the same input.

The function ``min_distances`` is the single definition of the distance from a
query patch to a set of reference patches. Everything else in this repository
(the full-bank score, the interval bounds, the screening decision) is expressed
in terms of it or of a conservative bound on it.
"""

from __future__ import annotations

import numpy as np

# Roughly 64 MiB of float32 per distance chunk: keeps peak RAM predictable on
# the 16 GB target machine without changing any numerical result.
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024


def as_features(x, name: str = "x") -> np.ndarray:
    """Validate and normalise a feature array to 2-D ``float32`` C-order."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return arr


def chunk_rows(n_ref: int, n_dim: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Number of query rows per chunk so that one distance block stays small."""
    bytes_per_row = max(1, n_ref * n_dim * 4)
    return max(1, int(chunk_bytes // bytes_per_row))


def pairwise_distances(
    queries: np.ndarray,
    references: np.ndarray,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> np.ndarray:
    """Exact Euclidean distance matrix, shape ``(len(queries), len(references))``.

    Uses the ``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b`` expansion, which is fast but
    slightly less accurate near zero. ``min_distances`` therefore *also* refines
    the winning candidates with a direct ``(a-b)`` difference; see there.
    """
    q = as_features(queries, "queries")
    r = as_features(references, "references")
    if q.shape[1] != r.shape[1]:
        raise ValueError(f"dimension mismatch: queries {q.shape[1]} vs references {r.shape[1]}")

    r_sq = np.einsum("ij,ij->i", r, r)
    out = np.empty((q.shape[0], r.shape[0]), dtype=np.float32)
    step = chunk_rows(r.shape[0], r.shape[1], chunk_bytes)
    for start in range(0, q.shape[0], step):
        stop = min(start + step, q.shape[0])
        block = q[start:stop]
        sq = r_sq[None, :] + np.einsum("ij,ij->i", block, block)[:, None]
        sq = sq - 2.0 * (block @ r.T)
        np.maximum(sq, 0.0, out=sq)
        out[start:stop] = np.sqrt(sq, dtype=np.float32)
    return out


def min_distances(
    queries: np.ndarray,
    references: np.ndarray,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    refine: bool = True,
    refine_candidates: int = 4,
) -> np.ndarray:
    """Exact ``min_m ||q - m||`` for every query, shape ``(len(queries),)``.

    The expansion form is used only to *rank* candidates cheaply. The reported
    value is then the direct Euclidean norm of the difference to the closest few
    ranked candidates, which removes the cancellation error of the expansion.

    ``refine_candidates`` controls how many top-ranked candidates are re-scored
    directly. Refining more than one matters because the expansion form can
    mis-order two references that are genuinely almost equidistant; refining only
    the expansion's argmin would then return a value slightly larger than the true
    minimum, i.e. it would *overstate* the reference score. Four is far more than
    enough in practice and costs ``O(4*P*D)``.

    Set ``refine=False`` only for a deliberate accuracy ablation; the unrefined
    value is not safe to use in any bound.
    """
    q = as_features(queries, "queries")
    r = as_features(references, "references")
    if q.shape[1] != r.shape[1]:
        raise ValueError(f"dimension mismatch: queries {q.shape[1]} vs references {r.shape[1]}")
    if r.shape[0] == 0:
        raise ValueError("references is empty; a minimum over an empty set is undefined")

    r_sq = np.einsum("ij,ij->i", r, r)
    out = np.empty(q.shape[0], dtype=np.float32)
    step = chunk_rows(r.shape[0], r.shape[1], chunk_bytes)
    n_cand = max(1, min(int(refine_candidates), r.shape[0])) if refine else 1

    for start in range(0, q.shape[0], step):
        stop = min(start + step, q.shape[0])
        block = q[start:stop]
        sq = r_sq[None, :] + np.einsum("ij,ij->i", block, block)[:, None]
        sq = sq - 2.0 * (block @ r.T)
        np.maximum(sq, 0.0, out=sq)

        if not refine:
            idx = np.argmin(sq, axis=1)
            out[start:stop] = np.sqrt(sq[np.arange(idx.shape[0]), idx], dtype=np.float32)
            continue

        cand = np.argpartition(sq, n_cand - 1, axis=1)[:, :n_cand]
        order = np.argsort(np.take_along_axis(sq, cand, axis=1), axis=1)
        cand = np.take_along_axis(cand, order, axis=1)
        refined = np.empty(cand.shape, dtype=np.float32)
        for c in range(cand.shape[1]):
            diff = block - r[cand[:, c]]
            refined[:, c] = np.sqrt(np.einsum("ij,ij->i", diff, diff), dtype=np.float32)
        out[start:stop] = refined.min(axis=1)
    return out


def min_distances_to_centers(
    queries: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Convenience wrapper: exact distances to a small center set (K << N)."""
    return min_distances(queries, centers, chunk_bytes=DEFAULT_CHUNK_BYTES)


def float32_epsilon() -> float:
    """Unit roundoff of float32 (2**-24)."""
    return float(np.finfo(np.float32).eps)
