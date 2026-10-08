"""master-research: decision-preserving edge screening over a compressed
normal-patch memory bank.

Week-1 scope (see docs/plan/scope_and_claims.md):

* exact Euclidean distance primitives,
* conservative lower/upper bounds from observed centers and coverage radii,
* the four-branch screening/escalation rule with an explicit UNRESOLVED state,
* version identifiers that make a decision-preservation claim falsifiable.

Nothing in this package trains a model, and nothing claims a new theorem.
"""

from .bank import Bank, greedy_coreset, random_subset, subset_size
from .bounds import (
    CoverageIndex,
    NumericalGuard,
    PatchBounds,
    aggregate_max,
    build_coverage_index,
    image_bounds,
    patch_bounds,
)
from .distances import min_distances, pairwise_distances
from .screening import Decision, FullBankDetector, ScreenOutcome, ScreenResult, screen_image, summarise_screening
from .versioning import VersionID, hash_bank, hash_weights, make_version_id

__all__ = [
    "Bank",
    "CoverageIndex",
    "Decision",
    "FullBankDetector",
    "NumericalGuard",
    "PatchBounds",
    "ScreenOutcome",
    "ScreenResult",
    "VersionID",
    "aggregate_max",
    "build_coverage_index",
    "greedy_coreset",
    "hash_bank",
    "hash_weights",
    "image_bounds",
    "make_version_id",
    "min_distances",
    "pairwise_distances",
    "patch_bounds",
    "random_subset",
    "screen_image",
    "subset_size",
    "summarise_screening",
]

__version__ = "0.1.0-week1"
