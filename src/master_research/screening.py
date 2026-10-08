"""The reference detector and the screening / escalation rule.

Two decisions must never be confused:

* ``S_M(x) > t`` — the **reference decision**, made by the frozen full bank.
* the **screened decision** — what the edge makes using only centers and radii,
  possibly after escalating to the full bank.

The research question is whether the second equals the first. It is not whether
either is *correct* about real defects: preservation cannot repair the reference
model's mistakes, and a preserved false alarm is still a false alarm.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bounds import CoverageIndex, NumericalGuard, aggregate_max
from .distances import as_features, min_distances
from .versioning import VersionID


class Decision(str, enum.Enum):
    """Outcome of a screening attempt."""

    NORMAL = "normal"
    ALARM = "alarm"
    UNRESOLVED = "unresolved"


class ScreenOutcome(str, enum.Enum):
    """How a :class:`Decision` was reached. Reported separately from the decision."""

    CERTIFIED_LOCAL = "certified_local"      # decided by the interval, no full-bank query
    ESCALATED_RESOLVED = "escalated_resolved"  # interval was ambiguous; full bank answered
    UNRESOLVED_NO_RESOLVER = "unresolved_no_resolver"
    UNRESOLVED_VERSION_MISMATCH = "unresolved_version_mismatch"
    UNRESOLVED_TIMEOUT = "unresolved_timeout"


@dataclass
class ScreenResult:
    """Per-image screening record. Every field is logged, none is inferred."""

    decision: Decision
    outcome: ScreenOutcome
    lower: float
    upper: float
    threshold: float
    exact_score: Optional[float] = None
    note: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.decision in (Decision.NORMAL, Decision.ALARM)

    @property
    def was_escalated(self) -> bool:
        return self.outcome is ScreenOutcome.ESCALATED_RESOLVED

    def as_row(self) -> dict:
        return {
            "decision": self.decision.value,
            "outcome": self.outcome.value,
            "lower": self.lower,
            "upper": self.upper,
            "threshold": self.threshold,
            "exact_score": self.exact_score,
            "note": self.note,
        }


@dataclass
class FullBankDetector:
    """The frozen reference detector: exact bank, fixed threshold, fixed version."""

    bank: np.ndarray
    threshold: float
    version: VersionID
    guard: NumericalGuard = field(default_factory=NumericalGuard)

    def __post_init__(self) -> None:
        self.bank = as_features(self.bank, "bank")
        self.threshold = float(self.threshold)

    def patch_scores(self, patches: np.ndarray) -> np.ndarray:
        """Exact ``s_M(q) = min_{m in M} d(q, m)`` for each query patch."""
        return min_distances(patches, self.bank)

    def image_score(self, patches: np.ndarray) -> float:
        """``S_M(x) = max_q s_M(q)`` — the max-score aggregation."""
        scores = self.patch_scores(patches)
        return float(aggregate_max(scores.reshape(1, -1))[0])

    def decide(self, image_score: float) -> Decision:
        """Alarm iff ``S_M(x) > t``. Strict inequality; equality is NORMAL."""
        return Decision.ALARM if image_score > self.threshold else Decision.NORMAL


def screen_image(
    lower: float,
    upper: float,
    threshold: float,
    resolver: Optional[FullBankDetector] = None,
    request_version: Optional[VersionID] = None,
    queried_patches: Optional[np.ndarray] = None,
) -> ScreenResult:
    """Apply the Part-A8 screening rule to one image's aggregated interval.

    1. ``U_x <= t``            -> NORMAL, locally certified.
    2. ``L_x >  t``            -> ALARM,  locally certified.
    3. otherwise               -> escalate to the full bank and adopt its decision.
    4. resolver missing / version mismatch / timeout
                               -> UNRESOLVED, never silently NORMAL.

    Note ``L_x > t`` is deliberately strict. When ``L_x == t`` the interval still
    straddles the boundary and the image must escalate, because the reference
    decision at exact equality is NORMAL.
    """
    lower = float(lower)
    upper = float(upper)
    threshold = float(threshold)
    if lower > upper:
        raise ValueError(f"invalid interval: lower {lower} > upper {upper}")

    # Branch 1: the whole interval is at or below the threshold.
    if upper <= threshold:
        return ScreenResult(
            decision=Decision.NORMAL,
            outcome=ScreenOutcome.CERTIFIED_LOCAL,
            lower=lower,
            upper=upper,
            threshold=threshold,
            note="U<=t",
        )

    # Branch 2: the whole interval is strictly above the threshold.
    if lower > threshold:
        return ScreenResult(
            decision=Decision.ALARM,
            outcome=ScreenOutcome.CERTIFIED_LOCAL,
            lower=lower,
            upper=upper,
            threshold=threshold,
            note="L>t",
        )

    # Branch 3: ambiguous -> full-bank query.
    if resolver is None:
        return ScreenResult(
            decision=Decision.UNRESOLVED,
            outcome=ScreenOutcome.UNRESOLVED_NO_RESOLVER,
            lower=lower,
            upper=upper,
            threshold=threshold,
            note="ambiguous and no resolver available",
        )
    if request_version is not None and not request_version.matches(resolver.version):
        fields = request_version.mismatch_fields(resolver.version)
        return ScreenResult(
            decision=Decision.UNRESOLVED,
            outcome=ScreenOutcome.UNRESOLVED_VERSION_MISMATCH,
            lower=lower,
            upper=upper,
            threshold=threshold,
            note=f"version mismatch in {fields}",
        )
    if queried_patches is None:
        raise ValueError("escalation requires queried_patches to score on the full bank")

    exact = resolver.image_score(queried_patches)
    return ScreenResult(
        decision=resolver.decide(exact),
        outcome=ScreenOutcome.ESCALATED_RESOLVED,
        lower=lower,
        upper=upper,
        threshold=threshold,
        exact_score=exact,
        note="ambiguous; resolved by full bank",
    )


def summarise_screening(results) -> dict:
    """Aggregate screening records into the rates the plan asks for."""
    results = list(results)
    n = len(results)
    if n == 0:
        return {"n": 0}
    from collections import Counter

    outcomes = Counter(r.outcome.value for r in results)
    decisions = Counter(r.decision.value for r in results)
    n_unresolved = decisions.get(Decision.UNRESOLVED.value, 0)
    n_local = outcomes.get(ScreenOutcome.CERTIFIED_LOCAL.value, 0)
    n_resolved = n - n_unresolved
    return {
        "n": n,
        "decisions": dict(decisions),
        "outcomes": dict(outcomes),
        "resolved_fraction": n_resolved / n,
        "locally_certified_fraction": n_local / n,
        "escalation_rate": outcomes.get(ScreenOutcome.ESCALATED_RESOLVED.value, 0) / n,
        "unresolved_rate": n_unresolved / n,
    }
