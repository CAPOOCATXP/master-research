"""Version identifiers.

The central systems claim of the project is *decision preservation*: the
screened decision must equal the decision the version-matched full bank would
have made. That claim is meaningless without a version contract, because a
screening index computed against encoder/preprocessing state A can silently be
compared against a full bank built under state B.

Every artifact therefore carries a :class:`VersionID`. Anything that changes the
feature space, the reference set, or the alarm threshold changes the id, and a
mismatch must produce ``UNRESOLVED`` rather than a guess.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .bank import serialize_bank


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_bank(bank: np.ndarray) -> str:
    """Content hash of a bank's float32 bytes. Order-sensitive by design."""
    return _sha256_hex(serialize_bank(bank))


def hash_weights(path_or_bytes) -> str:
    """Content hash of a weights file (path string/Path) or raw bytes."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return _sha256_hex(bytes(path_or_bytes))
    with open(path_or_bytes, "rb") as fh:
        digest = hashlib.sha256()
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class VersionID:
    """Immutable identity of a (encoder, preprocessing, bank, threshold) tuple."""

    encoder: str
    preprocessing: str
    bank_id: str
    bank_size: int
    threshold: float
    threshold_policy: str = ""

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return _sha256_hex(payload)[:16]

    def matches(self, other: "VersionID") -> bool:
        """Full equality on every field that can change a decision."""
        return self == other

    def mismatch_fields(self, other: "VersionID") -> list:
        a, b = asdict(self), asdict(other)
        return sorted(k for k in a if a[k] != b[k])

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.encoder}|{self.preprocessing}|{self.bank_id[:12]}|n={self.bank_size}|t={self.threshold:g}"


def make_version_id(
    encoder: str,
    preprocessing: str,
    bank: Optional[np.ndarray] = None,
    bank_id: Optional[str] = None,
    threshold: float = float("nan"),
    threshold_policy: str = "",
) -> VersionID:
    """Build a :class:`VersionID`, hashing the bank when one is supplied."""
    if bank is not None:
        bank_id = hash_bank(bank)
        bank_size = int(np.asarray(bank).shape[0])
    else:
        if bank_id is None:
            raise ValueError("provide either bank or bank_id")
        bank_size = -1
    return VersionID(
        encoder=encoder,
        preprocessing=preprocessing,
        bank_id=bank_id,
        bank_size=bank_size,
        threshold=float(threshold),
        threshold_policy=threshold_policy,
    )
