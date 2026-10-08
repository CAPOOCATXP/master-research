#!/usr/bin/env python
"""Immediate Assignment #4 — draft reference-fit / development / calibration manifests.

The professor asked for:

    "Draft separate reference-fit, development and final-calibration image
     manifests for the three pilot categories; check that no original image
     appears in more than one role."

Why this matters more than it looks: if the threshold is calibrated on an image
that also sits in the reference bank, the reported false-alarm rate is optimistically
biased, and *every* downstream disagreement number inherits that bias. The role
split is therefore a correctness requirement, not bookkeeping.

Behaviour
---------
* If ``data/raw/mvtec_ad/<category>/train/good/`` exists, the script enumerates the
  real files and writes manifests whose ``original_id`` values are real.
* If the data is absent (the current state — MVTec AD requires registration), the
  script writes **PROVISIONAL** manifests using the category sizes published with
  the dataset and placeholder ids. Every file records
  ``provenance=PROVISIONAL_NO_DATA``, so a provisional manifest can never be
  mistaken for a real one. Re-run after download to replace them.

The split is deterministic given ``--seed`` and is validated for:
complete coverage of the normal training set, pairwise role disjointness, and
non-empty roles.

Usage
-----
    .venv/bin/python scripts/03_build_manifests.py
    .venv/bin/python scripts/03_build_manifests.py --mvtec-root /path/to/mvtec
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PILOT_CATEGORIES = ("bottle", "tile", "screw")

# Normal training-image counts as published with MVTec AD. Used ONLY to size the
# provisional manifests; the script re-derives everything from the filesystem
# once the data is present, and `env/data_inventory.md` records the verification
# status.
PUBLISHED_TRAIN_GOOD_COUNTS = {"bottle": 209, "tile": 230, "screw": 320}

ROLE_FRACTIONS = {"reference_fit": 0.60, "development": 0.20, "final_calibration": 0.20}
FEW_SHOT_COUNTS = (1, 4, 16)
FEW_SHOT_SEEDS = (0, 1, 2, 3, 4)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_normal_train_images(mvtec_root: Path, category: str) -> list:
    """Return sorted real normal-training image paths, or [] if unavailable."""
    for candidate in (
        mvtec_root / category / "train" / "good",
        mvtec_root / "mvtec" / category / "train" / "good",
    ):
        if candidate.is_dir():
            files = sorted(p for p in candidate.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if files:
                return files
    return []


def split_ids(ids: list, seed: int) -> dict:
    """Deterministic 60/20/20 split of normal images into the three roles.

    Rounding is handled by taking the largest remainder: the three role sizes sum
    to exactly ``len(ids)`` so that no normal image is left unassigned, and every
    image is used in exactly one role.
    """
    n = len(ids)
    if n < 5:
        raise ValueError(f"need at least 5 normal images to form three non-empty roles, got {n}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    raw = {role: frac * n for role, frac in ROLE_FRACTIONS.items()}
    sizes = {role: int(np.floor(v)) for role, v in raw.items()}
    remainder = n - sum(sizes.values())
    for role in sorted(raw, key=lambda r: (-(raw[r] - sizes[r]), r))[:remainder]:
        sizes[role] += 1
    for role in sizes:
        sizes[role] = max(1, sizes[role])

    while sum(sizes.values()) > n:  # shrink the largest role if min-1 pushed us over
        largest = max(sizes, key=lambda r: sizes[r])
        sizes[largest] -= 1

    out, cursor = {}, 0
    for role in ROLE_FRACTIONS:
        k = sizes[role]
        out[role] = [ids[i] for i in order[cursor : cursor + k]]
        cursor += k
    assert cursor == n, "internal error: not every normal image was assigned"
    return out


def select_few_shot(reference_fit: list, seed: int) -> dict:
    """Sample 1/4/16 references **from reference-fit only**.

    The plan is explicit that additional development/calibration normals are
    counted separately: "few-shot references" never means only those images were
    used in total.
    """
    rng = np.random.default_rng(10_000 + seed)
    out = {}
    for k in FEW_SHOT_COUNTS:
        if k > len(reference_fit):
            continue
        idx = np.sort(rng.choice(len(reference_fit), size=k, replace=False))
        out[str(k)] = [reference_fit[i] for i in idx]
    return out


def validate(records: list, roles: dict) -> dict:
    """The disjointness check the assignment asks for, plus coverage."""
    all_ids = [r["original_id"] for r in records]
    per_role = {role: [r["original_id"] for r in records if r["role"] == role] for role in roles}
    seen = Counter(all_ids)
    duplicated = sorted(i for i, c in seen.items() if c > 1)

    checks = {
        "total_images": len(all_ids),
        "role_counts": {role: len(v) for role, v in per_role.items()},
        "any_image_in_more_than_one_role": bool(duplicated),
        "duplicated_ids": duplicated[:10],
        "any_role_empty": any(len(v) == 0 for v in per_role.values()),
        "roles_sum_to_total": sum(len(v) for v in per_role.values()) == len(all_ids),
        "roles_pairwise_disjoint": (
            len(set(per_role["reference_fit"]) & set(per_role["development"])) == 0
            and len(set(per_role["reference_fit"]) & set(per_role["final_calibration"])) == 0
            and len(set(per_role["development"]) & set(per_role["final_calibration"])) == 0
        ),
        "all_roles_nonempty": all(len(v) > 0 for v in per_role.values()),
    }
    checks["PASS"] = bool(
        not checks["any_image_in_more_than_one_role"]
        and checks["roles_pairwise_disjoint"]
        and checks["roles_sum_to_total"]
        and checks["all_roles_nonempty"]
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", type=Path, default=REPO / "data" / "raw" / "mvtec_ad")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "manifests")
    ap.add_argument("--seed", type=int, default=0, help="split seed; must stay fixed for the whole project")
    ap.add_argument("--categories", nargs="+", default=list(PILOT_CATEGORIES))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"stage": "immediate_assignment_4_manifests", "seed": args.seed, "categories": {}, "provenance": None}

    for category in args.categories:
        real = find_normal_train_images(args.mvtec_root, category)
        if real:
            provenance = "REAL"
            ids = [str(p.relative_to(args.mvtec_root)) for p in real]
            note = ""
        else:
            provenance = "PROVISIONAL_NO_DATA"
            count = PUBLISHED_TRAIN_GOOD_COUNTS.get(category)
            if count is None:
                raise SystemExit(f"no data and no published count for category {category!r}")
            ids = [f"PROVISIONAL/{category}/train/good/{i:04d}.png" for i in range(count)]
            note = "placeholder ids; replace by re-running after MVTec AD download"

        if summary["provenance"] is None:
            summary["provenance"] = provenance
        elif summary["provenance"] != provenance:
            summary["provenance"] = "MIXED"

        roles = split_ids(ids, args.seed)
        records = []
        for role, role_ids in roles.items():
            for oid in role_ids:
                records.append(
                    {
                        "category": category,
                        "role": role,
                        "original_id": oid,
                        "split": "train/good",
                        "provenance": provenance,
                        "note": note,
                    }
                )

        checks = validate(records, roles)
        checks["expected_count"] = PUBLISHED_TRAIN_GOOD_COUNTS.get(category)
        checks["count_matches_published"] = (
            None if real else len(ids) == PUBLISHED_TRAIN_GOOD_COUNTS.get(category)
        )
        if checks["count_matches_published"] is False:
            raise SystemExit(f"count mismatch for {category}: {len(ids)} != published")

        csv_path = args.out / f"{category}_roles.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["category", "role", "original_id", "split", "provenance", "note"])
            writer.writeheader()
            writer.writerows(records)

        few_shot = select_few_shot(roles["reference_fit"], args.seed)
        fs_path = args.out / f"{category}_fewshot.csv"
        with fs_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["category", "shot", "seed", "original_id", "source_role", "provenance"])
            writer.writeheader()
            for seed in FEW_SHOT_SEEDS:
                sel = select_few_shot(roles["reference_fit"], seed)
                for shot, items in sel.items():
                    for oid in items:
                        writer.writerow(
                            {
                                "category": category,
                                "shot": shot,
                                "seed": seed,
                                "original_id": oid,
                                "source_role": "reference_fit",
                                "provenance": provenance,
                            }
                        )

        summary["categories"][category] = {
            "provenance": provenance,
            "n_normal_train": len(ids),
            "checks": checks,
            "csv": str(csv_path),
            "fewshot_csv": str(fs_path),
            "few_shot_pool_size": len(roles["reference_fit"]),
            "few_shot_available": {shot: (int(shot) <= len(roles["reference_fit"])) for shot in map(str, FEW_SHOT_COUNTS)},
        }
        print(
            f"[{category}] {provenance}: {len(ids)} normal images -> "
            f"reference_fit={checks['role_counts']['reference_fit']} "
            f"development={checks['role_counts']['development']} "
            f"final_calibration={checks['role_counts']['final_calibration']} | "
            f"disjoint={checks['roles_pairwise_disjoint']} PASS={checks['PASS']}"
        )

    summary["all_categories_pass"] = all(c["checks"]["PASS"] for c in summary["categories"].values())
    out_json = args.out / "manifest_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nsummary: {out_json}")
    print("RESULT:", "PASS" if summary["all_categories_pass"] else "FAIL")
    return 0 if summary["all_categories_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
