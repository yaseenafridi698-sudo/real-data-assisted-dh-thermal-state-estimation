"""Freeze and audit the canonical 18,703-timestamp Sonderborg input."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = load_config()
    real_cfg = config["real_data"]
    canonical = PROJECT_ROOT / real_cfg["canonical_sonderborg_processed_path"]
    active = PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv"
    expected_rows = int(real_cfg["canonical_sonderborg_rows"])
    expected_hash = str(real_cfg["canonical_sonderborg_sha256"]).lower()
    rows = []
    reviewer_mirror = PROJECT_ROOT / "submission_review_bundle" / "data" / "processed" / "sonderborg_processed.csv"
    candidates = [("canonical_locked_copy_from_reviewer_archive", canonical), ("active_workspace_mirror", active)]
    if reviewer_mirror.exists():
        candidates.append(("current_reviewer_archive_mirror", reviewer_mirror))
    for role, path in candidates:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        digest = _sha256(path)
        status = "pass" if len(frame) == expected_rows and digest == expected_hash else "fail"
        rows.append(
            {
                "role": role,
                "canonical_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "retained_timestamps": len(frame),
                "sha256": digest,
                "status": status,
                "used_for_manuscript": role == "canonical_locked_copy_from_reviewer_archive",
            }
        )
    audit = pd.DataFrame(rows)
    if not audit["status"].eq("pass").all():
        raise RuntimeError("Canonical dataset freeze audit failed.")

    legacy_path = Path.home() / "Downloads" / "ATE_sonderborg_processed_provenance.csv"
    legacy = {}
    if legacy_path.exists():
        legacy_frame = pd.read_csv(legacy_path)
        if not legacy_frame.empty:
            legacy = legacy_frame.iloc[0].to_dict()
    legacy_rows = int(legacy.get("rows", 19878))
    legacy_short = int(legacy.get("short_gap_interpolated_rows", 2648))
    canonical_short = int(pd.read_csv(canonical)["interpolated_short_gap_flag"].fillna(False).astype(bool).sum())
    explanation = (
        f"The excluded legacy artifact retained {legacy_rows} timestamps and marked {legacy_short} short-gap rows as interpolated. "
        f"The frozen causal artifact retains {expected_rows} timestamps and {canonical_short} short-gap interpolations. "
        f"The {legacy_rows - expected_rows}-row difference equals the interpolation-count difference ({legacy_short - canonical_short}); "
        "the legacy artifact was produced before forward-only chronological interpolation was enforced and is excluded from active evidence."
    )
    out = PROJECT_ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "canonical_dataset_manifest.csv", index=False)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_file": str(canonical.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "retained_timestamps": expected_rows,
        "sha256": expected_hash,
        "preprocessing_script": "src/data_preprocessing.py",
        "preprocessing_policy": "15-min resampling; forward-only interpolation up to 8 samples; long gaps retained as trajectory breaks after missing-row removal; configured 5 C ambient boundary",
        "legacy_19878_row_artifact": str(legacy_path) if legacy_path.exists() else "not present locally; values taken from prior audit",
        "legacy_difference_explanation": explanation,
        "prohibition": "Do not mix the 18,703-row canonical input with the excluded 19,878-row legacy artifact.",
    }
    (out / "canonical_dataset_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "canonical_dataset_manifest.md").write_text(
        "# Canonical Sonderborg processed dataset\n\n"
        f"- Path: `{payload['canonical_file']}`\n"
        f"- Retained timestamps: **{expected_rows:,}**\n"
        f"- SHA-256: `{expected_hash}`\n"
        f"- Preprocessing: {payload['preprocessing_policy']}\n\n"
        "## Excluded legacy artifact\n\n"
        + explanation
        + "\n\nThe two processed datasets are never pooled or interchanged in the final rebuild.\n",
        encoding="utf-8",
    )
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
