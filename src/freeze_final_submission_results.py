"""Freeze only the active, post-causality-correction submission evidence.

The repository contains exploratory and superseded outputs.  A publication lock
must therefore be allowlist based: copying ``results/*`` would mix incompatible
simulator trajectories and old observer names into the evidence package.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.create_submission_review_bundle import EVIDENCE_FILES, SOURCE_FILES


LOCK_ROOT = PROJECT_ROOT / "results_locked" / "post_causality_ate_submission"
CANONICAL_DATA = "data/locked/sonderborg_processed_18703.csv"
ACTIVE_DATA = (
    CANONICAL_DATA,
    "data/locked/sonderborg_processed_18703_era5_land.csv",
    "data/external_weather/sonderborg_era5_land_2016_2019_hourly.csv",
)
ACTIVE_PAPER = (
    "paper/main_ate_submission_candidate.tex",
    "paper/supplementary_material.tex",
    "paper/references.bib",
    "paper/highlights_ate.txt",
    "paper/cover_letter_ate_draft.tex",
    "paper/compile_submission.ps1",
    "paper/SUBMISSION_METADATA_REQUIRED.md",
)
ACTIVE_RESULT_EXTRA = (
    "final_integrity_audit.csv",
    "final_integrity_audit.txt",
    "final_integrity_active_file_manifest.csv",
    "final_integrity_active_file_manifest.json",
    "full_dependent_regeneration_manifest.csv",
    "full_dependent_regeneration_protocol.json",
    "repeated_seed_protocol.json",
    "repeated_seed_raw_metrics.csv",
    "repeated_seed_statistics.csv",
    "repeated_seed_completeness_audit.csv",
    "repeated_seed_checkpoint_audit.csv",
    "second_chronological_window_metrics.csv",
    "second_chronological_window_summary.csv",
    "second_chronological_window_protocol.json",
    "verification_campaign_status.csv",
    "verification_campaign_status.json",
)
CHECKPOINT_PATTERNS = (
    "seed_*_best.pt",
    "mw_*_best.pt",
)
HISTORY_PATTERNS = ("seed_*_training_history.csv", "mw_*_training_history.csv")
FORBIDDEN_LOCK_TOKENS = (
    "legacy_audits",
    "superseded",
    "baseline_comparison_with_enkf",
    "enkf_baseline",
    "table_enkf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(relative: str, copied: list[Path]) -> None:
    source = PROJECT_ROOT / relative
    if not source.is_file():
        return
    destination = LOCK_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(destination)


def _tex_dependencies(relative: str) -> set[str]:
    """Return active table/figure dependencies already curated by the TeX source."""
    import re

    source = PROJECT_ROOT / relative
    text = source.read_text(encoding="utf-8")
    dependencies: set[str] = set()
    for item in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = (source.parent / (item if item.endswith(".tex") else f"{item}.tex")).resolve()
        if candidate.is_file():
            dependencies.add(candidate.relative_to(PROJECT_ROOT).as_posix())
    for item in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        candidates = [source.parent / item, PROJECT_ROOT / item]
        if not Path(item).suffix:
            candidates += [candidate.with_suffix(".pdf") for candidate in list(candidates)]
        candidate = next((path.resolve() for path in candidates if path.is_file()), None)
        if candidate is not None:
            dependencies.add(candidate.relative_to(PROJECT_ROOT).as_posix())
    return dependencies


def freeze_final_submission_results() -> pd.DataFrame:
    if LOCK_ROOT.exists():
        shutil.rmtree(LOCK_ROOT)
    LOCK_ROOT.mkdir(parents=True)
    copied: list[Path] = []

    active = {*ACTIVE_DATA, *ACTIVE_PAPER, *SOURCE_FILES}
    active.update(f"results/{name}" for name in {*EVIDENCE_FILES, *ACTIVE_RESULT_EXTRA})
    # Archive-provenance helpers remain in the working repository but are
    # intentionally outside the current-evidence lock because the integrity
    # policy rejects every path carrying a superseded-evidence token.
    active.discard("results/superseded_archive_status.txt")
    active.discard("src/hash_superseded_archive.py")
    active.update(_tex_dependencies("paper/main_ate_submission_candidate.tex"))
    active.update(_tex_dependencies("paper/supplementary_material.tex"))
    active.update(_tex_dependencies("paper/cover_letter_ate_draft.tex"))
    active.update(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "config").rglob("*")
        if path.is_file()
    )
    for pattern in (*CHECKPOINT_PATTERNS, *HISTORY_PATTERNS):
        active.update(path.relative_to(PROJECT_ROOT).as_posix() for path in (PROJECT_ROOT / "results").glob(pattern))

    for relative in sorted(active):
        lowered = relative.lower()
        if any(token in lowered for token in FORBIDDEN_LOCK_TOKENS):
            raise RuntimeError(f"Forbidden legacy artifact entered active allowlist: {relative}")
        _copy(relative, copied)

    rows = []
    for path in sorted(set(copied)):
        relative = path.relative_to(LOCK_ROOT).as_posix()
        rows.append({"locked_relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    inventory = pd.DataFrame(rows)
    inventory.to_csv(LOCK_ROOT / "final_submission_locked_inventory.csv", index=False)
    (LOCK_ROOT / "manifest_sha256.txt").write_text(
        "\n".join(f"{row['sha256']}  {row['locked_relative_path']}" for row in rows) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "post-causality-correction active evidence only",
        "canonical_dataset": CANONICAL_DATA,
        "canonical_rows": 18703,
        "canonical_sha256": "35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e",
        "file_count": len(rows),
        "excluded_by_design": list(FORBIDDEN_LOCK_TOKENS),
    }
    (LOCK_ROOT / "LOCK_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (LOCK_ROOT / "README_LOCKED_FINAL_SUBMISSION.md").write_text(
        "# Locked post-causality ATE evidence\n\n"
        "This allowlist-based package contains only active evidence generated after the full-state causality correction. "
        "Legacy audits, superseded trajectories, and outputs carrying the former observer name are excluded. "
        "The canonical input has 18,703 rows and the SHA-256 recorded in `LOCK_METADATA.json`.\n",
        encoding="utf-8",
    )
    report = PROJECT_ROOT / "results" / "final_submission_lock_report.txt"
    report.write_text(
        f"Final post-causality ATE lock\nstatus: PASS\nlocked_folder: {LOCK_ROOT}\nlocked_file_count: {len(rows)}\n",
        encoding="utf-8",
    )
    inventory.to_csv(PROJECT_ROOT / "results" / "final_submission_locked_inventory.csv", index=False)
    return inventory


if __name__ == "__main__":
    frame = freeze_final_submission_results()
    print(f"Locked {len(frame)} active files into {LOCK_ROOT}")
