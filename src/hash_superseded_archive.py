"""Hash the pre-correction evidence archive without changing archived artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "superseded_pre_full_state_causality_20260807"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not ARCHIVE.is_dir():
        raise FileNotFoundError(ARCHIVE)
    manifest_path = ARCHIVE / "superseded_archive_manifest.csv"
    rows = []
    for path in sorted(ARCHIVE.rglob("*")):
        if path.is_file() and path != manifest_path:
            rows.append(
                {
                    "relative_path": path.relative_to(ARCHIVE).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "status": "superseded_pre_full_state_causality",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_path, index=False)
    (ROOT / "results" / "superseded_archive_status.txt").write_text(
        f"status: archived\narchive: {ARCHIVE}\nfile_count_excluding_manifest: {len(frame)}\nactive_publication_use: prohibited\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
