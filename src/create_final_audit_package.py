"""Create the synchronized final review package with current PDFs and sources."""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission_review_bundle"
OUT = ROOT / "final_ate_submission_package.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_final_audit_package() -> Path:
    if not BUNDLE.exists():
        raise FileNotFoundError("submission_review_bundle is missing; rebuild the source reviewer archive first")
    staging = ROOT / "final_ate_submission_package"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(BUNDLE, staging / "source")

    required = [
        ROOT / "FINAL_AUDIT_REPORT.md",
        ROOT / "paper" / "main_ate_submission_candidate.pdf",
        ROOT / "paper" / "supplementary_material.pdf",
        ROOT / "results" / "latex_compile_report.txt",
        ROOT / "results" / "font_preflight_report.txt",
        ROOT / "results" / "paper_quality_gate_report_final.txt",
        ROOT / "results" / "audit_submission_consistency.txt",
        ROOT / "results" / "audit_targeted_remaining_issues.txt",
        ROOT / "results" / "gap_handling_audit.csv",
        ROOT / "results" / "gap_handling_audit.json",
    ]
    for source in required:
        if not source.exists():
            raise FileNotFoundError(f"Required final-audit artifact is missing: {source}")
        destination = staging / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    (staging / "PACKAGE_STATUS.txt").write_text(
        "The scientific evidence, sources, figures, and current PDFs are synchronized and audited.\n"
        "The package is pending factual author metadata, funding, competing-interest, and CRediT declarations.\n"
        "Do not submit with placeholder author or affiliation fields.\n",
        encoding="utf-8",
    )

    files = sorted(path for path in staging.rglob("*") if path.is_file())
    manifest_lines = [f"{_sha256(path)}  {path.relative_to(staging).as_posix()}" for path in files]
    (staging / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for source in sorted(path for path in staging.rglob("*") if path.is_file()):
            zipped.write(source, source.relative_to(staging.parent).as_posix())
    shutil.rmtree(staging)
    return OUT


if __name__ == "__main__":
    print(create_final_audit_package())
