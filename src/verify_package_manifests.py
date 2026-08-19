"""Verify SHA-256 manifests in the source archive and delivery ZIP."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "package_manifest_verification.txt"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify(zip_path: Path, manifest_member: str, prefix: str) -> tuple[int, list[str]]:
    failures: list[str] = []
    with zipfile.ZipFile(zip_path) as zipped:
        lines = zipped.read(manifest_member).decode("utf-8").splitlines()
        checked = 0
        for line in lines:
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            member = f"{prefix}/{relative}" if prefix else relative
            try:
                actual = _sha256(zipped.read(member))
            except KeyError:
                failures.append(f"missing: {member}")
                continue
            checked += 1
            if actual != expected:
                failures.append(f"hash mismatch: {member}")
    return checked, failures


def main() -> int:
    review = ROOT / "paper" / "real_data_assisted_dh_review_archive.zip"
    delivery = ROOT / "final_ate_submission_package.zip"
    review_count, review_failures = _verify(
        review,
        "submission_review_bundle/manifest_sha256.txt",
        "submission_review_bundle",
    )
    delivery_count, delivery_failures = _verify(
        delivery,
        "final_ate_submission_package/SHA256SUMS.txt",
        "final_ate_submission_package",
    )
    failures = review_failures + delivery_failures
    lines = [
        "Package manifest verification",
        "",
        f"review archive checked entries: {review_count}",
        f"delivery ZIP checked entries: {delivery_count}",
        f"status: {'PASS' if not failures else 'FAIL'}",
    ]
    lines.extend(f"- {item}" for item in failures)
    RESULT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(RESULT)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
