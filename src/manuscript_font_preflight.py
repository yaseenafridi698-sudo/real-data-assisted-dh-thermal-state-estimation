"""Audit fonts embedded in the compiled main and supplementary PDFs."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.figure_font_preflight import _font_records


PDFS = (
    PROJECT_ROOT / "paper" / "main_ate_submission_candidate.pdf",
    PROJECT_ROOT / "paper" / "supplementary_material.pdf",
)


def main() -> int:
    results = PROJECT_ROOT / "results"
    results.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in PDFS:
        row = {
            "pdf": path.relative_to(PROJECT_ROOT).as_posix(),
            "exists": str(path.is_file()).lower(),
            "font_resources": "0",
            "type3_present": "unknown",
            "used_unembedded_font_present": "unknown",
            "status": "FAIL",
        }
        if path.is_file():
            records, _ = _font_records(path)
            used = [record for record in records if record["text_used"] == "true"]
            type3 = any(record["type3"] == "true" for record in used)
            unembedded = any(record["embedded"] == "false" for record in used)
            row.update(
                {
                    "font_resources": str(len(records)),
                    "type3_present": str(type3).lower(),
                    "used_unembedded_font_present": str(unembedded).lower(),
                    "status": "PASS" if not type3 and not unembedded else "FAIL",
                }
            )
        rows.append(row)

    with (results / "manuscript_font_preflight.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = all(row["status"] == "PASS" for row in rows)
    lines = [
        "PDF font preflight report",
        "",
        "Method: direct PDF font-resource inspection; only fonts used by text operators are assessed.",
        "Acceptance: no used Type 3 font and no used font without an embedded font stream.",
        "",
    ]
    lines.extend(
        f"[{row['status']}] {row['pdf']}: resources={row['font_resources']}; "
        f"Type3={row['type3_present']}; used_unembedded={row['used_unembedded_font_present']}"
        for row in rows
    )
    lines.extend(["", f"status: {'PASS' if passed else 'FAIL'}"])
    (results / "font_preflight_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
