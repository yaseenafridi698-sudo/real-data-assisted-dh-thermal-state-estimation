"""Verify fonts in the vector PDFs imported by the active ATE TeX sources.

The check is deliberately limited to active figures referenced by the current
main manuscript and supplement.  Archived exploratory figures are not part of
the submission asset set and are therefore not allowed to affect a production
preflight result.
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
import zlib
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # The locked Torch runtime intentionally has no PDF dependency.
    PdfReader = None


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
TEX_SOURCES = ("main_ate_submission_candidate.tex", "supplementary_material.tex")


def _deref(value):
    return value.get_object() if hasattr(value, "get_object") else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_figures() -> list[tuple[str, Path]]:
    active: list[tuple[str, Path]] = []
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    for tex_name in TEX_SOURCES:
        tex = PAPER / tex_name
        text = tex.read_text(encoding="utf-8", errors="ignore")
        for item in pattern.findall(text):
            source = PAPER / item
            if source.suffix.lower() != ".pdf":
                source = source.with_suffix(".pdf")
            active.append((tex_name, source))
    return active


def _font_records_pypdf(path: Path) -> tuple[list[dict[str, str]], str]:
    reader = PdfReader(path)
    records: list[dict[str, str]] = []
    for page in reader.pages:
        resources = _deref(page.get("/Resources", {}))
        fonts = _deref(resources.get("/Font", {})) if resources else {}
        content = page.get_contents().get_data().decode("latin-1", errors="ignore")
        text_blocks = re.findall(r"BT(.*?)ET", content, flags=re.DOTALL)
        for resource_name, reference in fonts.items():
            font = _deref(reference)
            subtype = str(font.get("/Subtype", "unknown"))
            base_font = str(font.get("/BaseFont", "unknown"))
            descriptor = _deref(font.get("/FontDescriptor")) if font.get("/FontDescriptor") else None
            if subtype == "/Type0" and font.get("/DescendantFonts"):
                descendant = _deref(font["/DescendantFonts"][0])
                descriptor = _deref(descendant.get("/FontDescriptor")) if descendant.get("/FontDescriptor") else descriptor
                base_font = str(descendant.get("/BaseFont", base_font))
            embedded = bool(descriptor and any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")))
            resource_pattern = re.escape(str(resource_name)) + r"\\s+[-+0-9.]+\\s+Tf"
            text_used = any(
                re.search(resource_pattern, block) and re.search(r"(?:\\bTj\\b|\\bTJ\\b|\\'|\\\")", block)
                for block in text_blocks
            )
            records.append(
                {
                    "resource": str(resource_name),
                    "subtype": subtype,
                    "base_font": base_font,
                    "embedded": str(embedded).lower(),
                    "type3": str(subtype == "/Type3").lower(),
                    "text_used": str(text_used).lower(),
                }
            )
    return records, "ok"


def _font_records_raw(path: Path) -> tuple[list[dict[str, str]], str]:
    """Conservative dependency-free audit of PDF font object dictionaries.

    PDF object dictionaries are outside compressed streams in Matplotlib PDF
    output. Every discovered font is treated as used. Type0 fonts are resolved
    through their descendant CID font before checking the FontDescriptor.
    """
    text = path.read_bytes().decode("latin-1", errors="ignore")
    objects = {
        int(match.group(1)): match.group(2)
        for match in re.finditer(r"(?ms)(\d+)\s+\d+\s+obj\s*(.*?)\s*endobj", text)
    }
    content_streams: list[str] = []
    for body in objects.values():
        stream_match = re.search(r"(?s)stream\r?\n(.*?)\r?\nendstream", body)
        if not stream_match:
            continue
        payload = stream_match.group(1).encode("latin-1", errors="ignore")
        if "/FlateDecode" in body:
            try:
                payload = zlib.decompress(payload)
            except zlib.error:
                continue
        content_streams.append(payload.decode("latin-1", errors="ignore"))

    def referenced(body: str, key: str) -> str:
        match = re.search(rf"/{key}\s+(\d+)\s+\d+\s+R", body)
        return objects.get(int(match.group(1)), "") if match else ""

    records: list[dict[str, str]] = []
    for object_id, body in objects.items():
        if not re.search(r"/Type\s*/Font(?:\s|/|>>)", body):
            continue
        subtype_match = re.search(r"/Subtype\s*/([A-Za-z0-9]+)", body)
        subtype = f"/{subtype_match.group(1)}" if subtype_match else "unknown"
        base_match = re.search(r"/BaseFont\s*/([^\s/<>{}\[\]()]+)", body)
        base_font = f"/{base_match.group(1)}" if base_match else f"object_{object_id}"
        descriptor_source = body
        if subtype == "/Type0":
            descendant_match = re.search(r"/DescendantFonts\s*\[\s*(\d+)\s+\d+\s+R", body)
            if descendant_match:
                descriptor_source = objects.get(int(descendant_match.group(1)), "")
                descendant_base = re.search(r"/BaseFont\s*/([^\s/<>{}\[\]()]+)", descriptor_source)
                if descendant_base:
                    base_font = f"/{descendant_base.group(1)}"
        descriptor = referenced(descriptor_source, "FontDescriptor")
        if not descriptor:
            direct = re.search(r"(?s)/FontDescriptor\s*<<(.*?)>>", descriptor_source)
            descriptor = direct.group(1) if direct else ""
        embedded = bool(re.search(r"/FontFile(?:2|3)?(?:\s|/)", descriptor))
        resource_names: set[str] = set()
        reference_pattern = re.compile(rf"/([A-Za-z0-9_.+\-]+)\s+{object_id}\s+\d+\s+R")
        for owner_body in objects.values():
            resource_names.update(reference_pattern.findall(owner_body))
        text_used = any(
            re.search(rf"/{re.escape(name)}\s+[-+0-9.]+\s+Tf", stream)
            for name in resource_names
            for stream in content_streams
        )
        records.append(
            {
                "resource": f"object_{object_id}",
                "subtype": subtype,
                "base_font": base_font,
                "embedded": str(embedded).lower(),
                "type3": str(subtype == "/Type3").lower(),
                "text_used": str(text_used).lower(),
            }
        )
    return records, "raw PDF object audit"


def _font_records(path: Path) -> tuple[list[dict[str, str]], str]:
    if PdfReader is not None:
        return _font_records_pypdf(path)
    return _font_records_raw(path)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for tex_name, paper_path in _active_figures():
        source_path = ROOT / "figures" / "final" / paper_path.name
        row = {
            "tex_source": tex_name,
            "figure": paper_path.name,
            "paper_figure_path": paper_path.relative_to(ROOT).as_posix(),
            "source_figure_path": source_path.relative_to(ROOT).as_posix(),
            "exists": str(paper_path.exists()).lower(),
            "mirror_hash_match": "false",
            "type3_present": "unknown",
            "unembedded_font_present": "unknown",
            "font_records": "",
            "status": "FAIL",
        }
        if not paper_path.exists() or not source_path.exists():
            row["font_records"] = "missing PDF asset"
            rows.append(row)
            continue
        row["mirror_hash_match"] = str(_sha256(paper_path) == _sha256(source_path)).lower()
        try:
            records, _ = _font_records(paper_path)
            type3 = any(item["type3"] == "true" for item in records)
            unembedded = any(item["embedded"] == "false" and item["text_used"] == "true" for item in records)
            row["type3_present"] = str(type3).lower()
            row["unembedded_font_present"] = str(unembedded).lower()
            row["font_records"] = "; ".join(
                f"{item['base_font']} ({item['subtype']}, embedded={item['embedded']}, text_used={item['text_used']})" for item in records
            ) or "outlined/vector drawing with no text font resource"
            row["status"] = "PASS" if not type3 and not unembedded and row["mirror_hash_match"] == "true" else "FAIL"
        except Exception as exc:  # pragma: no cover - production preflight must report parse failures.
            row["font_records"] = f"PDF parse error: {exc}"
        rows.append(row)

    fieldnames = list(rows[0]) if rows else ["tex_source", "figure", "status"]
    csv_path = RESULTS / "figure_font_preflight.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = bool(rows) and all(row["status"] == "PASS" for row in rows)
    report_lines = [
        "Active vector-figure font preflight",
        "",
        "Scope: PDFs referenced by main_ate_submission_candidate.tex and supplementary_material.tex.",
        "Method: pypdf when available, otherwise a conservative raw-PDF object audit; Type 3 fonts and font descriptors without embedded FontFile streams fail. Source/paper mirrors must have identical SHA-256 hashes.",
        f"active_figure_count: {len(rows)}",
        f"status: {'PASS' if passed else 'FAIL'}",
        "",
    ]
    for row in rows:
        report_lines.append(
            f"[{row['status']}] {row['figure']}: Type3={row['type3_present']}; "
            f"unembedded={row['unembedded_font_present']}; mirror_match={row['mirror_hash_match']}"
        )
    (RESULTS / "figure_font_preflight.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
