from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT

try:
    from src.utils import ensure_dir
except Exception:
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


def _refs(tex_path: Path) -> tuple[list[str], list[str]]:
    text = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    inputs = re.findall(r"\\input\{([^}]+)\}", text)
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    return inputs, graphics


def _resolve_ref(base: Path, ref: str, default_suffix: str) -> Path:
    path = base / ref
    if path.suffix:
        return path
    return path.with_suffix(default_suffix)


def _missing_refs(tex_name: str) -> tuple[list[str], list[str]]:
    tex_path = PROJECT_ROOT / "paper" / tex_name
    inputs, graphics = _refs(tex_path)
    missing_tables = []
    missing_figures = []
    for item in inputs:
        path = _resolve_ref(tex_path.parent, item, ".tex")
        if not path.exists():
            missing_tables.append(str(path))
    for item in graphics:
        path = _resolve_ref(tex_path.parent, item, ".pdf")
        if not path.exists():
            missing_figures.append(str(path))
    return missing_tables, missing_figures


def _count_log_patterns(log_path: Path) -> tuple[int, int, bool]:
    if not log_path.exists():
        return 0, 0, False
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    undefined = len(re.findall(r"(Citation .* undefined|Reference .* undefined|undefined references)", text, flags=re.IGNORECASE))
    overfull = len(re.findall(r"Overfull \\hbox", text))
    fatal = "Fatal error" in text or "Emergency stop" in text
    return undefined, overfull, fatal


def _pdf_page_count(path: Path) -> int | str:
    if not path.exists():
        return "not available"
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        data = path.read_bytes()
        count = len(re.findall(rb"/Type\s*/Page\b", data))
        return count if count else "not available"
    except Exception:
        return "not available"


def write_latex_compile_report() -> Path:
    out_dir = ensure_dir(PROJECT_ROOT / "results")
    main_tex = "main_ate_submission_candidate.tex"
    supp_tex = "supplementary_material.tex"
    main_pdf = PROJECT_ROOT / "paper" / "main_ate_submission_candidate.pdf"
    supp_pdf = PROJECT_ROOT / "paper" / "supplementary_material.pdf"
    main_tex_path = PROJECT_ROOT / "paper" / main_tex
    supp_tex_path = PROJECT_ROOT / "paper" / supp_tex
    main_missing_tables, main_missing_figures = _missing_refs(main_tex)
    supp_missing_tables, supp_missing_figures = _missing_refs(supp_tex)
    undefined_main, overfull_main, fatal_main = _count_log_patterns(PROJECT_ROOT / "paper" / "main_ate_submission_candidate.log")
    undefined_supp, overfull_supp, fatal_supp = _count_log_patterns(PROJECT_ROOT / "paper" / "supplementary_material.log")
    pdflatex = shutil.which("pdflatex")
    tectonic = shutil.which("tectonic")
    main_pdf_older = main_pdf.exists() and main_tex_path.exists() and main_pdf.stat().st_mtime < main_tex_path.stat().st_mtime
    supp_pdf_older = supp_pdf.exists() and supp_tex_path.exists() and supp_pdf.stat().st_mtime < supp_tex_path.stat().st_mtime
    if main_pdf.exists() and supp_pdf.exists() and not pdflatex and not tectonic:
        status = "PDFs exist, but not recompiled in this run because no LaTeX engine was found on PATH"
    elif main_pdf.exists() and supp_pdf.exists():
        status = "compiled: main and supplementary PDFs exist"
    elif main_pdf.exists():
        status = "partially compiled: main PDF exists, supplementary PDF missing"
    elif not pdflatex and not tectonic:
        status = "not compiled: no LaTeX engine found on PATH"
    elif fatal_main:
        status = "compile failed: fatal LaTeX error in main manuscript log"
    else:
        status = "not compiled"
    lines = [
        "LaTeX compile report",
        "",
        f"compilation status: {status}",
        f"pdflatex path: {pdflatex or 'not found'}",
        f"tectonic path: {tectonic or 'not found on PATH; bundled runtime executable may have been used'}",
        f"main manuscript: paper/{main_tex}",
        f"main PDF path: {main_pdf}",
        f"main PDF exists: {main_pdf.exists()}",
        f"main PDF older than TeX source: {main_pdf_older}",
        f"supplementary manuscript: paper/{supp_tex}",
        f"supplementary PDF path: {supp_pdf}",
        f"supplementary PDF exists: {supp_pdf.exists()}",
        f"supplementary PDF older than TeX source: {supp_pdf_older}",
        f"page count main: {_pdf_page_count(main_pdf)}",
        f"page count supplementary: {_pdf_page_count(supp_pdf)}",
        f"undefined citations/references count: {undefined_main + undefined_supp}",
        f"overfull hbox count: {overfull_main + overfull_supp}",
        f"LaTeX fatal error found in logs: {fatal_main or fatal_supp}",
        f"missing figure count: {len(main_missing_figures) + len(supp_missing_figures)}",
        f"missing table/input count: {len(main_missing_tables) + len(supp_missing_tables)}",
    ]
    if main_missing_figures or supp_missing_figures:
        lines.append("missing figures:")
        lines.extend([f"- {item}" for item in main_missing_figures + supp_missing_figures])
    if main_missing_tables or supp_missing_tables:
        lines.append("missing table/input files:")
        lines.extend([f"- {item}" for item in main_missing_tables + supp_missing_tables])
    if not main_pdf.exists():
        lines.append("")
        lines.append("A local main PDF was not produced. Compile in Overleaf, TeX Live, MiKTeX, or Tectonic.")
    report = out_dir / "latex_compile_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "latex_compile_report_final_improved.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    path = write_latex_compile_report()
    print(path)
