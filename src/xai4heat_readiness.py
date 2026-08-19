from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.data_registry import DataRegistry, list_available_raw_files, processed_file_exists


def _latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(df.columns) + r"}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row[c]) for c in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_xai4heat_readiness_report() -> pd.DataFrame:
    registry = DataRegistry()
    info = registry.get("xai4heat")
    raw_files = list_available_raw_files("xai4heat")
    processed = processed_file_exists("xai4heat")
    validation_csv = PROJECT_ROOT / "results" / "xai4heat_sparse_substation_validation.csv"

    if raw_files and processed and validation_csv.exists():
        status = "validation_completed"
        action = "Use measured-node validation table and figure in the main/supplementary results."
        limitation = "No limitation beyond the measured-node scope; pressure/head and flow remain unmeasured."
    elif raw_files:
        status = "raw_available_processing_needed"
        action = "Run run_real_data_study.py in a Torch-enabled environment to preprocess and evaluate XAI4HEAT measured nodes."
        limitation = "Sparse-substation validation is not yet included until preprocessing and evaluation complete."
    else:
        status = "not_run_raw_files_missing"
        action = (
            "Download XAI4HEAT from Mendeley Data, place CSV/XLSX/TXT/ZIP files in "
            "data/raw/xai4heat/, then rerun run_real_data_study.py."
        )
        limitation = "XAI4HEAT is workflow-ready but not a completed validation result in this run."

    rows = [
        {
            "Item": "raw directory",
            "Status": "data/raw/xai4heat/",
            "Interpretation": "Expected location for manually downloaded XAI4HEAT files.",
        },
        {
            "Item": "raw file count",
            "Status": len(raw_files),
            "Interpretation": "Nonzero count is required before measured-substation validation can run.",
        },
        {
            "Item": "processed file",
            "Status": "yes" if processed else "no",
            "Interpretation": "Processed CSV is required for repeatable validation.",
        },
        {
            "Item": "validation status",
            "Status": status,
            "Interpretation": action,
        },
        {
            "Item": "measured variables",
            "Status": "; ".join(info.measured_variables),
            "Interpretation": "Only measured supply/return/energy/outdoor variables may be validated; no pressure/head or flow fields are invented.",
        },
        {
            "Item": "evidence boundary",
            "Status": limitation,
            "Interpretation": "XAI4HEAT can support sparse measured-node thermal/energy validation, not dense distributed hydraulic validation.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    results = PROJECT_ROOT / "results"
    tables = PROJECT_ROOT / "paper" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = build_xai4heat_readiness_report()
    df.to_csv(results / "xai4heat_readiness_report.csv", index=False)
    _write_latex_table(
        df,
        tables / "table_xai4heat_validation_readiness.tex",
        "XAI4HEAT validation status. The current artifact set includes measured-node thermal/energy validation when raw files, processed data, and validation CSV outputs are present.",
        "tab:xai4heat_readiness",
    )

    status = str(df.loc[df["Item"].eq("validation status"), "Status"].iloc[0])
    marker = results / "XAI4HEAT_NOT_RUN.txt"
    if status != "validation_completed":
        marker.write_text(
            "XAI4HEAT sparse-substation validation was not run because local raw files were not available or the preprocessing/evaluation step was incomplete. "
            "Place raw files in data/raw/xai4heat/ and rerun run_real_data_study.py in a Torch-enabled environment.\n",
            encoding="utf-8",
        )
    elif marker.exists():
        marker.unlink()
    print(results / "xai4heat_readiness_report.csv")


if __name__ == "__main__":
    main()
