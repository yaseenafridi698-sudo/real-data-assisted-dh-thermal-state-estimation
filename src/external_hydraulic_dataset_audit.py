"""Audit external MATLAB hydraulic datasets for possible paper use.

The datasets supplied by the user are not assumed to be district-heating data.
This script reads MATLAB v5 files without SciPy, summarizes measured hydraulic
signals, runs a simple sparse-node reconstruction sanity check where multiple
measured hydraulic nodes are available, and writes claim-safe outputs.
"""

from __future__ import annotations

import math
import os
import struct
import zipfile
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "final"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

DOWNLOADS = Path(os.environ.get("ATE_EXTERNAL_DATA_DIR", str(Path.home() / "Downloads")))
USER_FILES = [
    DOWNLOADS / "P036_data.mat",
    DOWNLOADS / "frequency_analysis.mat",
    DOWNLOADS / "test_repository.zip",
    DOWNLOADS / "10417585.zip",
    DOWNLOADS / "5535442.zip",
]

MI_COMPRESSED = 15
MI_MATRIX = 14
MI_DOUBLE = 9
CLASS_NAMES = {
    1: "cell",
    2: "struct",
    3: "object",
    4: "char",
    5: "sparse",
    6: "double",
    7: "single",
    8: "int8",
    9: "uint8",
    10: "int16",
    11: "uint16",
    12: "int32",
    13: "uint32",
    14: "int64",
    15: "uint64",
}

PALETTE = {
    "blue": "#0000E6",
    "orange": "#FF6626",
    "green": "#55D600",
    "yellow": "#F2E600",
    "magenta": "#E600E6",
    "black": "#111111",
    "gray": "#555555",
}


@dataclass
class MatVariable:
    source_file: str
    inner_path: str
    variable: str
    shape: tuple[int, ...]
    matlab_class: str
    data: np.ndarray | None


def _tag(buf: bytes, pos: int, endian: str = "<"):
    if pos + 4 > len(buf):
        return None
    first = struct.unpack(endian + "I", buf[pos : pos + 4])[0]
    small_type = first & 0xFFFF
    small_bytes = (first >> 16) & 0xFFFF
    if small_bytes:
        return small_type, small_bytes, pos + 4, pos + 4 + small_bytes, pos + 8, True
    if pos + 8 > len(buf):
        return None
    dtype, nbytes = struct.unpack(endian + "II", buf[pos : pos + 8])
    start = pos + 8
    end = start + nbytes
    padded_next = end + ((8 - nbytes % 8) % 8)
    return dtype, nbytes, start, end, padded_next, False


def _parse_matrix(payload: bytes, endian: str = "<", load_data: bool = True):
    pos = 0
    first = _tag(payload, pos, endian)
    if first and first[0] == MI_MATRIX:
        pos = first[2]

    flags_tag = _tag(payload, pos, endian)
    if not flags_tag:
        return "", (), "unknown", None
    _, _, ds, de, pos, _ = flags_tag
    flags = payload[ds:de]
    class_code = struct.unpack(endian + "I", flags[:4])[0] & 0xFF if len(flags) >= 4 else -1
    matlab_class = CLASS_NAMES.get(class_code, f"class_{class_code}")

    dims_tag = _tag(payload, pos, endian)
    if not dims_tag:
        return "", (), matlab_class, None
    dtype, nbytes, ds, de, pos, _ = dims_tag
    dims_raw = payload[ds:de]
    dims = ()
    if dtype in {5, 6} and len(dims_raw) % 4 == 0:
        dims = tuple(int(v) for v in struct.unpack(endian + "i" * (len(dims_raw) // 4), dims_raw))

    name_tag = _tag(payload, pos, endian)
    if not name_tag:
        return "", dims, matlab_class, None
    _, nbytes, ds, _, pos, _ = name_tag
    name = payload[ds : ds + nbytes].decode("utf-8", "replace").rstrip("\x00")

    if not load_data:
        return name, dims, matlab_class, None

    data_tag = _tag(payload, pos, endian)
    if not data_tag:
        return name, dims, matlab_class, None
    dtype, nbytes, ds, de, _, _ = data_tag
    arr = None
    if dtype == MI_DOUBLE:
        raw = payload[ds:de]
        count = nbytes // 8
        arr = np.frombuffer(raw[: count * 8], dtype=endian + "f8").copy()
        if dims and int(np.prod(dims)) == arr.size:
            # MATLAB is column-major. Keep vectors simple; reshape matrices if any appear.
            arr = arr.reshape(dims, order="F")
    return name, dims, matlab_class, arr


def _read_mat_bytes(raw: bytes, source_file: str, inner_path: str, load_limit_bytes: int = 60_000_000) -> list[MatVariable]:
    endian = "<" if raw[126:128] == b"IM" else ">"
    pos = 128
    variables: list[MatVariable] = []
    while pos + 8 <= len(raw):
        item = _tag(raw, pos, endian)
        if not item:
            break
        dtype, nbytes, ds, de, next_pos, _ = item
        if dtype == MI_COMPRESSED:
            comp = raw[ds:de]
            load_data = nbytes <= load_limit_bytes
            try:
                payload = zlib.decompress(comp) if load_data else zlib.decompressobj().decompress(comp, 512_000)
                name, dims, matlab_class, arr = _parse_matrix(payload, endian, load_data=load_data)
            except Exception:
                name, dims, matlab_class, arr = "", (), "unreadable", None
            if name:
                variables.append(MatVariable(source_file, inner_path, name, dims, matlab_class, arr))
            # MATLAB compressed blocks in these supplied files are consecutive without
            # 8-byte top-level padding, so use the raw end offset.
            pos = de
        elif dtype == MI_MATRIX:
            payload = raw[ds:de]
            name, dims, matlab_class, arr = _parse_matrix(payload, endian, load_data=nbytes <= load_limit_bytes)
            if name:
                variables.append(MatVariable(source_file, inner_path, name, dims, matlab_class, arr))
            pos = next_pos
        else:
            break
    return variables


def _iter_supplied_mat_payloads(paths: Iterable[Path]):
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".mat":
            yield path.name, path.name, path.read_bytes()
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".mat"):
                        yield path.name, name, zf.read(name)
                    elif name.lower().endswith(".zip"):
                        with zipfile.ZipFile(BytesIO(zf.read(name))) as nested:
                            for nested_name in nested.namelist():
                                if nested_name.lower().endswith(".mat"):
                                    yield path.name, f"{name}:{nested_name}", nested.read(nested_name)


def _load_variables() -> list[MatVariable]:
    variables: list[MatVariable] = []
    for source_file, inner_path, raw in _iter_supplied_mat_payloads(USER_FILES):
        variables.extend(_read_mat_bytes(raw, source_file, inner_path))
    return variables


def _signal_summary(variables: list[MatVariable]) -> pd.DataFrame:
    rows = []
    for var in variables:
        arr = var.data
        if arr is None:
            rows.append(
                {
                    "source_file": var.source_file,
                    "inner_path": var.inner_path,
                    "variable": var.variable,
                    "n_samples": int(np.prod(var.shape)) if var.shape else 0,
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "range": "",
                    "note": "metadata only; variable too large or not fully loaded",
                }
            )
            continue
        flat = np.asarray(arr, dtype=float).reshape(-1)
        rows.append(
            {
                "source_file": var.source_file,
                "inner_path": var.inner_path,
                "variable": var.variable,
                "n_samples": int(flat.size),
                "mean": float(np.nanmean(flat)),
                "std": float(np.nanstd(flat)),
                "min": float(np.nanmin(flat)),
                "max": float(np.nanmax(flat)),
                "range": float(np.nanmax(flat) - np.nanmin(flat)),
                "note": "loaded numeric vector",
            }
        )
    return pd.DataFrame(rows)


def _inventory(variables: list[MatVariable]) -> pd.DataFrame:
    rows = []
    for var in variables:
        lname = var.variable.lower()
        if lname == "t":
            inferred_role = "time_vector"
        elif lname.startswith("h"):
            inferred_role = "measured_head_or_pressure_like_signal"
        elif lname.startswith("node_"):
            inferred_role = "measured_node_hydraulic_signal"
        elif lname.startswith("hf") or lname == "omega":
            inferred_role = "frequency_response_or_frequency_axis"
        elif "p" in lname:
            inferred_role = "pressure_like_signal"
        else:
            inferred_role = "unknown_numeric_signal"
        rows.append(
            {
                "source_file": var.source_file,
                "inner_path": var.inner_path,
                "variable": var.variable,
                "shape": "x".join(str(v) for v in var.shape),
                "matlab_class": var.matlab_class,
                "loaded": var.data is not None,
                "inferred_role_from_name": inferred_role,
                "compatibility_for_dh_paper": (
                    "auxiliary hydraulic signal evidence only; not district-heating field validation"
                    if inferred_role != "time_vector"
                    else "time axis for auxiliary dataset"
                ),
            }
        )
    return pd.DataFrame(rows)


def _time_info(summary: pd.DataFrame, variables: list[MatVariable]) -> dict[str, dict[str, float]]:
    out = {}
    for var in variables:
        if var.variable.lower() != "t" or var.data is None:
            continue
        t = np.asarray(var.data, dtype=float).reshape(-1)
        dt = np.diff(t)
        label = f"{var.source_file}:{var.inner_path}"
        out[label] = {
            "n_time_samples": float(t.size),
            "time_start": float(t[0]),
            "time_end": float(t[-1]),
            "median_dt": float(np.nanmedian(dt)) if dt.size else math.nan,
            "duration": float(t[-1] - t[0]) if t.size else math.nan,
        }
    return out


def _sparse_reconstruction(variables: list[MatVariable]) -> pd.DataFrame:
    """Chronological linear sparse-node reconstruction for datasets with >=3 signals.

    This is deliberately simple and is not presented as PI-GNN validation.
    It asks whether external measured hydraulic nodes can be reconstructed from
    other measured hydraulic nodes after a chronological train/test split.
    """
    rows = []
    by_dataset: dict[str, list[MatVariable]] = {}
    for var in variables:
        if var.data is None:
            continue
        if var.variable.lower() == "t" or var.variable.lower().startswith("hf") or var.variable.lower() == "omega":
            continue
        by_dataset.setdefault(f"{var.source_file}:{var.inner_path}", []).append(var)

    for dataset, vars_ in by_dataset.items():
        if len(vars_) < 3:
            continue
        # Keep only equal-length vectors.
        lengths = [np.asarray(v.data).reshape(-1).size for v in vars_]
        n = min(lengths)
        if n < 20:
            continue
        names = [v.variable for v in vars_]
        x = np.column_stack([np.asarray(v.data, dtype=float).reshape(-1)[:n] for v in vars_])
        split = int(n * 0.7)
        for target_idx, target_name in enumerate(names):
            sensor_idx = [i for i in range(len(names)) if i != target_idx]
            # Use up to three other sensors to avoid overfitting tiny node sets.
            sensor_idx = sensor_idx[: min(3, len(sensor_idx))]
            x_train = x[:split, sensor_idx]
            y_train = x[:split, target_idx]
            x_test = x[split:, sensor_idx]
            y_test = x[split:, target_idx]
            x_aug = np.column_stack([np.ones(x_train.shape[0]), x_train])
            coef, *_ = np.linalg.lstsq(x_aug, y_train, rcond=None)
            y_pred = np.column_stack([np.ones(x_test.shape[0]), x_test]) @ coef
            err = y_pred - y_test
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            denom = float(np.nanmax(y_test) - np.nanmin(y_test))
            rows.append(
                {
                    "dataset": dataset,
                    "target_node": target_name,
                    "sensor_nodes": ";".join(names[i] for i in sensor_idx),
                    "n_train": int(split),
                    "n_test": int(n - split),
                    "RMSE_external_hydraulic_units": rmse,
                    "MAE_external_hydraulic_units": mae,
                    "nRMSE_percent_of_test_range": 100.0 * rmse / denom if denom > 0 else math.nan,
                    "validation_type": "auxiliary_measured_hydraulic_signal_reconstruction",
                    "safe_interpretation": "Useful as external hydraulic sparse-node sanity check; not district-heating pressure/head or flow field validation.",
                }
            )
    return pd.DataFrame(rows)


def _write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, max_rows: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
    cols = list(show.columns)
    def esc(v: object) -> str:
        s = str(v)
        for a, b in {
            "\\": r"\textbackslash{}",
            "_": r"\_",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
        }.items():
            s = s.replace(a, b)
        return s
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\toprule",
        " & ".join(esc(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in show.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_summary_figure(inventory: pd.DataFrame, sparse: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / "fig_external_hydraulic_dataset_audit.pdf"
    png_path = FIGURES_DIR / "fig_external_hydraulic_dataset_audit.png"

    width, height = landscape(letter)
    c = canvas.Canvas(str(pdf_path), pagesize=landscape(letter))
    try:
        pdfmetrics.registerFont(TTFont("DejaVuSerif", "DejaVuSerif.ttf"))
        font = "DejaVuSerif"
    except Exception:
        font = "Times-Roman"
    c.setFont(font, 18)
    c.setFillColor(colors.HexColor(PALETTE["black"]))
    c.drawString(0.55 * inch, height - 0.55 * inch, "External hydraulic dataset audit")
    c.setFont(font, 9)
    c.drawString(0.55 * inch, height - 0.82 * inch, "Auxiliary measured hydraulic signals; not direct district-heating field validation.")

    grouped = inventory.groupby("source_file").agg(
        variables=("variable", "count"),
        loaded=("loaded", "sum"),
    ).reset_index()
    x0, y0 = 0.55 * inch, height - 1.45 * inch
    bar_w = 1.25 * inch
    colors_order = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"], PALETTE["yellow"]]
    for i, row in grouped.iterrows():
        x = x0 + i * (bar_w + 0.2 * inch)
        h = 0.18 * inch * float(row["variables"])
        c.setFillColor(colors.HexColor(colors_order[i % len(colors_order)]))
        c.setStrokeColor(colors.HexColor(PALETTE["black"]))
        c.rect(x, y0 - h, bar_w, h, fill=1, stroke=1)
        c.setFillColor(colors.HexColor(PALETTE["black"]))
        c.setFont(font, 7.5)
        c.drawCentredString(x + bar_w / 2, y0 + 0.08 * inch, str(row["source_file"])[:18])
        c.drawCentredString(x + bar_w / 2, y0 - h - 0.16 * inch, f"{int(row['variables'])} vars")

    c.setFont(font, 11)
    c.drawString(0.55 * inch, 2.35 * inch, "Compatibility decision")
    c.setFont(font, 9)
    decision_lines = [
        "1. Multiple measured hydraulic signals are available in 5u_q015.mat and test_pead_dapubblicare.mat.",
        "2. These files do not include district-heating heat load, temperature, pipe topology, or flow fields.",
        "3. Use them only as auxiliary sparse hydraulic signal checks, not full DH hydraulic validation.",
    ]
    for j, line in enumerate(decision_lines):
        c.drawString(0.75 * inch, 2.05 * inch - j * 0.25 * inch, line)

    if not sparse.empty:
        best = sparse.sort_values("nRMSE_percent_of_test_range").iloc[0]
        c.setFillColor(colors.HexColor(PALETTE["blue"]))
        c.roundRect(7.35 * inch, 1.25 * inch, 3.0 * inch, 1.1 * inch, 8, fill=0, stroke=1)
        c.setFillColor(colors.HexColor(PALETTE["black"]))
        c.setFont(font, 9)
        c.drawString(7.55 * inch, 2.05 * inch, "Best auxiliary sparse-node check")
        c.drawString(7.55 * inch, 1.78 * inch, f"Target: {best['target_node']}")
        c.drawString(7.55 * inch, 1.53 * inch, f"nRMSE: {best['nRMSE_percent_of_test_range']:.2f}% of test range")

    c.showPage()
    c.save()

    # PNG preview drawn with PIL for quick review.
    im = Image.new("RGB", (1600, 900), "white")
    d = ImageDraw.Draw(im)
    try:
        font_title = ImageFont.truetype("times.ttf", 36)
        font_small = ImageFont.truetype("times.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()
    d.text((60, 45), "External hydraulic dataset audit", fill=PALETTE["black"], font=font_title)
    d.text((60, 92), "Auxiliary measured hydraulic signals; not direct district-heating field validation.", fill=PALETTE["gray"], font=font_small)
    max_vars = max(1, int(grouped["variables"].max())) if not grouped.empty else 1
    for i, row in grouped.iterrows():
        x = 70 + i * 285
        bar_h = int(330 * row["variables"] / max_vars)
        color = colors_order[i % len(colors_order)]
        d.rectangle([x, 480 - bar_h, x + 185, 480], fill=color, outline=PALETTE["black"], width=3)
        d.text((x, 500), str(row["source_file"])[:20], fill=PALETTE["black"], font=font_small)
        d.text((x + 35, 455 - bar_h), f"{int(row['variables'])} vars", fill=PALETTE["black"], font=font_small)
    d.text((60, 650), "Compatibility decision:", fill=PALETTE["black"], font=font_title)
    for j, line in enumerate(decision_lines):
        d.text((90, 710 + j * 38), line, fill=PALETTE["black"], font=font_small)
    im.save(png_path, dpi=(300, 300))


def _assessment_md(inventory: pd.DataFrame, summary: pd.DataFrame, sparse: pd.DataFrame, time_info: dict[str, dict[str, float]]) -> str:
    loaded_vars = int(inventory["loaded"].sum())
    total_vars = int(len(inventory))
    measured_like = inventory[inventory["inferred_role_from_name"].str.contains("head|hydraulic|pressure", case=False, na=False)]
    lines = [
        "# External hydraulic dataset audit",
        "",
        "## Honest conclusion",
        "The supplied files contain useful hydraulic pressure/head-like time series and frequency-response variables, but they are not direct district-heating field-validation datasets for the present paper. They do not provide the combination of district-heating heat load, supply/return temperature, network topology, measured distributed pressure/head, measured distributed flow, and pipe heat-loss fields required to validate the thermo-hydraulic digital twin as a district-heating hydraulic field model.",
        "",
        "They can be used as an auxiliary external hydraulic signal sanity check and as supporting evidence that sparse hydraulic node reconstruction is feasible on measured hydraulic signals. They should not be described as full field validation of the district-heating pressure/head or flow states.",
        "",
        "## What was read",
        f"- MATLAB variables discovered: {total_vars}",
        f"- Variables fully loaded for numeric summaries: {loaded_vars}",
        f"- Pressure/head/hydraulic-like variables inferred from names: {len(measured_like)}",
        "",
        "## Time-vector information",
    ]
    if time_info:
        for key, val in time_info.items():
            lines.append(f"- `{key}`: n={int(val['n_time_samples'])}, median dt={val['median_dt']:.6g}, duration={val['duration']:.6g}")
    else:
        lines.append("- No explicit time vectors were loaded for the largest P036 signal.")
    lines.extend(
        [
            "",
            "## Auxiliary sparse-node reconstruction",
        ]
    )
    if sparse.empty:
        lines.append("No multi-node dataset with enough equal-length loaded hydraulic signals was available for sparse-node reconstruction.")
    else:
        best = sparse.sort_values("nRMSE_percent_of_test_range").head(5)
        for _, row in best.iterrows():
            lines.append(
                f"- {row['dataset']} target `{row['target_node']}` from sensors `{row['sensor_nodes']}`: "
                f"RMSE={row['RMSE_external_hydraulic_units']:.4g}, nRMSE={row['nRMSE_percent_of_test_range']:.2f}% of test range."
            )
    lines.extend(
        [
            "",
            "## Paper-use recommendation",
            "Use these data only in the supplementary material as an external hydraulic data audit or auxiliary sparse hydraulic signal check. The main paper should continue to state that pressure/head and flow in the district-heating network are simulator-assisted hidden hydraulic states.",
            "",
            "Safe wording: `External hydraulic signal datasets were audited and used only as auxiliary sparse hydraulic signal checks; they do not constitute district-heating pressure/head or flow field validation.`",
            "",
            "Unsafe wording to avoid: `The district-heating hydraulic model was field validated with real distributed pressure and flow data.`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    variables = _load_variables()
    inventory = _inventory(variables)
    summary = _signal_summary(variables)
    sparse = _sparse_reconstruction(variables)
    time_info = _time_info(summary, variables)

    inventory.to_csv(RESULTS_DIR / "external_hydraulic_dataset_inventory.csv", index=False)
    summary.to_csv(RESULTS_DIR / "external_hydraulic_signal_summary.csv", index=False)
    sparse.to_csv(RESULTS_DIR / "external_hydraulic_sparse_reconstruction.csv", index=False)
    (RESULTS_DIR / "external_hydraulic_validation_assessment.md").write_text(
        _assessment_md(inventory, summary, sparse, time_info), encoding="utf-8"
    )

    inv_table = inventory[
        ["source_file", "variable", "shape", "inferred_role_from_name", "compatibility_for_dh_paper"]
    ].copy()
    _write_latex_table(
        inv_table,
        TABLES_DIR / "table_external_hydraulic_dataset_audit.tex",
        "External hydraulic dataset audit. These datasets provide auxiliary measured hydraulic signals but do not provide district-heating heat-load, temperature, topology, dense pressure/head, dense flow, or heat-loss field validation.",
        "tab:external_hydraulic_dataset_audit",
        max_rows=16,
    )
    if not sparse.empty:
        sparse_table = sparse[
            [
                "dataset",
                "target_node",
                "sensor_nodes",
                "RMSE_external_hydraulic_units",
                "nRMSE_percent_of_test_range",
                "safe_interpretation",
            ]
        ].sort_values("nRMSE_percent_of_test_range")
        _write_latex_table(
            sparse_table,
            TABLES_DIR / "table_external_hydraulic_sparse_check.tex",
            "Auxiliary sparse-node reconstruction on external measured hydraulic signals. This is not district-heating hydraulic field validation.",
            "tab:external_hydraulic_sparse_check",
            max_rows=10,
        )

    _draw_summary_figure(inventory, sparse)
    print(RESULTS_DIR / "external_hydraulic_validation_assessment.md")


if __name__ == "__main__":
    main()
