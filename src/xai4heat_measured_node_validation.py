from __future__ import annotations

import math
import re
import sys
import hashlib
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
try:  # Legacy fallback drawing only; primary validation uses Matplotlib.
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - optional fallback dependency.
    Image = ImageDraw = ImageFont = None
try:  # Legacy fallback drawing only; primary validation uses Matplotlib.
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:  # pragma: no cover - optional fallback dependency.
    colors = canvas = None
    landscape = letter = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT


PALETTE = {
    "blue": "#0000E6",
    "orange": "#FF6626",
    "green": "#55D600",
    "yellow": "#F2E600",
    "magenta": "#E600E6",
    "black": "#111111",
    "gray": "#555555",
}

VARIABLES = {
    "t_sup_prim": ("Primary supply temperature", "C", "temperature"),
    "t_ret_prim": ("Primary return temperature", "C", "temperature"),
    "t_sup_sec": ("Secondary supply temperature", "C", "temperature"),
    "t_ret_sec": ("Secondary return temperature", "C", "temperature"),
    "e": ("Energy metric e", "source units", "energy"),
    "pe": ("Power/energy metric pe", "source units", "energy"),
}


def discover_raw_files() -> tuple[list[Path], pd.DataFrame]:
    """Select one raw file for each unique content hash.

    Nested duplicate extractions are common for manually downloaded archives.
    They must not silently inflate counts in the measured-node validation.
    """
    raw = PROJECT_ROOT / "data" / "raw" / "xai4heat"
    candidates = sorted(raw.rglob("xai4heat_scada_L*.csv"))
    groups: dict[str, list[Path]] = {}
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.setdefault(digest, []).append(path)

    selected: list[Path] = []
    rows: list[dict[str, object]] = []
    for digest, paths in groups.items():
        preferred = sorted(paths, key=lambda p: (len(p.parts), str(p).lower()))[0]
        selected.append(preferred)
        for path in paths:
            rows.append(
                {
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "substation_id": _substation_id(path),
                    "sha256": digest,
                    "selected_for_validation": path == preferred,
                    "duplicate_of": "" if path == preferred else str(preferred.relative_to(PROJECT_ROOT)),
                    "file_bytes": path.stat().st_size,
                }
            )
    inventory = pd.DataFrame(rows).sort_values(
        ["substation_id", "selected_for_validation", "relative_path"],
        ascending=[True, False, True],
    )
    return sorted(selected), inventory


def _substation_id(path: Path) -> str:
    match = re.search(r"L\d+", path.stem)
    return match.group(0) if match else path.stem


def _position_from_id(substation_id: str) -> float:
    match = re.search(r"\d+", substation_id)
    return float(match.group(0)) if match else float("nan")


def load_xai4heat_scada() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    raw_files, inventory = discover_raw_files()
    for path in raw_files:
        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            continue
        sid = _substation_id(path)
        df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["substation_id"] = sid
        df["ordered_position"] = _position_from_id(sid)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No xai4heat_scada_L*.csv files found in data/raw/xai4heat/.")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp", "substation_id"])
    keep = [
        "timestamp",
        "substation_id",
        "ordered_position",
        "t_amb",
        "t_ref",
        "t_sup_prim",
        "t_ret_prim",
        "t_sup_sec",
        "t_ret_sec",
        "e",
        "pe",
    ]
    present = [c for c in keep if c in out.columns]
    out = out[present].copy()
    for c in [c for c in present if c not in {"timestamp", "substation_id"}]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.sort_values(["timestamp", "ordered_position", "substation_id"]).reset_index(drop=True)
    primary_valid = out["t_sup_prim"].between(5.0, 120.0) & out["t_ret_prim"].between(5.0, 120.0)
    out["primary_temperature_order_ok"] = primary_valid & (out["t_sup_prim"] >= out["t_ret_prim"])
    return out, inventory


def write_processed(df: pd.DataFrame) -> Path:
    out = df.rename(
        columns={
            "t_amb": "ambient_temp_C",
            "t_sup_prim": "supply_temp_C",
            "t_ret_prim": "return_temp_C",
            "t_sup_sec": "secondary_supply_temp_C",
            "t_ret_sec": "secondary_return_temp_C",
            "e": "energy_metric_e",
            "pe": "power_energy_metric_pe",
        }
    ).copy()
    out["source_dataset"] = "xai4heat"
    out["plant_id"] = ""
    out["return_temp_assumed"] = False
    path = PROJECT_ROOT / "data" / "processed" / "xai4heat_processed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def _predict_holdout(group: pd.DataFrame, holdout: str, variable: str) -> tuple[float, float] | None:
    sub = group[["substation_id", "ordered_position", variable]].dropna()
    if holdout not in set(sub["substation_id"]):
        return None
    train = sub[sub["substation_id"] != holdout].sort_values("ordered_position")
    test = sub[sub["substation_id"] == holdout].iloc[0]
    if train.empty or pd.isna(test[variable]) or pd.isna(test["ordered_position"]):
        return None
    x = train["ordered_position"].to_numpy(dtype=float)
    y = train[variable].to_numpy(dtype=float)
    if len(np.unique(x)) == 1:
        pred = float(np.nanmean(y))
    else:
        pred = float(np.interp(float(test["ordered_position"]), x, y))
    return pred, float(test[variable])


def leave_one_substation_out_validation(
    df: pd.DataFrame,
    primary_temperature_protocol: str = "all_valid_range",
    variables: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    details = []
    substations = sorted(df["substation_id"].dropna().unique(), key=_position_from_id)
    ordered_columns = substations
    ordered_positions = {_sid: _position_from_id(_sid) for _sid in ordered_columns}
    selected_variables = variables or tuple(VARIABLES)
    for variable in selected_variables:
        if variable not in VARIABLES:
            continue
        label, unit, category = VARIABLES[variable]
        if variable not in df.columns:
            continue
        work = df
        if variable in {"t_sup_prim", "t_ret_prim"} and primary_temperature_protocol in {
            "primary_ordered_subset",
            "physically_ordered_sensitivity_subset",
        }:
            # This target-pair filter is retained only as a sensitivity study.
            # It is not the primary measured-node validation because it uses
            # the relationship between the two withheld target channels.
            work = df[df["primary_temperature_order_ok"].fillna(False)].copy()
        pivot = work.pivot_table(
            index="timestamp",
            columns="substation_id",
            values=variable,
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=ordered_columns)
        position_cols = sorted(pivot.columns, key=lambda sid: ordered_positions.get(sid, float("inf")))
        pivot = pivot[position_cols]
        if category == "temperature":
            pivot = pivot.where((pivot >= 5.0) & (pivot <= 120.0))
            data_quality_filter = "temperature values outside 5-120 C treated as missing"
            if variable in {"t_sup_prim", "t_ret_prim"} and primary_temperature_protocol in {
                "primary_ordered_subset",
                "physically_ordered_sensitivity_subset",
            }:
                data_quality_filter += "; physically ordered sensitivity subset (primary supply >= primary return)"
        elif category == "energy":
            upper = float(pivot.stack().quantile(0.999)) if pivot.notna().any().any() else float("nan")
            if np.isfinite(upper) and upper > 0:
                pivot = pivot.where((pivot >= 0.0) & (pivot <= upper))
                data_quality_filter = "nonnegative values retained; upper 0.1 percent treated as source-unit outliers"
            else:
                pivot = pivot.where(pivot >= 0.0)
                data_quality_filter = "nonnegative source-unit values retained"
        else:
            data_quality_filter = "finite values retained"
        for holdout in substations:
            if holdout not in pivot.columns:
                continue
            actual = pivot[holdout].to_numpy(dtype=float)
            train = pivot.copy()
            train[holdout] = np.nan
            # Spatial interpolation across ordered substations. Boundary holdouts
            # use the nearest measured substation through limit_direction="both".
            predicted = (
                train.interpolate(axis=1, method="linear", limit_direction="both")
                .loc[:, holdout]
                .to_numpy(dtype=float)
            )
            mask = np.isfinite(predicted) & np.isfinite(actual)
            if not mask.any():
                continue
            pred_arr = predicted[mask]
            actual_arr = actual[mask]
            err = pred_arr - actual_arr
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            bias = float(np.mean(err))
            if category == "temperature":
                denom = np.maximum(np.abs(actual_arr), 5.0)
                mape = float(np.mean(np.abs(err) / denom) * 100.0)
            else:
                usable = np.abs(actual_arr) > 1.0
                mape = float(np.mean(np.abs(err[usable]) / np.abs(actual_arr[usable])) * 100.0) if usable.any() else float("nan")
            scale = float(np.nanpercentile(actual_arr, 95) - np.nanpercentile(actual_arr, 5))
            nrmse = float((rmse / scale) * 100.0) if scale > 1e-9 else float("nan")
            details.append(
                {
                    "substation_id": holdout,
                    "variable": variable,
                    "variable_label": label,
                    "category": category,
                    "unit": unit,
                    "n_samples": int(len(actual_arr)),
                    "RMSE": rmse,
                    "MAE": mae,
                    "bias": bias,
                    "nRMSE_percent": nrmse,
                    "MAPE_percent": mape,
                    "validation_method": "leave_one_substation_out_spatial_interpolation",
                    "primary_temperature_protocol": primary_temperature_protocol,
                    "data_quality_filter": data_quality_filter,
                    "state_type": "real_measured_node",
                    "safe_claim": "XAI4HEAT supports sparse measured-node temperature/energy consistency, not dense distributed hydraulic validation.",
                }
            )
    details_df = pd.DataFrame(details)
    if details_df.empty:
        return pd.DataFrame(), details_df
    for variable, group in details_df.groupby("variable", sort=False):
        label, unit, category = VARIABLES[variable]
        rows.append(
            {
                "variable": variable,
                "variable_label": label,
                "category": category,
                "unit": unit,
                "n_substations": int(group["substation_id"].nunique()),
                "n_total_samples": int(group["n_samples"].sum()),
                "mean_RMSE": float(group["RMSE"].mean()),
                "mean_MAE": float(group["MAE"].mean()),
                "mean_abs_bias": float(group["bias"].abs().mean()),
                "mean_nRMSE_percent": float(group["nRMSE_percent"].replace([np.inf, -np.inf], np.nan).mean()),
                "mean_MAPE_percent": float(group["MAPE_percent"].replace([np.inf, -np.inf], np.nan).mean()),
                "validation_method": "leave_one_substation_out_spatial_interpolation",
                "primary_temperature_protocol": primary_temperature_protocol,
                "data_quality_filter": "; ".join(sorted(set(group["data_quality_filter"].dropna()))),
                "state_type": "real_measured_node",
                "safe_claim": "Measured-node validation only; pressure/head and flow are not measured by XAI4HEAT.",
            }
        )
    return pd.DataFrame(rows), details_df


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, group in df.groupby("substation_id"):
        row = {
            "substation_id": sid,
            "start": group["timestamp"].min(),
            "end": group["timestamp"].max(),
            "samples": len(group),
            "primary_temperature_order_ok_percent": float(group["primary_temperature_order_ok"].mean() * 100.0),
        }
        for col in ["t_sup_prim", "t_ret_prim", "t_sup_sec", "t_ret_sec", "e", "pe"]:
            if col in group:
                row[f"{col}_mean"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def station_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Describe rather than conceal station-level primary-temperature reversals."""
    rows = []
    for sid, group in df.groupby("substation_id", sort=True):
        supply = pd.to_numeric(group["t_sup_prim"], errors="coerce")
        ret = pd.to_numeric(group["t_ret_prim"], errors="coerce")
        valid_pair = supply.between(5.0, 120.0) & ret.between(5.0, 120.0)
        reversed_pair = valid_pair & (supply < ret)
        rows.append(
            {
                "substation_id": sid,
                "raw_rows_after_file_deduplication": int(len(group)),
                "valid_primary_temperature_pairs": int(valid_pair.sum()),
                "primary_supply_below_return_count": int(reversed_pair.sum()),
                "primary_supply_below_return_percent": float(100.0 * reversed_pair.sum() / max(valid_pair.sum(), 1)),
                "primary_ordered_rows": int((valid_pair & ~reversed_pair).sum()),
                "primary_ordered_percent": float(100.0 * (valid_pair & ~reversed_pair).sum() / max(valid_pair.sum(), 1)),
                "primary_supply_median_C": float(supply[valid_pair].median()) if valid_pair.any() else float("nan"),
                "primary_return_median_C": float(ret[valid_pair].median()) if valid_pair.any() else float("nan"),
                "interpretation": (
                    "Primary-temperature reversal frequency is reported for forensic review; the physically ordered subset is a target-conditioned sensitivity analysis, not a claim of instrument fault."
                ),
            }
        )
    return pd.DataFrame(rows)


def _latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in value)


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, max_rows: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_df = df.head(max_rows).copy()
    for col in table_df.columns:
        if pd.api.types.is_numeric_dtype(table_df[col]):
            table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(table_df.columns) + r"}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in table_df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        lines.append(" & ".join(_latex_escape(row[c]) for c in table_df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_png(summary: pd.DataFrame, path: Path) -> None:
    width, height = 2400, 1500
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(58, True)
    label_font = _font(34, True)
    small_font = _font(26)
    d.text((90, 60), "XAI4HEAT sparse-substation measured-node validation", fill=PALETTE["black"], font=title_font)
    d.text((90, 135), "Leave-one-substation-out interpolation using real measured supply/return and energy variables", fill=PALETTE["gray"], font=small_font)
    plot = summary.copy()
    plot["plot_label"] = plot["variable_label"].str.replace(" temperature", "").str.replace(" metric", "")
    x0, y0, w, h = 190, 260, 2050, 930
    d.rectangle((x0, y0, x0 + w, y0 + h), outline=PALETTE["black"], width=3)
    max_val = max(float(plot["mean_RMSE"].max()), 1e-6)
    colors_list = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"], PALETTE["yellow"], PALETTE["gray"]]
    bar_gap = 24
    bar_h = (h - 140 - bar_gap * (len(plot) - 1)) / max(len(plot), 1)
    for i, (_, row) in enumerate(plot.iterrows()):
        y = y0 + 70 + i * (bar_h + bar_gap)
        value = float(row["mean_RMSE"])
        bw = int((value / max_val) * (w - 760))
        color = colors_list[i % len(colors_list)]
        d.text((x0 + 25, int(y + bar_h * 0.18)), str(row["plot_label"])[:34], fill=PALETTE["black"], font=label_font)
        d.rectangle((x0 + 650, int(y), x0 + 650 + bw, int(y + bar_h)), fill=color, outline=PALETTE["black"], width=3)
        d.text((x0 + 670 + bw, int(y + bar_h * 0.18)), f"RMSE {value:.2f} {row['unit']}", fill=PALETTE["black"], font=label_font)
    note = "Pressure/head and flow are not measured by XAI4HEAT; validation is real measured-node thermal/energy consistency only."
    d.text((90, 1260), note, fill=PALETTE["black"], font=small_font)
    d.text((90, 1315), "State type: real_measured_node. Distributed hydraulic and heat-loss field validation is not claimed.", fill=PALETTE["gray"], font=small_font)
    img.save(path, dpi=(1200, 1200))


def _draw_pdf(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    W, H = landscape(letter)
    c.setTitle("XAI4HEAT sparse-substation measured-node validation")
    c.setFont("Times-Bold", 20)
    c.setFillColor(colors.HexColor(PALETTE["black"]))
    c.drawString(42, H - 50, "XAI4HEAT sparse-substation measured-node validation")
    c.setFont("Times-Roman", 10.5)
    c.setFillColor(colors.HexColor(PALETTE["gray"]))
    c.drawString(42, H - 68, "Leave-one-substation-out interpolation using real measured supply/return and energy variables")
    plot = summary.copy()
    plot["plot_label"] = plot["variable_label"].str.replace(" temperature", "").str.replace(" metric", "")
    x0, y0, w, h = 70, 105, 650, 360
    max_val = max(float(plot["mean_RMSE"].max()), 1e-6)
    color_keys = ["blue", "orange", "green", "magenta", "yellow", "gray"]
    bar_h = 34
    gap = 18
    c.setStrokeColor(colors.HexColor(PALETTE["black"]))
    c.rect(x0, y0, w, h, stroke=1, fill=0)
    for i, (_, row) in enumerate(plot.iterrows()):
        y = y0 + h - 60 - i * (bar_h + gap)
        value = float(row["mean_RMSE"])
        bw = (value / max_val) * 360
        c.setFillColor(colors.HexColor(PALETTE["black"]))
        c.setFont("Times-Bold", 10)
        c.drawString(x0 + 15, y + 10, str(row["plot_label"])[:30])
        c.setFillColor(colors.HexColor(PALETTE[color_keys[i % len(color_keys)]]))
        c.rect(x0 + 230, y, bw, bar_h, stroke=1, fill=1)
        c.setFillColor(colors.HexColor(PALETTE["black"]))
        c.drawString(x0 + 240 + bw, y + 10, f"RMSE {value:.2f} {row['unit']}")
    c.setFont("Times-Roman", 9.5)
    c.drawString(42, 55, "Pressure/head and flow are not measured by XAI4HEAT; validation is real measured-node thermal/energy consistency only.")
    c.drawString(42, 40, "State type: real_measured_node. Distributed hydraulic and heat-loss field validation is not claimed.")
    c.showPage()
    c.save()


def _draw_primary_temperature_figure(validation: pd.DataFrame, out: Path) -> None:
    """Plot the all-valid-range primary-temperature LOSO protocol used in the main text."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.edgecolor": PALETTE["black"],
            "axes.linewidth": 0.8,
        }
    )
    primary = validation[validation["variable"].isin(["supply_temp_C", "return_temp_C"])].copy()
    stations = sorted(primary["target_substation"].astype(str).unique(), key=lambda x: int(x[1:]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True, constrained_layout=True)
    specs = [
        ("supply_temp_C", "Primary supply", PALETTE["blue"]),
        ("return_temp_C", "Primary return", PALETTE["orange"]),
    ]
    for panel, (ax, (variable, title, color)) in enumerate(zip(axes, specs)):
        subset = primary[primary["variable"].eq(variable)].set_index("target_substation").reindex(stations)
        values = subset["RMSE_C"].to_numpy(float)
        bars = ax.bar(
            np.arange(len(stations)),
            values,
            color=color,
            edgecolor=PALETTE["black"],
            linewidth=0.9,
            width=0.68,
        )
        mean_fold_rmse = float(np.mean(values))
        ax.axhline(
            mean_fold_rmse,
            color=PALETTE["magenta"],
            lw=1.6,
            ls="--",
            label=f"Mean fold RMSE = {mean_fold_rmse:.2f} °C",
        )
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel("Withheld-node RMSE (°C)")
        ax.set_xticks(np.arange(len(stations)), stations)
        ax.grid(axis="y", color="#D8D8D8", lw=0.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PALETTE["black"])
            spine.set_linewidth(1.0)
        ax.legend(frameon=True, fancybox=False, edgecolor=PALETTE["black"], framealpha=1.0, fontsize=8, loc="upper left")
        ax.text(-0.10, 1.04, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold", fontsize=10)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax.set_ylim(0, max(values) * 1.22)
    fig.suptitle(
        "XAI4HEAT blind leave-one-substation-out validation",
        fontsize=12,
        fontweight="bold",
    )
    for suffix, dpi in [(".pdf", None), (".svg", None), (".png", 1200)]:
        target = out / f"fig12_xai4heat_validation_final{suffix}"
        fig.savefig(target, dpi=dpi, bbox_inches="tight", facecolor="white")
    # Compatibility names used by earlier paper assets.
    fig.savefig(out / "fig12_xai4heat_sparse_substations.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / "fig12_xai4heat_sparse_substations.png", dpi=1200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_figure(summary: pd.DataFrame, by_substation: pd.DataFrame | None = None) -> None:
    out = PROJECT_ROOT / "figures" / "final"
    out.mkdir(parents=True, exist_ok=True)
    if by_substation is not None and not by_substation.empty:
        primary = by_substation[by_substation["variable"].isin(["t_sup_prim", "t_ret_prim"])].copy()
        if not primary.empty:
            primary_validation = primary.rename(
                columns={
                    "substation_id": "target_substation",
                    "n_samples": "samples",
                    "RMSE": "RMSE_C",
                }
            )
            primary_validation["variable"] = primary_validation["variable"].replace(
                {"t_sup_prim": "supply_temp_C", "t_ret_prim": "return_temp_C"}
            )
            _draw_primary_temperature_figure(primary_validation, out)
            return
    _draw_png(summary, out / "fig12_xai4heat_validation_final.png")
    _draw_pdf(summary, out / "fig12_xai4heat_validation_final.pdf")
    # Compatibility names used by earlier paper assets.
    _draw_png(summary, out / "fig12_xai4heat_sparse_substations.png")
    _draw_pdf(summary, out / "fig12_xai4heat_sparse_substations.pdf")


def main() -> None:
    results = PROJECT_ROOT / "results"
    tables = PROJECT_ROOT / "paper" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    df, inventory = load_xai4heat_scada()
    processed_path = write_processed(df)
    summary = dataset_summary(df)
    quality = station_quality_audit(df)
    primary_variables = ("t_sup_prim", "t_ret_prim")
    all_valid, all_valid_by_substation = leave_one_substation_out_validation(
        df, "all_valid_range", variables=primary_variables
    )
    physically_ordered, physically_ordered_by_substation = leave_one_substation_out_validation(
        df, "physically_ordered_sensitivity_subset", variables=primary_variables
    )
    # All valid-range records form the primary measured-node result. The
    # physically ordered subset remains an explicitly target-conditioned
    # sensitivity analysis, not a headline validation score.
    validation = all_valid[all_valid["variable"].isin(["t_sup_prim", "t_ret_prim"])].copy()
    by_substation = all_valid_by_substation[
        all_valid_by_substation["variable"].isin(["t_sup_prim", "t_ret_prim"])
    ].copy()
    if validation.empty or by_substation.empty:
        raise RuntimeError("XAI4HEAT validation produced no metric rows.")
    summary.to_csv(results / "xai4heat_sparse_substation_dataset_summary.csv", index=False)
    inventory.to_csv(results / "xai4heat_raw_file_inventory.csv", index=False)
    quality.to_csv(results / "xai4heat_station_quality_audit.csv", index=False)
    all_valid.to_csv(results / "xai4heat_sparse_substation_validation_all_valid_range.csv", index=False)
    all_valid_by_substation.to_csv(results / "xai4heat_sparse_substation_validation_by_substation_all_valid_range.csv", index=False)
    # Archive obsolete filenames that described the sensitivity result as the
    # primary protocol. They are retained for traceability, never re-issued.
    archive = results / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    for legacy_name in [
        "xai4heat_sparse_substation_validation_primary_ordered_subset.csv",
        "xai4heat_sparse_substation_validation_by_substation_primary_ordered_subset.csv",
    ]:
        legacy = results / legacy_name
        if legacy.exists():
            shutil.move(str(legacy), str(archive / legacy_name))
    physically_ordered.to_csv(results / "xai4heat_sparse_substation_validation_physically_ordered_sensitivity.csv", index=False)
    physically_ordered_by_substation.to_csv(results / "xai4heat_sparse_substation_validation_by_substation_physically_ordered_sensitivity.csv", index=False)
    (results / "XAI4HEAT_PROTOCOL_NOTE.txt").write_text(
        "Primary XAI4HEAT headline metrics use all valid-range temperature observations.\n"
        "The physically ordered subset uses both measured target channels and is retained only as sensitivity evidence.\n",
        encoding="utf-8",
    )
    validation.to_csv(results / "xai4heat_sparse_substation_validation.csv", index=False)
    validation.to_csv(results / "xai4heat_sparse_substation_validation_final.csv", index=False)
    by_substation.to_csv(results / "xai4heat_sparse_substation_validation_by_substation.csv", index=False)
    write_latex_table(
        validation[
            [
                "variable_label",
                "category",
                "unit",
                "n_substations",
                "n_total_samples",
                "mean_RMSE",
                "mean_MAE",
                "mean_nRMSE_percent",
                "state_type",
            ]
        ],
        tables / "table12_xai4heat_validation.tex",
        "XAI4HEAT primary-temperature measured-node validation across all valid-range records. A physically ordered, target-conditioned sensitivity subset is reported separately because primary supply is below primary return for a substantial fraction of L22 observations. Pressure/head, flow, distributed pipe temperature, and heat loss are not measured by XAI4HEAT.",
        "tab:xai4heat_validation",
    )
    write_latex_table(
        validation[
            [
                "variable_label",
                "category",
                "unit",
                "n_substations",
                "n_total_samples",
                "mean_RMSE",
                "mean_MAE",
                "mean_nRMSE_percent",
                "state_type",
            ]
        ],
        tables / "table_xai4heat_validation.tex",
        "XAI4HEAT primary-temperature measured-node validation across all valid-range records. The physically ordered sensitivity subset and raw reversal frequencies are retained separately; no hydraulic field validation is claimed.",
        "tab:xai4heat_validation_full",
    )
    write_latex_table(
        physically_ordered[
            [
                "variable_label",
                "category",
                "unit",
                "n_substations",
                "n_total_samples",
                "mean_RMSE",
                "mean_MAE",
                "mean_nRMSE_percent",
                "state_type",
            ]
        ],
        tables / "table_xai4heat_physically_ordered_sensitivity.tex",
        "XAI4HEAT physically ordered sensitivity analysis. This subset retains only primary-temperature pairs satisfying supply greater than or equal to return; because the rule uses the two measured target channels, it is not the headline measured-node validation result.",
        "tab:xai4heat_ordered_sensitivity",
    )
    provenance = pd.DataFrame(
        [
            {
                "item": "source package",
                "status": "local zip extracted",
                "note": f"{len(inventory)} discovered SCADA files reduced to {int(inventory['selected_for_validation'].sum())} unique-content files by SHA-256 before validation.",
            },
            {
                "item": "processed file",
                "status": str(processed_path.relative_to(PROJECT_ROOT)),
                "note": "Processed long-form measured-substation CSV generated for reproducibility.",
            },
            {
                "item": "validation scope",
                "status": "primary-temperature measured-node validation",
                "note": "Primary supply/return headline results use all valid-range records. The physically ordered sensitivity subset uses the measured supply-return relationship and is retained separately. No pressure/head or flow measurements are provided.",
            },
        ]
    )
    provenance.to_csv(results / "xai4heat_validation_provenance.csv", index=False)
    write_latex_table(
        provenance,
        tables / "table_xai4heat_status.tex",
        "XAI4HEAT validation status and provenance. The package is used for measured-node thermal/energy validation only; no real pressure/head or flow field validation is claimed.",
        "tab:xai4heat_status",
    )
    write_latex_table(
        quality,
        tables / "table_xai4heat_station_quality_audit.tex",
        "XAI4HEAT station-level primary-temperature quality audit after file-content deduplication. Supply-below-return observations are retained in the primary all-valid-range evaluation; their exclusion is considered only in a separately reported physically ordered sensitivity analysis.",
        "tab:xai4heat_quality_audit",
    )
    write_figure(validation, by_substation)
    for marker_name in ["XAI4HEAT_NOT_RUN.txt", "XAI4HEAT_NOT_AVAILABLE_FINAL.txt"]:
        marker = results / marker_name
        if marker.exists():
            marker.unlink()
    print(results / "xai4heat_sparse_substation_validation.csv")


if __name__ == "__main__":
    main()
