from __future__ import annotations

import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor, white as pdf_white
from reportlab.pdfgen import canvas as pdf_canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures" / "final"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

PALETTE = {
    "blue": "#0000E6",
    "orange": "#FF6626",
    "green": "#55D600",
    "yellow": "#F2E600",
    "magenta": "#E600E6",
    "black": "#111111",
    "gray": "#555555",
    "grid": "#D7D7D7",
    "panel": "#F8FAFF",
}


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_json(name: str, payload: dict) -> None:
    ensure_dir(RESULTS_DIR)
    (RESULTS_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, *, resize: bool = True) -> None:
    ensure_dir(path.parent)
    if df.empty:
        df = pd.DataFrame([{"status": "not run", "note": "Source result file was not available."}])
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    def esc(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    align = "l" * len(out.columns)
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{esc(caption)}}}",
        rf"\label{{{esc(label)}}}",
    ]
    if resize or len(out.columns) > 5:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(esc(c) for c in out.columns) + r" \\",
        r"\midrule",
    ])
    for _, row in out.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in out.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize or len(out.columns) > 5:
        lines.append(r"}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines) + "\n"
    latex = latex.replace(r"\$\textasciicircum \textbackslash circ\$C", r"$^\circ$C")
    path.write_text(latex, encoding="utf-8")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(names: Iterable[str]) -> pd.DataFrame:
    for name in names:
        df = read_csv(name)
        if not df.empty:
            return df
    return pd.DataFrame()


def compact_model_name(value: object) -> str:
    text = str(value)
    return (
        text.replace("Proposed PI-GNN-GRU-v3 accuracy_mode", "PI-GNN-v3 acc.")
        .replace("Proposed PI-GNN-GRU-v3 balanced_mode", "PI-GNN-v3 bal.")
        .replace("Proposed PI-GNN-GRU-v3 physics_mode", "PI-GNN-v3 phys.")
        .replace("Transformer-MSE", "Transformer")
        .replace("GRU-MSE", "GRU")
        .replace("LSTM-MSE", "LSTM")
    )


def add_safe_claim_columns(df: pd.DataFrame, state_type: str, interpretation: str, safe_claim: str) -> pd.DataFrame:
    out = df.copy()
    if "state_type" not in out.columns:
        out["state_type"] = state_type
    if "interpretation" not in out.columns:
        out["interpretation"] = interpretation
    if "safe_claim" not in out.columns:
        out["safe_claim"] = safe_claim
    return out


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    named = {"white": "#FFFFFF", "black": "#000000"}
    color = named.get(str(color).lower(), str(color))
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _pdf_color(color: str):
    if str(color).lower() == "white":
        return pdf_white
    return HexColor(color)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/cambriaz.ttf" if bold else "C:/Windows/Fonts/cambria.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


class _DrawAdapter:
    def __init__(self, mode: str, target, width: int, height: int):
        self.mode = mode
        self.target = target
        self.width = width
        self.height = height
        self.draw = target if mode == "png" else None

    def _pdf_y(self, y: float) -> float:
        return self.height - y

    def rect(self, x: float, y: float, w: float, h: float, fill: str, outline: str = "#111111", width: float = 1.0) -> None:
        if self.mode == "png":
            self.target.rectangle([x, y, x + w, y + h], fill=_hex_to_rgb(fill), outline=_hex_to_rgb(outline), width=max(1, int(width)))
        else:
            c = self.target
            c.setFillColor(_pdf_color(fill))
            c.setStrokeColor(_pdf_color(outline))
            c.setLineWidth(width)
            c.rect(x, self._pdf_y(y + h), w, h, fill=1, stroke=1)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#111111", width: float = 1.0) -> None:
        if self.mode == "png":
            self.target.line([x1, y1, x2, y2], fill=_hex_to_rgb(color), width=max(1, int(width)))
        else:
            c = self.target
            c.setStrokeColor(_pdf_color(color))
            c.setLineWidth(width)
            c.line(x1, self._pdf_y(y1), x2, self._pdf_y(y2))

    def text(self, x: float, y: float, text: str, size: int = 18, color: str = "#111111", bold: bool = False, anchor: str = "la") -> None:
        text = str(text)
        if self.mode == "png":
            font = _font(size, bold)
            self.target.text((x, y), text, fill=_hex_to_rgb(color), font=font, anchor=anchor)
        else:
            c = self.target
            c.setFillColor(_pdf_color(color))
            c.setFont("Times-Bold" if bold else "Times-Roman", size)
            if anchor in {"ma", "mm"}:
                c.drawCentredString(x, self._pdf_y(y + size), text)
            elif anchor in {"ra", "rm"}:
                c.drawRightString(x, self._pdf_y(y + size), text)
            else:
                c.drawString(x, self._pdf_y(y + size), text)


def _save_drawn(stem: str, width: int, height: int, draw_func) -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(PAPER_FIGURES_DIR)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    c = pdf_canvas.Canvas(str(pdf_path), pagesize=(width, height))
    draw_func(_DrawAdapter("pdf", c, width, height))
    c.showPage()
    c.save()
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_func(_DrawAdapter("png", draw, width, height))
    img.save(png_path, dpi=(1200, 1200))
    for path in [pdf_path, png_path]:
        shutil.copy2(path, PAPER_FIGURES_DIR / path.name)


def _wrap(label: object, width: int = 16) -> list[str]:
    return textwrap.wrap(str(label).replace("_", " "), width=width)[:3] or [""]


def save_grouped_bar_figure(
    stem: str,
    panels: list[dict],
    *,
    title: str,
    width: int = 1700,
    height: int = 850,
    ncols: int = 3,
) -> None:
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"], PALETTE["yellow"], PALETTE["gray"]]
    n = len(panels)
    nrows = int(math.ceil(n / ncols))
    margin_x, top, bottom = 70, 95, 80
    gap_x, gap_y = 45, 80
    panel_w = (width - 2 * margin_x - gap_x * (ncols - 1)) / ncols
    panel_h = (height - top - bottom - gap_y * (nrows - 1)) / nrows

    def draw(g: _DrawAdapter) -> None:
        g.rect(0, 0, width, height, "white", "white")
        g.text(width / 2, 28, title, size=28, bold=True, anchor="ma")
        for i, panel in enumerate(panels):
            row, col = divmod(i, ncols)
            x0 = margin_x + col * (panel_w + gap_x)
            y0 = top + row * (panel_h + gap_y)
            g.rect(x0, y0, panel_w, panel_h, PALETTE["panel"], "#CFCFCF", 1)
            g.text(x0 + 12, y0 + 12, f"({chr(97+i)}) {panel.get('title','')}", size=17, bold=True)
            df = panel.get("data", pd.DataFrame()).copy()
            cat = panel.get("category")
            series = panel.get("series")
            value = panel.get("value")
            ylabel = panel.get("ylabel", "")
            plot_x0, plot_y0 = x0 + 55, y0 + 60
            plot_w, plot_h = panel_w - 85, panel_h - 120
            g.line(plot_x0, plot_y0 + plot_h, plot_x0 + plot_w, plot_y0 + plot_h, PALETTE["black"], 1.4)
            g.line(plot_x0, plot_y0, plot_x0, plot_y0 + plot_h, PALETTE["black"], 1.4)
            if df.empty or cat not in df.columns or value not in df.columns:
                g.text(x0 + panel_w / 2, y0 + panel_h / 2, "not run", size=20, anchor="ma")
                continue
            df[value] = pd.to_numeric(df[value], errors="coerce")
            df = df.dropna(subset=[value])
            if df.empty:
                g.text(x0 + panel_w / 2, y0 + panel_h / 2, "N/A", size=20, anchor="ma")
                continue
            if series and series in df.columns:
                pivot = df.pivot_table(index=cat, columns=series, values=value, aggfunc="mean").head(6)
            else:
                pivot = df.groupby(cat)[value].mean().head(8).to_frame("value")
            pivot = pivot.apply(pd.to_numeric, errors="coerce").fillna(0)
            ymax = max(float(np.nanmax(pivot.to_numpy())), 1e-9) * 1.18
            cats = list(pivot.index.astype(str))
            ser_names = list(pivot.columns.astype(str))
            group_w = plot_w / max(len(cats), 1)
            bar_w = group_w * 0.72 / max(len(ser_names), 1)
            for ci, cname in enumerate(cats):
                cx = plot_x0 + ci * group_w + group_w * 0.14
                for si, sname in enumerate(ser_names):
                    val = float(pivot.iloc[ci, si])
                    bh = max(0, val / ymax * (plot_h - 8))
                    bx = cx + si * bar_w
                    by = plot_y0 + plot_h - bh
                    g.rect(bx, by, bar_w * 0.92, bh, colors[si % len(colors)], PALETTE["black"], 0.8)
                for li, line in enumerate(_wrap(cname, 12)):
                    g.text(plot_x0 + ci * group_w + group_w / 2, plot_y0 + plot_h + 14 + li * 14, line, size=10, anchor="ma")
            g.text(plot_x0 - 42, plot_y0 + plot_h / 2, ylabel, size=12, anchor="ma")
            for si, sname in enumerate(ser_names[:5]):
                lx = plot_x0 + si * 105
                ly = y0 + panel_h - 35
                g.rect(lx, ly, 14, 10, colors[si % len(colors)], PALETTE["black"], 0.5)
                g.text(lx + 18, ly - 2, str(sname)[:15], size=10)

    _save_drawn(stem, width, height, draw)


def save_text_panel_figure(stem: str, rows: list[tuple[str, str]], *, title: str, width: int = 1400, height: int = 780) -> None:
    def draw(g: _DrawAdapter) -> None:
        g.rect(0, 0, width, height, "white", "white")
        g.text(width / 2, 34, title, size=28, bold=True, anchor="ma")
        y = 115
        colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"], PALETTE["yellow"]]
        for i, (left, right) in enumerate(rows):
            g.rect(70, y, width - 140, 78, "#F8FAFF", colors[i % len(colors)], 2)
            g.text(95, y + 18, left, size=18, bold=True, color=colors[i % len(colors)])
            for li, line in enumerate(_wrap(right, 80)):
                g.text(420, y + 16 + li * 18, line, size=15)
            y += 100

    _save_drawn(stem, width, height, draw)
