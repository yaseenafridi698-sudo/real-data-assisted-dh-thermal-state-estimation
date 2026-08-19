from __future__ import annotations

import math
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
DATA = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures" / "final"

W, H = landscape((9.4 * inch, 5.75 * inch))
PNG_DPI = 1200

PALETTE = {
    "blue": "#0000E6",
    "orange": "#FF6626",
    "green": "#55D600",
    "yellow": "#F2E600",
    "magenta": "#E600E6",
    "black": "#111111",
    "gray": "#555555",
    "light_gray": "#E8E8E8",
    "pale_blue": "#E9F0FF",
    "pale_orange": "#FFF0EA",
    "pale_green": "#EFFBE8",
    "pale_magenta": "#FFF0FF",
    "white": "#FFFFFF",
}

MODEL_COLORS = {
    "Measured/reference": PALETTE["black"],
    "Calibration/reference": PALETTE["black"],
    "GRU-MSE": PALETTE["blue"],
    "Transformer-MSE": PALETTE["orange"],
    "PureGNN-MSE": PALETTE["gray"],
    "PI-GNN-GRU-v3": PALETTE["magenta"],
    "PI-GNN-GRU-v3 accuracy": PALETTE["magenta"],
    "PI-GNN-GRU-v3 balanced": PALETTE["green"],
}

FIGURE_SPECS: list[dict[str, str]] = []


def _hex(value: str) -> colors.Color:
    value = value.lstrip("#")
    return colors.Color(int(value[0:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:6], 16) / 255)


def _font_path(name: str) -> Path | None:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def setup_fonts() -> tuple[str, str]:
    regular = _font_path("times.ttf") or _font_path("DejaVuSerif.ttf")
    bold = _font_path("timesbd.ttf") or _font_path("DejaVuSerif-Bold.ttf")
    if regular and bold:
        pdfmetrics.registerFont(TTFont("ATE-Serif", str(regular)))
        pdfmetrics.registerFont(TTFont("ATE-Serif-Bold", str(bold)))
        return "ATE-Serif", "ATE-Serif-Bold"
    return "Times-Roman", "Times-Bold"


FONT, FONT_BOLD = setup_fonts()


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_processed(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(series: Iterable, default: float = np.nan) -> np.ndarray:
    return pd.to_numeric(pd.Series(series), errors="coerce").fillna(default).to_numpy(dtype=float)


def short_model(name: object) -> str:
    text = str(name)
    return (
        text.replace("Proposed ", "")
        .replace("PI-GNN-GRU-v3 accuracy_mode", "PI-GNN-GRU-v3")
        .replace("PI-GNN-GRU-v3 balanced_mode", "PI-GNN-GRU-v3")
        .replace("Transformer-MSE", "Transformer")
    )


def wrap(text: str, width: int = 26) -> list[str]:
    return textwrap.wrap(str(text), width=width, break_long_words=False)


def add_spec(stem: str, source: str, panels: str, label: str, note: str) -> None:
    FIGURE_SPECS.append(
        {
            "figure_file": f"figures/final/{stem}.pdf",
            "source": source,
            "panels": panels,
            "latex_label": label,
            "page": "pending LaTeX compilation",
            "note": note,
        }
    )


def finish_pdf(c: canvas.Canvas, stem: str) -> None:
    c.showPage()
    c.save()
    pdf_path = FIGURES / f"{stem}.pdf"
    png_path = FIGURES / f"{stem}.png"
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[0]
        bitmap = page.render(scale=PNG_DPI / 72)
        image = bitmap.to_pil()
        image.save(png_path)
        pdf.close()
    except Exception:
        # Fallback: keep PDF even if local rendering is unavailable.
        pass
    for suffix in [".pdf", ".png"]:
        src = FIGURES / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, PAPER_FIGURES / src.name)


def new_canvas(stem: str, title: str) -> canvas.Canvas:
    FIGURES.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(FIGURES / f"{stem}.pdf"), pagesize=(W, H))
    c.setTitle(title)
    c.setFillColor(_hex(PALETTE["black"]))
    c.setFont(FONT_BOLD, 15)
    c.drawString(28, H - 28, title)
    c.setStrokeColor(_hex(PALETTE["black"]))
    c.setLineWidth(0.7)
    c.line(28, H - 35, W - 28, H - 35)
    return c


def text(c: canvas.Canvas, x: float, y: float, value: str, size: float = 8.5, *, bold: bool = False, color: str = "black", align: str = "left") -> None:
    c.setFillColor(_hex(PALETTE[color] if color in PALETTE else color))
    c.setFont(FONT_BOLD if bold else FONT, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def box(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, *, fill: str, stroke: str = "black", txt: str = "white", size: float = 8.4) -> None:
    c.setFillColor(_hex(PALETTE[fill] if fill in PALETTE else fill))
    c.setStrokeColor(_hex(PALETTE[stroke] if stroke in PALETTE else stroke))
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 8, stroke=1, fill=1)
    lines = wrap(title, max(10, int(w / 5.2)))
    y0 = y + h / 2 + (len(lines) - 1) * size * 0.55
    for i, line in enumerate(lines):
        text(c, x + w / 2, y0 - i * size * 1.08, line, size, bold=True, color=txt, align="center")


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color: str = "black", width: float = 1.4) -> None:
    c.setStrokeColor(_hex(PALETTE[color] if color in PALETTE else color))
    c.setFillColor(_hex(PALETTE[color] if color in PALETTE else color))
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    ang = math.atan2(y2 - y1, x2 - x1)
    length = 8
    a1 = ang + math.pi * 0.82
    a2 = ang - math.pi * 0.82
    p1 = (x2 + length * math.cos(a1), y2 + length * math.sin(a1))
    p2 = (x2 + length * math.cos(a2), y2 + length * math.sin(a2))
    c.line(x2, y2, p1[0], p1[1])
    c.line(x2, y2, p2[0], p2[1])


def panel_label(c: canvas.Canvas, x: float, y: float, label: str) -> None:
    text(c, x, y, label, 10, bold=True)


def axes(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, xlabel: str = "", ylabel: str = "") -> None:
    c.setStrokeColor(_hex(PALETTE["black"]))
    c.setLineWidth(0.9)
    c.rect(x, y, w, h, stroke=1, fill=0)
    c.setStrokeColor(_hex("#D7D7D7"))
    c.setLineWidth(0.35)
    for i in range(1, 4):
        c.line(x, y + h * i / 4, x + w, y + h * i / 4)
        c.line(x + w * i / 4, y, x + w * i / 4, y + h)
    text(c, x + w / 2, y + h + 12, title, 9.2, bold=True, align="center")
    if xlabel:
        text(c, x + w / 2, y - 16, xlabel, 7.6, align="center")
    if ylabel:
        c.saveState()
        c.translate(x - 22, y + h / 2)
        c.rotate(90)
        text(c, 0, 0, ylabel, 7.6, align="center")
        c.restoreState()


def _scale(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lo = 0.0 if lo is None else lo
        hi = 1.0 if hi is None else hi
    else:
        lo = float(np.nanmin(finite)) if lo is None else float(lo)
        hi = float(np.nanmax(finite)) if hi is None else float(hi)
        if abs(hi - lo) < 1e-12:
            hi = lo + 1.0
    return (arr - lo) / (hi - lo), lo, hi


def line_chart(c: canvas.Canvas, x: float, y: float, w: float, h: float, series: list[dict], title: str, xlabel: str, ylabel: str, *, y_min: float | None = None, y_max: float | None = None) -> None:
    axes(c, x, y, w, h, title, xlabel, ylabel)
    all_y = np.concatenate([np.asarray(s["y"], dtype=float) for s in series if len(s["y"])])
    y_norm, lo, hi = _scale(all_y, y_min, y_max)
    text(c, x - 4, y - 2, f"{lo:.1f}", 6.5, align="right", color="gray")
    text(c, x - 4, y + h - 2, f"{hi:.1f}", 6.5, align="right", color="gray")
    for s in series:
        yy = np.asarray(s["y"], dtype=float)
        if yy.size == 0:
            continue
        step = max(1, int(math.ceil(yy.size / 180)))
        yy = yy[::step]
        xx = np.linspace(0, 1, len(yy))
        norm, _, _ = _scale(yy, lo, hi)
        pts = [(x + w * a, y + h * b) for a, b in zip(xx, norm)]
        c.setStrokeColor(_hex(s.get("color", PALETTE["blue"])))
        c.setLineWidth(s.get("width", 1.4))
        p = c.beginPath()
        p.moveTo(*pts[0])
        for pt in pts[1:]:
            p.lineTo(*pt)
        c.drawPath(p)
    # compact legend
    lx = x + 4
    ly = y + h - 12
    for i, s in enumerate(series[:4]):
        c.setStrokeColor(_hex(s.get("color", PALETTE["blue"])))
        c.setLineWidth(2)
        c.line(lx + i * w / 4, ly, lx + i * w / 4 + 12, ly)
        text(c, lx + i * w / 4 + 15, ly - 3, s.get("label", ""), 6.3)


def bar_chart(c: canvas.Canvas, x: float, y: float, w: float, h: float, labels: list[str], values: list[float], title: str, ylabel: str, *, colors_: list[str] | None = None, fmt: str = "{:.2f}") -> None:
    axes(c, x, y, w, h, title, "", ylabel)
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    ymax = max(float(np.nanmax(finite)) * 1.22, 1.0) if finite.size else 1.0
    colors_ = colors_ or [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["yellow"], PALETTE["magenta"], PALETTE["gray"]]
    n = max(1, len(labels))
    bw = w / (n * 1.45)
    for i, (lab, val) in enumerate(zip(labels, vals)):
        cx = x + w * (i + 0.5) / n
        bh = 0 if not np.isfinite(val) else h * max(0, val) / ymax
        c.setFillColor(_hex(colors_[i % len(colors_)]))
        c.setStrokeColor(_hex(PALETTE["black"]))
        c.setLineWidth(0.65)
        c.rect(cx - bw / 2, y, bw, bh, stroke=1, fill=1)
        if np.isfinite(val):
            text(c, cx, y + bh + 5, fmt.format(val), 6.5, align="center")
        else:
            text(c, cx, y + 6, "N/A", 6.5, bold=True, color="gray", align="center")
        for j, line in enumerate(wrap(lab, 9)[:2]):
            text(c, cx, y - 12 - j * 7, line, 6.3, align="center")


def heatmap(c: canvas.Canvas, x: float, y: float, w: float, h: float, matrix: np.ndarray, title: str, xlabels: list[str], ylabels: list[str], *, invert: bool = False) -> None:
    axes(c, x, y, w, h, title)
    m = np.asarray(matrix, dtype=float)
    maxv = np.nanmax(m) if np.isfinite(m).any() else 1.0
    minv = np.nanmin(m) if np.isfinite(m).any() else 0.0
    if abs(maxv - minv) < 1e-9:
        maxv = minv + 1.0
    rows, cols = m.shape
    cell_w, cell_h = w / cols, h / rows
    ramp = [PALETTE["blue"], PALETTE["green"], PALETTE["yellow"], PALETTE["orange"], PALETTE["magenta"]]
    for r in range(rows):
        for col in range(cols):
            val = m[r, col]
            idx = 0 if not np.isfinite(val) else int(np.clip((val - minv) / (maxv - minv) * (len(ramp) - 1), 0, len(ramp) - 1))
            if invert:
                idx = len(ramp) - 1 - idx
            c.setFillColor(_hex(ramp[idx]))
            c.setStrokeColor(_hex(PALETTE["black"]))
            c.setLineWidth(0.35)
            c.rect(x + col * cell_w, y + (rows - 1 - r) * cell_h, cell_w, cell_h, stroke=1, fill=1)
            if rows <= 7 and cols <= 7:
                text(c, x + (col + 0.5) * cell_w, y + (rows - r - 0.55) * cell_h, f"{val:.0f}", 6.5, align="center", color="black")
    for col, lab in enumerate(xlabels):
        text(c, x + (col + 0.5) * cell_w, y - 13, lab, 6.4, align="center")
    for r, lab in enumerate(ylabels):
        text(c, x - 5, y + (rows - r - 0.55) * cell_h, lab, 6.4, align="right")


def embed_image(c: canvas.Canvas, path: Path, x: float, y: float, w: float, h: float, title: str = "") -> None:
    if title:
        text(c, x + w / 2, y + h + 10, title, 9, bold=True, align="center")
    if not path.exists():
        c.setStrokeColor(_hex(PALETTE["gray"]))
        c.rect(x, y, w, h, stroke=1, fill=0)
        text(c, x + w / 2, y + h / 2, "source panel not available", 8, color="gray", align="center")
        return
    img = Image.open(path).convert("RGB")
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(ImageReader(img), x + (w - dw) / 2, y + (h - dh) / 2, dw, dh, preserveAspectRatio=True, mask="auto")


def crop_image(stem: str, suffix: str, box_frac: tuple[float, float, float, float]) -> Path:
    src = FIGURES / f"{stem}.png"
    out_dir = FIGURES / "_ordered_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_{suffix}.png"
    if src.exists():
        img = Image.open(src).convert("RGB")
        w, h = img.size
        l, t, r, b = box_frac
        img.crop((int(l * w), int(t * h), int(r * w), int(b * h))).save(out)
    return out


def panel_image_ref(spec: str) -> Path:
    if "::" not in spec:
        return FIGURES / f"{spec}.png"
    stem, part = spec.split("::", 1)
    thirds = {
        "first": (0.02, 0.10, 0.335, 0.92),
        "second": (0.34, 0.10, 0.665, 0.92),
        "third": (0.67, 0.10, 0.995, 0.92),
    }
    return crop_image(stem, part, thirds.get(part, (0.0, 0.0, 1.0, 1.0)))


def draw_projected_surface(c: canvas.Canvas, x: float, y: float, w: float, h: float, z: np.ndarray, title: str, unit: str) -> None:
    text(c, x + w / 2, y + h + 10, title, 9, bold=True, align="center")
    c.setStrokeColor(_hex(PALETTE["light_gray"]))
    c.setLineWidth(0.4)
    z = np.asarray(z, dtype=float)
    rows, cols = z.shape
    finite = z[np.isfinite(z)]
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 1.0
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    def proj(i: float, j: float, val: float) -> tuple[float, float]:
        px = x + w * (0.14 + 0.68 * j / max(cols - 1, 1) + 0.18 * i / max(rows - 1, 1))
        py = y + h * (0.18 + 0.56 * i / max(rows - 1, 1) - 0.23 * j / max(cols - 1, 1))
        py += h * 0.22 * (val - lo) / (hi - lo)
        return px, py
    ramp = [_hex(PALETTE["blue"]), _hex(PALETTE["green"]), _hex(PALETTE["yellow"]), _hex(PALETTE["orange"]), _hex(PALETTE["magenta"])]
    for i in range(rows - 1):
        for j in range(cols - 1):
            val = float(np.nanmean(z[i : i + 2, j : j + 2]))
            idx = int(np.clip((val - lo) / (hi - lo) * (len(ramp) - 1), 0, len(ramp) - 1))
            pts = [proj(i, j, z[i, j]), proj(i, j + 1, z[i, j + 1]), proj(i + 1, j + 1, z[i + 1, j + 1]), proj(i + 1, j, z[i + 1, j])]
            p = c.beginPath()
            p.moveTo(*pts[0])
            for pt in pts[1:]:
                p.lineTo(*pt)
            p.close()
            c.setFillColor(ramp[idx])
            c.setStrokeColor(_hex(PALETTE["black"]))
            c.setLineWidth(0.18)
            c.drawPath(p, stroke=1, fill=1)
    c.setStrokeColor(_hex(PALETTE["black"]))
    c.setLineWidth(0.8)
    c.line(x + w * 0.14, y + h * 0.16, x + w * 0.85, y + h * 0.16)
    c.line(x + w * 0.14, y + h * 0.16, x + w * 0.28, y + h * 0.74)
    c.line(x + w * 0.14, y + h * 0.16, x + w * 0.02, y + h * 0.39)
    text(c, x + w * 0.5, y + 2, "distance / time", 6.5, align="center")
    text(c, x + w - 3, y + h * 0.87, f"{unit}: {lo:.2g}-{hi:.2g}", 6.4, align="right", color="gray")


def metric_value(file: str, metric: str, model_contains: str | None = None) -> float:
    df = read_csv(file)
    if df.empty or "metric" not in df.columns:
        return float("nan")
    sub = df[df["metric"].astype(str).eq(metric)].copy()
    if model_contains and "model" in sub.columns:
        sub = sub[sub["model"].astype(str).str.contains(model_contains, regex=False)]
    if sub.empty:
        return float("nan")
    if "value" in sub.columns:
        return float(pd.to_numeric(sub.iloc[0]["value"], errors="coerce"))
    return float("nan")


def make_physical_system() -> None:
    stem = "fig_physical_system"
    c = new_canvas(stem, "Physical district-heating system and evidence boundary")
    # light background bands
    c.setFillColor(_hex(PALETTE["pale_blue"]))
    c.roundRect(26, 48, W - 52, H - 96, 14, fill=1, stroke=0)
    # supply/return pipes
    y_sup, y_ret = H * 0.57, H * 0.33
    x0, x1 = 96, W - 105
    c.setStrokeColor(_hex(PALETTE["orange"]))
    c.setLineWidth(8)
    c.line(x0, y_sup, x1, y_sup)
    c.setStrokeColor(_hex(PALETTE["blue"]))
    c.line(x1, y_ret, x0, y_ret)
    arrow(c, x0 + 15, y_sup + 14, x1 - 10, y_sup + 14, "orange", 2)
    arrow(c, x1 - 15, y_ret - 14, x0 + 10, y_ret - 14, "blue", 2)
    box(c, 34, y_sup - 30, 92, 60, "Heat source\n+ pump", fill="black")
    box(c, W - 126, (y_sup + y_ret) / 2 - 33, 92, 66, "Consumer\nload", fill="magenta")
    text(c, (x0 + x1) / 2, y_sup + 33, "Supply pipe: measured sparse thermal nodes + virtual sensors", 9, bold=True, color="orange", align="center")
    text(c, (x0 + x1) / 2, y_ret - 42, "Return pipe: heat extraction and return-temperature reconstruction", 9, bold=True, color="blue", align="center")
    # sensors and virtuals
    sensor_x = [x0, (x0 + x1) / 2, x1]
    for i, sx in enumerate(sensor_x):
        c.setFillColor(_hex(PALETTE["black"]))
        c.circle(sx, y_sup, 5.5, fill=1, stroke=0)
        c.circle(sx, y_ret, 5.5, fill=1, stroke=0)
        text(c, sx, y_sup + 15, ["inlet", "middle", "outlet"][i], 7.2, align="center")
    for sx in np.linspace(x0 + 50, x1 - 50, 7):
        c.setStrokeColor(_hex(PALETTE["gray"]))
        c.circle(float(sx), y_sup - 25, 3.5, stroke=1, fill=0)
    for sx in np.linspace(x0 + 75, x1 - 75, 5):
        arrow(c, float(sx), y_sup - 7, float(sx), y_sup - 31, "green", 0.9)
    text(c, W / 2, 72, "Pressure/head and flow are simulator-assisted hidden hydraulic states; dense real hydraulic fields are not measured in public data.", 8.3, bold=True, color="black", align="center")
    text(c, W / 2, H - 54, "20 km virtual pipeline | heat-loss arrows | measured nodes separated from hidden distributed states", 9.2, bold=True, color="gray", align="center")
    finish_pdf(c, stem)
    add_spec(stem, "conceptual schematic + evidence boundary from manuscript/results", "(a) physical pipe layout", "fig:physical_system", "Conceptual, no numeric result alteration.")


def make_dt_framework() -> None:
    stem = "fig_dt_framework"
    c = new_canvas(stem, "Real-data-assisted digital-twin framework")
    xs = [34, 132, 238, 350, 458, 560]
    ys = [H * 0.56] * len(xs)
    labels = [
        "Real data\nSønderborg\nFlensburg\nXAI4HEAT",
        "Sparse sensors\nboundary + measured nodes",
        "Calibrated\nthermo-hydraulic\nsimulator",
        "PI-GNN-GRU-v3\nbenchmark estimator",
        "Virtual sensors\nhidden-state\nreconstruction",
        "Uncertainty\nanomaly\nKPIs",
    ]
    fills = ["blue", "orange", "green", "magenta", "yellow", "black"]
    widths = [82, 88, 94, 92, 84, 78]
    for i, (x, y, lab, fill, ww) in enumerate(zip(xs, ys, labels, fills, widths)):
        txt = "black" if fill == "yellow" else "white"
        box(c, x, y, ww, 76, lab, fill=fill, txt=txt, size=7.6)
        if i < len(xs) - 1:
            arrow(c, x + ww + 5, y + 38, xs[i + 1] - 7, y + 38, "black", 1.4)
    # lower outputs
    lower = [
        ("Measured-node\nvalidation", "black"),
        ("Simulator-assisted\nhidden states", "gray"),
        ("External transfer\nand domain shift", "orange"),
        ("Operator\nsensor guidance", "green"),
    ]
    lx = [80, 225, 370, 515]
    for x, (lab, fill) in zip(lx, lower):
        box(c, x, 85, 115, 46, lab, fill=fill, txt="white" if fill != "yellow" else "black", size=7.2)
        arrow(c, x + 57, H * 0.56, x + 57, 136, fill, 1.1)
    text(c, W / 2, H - 58, "Sparse real data -> calibrated simulator -> graph-temporal virtual sensing -> uncertainty/anomaly/KPI layer", 9, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "conceptual workflow; datasets documented in data_availability_report.csv and XAI4HEAT validation CSV", "(a) workflow blocks", "fig:dt_framework", "Conceptual, evidence boundaries shown.")


def make_real_data_overview() -> None:
    stem = "fig_real_data_overview"
    c = new_canvas(stem, "Real operating-data overview: Sønderborg")
    df = read_processed("sonderborg_processed.csv")
    if not df.empty:
        n = min(len(df), 1536)
        sub = df.iloc[:n].copy()
        t = np.arange(n) / 96.0
        panels = [
            ("heat_load_kw", "Heat load", "MW", 1 / 1000),
            ("supply_temp_C", "Supply temperature", "°C", 1),
            ("return_temp_C", "Return temperature", "°C", 1),
            ("ambient_temp_C", "Ambient temperature", "°C", 1),
        ]
        positions = [(58, 250), (383, 250), (58, 78), (383, 78)]
        colors_ = [PALETTE["orange"], PALETTE["blue"], PALETTE["green"], PALETTE["magenta"]]
        for i, ((col, title, unit, scale), (x, y)) in enumerate(zip(panels, positions)):
            panel_label(c, x - 22, y + 134, f"({chr(97+i)})")
            yv = numeric(sub[col]) * scale
            line_chart(c, x, y, 255, 120, [{"x": t, "y": yv, "label": title, "color": colors_[i], "width": 1.8}], title, "time (days)", unit)
        text(c, W / 2, 50, "Real measured plant-level data used for boundary conditions, calibration, and measured-node thermal validation.", 8.2, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "data/processed/sonderborg_processed.csv", "(a) heat load; (b) supply; (c) return; (d) ambient", "fig:real_data", "Real measured plant-level operating data.")


def make_architecture() -> None:
    stem = "fig_pignn_architecture"
    c = new_canvas(stem, "PI-GNN-GRU-v3 architecture and physics-informed loss")
    # inputs
    box(c, 35, 260, 112, 56, "Sparse sensor\nfeatures\n+ mask", fill="blue", size=7.5)
    box(c, 35, 176, 112, 56, "Graph + edge\nfeatures\nlength, U, f", fill="orange", size=7.5)
    box(c, 35, 92, 112, 56, "Interpolation\nbaseline", fill="gray", size=7.5)
    box(c, 205, 220, 125, 82, "Residual graph\nconvolution blocks\n+ LayerNorm", fill="green", txt="black", size=7.4)
    box(c, 378, 220, 118, 82, "Temporal GRU\nmemory block", fill="magenta", size=7.4)
    box(c, 546, 220, 118, 82, "Multi-head\ndecoders", fill="yellow", txt="black", size=7.4)
    box(
        c,
        552,
        82,
        95,
        100,
        "Outputs\nTs, Tr\nH, q\nheat loss\nuncertainty\nanomaly",
        fill="yellow",
        txt="black",
        size=6.2,
    )
    for yy in [288, 204, 120]:
        arrow(c, 150, yy, 203, 260 if yy != 120 else 226, "black")
    arrow(c, 330, 260, 378, 260, "black")
    arrow(c, 496, 260, 546, 260, "black")
    arrow(c, 608, 220, 608, 184, "black")
    # losses
    loss_items = [("state", "blue"), ("sensor", "orange"), ("thermal", "green"), ("hydraulic", "magenta"), ("boundary", "yellow"), ("energy", "gray")]
    x0, y0 = 160, 58
    for i, (lab, fill) in enumerate(loss_items):
        box(c, x0 + i * 64, y0, 52, 30, lab, fill=fill, txt="black" if fill == "yellow" else "white", size=6.2)
    text(c, W / 2, 35, "Physics losses are normalized and curriculum-weighted; PI-GNN-GRU-v3 is benchmarked, not claimed as a universal best model.", 8.1, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "model/loss design from src/models.py, src/losses.py and manuscript", "(a) architecture; (b) normalized residual losses", "fig:pignn_architecture", "Conceptual architecture, no numeric values.")


def make_three_panel_from_existing(stem: str, title: str, profile_stem: str, error_stem: str, label: str, source: str) -> None:
    c = new_canvas(stem, title)
    left = crop_image(profile_stem, "left", (0.06, 0.13, 0.53, 0.91))
    right = crop_image(profile_stem, "right", (0.54, 0.13, 0.995, 0.91))
    paths = [left, right, FIGURES / f"{error_stem}.png"]
    titles = ["normal-load profile", "high-demand profile", "absolute-error field"]
    xs = [34, 260, 486]
    for i, (p, ttl, x) in enumerate(zip(paths, titles, xs)):
        panel_label(c, x, H - 59, f"({chr(97+i)})")
        embed_image(c, p, x, 79, 205, 278, ttl)
    text(c, W / 2, 54, "Distributed temperature labels are calibrated-simulator-assisted hidden states; real data provide boundary conditions and measured-node thermal validation.", 7.8, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, source, "(a) normal-load profile; (b) high-demand profile; (c) error field/surface", label, "Composed from existing reconstruction artifacts generated by the repository.")


def make_pressure_flow_three_panel(stem: str, title: str, stems: list[str], label: str, note: str) -> None:
    c = new_canvas(stem, title)
    xs = [34, 260, 486]
    titles = ["reference / profile", "reconstruction / response", "residual field"]
    for i, (s, ttl, x) in enumerate(zip(stems, titles, xs)):
        panel_label(c, x, H - 59, f"({chr(97+i)})")
        embed_image(c, panel_image_ref(s), x, 79, 205, 278, ttl)
    text(c, W / 2, 54, note, 7.6, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "existing generated hydraulic/flow reconstruction figures and CSV-backed diagnostics", "(a) reference/profile; (b) reconstruction/response; (c) residual field", label, note)


def make_heat_energy() -> None:
    stem = "fig_heat_energy"
    c = new_canvas(stem, "Heat-loss, delivered-heat, and energy-balance response")
    ts = read_csv("energy_balance_time_series.csv")
    op = read_csv("operational_energy_impact_timeseries.csv")
    loss = read_csv("heat_loss_profile_metrics.csv")
    if not ts.empty:
        n = min(len(ts), 384)
        sub = ts.iloc[:n]
        line_chart(c, 48, 246, 260, 115, [
            {"y": numeric(sub["measured_boundary_heat_load_kw"]) / 1000, "label": "heat load", "color": PALETTE["black"]},
            {"y": numeric(sub["simulator_delivered_heat_kw"]) / 1000, "label": "simulator", "color": PALETTE["blue"]},
            {"y": numeric(sub["pignn_v3_delivered_heat_kw"]) / 1000, "label": "PI-GNN", "color": PALETTE["magenta"]},
        ], "Delivered heat tracking", "time step", "MW")
        panel_label(c, 30, 365, "(a)")
        line_chart(c, 380, 246, 260, 115, [
            {"y": np.abs(numeric(sub["pignn_v3_energy_residual_kw"])) / 1000, "label": "residual", "color": PALETTE["orange"]},
        ], "Energy-balance residual", "time step", "MW")
        panel_label(c, 360, 365, "(b)")
    if not loss.empty:
        bar_chart(c, 48, 80, 260, 108, [f"{v:.0f}" for v in numeric(loss["segment_midpoint_km"])[:10]], numeric(loss["simulator_heat_loss_kW"])[:10].tolist(), "Segment heat loss", "kW", colors_=[PALETTE["green"]] * 10, fmt="{:.0f}")
        panel_label(c, 30, 193, "(c)")
        line_chart(c, 380, 80, 260, 108, [
            {"y": np.cumsum(numeric(loss["simulator_heat_loss_kW"])), "label": "simulator", "color": PALETTE["black"]},
            {"y": np.cumsum(numeric(loss["pignn_v3_heat_loss_kW"])), "label": "PI-GNN", "color": PALETTE["magenta"]},
        ], "Cumulative heat loss", "segment", "kW")
        panel_label(c, 360, 193, "(d)")
    if not op.empty and "heat_loss_kw" in op.columns:
        ratio = 100 * numeric(op["heat_loss_kw"]) / np.maximum(numeric(op["delivered_heat_kw"]), 1)
        line_chart(c, 230, 31, 220, 44, [{"y": ratio[:384], "label": "ratio", "color": PALETTE["yellow"], "width": 1.5}], "Operational heat-loss ratio", "time", "%")
    finish_pdf(c, stem)
    add_spec(stem, "results/energy_balance_time_series.csv; results/heat_loss_profile_metrics.csv; results/operational_energy_impact_timeseries.csv", "(a) delivered heat; (b) residual; (c) heat-loss profile; (d) cumulative loss; small ratio panel", "fig:heat_energy", "Heat loss is calibrated-model-derived engineering KPI.")


def make_model_tradeoff() -> None:
    stem = "fig_model_tradeoff"
    c = new_canvas(stem, "Objective-dependent model comparison")
    rank = read_csv("model_ranking_by_metric_final.csv")
    base = read_csv("baseline_comparison_final.csv")
    key_models = ["GRU-MSE", "Transformer-MSE", "PureGNN-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"]
    metrics = ["RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean", "RMSE_q_full"]
    if not rank.empty:
        piv = rank[rank["model"].isin(key_models) & rank["metric"].isin(metrics)].pivot_table(index="model", columns="metric", values="rank", aggfunc="min").reindex(key_models)
        heatmap(c, 64, 90, 320, 250, piv.to_numpy(dtype=float), "Rank heatmap (1 = best)", ["Ts", "Tr", "loss", "energy", "boundary", "flow"], [short_model(m).replace("\n", " ")[:15] for m in key_models], invert=True)
        panel_label(c, 35, 347, "(a)")
    if not base.empty:
        sub = base[base["model"].isin(key_models)].copy()
        direct = numeric(sub["RMSE_Ts_full"]) + numeric(sub["RMSE_Tr_full"])
        physics = numeric(sub["heat_loss_error_percent"]) + numeric(sub["energy_balance_residual"]) + 2 * numeric(sub["boundary_residual_mean"])
        axes(c, 445, 106, 165, 200, "Accuracy-physics tradeoff", "Ts+Tr RMSE (°C)", "physics score")
        xnorm, xlo, xhi = _scale(direct)
        ynorm, ylo, yhi = _scale(physics)
        for model, xn, yn in zip(sub["model"], xnorm, ynorm):
            label = short_model(model).split("\n")[0]
            col = MODEL_COLORS.get(label, PALETTE["magenta"] if "PI-GNN" in str(model) else PALETTE["gray"])
            c.setFillColor(_hex(col))
            c.setStrokeColor(_hex(PALETTE["black"]))
            c.circle(445 + 165 * xn, 106 + 200 * yn, 5.2, fill=1, stroke=1)
            text(c, 445 + 165 * xn + 7, 106 + 200 * yn - 2, label[:11], 6.1)
        panel_label(c, 420, 316, "(b)")
        text(c, 448, 88, f"{xlo:.2f}-{xhi:.2f}", 6.4, color="gray")
        text(c, 586, 88, f"{ylo:.2f}-{yhi:.2f}", 6.4, color="gray", align="right")
    text(c, W / 2, 54, "Lower pointwise RMSE and lower thermo-hydraulic residuals are different objectives; no universal best model is claimed.", 8, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "results/model_ranking_by_metric_final.csv; results/baseline_comparison_final.csv", "(a) rank heatmap; (b) direct accuracy versus physics score", "fig:model_tradeoff", "Metric-dependent ranking, no best-overall claim.")


def make_xai4heat() -> None:
    stem = "fig_xai4heat"
    c = new_canvas(stem, "XAI4HEAT sparse-substation measured-node validation")
    df = read_csv("xai4heat_sparse_substation_validation_final.csv")
    if not df.empty:
        temps = df[df["category"].astype(str).eq("temperature")].copy()
        labs = [str(v).replace(" temperature", "").replace("Primary", "Prim.").replace("Secondary", "Sec.") for v in temps["variable_label"]]
        bar_chart(c, 50, 118, 280, 190, labs, numeric(temps["mean_RMSE"]).tolist(), "Measured-node temperature RMSE", "°C", colors_=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"]], fmt="{:.2f}")
        panel_label(c, 30, 314, "(a)")
        if "mean_nRMSE_percent" in temps.columns:
            bar_chart(c, 392, 118, 220, 190, labs, numeric(temps["mean_nRMSE_percent"]).tolist(), "Normalized RMSE", "%", colors_=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"]], fmt="{:.1f}")
            panel_label(c, 372, 314, "(b)")
        text(c, W / 2, 72, "Real sparse-substation thermal/energy validation only; XAI4HEAT does not measure dense pressure/head, flow, heat-loss, or pipe-state fields.", 8, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "results/xai4heat_sparse_substation_validation_final.csv", "(a) temperature RMSE; (b) normalized error", "fig:xai4heat", "Real measured-node validation; no hydraulic field validation.")


def make_flensburg() -> None:
    stem = "fig_flensburg"
    c = new_canvas(stem, "Flensburg external transfer and domain shift")
    ts = read_csv("external_validation_flensburg_timeseries.csv")
    modes = read_csv("external_validation_flensburg_modes_final.csv")
    shift = read_csv("flensburg_domain_shift_analysis.csv")
    if not ts.empty:
        sub = ts.iloc[: min(12, len(ts))]
        line_chart(c, 45, 230, 270, 110, [
            {"y": numeric(sub["measured_or_boundary_supply_C"]), "label": "measured/boundary", "color": PALETTE["black"], "width": 1.7},
            {"y": numeric(sub["predicted_supply_C"]), "label": "predicted", "color": PALETTE["magenta"], "width": 1.7},
        ], "Supply transfer time series", "window", "°C")
        panel_label(c, 28, 345, "(a)")
        line_chart(c, 45, 76, 270, 92, [{"y": numeric(sub["residual_C"]), "label": "residual", "color": PALETTE["orange"], "width": 1.7}], "Residual", "window", "°C")
        panel_label(c, 28, 172, "(b)")
    if not modes.empty:
        labels = modes["mode"].astype(str).str.replace("_", "\n").tolist()[:4]
        vals = numeric(modes["RMSE_supply_measured_C"])[:4].tolist()
        bar_chart(c, 385, 208, 220, 130, labels, vals, "Transfer modes", "supply RMSE (°C)", colors_=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"]], fmt="{:.2f}")
        panel_label(c, 365, 342, "(c)")
    if not shift.empty:
        def get(metric: str) -> float:
            sub = shift[shift["metric"].astype(str).eq(metric)]
            return float(pd.to_numeric(sub.iloc[0]["value"], errors="coerce")) if not sub.empty else np.nan
        labels = ["Sønd.\nheat MW", "Flens.\nheat MW", "Sønd.\nsupply °C", "Flens.\nsupply °C"]
        vals = [get("mean_heat_load_sonderborg_kw") / 1000, get("mean_heat_load_flensburg_kw") / 1000, get("supply_temp_mean_sonderborg_C"), get("supply_temp_mean_flensburg_C")]
        bar_chart(c, 385, 76, 220, 82, labels, vals, "Domain-shift anchors", "value", colors_=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"]], fmt="{:.1f}")
        panel_label(c, 365, 164, "(d)")
    text(c, W / 2, 52, "Flensburg is a domain-shift stress test; local calibration/adaptation is required for cross-network use.", 8, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "results/external_validation_flensburg_timeseries.csv; results/external_validation_flensburg_modes_final.csv; results/flensburg_domain_shift_analysis.csv", "(a) supply transfer; (b) residual; (c) transfer modes; (d) domain-shift anchors", "fig:flensburg", "External domain-shift validation; return may be assumed where unavailable.")


def make_robustness() -> None:
    stem = "fig_robustness"
    c = new_canvas(stem, "Robustness, uncertainty, and anomaly indicators")
    unc = read_csv("uncertainty_quantification_metrics.csv")
    rob = read_csv("thermo_hydraulic_robustness.csv")
    anom = read_csv("anomaly_detection_metrics_improved.csv")
    if not unc.empty:
        q90 = unc[unc["interval"].astype(str).eq("90%")].head(5)
        labels = q90["quantity"].astype(str).str.replace("_", "\n").tolist()
        bar_chart(c, 42, 216, 265, 120, labels, numeric(q90["coverage"]).tolist(), "Uncertainty coverage", "%", colors_=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["magenta"], PALETTE["yellow"]], fmt="{:.1f}")
        panel_label(c, 25, 340, "(a)")
    if not rob.empty:
        sub = rob[rob["model"].astype(str).str.contains("PI-GNN-GRU-v3", regex=False)].head(6)
        labels = sub["condition"].astype(str).str.replace("_", "\n").tolist()
        bar_chart(c, 372, 216, 265, 120, labels, numeric(sub["heat_loss_error_percent"]).tolist(), "Heat-loss robustness", "%", colors_=[PALETTE["magenta"]] * max(1, len(sub)), fmt="{:.2f}")
        panel_label(c, 354, 340, "(b)")
    if not anom.empty:
        labels = anom["case"].astype(str).str.replace("_", "\n").tolist()[:6]
        vals = numeric(anom["detection_rate_percent"])[:6].tolist()
        bar_chart(c, 120, 70, 420, 95, labels, vals, "Residual-anomaly detection rate", "%", colors_=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["yellow"], PALETTE["magenta"], PALETTE["gray"]], fmt="{:.0f}")
        panel_label(c, 100, 170, "(c)")
    text(c, W / 2, 44, "Uncertainty bands and anomaly cases are operational diagnostics; anomalies are controlled perturbations, not documented field faults.", 7.8, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "results/uncertainty_quantification_metrics.csv; results/thermo_hydraulic_robustness.csv; results/anomaly_detection_metrics_improved.csv", "(a) coverage; (b) robustness; (c) anomaly indicators", "fig:robustness", "Operational diagnostics; hydraulic quantities remain simulator-assisted.")


def make_combined_3d() -> None:
    stem = "fig_combined_3d"
    c = new_canvas(stem, "Combined 3D thermo-hydraulic reconstruction summary")
    # Use existing generated 3D field surfaces where available and create a flow diagnostic surface from saved flow time series.
    positions = [(38, 222), (350, 222), (38, 64), (350, 64)]
    panels = [
        ("main_3d_supply_temperature_surface", "supply-temperature field"),
        ("main_3d_return_temperature_surface", "return-temperature field"),
        ("main_3d_pressure_surface", "pressure/head field"),
        (None, "flow proxy diagnostic surface"),
    ]
    for i, ((stem_img, ttl), (x, y)) in enumerate(zip(panels, positions)):
        panel_label(c, x - 12, y + 135, f"({chr(97+i)})")
        if stem_img is not None and (FIGURES / f"{stem_img}.png").exists():
            embed_image(c, FIGURES / f"{stem_img}.png", x, y, 260, 130, ttl)
        else:
            op = read_csv("operational_energy_impact_timeseries.csv")
            flow = numeric(op["flow_m3_s"])[:160] if not op.empty and "flow_m3_s" in op.columns else np.linspace(0.2, 0.4, 160)
            rows, cols = 18, 20
            time = np.linspace(0, len(flow) - 1, rows).astype(int)
            base = flow[time]
            dist = np.linspace(0.98, 1.02, cols)
            z = np.outer(base, dist)
            draw_projected_surface(c, x, y, 260, 130, z, ttl, "m³/s")
    text(c, W / 2, 44, "3D panels summarize calibrated-simulator/reconstruction artifacts. Pressure/head and flow are simulator-assisted hidden hydraulic states.", 7.6, bold=True, align="center")
    finish_pdf(c, stem)
    add_spec(stem, "figures/final/main_3d_*_surface.png; results/operational_energy_impact_timeseries.csv for flow proxy surface", "(a) supply; (b) return; (c) pressure/head; (d) flow proxy", "fig:combined_3d", "3D field summary; dense flow measurements unavailable.")


def make_contact_sheet() -> None:
    stems = [
        "fig_physical_system",
        "fig_dt_framework",
        "fig_real_data_overview",
        "fig_pignn_architecture",
        "fig_supply_three_panel",
        "fig_return_three_panel",
        "fig_pressure_three_panel",
        "fig_flow_three_panel",
        "fig_heat_energy",
        "fig_model_tradeoff",
        "fig_xai4heat",
        "fig_flensburg",
        "fig_robustness",
        "fig_combined_3d",
    ]
    thumbs = []
    for s in stems:
        p = FIGURES / f"{s}.png"
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((540, 330), Image.Resampling.LANCZOS)
            thumbs.append((s, im.copy()))
    if not thumbs:
        return
    cols = 2
    cell_w, cell_h = 590, 385
    rows = math.ceil(len(thumbs) / cols)
    out = Image.new("RGB", (cols * cell_w, rows * cell_h + 60), "white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype(str(_font_path("times.ttf") or _font_path("DejaVuSerif.ttf")), 22)
        bold = ImageFont.truetype(str(_font_path("timesbd.ttf") or _font_path("DejaVuSerif-Bold.ttf")), 28)
    except Exception:
        font = bold = ImageFont.load_default()
    draw.text((24, 18), "Ordered 14-figure ATE main-paper package", fill=PALETTE["black"], font=bold)
    for idx, (s, im) in enumerate(thumbs):
        r, col = divmod(idx, cols)
        x = col * cell_w + 24
        y = 60 + r * cell_h
        draw.rectangle((x - 6, y - 6, x + cell_w - 30, y + cell_h - 20), outline=PALETTE["black"], width=2)
        out.paste(im, (x + 8, y + 24))
        draw.text((x + 8, y + 4), f"{idx+1}. {s}", fill=PALETTE["black"], font=font)
    out.save(FIGURES / "contact_sheet_ordered_14_main_figures.png")
    shutil.copy2(FIGURES / "contact_sheet_ordered_14_main_figures.png", PAPER_FIGURES / "contact_sheet_ordered_14_main_figures.png")


def write_report() -> None:
    df = pd.DataFrame(FIGURE_SPECS)
    df.to_csv(RESULTS / "main_figure_package_report.csv", index=False)
    lines = ["Ordered 14-figure ATE package report", ""]
    for i, row in enumerate(FIGURE_SPECS, 1):
        lines.append(f"{i}. {row['figure_file']} | label={row['latex_label']} | source={row['source']} | panels={row['panels']} | page={row['page']}")
        lines.append(f"   note: {row['note']}")
    (RESULTS / "main_figure_package_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    make_physical_system()
    make_dt_framework()
    make_real_data_overview()
    make_architecture()
    make_three_panel_from_existing(
        "fig_supply_three_panel",
        "Supply-temperature reconstruction: profiles and error field",
        "fig5_supply_temperature_reconstruction",
        "fig_temperature_error_spacetime",
        "fig:supply_three_panel",
        "figures/final/fig5_supply_temperature_reconstruction.png; figures/final/fig_temperature_error_spacetime.png",
    )
    make_three_panel_from_existing(
        "fig_return_three_panel",
        "Return-temperature reconstruction: profiles and error field",
        "fig6_return_temperature_reconstruction",
        "fig_return_temperature_error_spacetime",
        "fig:return_three_panel",
        "figures/final/fig6_return_temperature_reconstruction.png; figures/final/fig_return_temperature_error_spacetime.png",
    )
    make_pressure_flow_three_panel(
        "fig_pressure_three_panel",
        "Pressure/head-field reconstruction diagnostics",
        ["fig7_pressure_head_reconstruction::first", "fig7_pressure_head_reconstruction::second", "fig_head_error_spacetime"],
        "fig:pressure_three_panel",
        "Pressure/head is a simulator-assisted hidden hydraulic state; dense real hydraulic measurements are unavailable.",
    )
    make_pressure_flow_three_panel(
        "fig_flow_three_panel",
        "Flow reconstruction and flow-balance diagnostics",
        ["fig8_flow_reconstruction::first", "fig8_flow_reconstruction::second", "fig8_flow_reconstruction::third"],
        "fig:flow_three_panel",
        "Flow is simulator/proxy-assisted and not directly measured in the public operating datasets.",
    )
    make_heat_energy()
    make_model_tradeoff()
    make_xai4heat()
    make_flensburg()
    make_robustness()
    make_combined_3d()
    make_contact_sheet()
    write_report()
    print(FIGURES)
    print(RESULTS / "main_figure_package_report.csv")


if __name__ == "__main__":
    main()
