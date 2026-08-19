from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures" / "final"

COLORS = {
    "blue": "#0000E6",
    "orange": "#FF6626",
    "green": "#55D600",
    "yellow": "#F2E600",
    "magenta": "#E600E6",
    "black": "#111111",
    "gray": "#555555",
    "light_gray": "#D9D9D9",
    "warning_red": "#9B2226",
    "dark_blue": "#001A8D",
    "dark_orange": "#C84A16",
    "dark_green": "#2E8B00",
    "dark_yellow": "#B8A900",
    "dark_magenta": "#9B009B",
    "white": "#FFFFFF",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 1200,
            "font.family": "serif",
            "font.serif": [
                "Latin Modern Roman",
                "CMU Serif",
                "STIX Two Text",
                "STIXGeneral",
                "Times New Roman",
                "DejaVu Serif",
            ],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["black"],
            "axes.linewidth": 1.0,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.2,
            "grid.color": COLORS["light_gray"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.65,
            "savefig.bbox": "tight",
        }
    )


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 1200) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        target = FIGURES / f"{stem}{suffix}"
        if suffix == ".pdf":
            fig.savefig(target, bbox_inches="tight")
        else:
            fig.savefig(target, dpi=dpi, bbox_inches="tight")
        shutil.copy2(target, PAPER_FIGURES / f"{stem}{suffix}")
    plt.close(fig)


def wrap(text: str, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str = "white",
    fontsize: float = 8.5,
    radius: float = 0.06,
    lw: float = 1.2,
    zorder: int = 2,
):
    edge = edgecolor or COLORS["black"]
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight="bold",
        linespacing=1.08,
        zorder=zorder + 1,
    )
    return patch


def add_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str | None = None, lw: float = 1.6, rad: float = 0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color or COLORS["black"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=3,
    )
    ax.add_patch(arrow)
    return arrow


def add_panel_label(ax, label: str, x: float = 0.0, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["black"],
        clip_on=False,
    )


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def model_short(label: str) -> str:
    text = str(label)
    replacements = {
        "Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-GRU-v3\naccuracy",
        "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-GRU-v3\nbalanced",
        "Proposed PI-GNN-GRU-v3 physics_mode": "PI-GNN-GRU-v3\nphysics",
        "Proposed PI-GNN-GRU-v3": "PI-GNN-GRU-v3",
        "Transformer-MSE": "Transformer",
        "GRU-MSE": "GRU-MSE",
        "Interpolation": "Interpolation",
    }
    return replacements.get(text, text.replace("Proposed ", "").replace("-MSE", ""))


def rank_cmap(max_rank: int = 6):
    colors = [
        COLORS["blue"],
        COLORS["green"],
        COLORS["yellow"],
        COLORS["orange"],
        COLORS["magenta"],
        COLORS["gray"],
    ]
    cmap = ListedColormap(colors[:max_rank])
    norm = BoundaryNorm(np.arange(0.5, max_rank + 1.5, 1), cmap.N)
    return cmap, norm


def contact_sheet(stems: list[str], output_name: str, *, title: str = "Conceptual and non-simulation figure package") -> None:
    images: list[tuple[str, Image.Image]] = []
    for stem in stems:
        path = FIGURES / f"{stem}.png"
        if path.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((900, 520), Image.Resampling.LANCZOS)
            images.append((stem, im.copy()))
    if not images:
        return
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    cell_w, cell_h = 980, 620
    header_h = 90
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((36, 28), title, fill=COLORS["black"])
    for i, (stem, im) in enumerate(images):
        col = i % cols
        row = i // cols
        x0 = col * cell_w + 35
        y0 = header_h + row * cell_h + 35
        draw.rectangle([x0 - 12, y0 - 12, x0 + 910, y0 + 545], outline=COLORS["light_gray"], width=2)
        sheet.paste(im, (x0, y0))
        draw.text((x0, y0 + im.height + 12), stem, fill=COLORS["black"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / output_name
    sheet.save(out, dpi=(300, 300))
    shutil.copy2(out, PAPER_FIGURES / output_name)


def draw_circle(ax, x: float, y: float, r: float, *, facecolor: str, edgecolor: str | None = None, lw: float = 1.2, zorder: int = 3):
    patch = Circle((x, y), r, facecolor=facecolor, edgecolor=edgecolor or COLORS["black"], linewidth=lw, zorder=zorder)
    ax.add_patch(patch)
    return patch
