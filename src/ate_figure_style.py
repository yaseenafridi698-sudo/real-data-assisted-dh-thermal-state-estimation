from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


PALETTE = {
    "proposed": "#0000E6",
    "gru": "#55D600",
    "lstm": "#55D600",
    "transformer": "#E600E6",
    "pilstm": "#FF6626",
    "pid": "#FF6626",
    "baseline": "#555555",
    "puregnn": "#F2E600",
    "pignn": "#0000E6",
    "measured": "#111111",
    "reference": "#111111",
    "safe": "#55D600",
    "warning": "#F2E600",
    "alarm": "#E600E6",
    "grid": "#D7D7D7",
    "edge": "#111111",
    "band": "#0000E6",
}

MODEL_COLORS = {
    "Proposed": PALETTE["proposed"],
    "Proposed PI-GNN-GRU-v3": PALETTE["proposed"],
    "Proposed PI-GNN-GRU-v3 accuracy_mode": PALETTE["proposed"],
    "Proposed PI-GNN-GRU-v3 balanced_mode": PALETTE["proposed"],
    "GRU": PALETTE["gru"],
    "GRU-MSE": PALETTE["gru"],
    "LSTM": PALETTE["lstm"],
    "LSTM-MSE": PALETTE["lstm"],
    "Transformer": PALETTE["transformer"],
    "Transformer-MSE": PALETTE["transformer"],
    "PI-LSTM": PALETTE["pilstm"],
    "PureGNN": PALETTE["puregnn"],
    "PureGNN-MSE": PALETTE["puregnn"],
    "Interpolation": PALETTE["baseline"],
    "Baseline": PALETTE["baseline"],
}


def set_ate_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
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
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": PALETTE["edge"],
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.4,
            "grid.alpha": 0.55,
            "savefig.bbox": "tight",
        }
    )


def short_model_label(label: str) -> str:
    text = str(label)
    replacements = {
        "Proposed PI-GNN-GRU-v3 accuracy_mode": "Proposed",
        "Proposed PI-GNN-GRU-v3 balanced_mode": "Proposed",
        "Proposed PI-GNN-GRU-v3 physics_mode": "Proposed",
        "Proposed PI-GNN-GRU-v3": "Proposed",
        "Transformer-MSE": "Transformer",
        "GRU-MSE": "GRU",
        "LSTM-MSE": "LSTM",
        "PureGNN-MSE": "PureGNN",
        "PI-GNN-no-temporal": "PI-GNN",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def model_color(label: str) -> str:
    short = short_model_label(label)
    return MODEL_COLORS.get(label, MODEL_COLORS.get(short, PALETTE["baseline"]))


def style_axes(ax, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["edge"])
    ax.spines["bottom"].set_color(PALETTE["edge"])


def style_legend(ax, **kwargs):
    defaults = {
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "#CFCFCF",
        "framealpha": 0.96,
        "borderpad": 0.35,
        "labelspacing": 0.35,
        "handlelength": 1.5,
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.055,
        1.035,
        label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=10.5,
        clip_on=False,
    )


def save_ate_figure(fig, out_dir: Path, stem: str, *, dpi: int = 1200) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ordered_colors(labels: Iterable[str]) -> list[str]:
    return [model_color(label) for label in labels]
