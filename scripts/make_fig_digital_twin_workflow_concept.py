from __future__ import annotations

from ate_concept_figure_style import COLORS, add_arrow, add_box, clean_axis, save_figure, set_style
import matplotlib.pyplot as plt


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(13.4, 5.1))
    clean_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.axhspan(0.50, 0.86, color="#F7F9FF", zorder=-5)
    ax.axhspan(0.16, 0.44, color="#F9FFF6", zorder=-5)

    blocks = [
        (0.030, 0.59, 0.135, 0.22, "Real DH data\nSonderborg\nFlensburg", COLORS["dark_blue"]),
        (0.198, 0.59, 0.125, 0.22, "Sparse sensors\nthermal nodes\nboundaries", COLORS["orange"]),
        (0.360, 0.59, 0.150, 0.22, "Calibrated\nthermo-hydraulic\nsimulator", COLORS["dark_green"]),
        (0.548, 0.59, 0.145, 0.22, "PI-GNN-GRU-v3\ngraph + GRU\nphysics", COLORS["dark_magenta"]),
        (0.730, 0.59, 0.125, 0.22, "Virtual sensors\nuncertainty\nanomaly", COLORS["blue"]),
        (0.888, 0.59, 0.090, 0.22, "Operational\nKPIs", COLORS["yellow"]),
    ]

    for x, y, w, h, text, color in blocks:
        text_color = COLORS["black"] if color == COLORS["yellow"] else "white"
        add_box(ax, x, y, w, h, text, facecolor=color, textcolor=text_color, fontsize=8.5, lw=1.35)

    for i in range(len(blocks) - 1):
        x, y, w, h, *_ = blocks[i]
        x2, y2, *_ = blocks[i + 1]
        add_arrow(ax, (x + w + 0.012, y + h / 2), (x2 - 0.012, y2 + h / 2), color=COLORS["black"], lw=1.5)

    lower = [
        (0.058, 0.23, 0.145, "Calibration\nmeasured-node\nvalidation", COLORS["dark_blue"]),
        (0.385, 0.23, 0.165, "Hidden fields\nTs, Tr, heat loss\nH/q assisted", COLORS["dark_green"]),
        (0.710, 0.23, 0.160, "Confidence bands\nresidual alarms\nsensor health", COLORS["magenta"]),
        (0.885, 0.23, 0.105, "Heat loss\nenergy\nproxy KPIs", COLORS["dark_orange"]),
    ]
    for x, y, w, text, color in lower:
        add_box(ax, x, y, w, 0.18, text, facecolor="white", edgecolor=color, textcolor=COLORS["black"], fontsize=7.6, radius=0.035, lw=1.5)

    add_arrow(ax, (0.42, 0.56), (0.46, 0.42), color=COLORS["dark_green"], lw=1.2, rad=0.15)
    add_arrow(ax, (0.77, 0.56), (0.78, 0.42), color=COLORS["magenta"], lw=1.2, rad=-0.12)

    ax.text(
        0.5,
        0.075,
        "Evidence boundary: real data support calibration and measured-node thermal validation; pressure/head and flow are simulator-assisted hidden hydraulic states.",
        ha="center",
        va="center",
        fontsize=8.3,
        color=COLORS["black"],
    )
    ax.set_title("Real-data-assisted sparse-sensor thermo-hydraulic digital-twin workflow", pad=12, fontweight="bold", fontsize=11.5)
    save_figure(fig, "fig_digital_twin_workflow_concept")


if __name__ == "__main__":
    main()
