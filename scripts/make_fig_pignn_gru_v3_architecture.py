from __future__ import annotations

from ate_concept_figure_style import COLORS, add_arrow, add_box, clean_axis, save_figure, set_style
import matplotlib.pyplot as plt


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    clean_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.axhspan(0.54, 0.83, color="#F7F4FF", zorder=-5)
    ax.axhspan(0.23, 0.45, color="#F7FFF3", zorder=-5)

    add_box(ax, 0.035, 0.59, 0.18, 0.20, "Inputs\nsparse sensors\nboundaries\nambient/load\ngraph topology", facecolor=COLORS["dark_blue"], fontsize=8.0)
    add_box(ax, 0.285, 0.61, 0.155, 0.16, "Residual graph\nencoder", facecolor=COLORS["dark_magenta"], fontsize=8.6)
    add_box(ax, 0.510, 0.61, 0.155, 0.16, "GRU temporal\nmemory", facecolor=COLORS["magenta"], fontsize=8.6)
    add_box(ax, 0.745, 0.59, 0.205, 0.20, "Decoder heads\nTs, Tr, H, q\nQloss, Qload\nuncertainty", facecolor=COLORS["dark_orange"], fontsize=8.0)

    add_arrow(ax, (0.225, 0.69), (0.278, 0.69), color=COLORS["black"], lw=1.7)
    add_arrow(ax, (0.448, 0.69), (0.502, 0.69), color=COLORS["black"], lw=1.7)
    add_arrow(ax, (0.673, 0.69), (0.738, 0.69), color=COLORS["black"], lw=1.7)

    losses = [
        ("data + sensor", 0.075),
        ("thermal transport", 0.235),
        ("heat loss", 0.395),
        ("energy balance", 0.555),
        ("boundary /\npressure drop", 0.715),
    ]
    for text, x in losses:
        add_box(ax, x, 0.355, 0.135, 0.090, text, facecolor="white", edgecolor=COLORS["green"], textcolor=COLORS["black"], fontsize=7.3, radius=0.025, lw=1.25)

    add_box(ax, 0.285, 0.205, 0.43, 0.075, "Physics-informed objective with normalized residual scales", facecolor=COLORS["green"], edgecolor=COLORS["black"], textcolor=COLORS["black"], fontsize=8.0, radius=0.025)
    for _, x in losses:
        add_arrow(ax, (x + 0.067, 0.355), (x + 0.067, 0.285), color=COLORS["dark_green"], lw=0.9, rad=0.0)

    add_box(ax, 0.055, 0.060, 0.205, 0.090, "Interpolation residual\nbaseline + correction", facecolor="white", edgecolor=COLORS["blue"], textcolor=COLORS["black"], fontsize=7.3, radius=0.025, lw=1.35)
    add_box(ax, 0.292, 0.060, 0.205, 0.090, "Sensor-mask-aware\nfeature fusion", facecolor="white", edgecolor=COLORS["blue"], textcolor=COLORS["black"], fontsize=7.3, radius=0.025, lw=1.35)
    add_box(ax, 0.528, 0.060, 0.205, 0.090, "Multi-head output\nstate + heat + residuals", facecolor="white", edgecolor=COLORS["blue"], textcolor=COLORS["black"], fontsize=7.3, radius=0.025, lw=1.35)

    add_box(ax, 0.760, 0.055, 0.195, 0.115, "Monitoring outputs\nvirtual sensors\nuncertainty\nanomaly residuals", facecolor="white", edgecolor=COLORS["warning_red"], textcolor=COLORS["black"], fontsize=7.5, radius=0.025, lw=1.35)
    add_arrow(ax, (0.850, 0.59), (0.855, 0.180), color=COLORS["warning_red"], lw=1.15)

    ax.text(
        0.5,
        0.925,
        "PI-GNN-GRU-v3: topology-aware sparse-sensor correction with physics-informed residuals",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color=COLORS["black"],
    )
    ax.text(
        0.5,
        0.006,
        "Hydraulic outputs H and q are evaluated as simulator-assisted hidden states when dense real hydraulic measurements are unavailable.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["black"],
    )
    save_figure(fig, "fig_pignn_gru_v3_architecture")


if __name__ == "__main__":
    main()
