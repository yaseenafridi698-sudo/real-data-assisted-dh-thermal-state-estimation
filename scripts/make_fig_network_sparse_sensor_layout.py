from __future__ import annotations

from ate_concept_figure_style import COLORS, add_arrow, add_box, clean_axis, draw_circle, save_figure, set_style
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    clean_axis(ax)
    ax.set_xlim(-1.2, 21.2)
    ax.set_ylim(-2.25, 3.95)

    ax.axhspan(0.55, 1.85, color="#FFF7F2", zorder=-5)
    ax.axhspan(-1.35, -0.25, color="#F5F7FF", zorder=-5)

    supply_y, return_y, hidden_y = 1.25, -0.78, 0.19
    ax.plot([0, 20], [supply_y, supply_y], color=COLORS["orange"], lw=5.5, solid_capstyle="round", zorder=1)
    ax.plot([20, 0], [return_y, return_y], color=COLORS["blue"], lw=5.5, solid_capstyle="round", zorder=1)
    ax.plot([0, 20], [hidden_y, hidden_y], color=COLORS["green"], lw=2.0, ls=(0, (6, 4)), alpha=0.95, zorder=0)

    add_box(ax, -0.85, 0.43, 1.42, 1.64, "Plant\npump\ninlet", facecolor=COLORS["dark_blue"], fontsize=8.1)
    add_box(ax, 19.42, -1.52, 1.18, 3.33, "Consumer\nload\nreturn", facecolor=COLORS["dark_orange"], fontsize=7.8)

    add_arrow(ax, (1.1, 1.68), (3.15, 1.68), color=COLORS["black"], lw=1.5)
    add_arrow(ax, (18.9, -1.16), (16.75, -1.16), color=COLORS["black"], lw=1.5)
    ax.text(3.35, 1.76, "Supply flow and thermal transport", fontsize=8.3, color=COLORS["black"])
    ax.text(12.3, -1.33, "Return flow and pressure-drop direction", fontsize=8.3, color=COLORS["black"])

    measured = [0, 10, 20]
    virtual = [5, 7.5, 12.5, 15]
    for x in measured:
        draw_circle(ax, x, supply_y, 0.15, facecolor=COLORS["black"], edgecolor=COLORS["black"], zorder=5)
        draw_circle(ax, x, return_y, 0.15, facecolor=COLORS["black"], edgecolor=COLORS["black"], zorder=5)
    for x in virtual:
        draw_circle(ax, x, supply_y, 0.14, facecolor="white", edgecolor=COLORS["blue"], lw=2.0, zorder=5)
        draw_circle(ax, x, return_y, 0.14, facecolor="white", edgecolor=COLORS["blue"], lw=2.0, zorder=5)

    for x in np.linspace(2, 18, 6):
        add_arrow(ax, (x, 1.05), (x, 0.66), color=COLORS["dark_orange"], lw=1.25)
        add_arrow(ax, (x, -0.59), (x, -0.28), color=COLORS["dark_orange"], lw=1.25)
    ax.text(7.55, 0.55, "Heat loss to ambient", fontsize=8.5, color=COLORS["dark_orange"], fontweight="bold")

    legend_y = 2.73
    add_box(ax, 0.8, legend_y, 4.0, 0.44, "Measured sparse sensors", facecolor="white", edgecolor=COLORS["black"], textcolor=COLORS["black"], fontsize=8.0, radius=0.025)
    draw_circle(ax, 1.18, legend_y + 0.22, 0.095, facecolor=COLORS["black"])
    add_box(ax, 5.4, legend_y, 4.0, 0.44, "Virtual sensor estimates", facecolor="white", edgecolor=COLORS["blue"], textcolor=COLORS["black"], fontsize=8.0, radius=0.025)
    draw_circle(ax, 5.78, legend_y + 0.22, 0.095, facecolor="white", edgecolor=COLORS["blue"], lw=1.9)
    add_box(ax, 10.0, legend_y, 6.7, 0.44, "Simulator-assisted distributed states", facecolor="white", edgecolor=COLORS["green"], textcolor=COLORS["black"], fontsize=8.0, radius=0.025)
    ax.plot([10.4, 11.05], [legend_y + 0.22, legend_y + 0.22], color=COLORS["green"], lw=2.0, ls=(0, (6, 4)))

    ax.text(0, -1.84, "0 km", ha="center", fontsize=8.5)
    ax.text(10, -1.84, "10 km", ha="center", fontsize=8.5)
    ax.text(20, -1.84, "20 km", ha="center", fontsize=8.5)
    ax.plot([0, 20], [-1.68, -1.68], color=COLORS["black"], lw=1.1)
    for x in [0, 10, 20]:
        ax.plot([x, x], [-1.74, -1.62], color=COLORS["black"], lw=1.1)

    ax.text(
        10,
        -2.17,
        "Pressure/head and flow are simulator-assisted hidden hydraulic states.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=COLORS["black"],
        fontweight="bold",
    )
    ax.set_title("Sparse-sensor district-heating network and virtual-sensing layout", fontweight="bold", pad=12, fontsize=11.5)
    save_figure(fig, "fig_network_sparse_sensor_layout")


if __name__ == "__main__":
    main()
