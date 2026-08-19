from __future__ import annotations

import matplotlib.pyplot as plt

from ate_concept_figure_style import COLORS, add_box, clean_axis, read_csv, save_figure, set_style, wrap


def _evidence(objective: str) -> str:
    guidelines = read_csv("operator_sensor_guidelines.csv")
    if guidelines.empty:
        return ""
    sub = guidelines[guidelines["Operator objective"].astype(str).str.contains(objective, case=False, na=False)]
    if sub.empty:
        return ""
    return str(sub["Key evidence"].iloc[0])


def _short_evidence(text: str, fallback: str) -> str:
    text = str(text or "").strip()
    if not text:
        return fallback
    parts = [p.strip() for p in text.split(";") if p.strip()]
    if len(parts) >= 2:
        return f"{parts[0]}; {parts[1]}."
    if len(text) > 92:
        return text[:89].rstrip(" .,") + "."
    return text


def main() -> None:
    set_style()
    rows = [
        ("Direct thermal accuracy", "S12 or S10", _short_evidence(_evidence("Direct thermal"), "RMSE_Ts + RMSE_Tr objective."), COLORS["blue"]),
        ("Return-temperature monitoring", "S12 / S10", "Middle and outlet sensing improves return-temperature tracking.", COLORS["magenta"]),
        ("Heat-loss / physical consistency", "Optimized five-sensor or S10", _short_evidence(_evidence("Physical consistency"), "Heat-loss + energy + boundary objective."), COLORS["green"]),
        ("Low-cost monitoring", "S2 inlet + outlet; S1 only for coarse monitoring", _short_evidence(_evidence("Low-cost"), "Boundary layouts trade coverage for lower sensor count."), COLORS["gray"]),
        ("Energy-impact monitoring", "Optimized five-sensor", _short_evidence(_evidence("Energy-impact"), "Energy-impact uses proxy assumptions."), COLORS["orange"]),
        ("Robustness", "Optimized / five-sensor layout", "Use more coverage when dropout, bias, or stress robustness is prioritized.", COLORS["warning_red"]),
    ]

    fig, ax = plt.subplots(figsize=(12.7, 6.6))
    clean_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhspan(0.15, 0.87, color="#FBFBFF", zorder=-5)
    ax.set_title("Objective-specific sensor-placement guidance for operators", fontweight="bold", pad=11, fontsize=11.5)

    headers = [("Operator objective", 0.035, 0.26), ("Recommended layout", 0.31, 0.22), ("Compact evidence / practical note", 0.55, 0.40)]
    for text, x, w in headers:
        add_box(ax, x, 0.88, w, 0.075, text, facecolor=COLORS["black"], textcolor="white", fontsize=8.6, radius=0.025)

    y = 0.78
    row_h = 0.105
    for objective, layout, evidence, color in rows:
        text_color = COLORS["black"] if color == COLORS["yellow"] else "white"
        add_box(ax, 0.035, y, 0.26, row_h, wrap(objective, 24), facecolor=color, textcolor=text_color, fontsize=7.8, radius=0.025, lw=1.25)
        add_box(ax, 0.31, y, 0.22, row_h, wrap(layout, 22), facecolor="white", edgecolor=color, textcolor=COLORS["black"], fontsize=7.4, radius=0.025, lw=1.25)
        add_box(ax, 0.55, y, 0.40, row_h, wrap(evidence, 46), facecolor="white", edgecolor=COLORS["light_gray"], textcolor=COLORS["black"], fontsize=7.0, radius=0.025, lw=1.0)
        y -= row_h + 0.018

    ax.text(
        0.5,
        0.035,
        "Recommendations are objective-specific. No sensor layout is claimed as universally optimal.\n"
        "Cost and CO2 are proxy indicators; pressure/head and flow are simulator-assisted hidden hydraulic states.",
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["black"],
        fontweight="bold",
    )
    save_figure(fig, "fig_operator_sensor_guidelines_matrix")


if __name__ == "__main__":
    main()
