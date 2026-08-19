from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ate_concept_figure_style import COLORS, contact_sheet, read_csv, save_figure, set_style


CONCEPT_STEMS = [
    "fig_digital_twin_workflow_concept",
    "fig_network_sparse_sensor_layout",
    "fig_pignn_gru_v3_architecture",
    "fig_model_value_rank_heatmap",
    "fig_operator_sensor_guidelines_matrix",
    "fig_operational_digital_twin_kpi_dashboard",
]


def _scenario_rows() -> pd.DataFrame:
    rows = []
    nominal = read_csv("scenario_nominal_winter_energy_impact.csv")
    stress = read_csv("scenario_combined_stress_energy_impact.csv")
    layout = read_csv("scenario_sensor_layout_energy_impact.csv")
    if not nominal.empty:
        r = nominal.iloc[0].copy()
        r["label"] = "Nominal winter\n24 h"
        rows.append(r)
    if not stress.empty:
        severe = stress[stress.get("scenario", pd.Series(dtype=str)).astype(str).eq("combined_stress_severe")]
        r = (severe.iloc[0] if not severe.empty else stress.iloc[-1]).copy()
        r["label"] = "Combined stress\nper day"
        rows.append(r)
    if not layout.empty:
        opt = layout[layout.get("sensor_layout", pd.Series(dtype=str)).astype(str).str.contains("S10", na=False)]
        r = (opt.iloc[0] if not opt.empty else layout.iloc[-1]).copy()
        r["label"] = "Optimized sensors\nper day"
        rows.append(r)
        low = layout[layout.get("sensor_layout", pd.Series(dtype=str)).astype(str).str.contains("S2", na=False)]
        if not low.empty:
            r2 = low.iloc[0].copy()
            r2["label"] = "Inlet + outlet\nper day"
            rows.append(r2)
    return pd.DataFrame(rows)


def _vals(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = []
    for _, row in df.iterrows():
        value = np.nan
        for col in cols:
            if col in df.columns:
                candidate = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
                if np.isfinite(candidate):
                    value = float(candidate)
                    break
        out.append(value)
    return np.array(out, dtype=float)


def _format_value(v: float) -> str:
    if not np.isfinite(v):
        return "N/A"
    av = abs(v)
    if av >= 100000:
        return f"{v/1000:.0f}k"
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def main() -> None:
    set_style()
    df = _scenario_rows()
    if df.empty:
        return
    labels = df["label"].astype(str).to_list()
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["gray"]][: len(labels)]
    edge = COLORS["black"]

    panels = [
        ("Delivered heat\n(MWh/day)", ["delivered_heat_MWh_per_day"]),
        ("Heat loss\n(MWh/day)", ["heat_loss_MWh_per_day"]),
        ("Pump-energy proxy\n(kWh/MWh delivered)", ["normalized_pump_energy_proxy_kWh_per_MWh"]),
        ("Pressure-drop\nresidual (%)", ["pressure_drop_residual_percent", "maximum_pressure_drop_residual_percent"]),
        ("Energy-balance\nresidual (%)", ["energy_balance_residual_percent", "mean_energy_balance_residual_percent", "maximum_energy_balance_residual_percent"]),
        ("Cost / CO2 proxy\n(EUR/day; kg/day)", ["cost_proxy_EUR_per_day"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.1))
    axes = axes.ravel()
    x = np.arange(len(labels))
    for ax, (title, cols) in zip(axes, panels):
        vals = _vals(df, cols)
        if title.startswith("Cost"):
            co2_vals = _vals(df, ["CO2_proxy_kg_per_day"])
            bars = ax.bar(x - 0.17, vals, width=0.34, color=colors, edgecolor=edge, linewidth=1.0, zorder=3)
            ax2 = ax.twinx()
            ax2.scatter(x + 0.17, co2_vals, marker="D", s=42, color=COLORS["magenta"], edgecolor=edge, linewidth=0.7, zorder=4)
            ax2.set_ylabel("CO2 proxy (kg/day)", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7.3)
            ax.set_ylabel("Cost proxy (EUR/day)", fontsize=8)
            if np.isfinite(co2_vals).any() and np.nanmax(np.abs(co2_vals)) > 10000:
                ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            for xi, v in zip(x + 0.17, co2_vals):
                if not np.isfinite(v):
                    ax2.text(xi, 0.5, "N/A", ha="center", va="bottom", fontsize=7.0, color=COLORS["gray"], transform=ax2.get_xaxis_transform())
        else:
            bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), color=colors, edgecolor=edge, linewidth=1.0, zorder=3)
        ax.set_title(title, fontweight="bold", fontsize=9.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=7.5)
        ax.grid(True, axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            ymax = max(np.nanmax(np.abs(finite)) * 1.20, 1.0)
            if np.nanmin(finite) >= 0:
                ax.set_ylim(0, ymax)
        for bar, value in zip(bars, vals):
            if not np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.04, "N/A", ha="center", va="bottom", fontsize=7.0, color=COLORS["gray"], transform=ax.get_xaxis_transform(), fontweight="bold")
            elif abs(value) < 1e-12:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.04, "0", ha="center", va="bottom", fontsize=7.0, color=COLORS["black"], transform=ax.get_xaxis_transform(), fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), _format_value(value), ha="center", va="bottom", fontsize=6.8, color=COLORS["black"])
        if finite.size and np.nanmax(np.abs(finite)) > 10000:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec=edge) for c in colors]
    handles.append(plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["magenta"], markeredgecolor=edge, markersize=6, label="CO2 proxy marker"))
    legend_labels = labels + ["CO2 proxy marker"]
    fig.legend(handles, legend_labels, loc="upper center", ncol=min(5, len(legend_labels)), frameon=True, bbox_to_anchor=(0.5, 1.00), fontsize=8.0)
    fig.suptitle("Operational digital-twin KPI dashboard", y=1.055, fontweight="bold", fontsize=11.7)
    fig.text(
        0.5,
        0.005,
        "Values use stated horizons or normalized units. Cost and CO2 are proxy indicators, not optimized dispatch results. "
        "Pump and pressure-drop quantities use simulator-assisted hidden hydraulic states.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["black"],
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.93])
    save_figure(fig, "fig_operational_digital_twin_kpi_dashboard")
    contact_sheet(CONCEPT_STEMS, "contact_sheet_conceptual_non_simulation_figures.png")


if __name__ == "__main__":
    main()
