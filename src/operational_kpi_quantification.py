from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, first_existing, read_csv, save_grouped_bar_figure, write_json, write_latex_table


def run() -> None:
    summary = first_existing(["operational_energy_impact_summary.csv", "digital_twin_kpis.csv"])
    scenarios = []
    for name in ["scenario_nominal_winter_energy_impact.csv", "scenario_combined_stress_energy_impact.csv", "scenario_sensor_layout_energy_impact.csv"]:
        df = read_csv(name)
        if not df.empty:
            scenarios.append(df)
    if scenarios:
        scenario_df = pd.concat(scenarios, ignore_index=True, sort=False)
    else:
        scenario_df = pd.DataFrame()

    if summary.empty and scenario_df.empty:
        out = pd.DataFrame([{"status": "not run", "note": "Operational KPI source results were not available."}])
    elif not scenario_df.empty:
        out = scenario_df.copy()
        out["source"] = out.get("scenario", "scenario")
        out["safe_claim"] = "Cost and CO2 are proxy indicators under stated assumptions, not optimized economic-dispatch results."
        out["state_type"] = out.get("state_type", "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state")
    else:
        out = summary.copy()
    out.to_csv(PROJECT_ROOT / "results" / "operational_kpi_quantification.csv", index=False)

    timeseries = read_csv("operational_energy_impact_timeseries.csv")
    if timeseries.empty:
        timeseries = pd.DataFrame([{"status": "not run", "note": "Operational KPI time series was not available."}])
    timeseries.to_csv(PROJECT_ROOT / "results" / "operational_kpi_timeseries.csv", index=False)

    assumptions_path = PROJECT_ROOT / "results" / "operational_energy_impact_assumptions.json"
    assumptions = {
        "pump_efficiency_eta": 0.75,
        "pump_power_proxy": "P_pump_proxy = rho*g*Q*Delta_H/eta",
        "safe_interpretation": "Cost and CO2 are proxy indicators under stated assumptions, not optimized economic-dispatch results.",
    }
    if assumptions_path.exists():
        try:
            assumptions.update(json.loads(assumptions_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    write_json("operational_kpi_assumptions.json", assumptions)

    table_cols = [
        "scenario",
        "sensor_layout",
        "delivered_heat_MWh_per_day",
        "heat_loss_MWh_per_day",
        "heat_loss_ratio_percent",
        "pump_energy_proxy_kWh_per_day",
        "pressure_drop_residual_percent",
        "energy_balance_residual_percent",
        "cost_proxy_EUR_per_day",
        "CO2_proxy_kg_per_day",
    ]
    write_latex_table(
        out[[c for c in table_cols if c in out.columns]],
        TABLES_DIR / "table_operational_kpi_quantification.tex",
        "Operational KPI quantification. Cost and CO2 are proxy indicators under stated assumptions, not optimized economic-dispatch results.",
        "tab:operational_kpi_quantification",
    )
    plot(out)


def plot(df: pd.DataFrame) -> None:
    label_col = "sensor_layout" if "sensor_layout" in df.columns and df["sensor_layout"].notna().any() else ("source" if "source" in df.columns else "scenario")
    plot_df = df.dropna(subset=[label_col]).head(8).copy() if label_col in df else df
    save_grouped_bar_figure(
        "fig_operational_kpi_quantification",
        [
            {"title": "Delivered heat", "data": plot_df, "category": label_col, "value": "delivered_heat_MWh_per_day", "ylabel": "MWh/day"},
            {"title": "Heat loss", "data": plot_df, "category": label_col, "value": "heat_loss_MWh_per_day", "ylabel": "MWh/day"},
            {"title": "Pump-energy proxy", "data": plot_df, "category": label_col, "value": "pump_energy_proxy_kWh_per_day", "ylabel": "kWh/day"},
            {"title": "Pressure-drop residual", "data": plot_df, "category": label_col, "value": "pressure_drop_residual_percent", "ylabel": "%"},
            {"title": "Energy residual", "data": plot_df, "category": label_col, "value": "energy_balance_residual_percent", "ylabel": "%"},
            {"title": "Cost proxy", "data": plot_df, "category": label_col, "value": "cost_proxy_EUR_per_day", "ylabel": "EUR/day"},
        ],
        title="Operational KPI quantification from reconstructed thermo-hydraulic states",
        ncols=3,
        width=1700,
        height=950,
    )


if __name__ == "__main__":
    run()
