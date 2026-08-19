from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, first_existing, save_grouped_bar_figure, save_text_panel_figure, write_latex_table


def run() -> None:
    ranking = first_existing(["sensor_layout_ranking_by_objective.csv"])
    guidelines = first_existing(["operator_sensor_guidelines.csv"])
    energy = first_existing(["scenario_sensor_layout_energy_impact.csv"])
    if ranking.empty:
        out = pd.DataFrame([{"status": "not run", "note": "Sensor-layout ranking source was not available."}])
    else:
        out = ranking.copy()
        out["recommended_for"] = out["objective"]
        out["safe_claim"] = "Recommended layouts are objective-specific; no single sensor layout is universally optimal."
        if not energy.empty and "sensor_layout" in energy.columns:
            energy_cols = [c for c in ["sensor_layout", "pump_energy_proxy_error_percent", "energy_balance_residual_percent", "heat_loss_error_percent"] if c in energy.columns]
            out = out.merge(energy[energy_cols].drop_duplicates("sensor_layout"), on="sensor_layout", how="left")
    out.to_csv(PROJECT_ROOT / "results" / "sensor_placement_operator_guidelines.csv", index=False)
    write_latex_table(
        out.head(18),
        TABLES_DIR / "table_sensor_placement_operator_guidelines.tex",
        "Sensor-placement operator guidelines. Recommended layouts are objective-specific; no single sensor layout is universally optimal.",
        "tab:sensor_placement_operator_guidelines",
    )
    plot(out, guidelines)


def plot(df: pd.DataFrame, guidelines: pd.DataFrame) -> None:
    best = df.sort_values("rank").groupby("objective", as_index=False).first() if not df.empty and "objective" in df else pd.DataFrame()
    save_grouped_bar_figure(
        "fig_sensor_placement_operator_guidelines",
        [{"title": "Best layout score by objective", "data": best, "category": "objective", "value": "score", "ylabel": "Score"}],
        title="Operator sensor-placement guidelines by monitoring objective",
        ncols=1,
        width=1300,
        height=780,
    )
    if not guidelines.empty:
        rows = [(row.get("Operator objective", ""), row.get("Recommended layout", "")) for _, row in guidelines.head(5).iterrows()]
        save_text_panel_figure("fig_sensor_placement_operator_guidelines_matrix", rows, title="Objective-specific sensor-placement recommendations")


if __name__ == "__main__":
    run()
