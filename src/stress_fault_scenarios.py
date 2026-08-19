from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, compact_model_name, first_existing, read_csv, save_grouped_bar_figure, write_latex_table


def run() -> None:
    stress = first_existing(["combined_stress_test.csv", "combined_stress_test_improved.csv"])
    anomaly = first_existing(["anomaly_detection_metrics_improved.csv", "anomaly_detection_metrics.csv"])
    if stress.empty:
        out = pd.DataFrame([{"status": "not run", "note": "Combined stress source results were not available."}])
    else:
        out = stress.copy()
        out["model_short"] = out["model"].map(compact_model_name)
        case_map = {
            "return_bias_only": "return_sensor_bias_plus_2C",
            "sensor_dropout_only": "outlet_sensor_dropout",
            "combined_stress_moderate": "combined_operational_stress",
            "combined_stress_severe": "combined_operational_stress",
        }
        out["anomaly_case"] = out["case"].map(case_map).fillna(out["case"])
        if not anomaly.empty:
            keep = anomaly[["case", "detection_rate_percent", "false_alarm_rate_percent", "detection_delay_min", "best_F1_score"]].copy()
            out = out.merge(keep, left_on="anomaly_case", right_on="case", how="left", suffixes=("", "_anomaly"))
        out["safe_claim"] = "Stress and fault scenarios are controlled perturbations applied to real operating profiles; they are not documented field fault events."
        out["state_type"] = "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state"
    out.to_csv(PROJECT_ROOT / "results" / "stress_fault_scenario_metrics.csv", index=False)

    ts = read_csv("anomaly_detection_category_scores.csv")
    if ts.empty:
        ts = read_csv("anomaly_detection_timeseries_improved.csv")
    if not ts.empty:
        ts.to_csv(PROJECT_ROOT / "results" / "stress_fault_scenario_timeseries.csv", index=False)
    else:
        pd.DataFrame([{"status": "not run", "note": "No anomaly/stress time series source was available."}]).to_csv(
            PROJECT_ROOT / "results" / "stress_fault_scenario_timeseries.csv", index=False
        )

    table_cols = [
        "case",
        "model_short",
        "supply_RMSE_C",
        "return_RMSE_C",
        "heat_loss_error_percent",
        "energy_balance_residual_percent",
        "pressure_drop_error_percent",
        "max_temperature_error_C",
        "max_head_error_m",
        "detection_rate_percent",
        "false_alarm_rate_percent",
        "recovery_time_min",
    ]
    write_latex_table(
        out[[c for c in table_cols if c in out.columns]],
        TABLES_DIR / "table_stress_fault_scenarios.tex",
        "Limited stress/fault scenarios. Perturbations are controlled modifications of real operating profiles, not documented field fault events.",
        "tab:stress_fault_scenarios",
    )
    plot_summary(out)
    plot_examples(ts)


def plot_summary(df: pd.DataFrame) -> None:
    sub = df[df.get("model_short", "").astype(str).isin(["GRU", "Transformer", "PI-GNN-v3 acc.", "PI-GNN-v3 bal."])].copy() if not df.empty else df
    save_grouped_bar_figure(
        "fig_stress_fault_scenario_summary",
        [
            {"title": "Heat-loss impact", "data": sub, "category": "case", "series": "model_short", "value": "heat_loss_error_percent", "ylabel": "Error (%)"},
            {"title": "Energy residual", "data": sub, "category": "case", "series": "model_short", "value": "energy_balance_residual_percent", "ylabel": "Residual (%)"},
            {"title": "Pressure-drop residual", "data": sub, "category": "case", "series": "model_short", "value": "pressure_drop_error_percent", "ylabel": "Residual (%)"},
        ],
        title="Controlled stress/fault scenarios from real operating profiles",
    )


def plot_examples(ts: pd.DataFrame) -> None:
    if not ts.empty:
        score_col = "return_ewma_score" if "return_ewma_score" in ts.columns else ("residual_score" if "residual_score" in ts.columns else ts.select_dtypes("number").columns[-1])
        agg = ts.groupby("case", as_index=False)[score_col].max()
    else:
        agg = pd.DataFrame()
    save_grouped_bar_figure(
        "fig_stress_fault_anomaly_examples",
        [{"title": "Maximum residual/anomaly score", "data": agg, "category": "case", "value": score_col if not ts.empty else "value", "ylabel": "Score"}],
        title="Anomaly score examples for controlled perturbations",
        ncols=1,
        width=1200,
        height=700,
    )


if __name__ == "__main__":
    run()
