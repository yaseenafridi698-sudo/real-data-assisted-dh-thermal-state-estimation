from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, compact_model_name, first_existing, save_grouped_bar_figure, write_latex_table


def run() -> None:
    raw = first_existing(["noise_dropout_robustness_final.csv", "noise_dropout_robustness.csv"])
    if raw.empty:
        stats = pd.DataFrame([{"status": "not run", "note": "Noise/dropout source results were not available."}])
        out = stats.copy()
    else:
        out = raw.copy()
        out["model_short"] = out.get("base_model", out.get("model", "")).map(compact_model_name)
        group_cols = [c for c in ["model_short", "condition", "sensor_layout", "noise_std_fraction"] if c in out.columns]
        metric_cols = [c for c in ["RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"] if c in out.columns]
        stats = out.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
        stats.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in stats.columns]
        stats["state_type"] = "real_measured_node + simulator_assisted_hidden_state"
        stats["safe_claim"] = "Noise/dropout tests are controlled robustness tests applied to real operating profiles."
    out.to_csv(PROJECT_ROOT / "results" / "sensor_noise_dropout_robustness_raw.csv", index=False)
    stats.to_csv(PROJECT_ROOT / "results" / "sensor_noise_dropout_robustness_stats.csv", index=False)
    write_latex_table(
        stats.head(30),
        TABLES_DIR / "table_sensor_noise_dropout_robustness.tex",
        "Sensor noise/dropout robustness. Tests are controlled perturbations applied to real operating profiles.",
        "tab:sensor_noise_dropout_robustness",
    )
    plot(stats)


def plot(stats: pd.DataFrame) -> None:
    sub = stats[stats["model_short"].astype(str).isin(["GRU", "Transformer", "PI-GNN-v3 bal.", "PI-GNN-v3 acc."])].copy() if not stats.empty and "model_short" in stats else stats
    save_grouped_bar_figure(
        "fig_sensor_noise_dropout_robustness",
        [
            {"title": "Supply-temperature robustness", "data": sub, "category": "condition", "series": "model_short", "value": "RMSE_Ts_full_mean", "ylabel": "RMSE (C)"},
            {"title": "Heat-loss robustness", "data": sub, "category": "condition", "series": "model_short", "value": "heat_loss_error_percent_mean", "ylabel": "Error (%)"},
        ],
        title="Sensor noise/dropout robustness under controlled perturbations",
        ncols=2,
        width=1350,
        height=680,
    )


if __name__ == "__main__":
    run()
