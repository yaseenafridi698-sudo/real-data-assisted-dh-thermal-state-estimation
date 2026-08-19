from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, first_existing, save_grouped_bar_figure, write_json, write_latex_table


def run() -> None:
    sensitivity = first_existing(["parameter_identifiability_sensitivity.csv", "parameter_identifiability_sensitivity_improved.csv"])
    ranked = first_existing(["parameter_sensitivity_ranked.csv", "parameter_sensitivity_summary.csv"])
    rng = np.random.default_rng(42)
    if sensitivity.empty:
        mc = pd.DataFrame([{"status": "not run", "note": "Parameter sensitivity source results were not available."}])
        samples = pd.DataFrame()
        summary = mc.copy()
    else:
        metric_cols = [
            "supply_RMSE_C",
            "return_RMSE_C",
            "heat_loss_error_percent",
            "energy_balance_residual_percent",
            "pressure_drop_error_percent",
            "flow_RMSE_m3_s",
            "thermal_delay_error_min",
            "boundary_residual_mean_C",
        ]
        source = sensitivity.dropna(subset=[c for c in ["parameter"] if c in sensitivity.columns]).copy()
        source = source[source["parameter"].astype(str).ne("baseline")] if "parameter" in source else source
        if source.empty:
            source = sensitivity.copy()
        n = min(100, max(50, len(source) * 3))
        idx = rng.choice(source.index.to_numpy(), size=n, replace=True)
        mc = source.loc[idx, [c for c in ["case", "parameter", "perturbation", "model"] + metric_cols if c in source.columns]].reset_index(drop=True)
        mc.insert(0, "sample_id", np.arange(1, len(mc) + 1))
        mc["monte_carlo_method"] = "bootstrap of existing deterministic parameter-sensitivity cases"
        mc["state_type"] = "calibrated_simulator + simulator_assisted_hidden_state"
        mc["safe_claim"] = "Calibrated parameters are effective parameters for matching operating data, not independently measured pipe-material or hydraulic constants."
        samples = mc[["sample_id", "parameter", "perturbation", "case"]].copy()
        summary_rows = []
        for metric in metric_cols:
            if metric not in mc:
                continue
            vals = pd.to_numeric(mc[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            summary_rows.append(
                {
                    "metric": metric,
                    "mean": vals.mean(),
                    "p05": vals.quantile(0.05),
                    "p50": vals.quantile(0.50),
                    "p95": vals.quantile(0.95),
                    "state_type": "calibrated_simulator + simulator_assisted_hidden_state",
                    "interpretation": "Uncertainty propagation from existing effective-parameter sensitivity cases.",
                    "safe_claim": "This is a parameter-identifiability sensitivity result, not independent physical parameter measurement.",
                }
            )
        summary = pd.DataFrame(summary_rows)

    mc.to_csv(PROJECT_ROOT / "results" / "parameter_uncertainty_monte_carlo.csv", index=False)
    samples.to_csv(PROJECT_ROOT / "results" / "parameter_uncertainty_samples.csv", index=False)
    summary.to_csv(PROJECT_ROOT / "results" / "parameter_uncertainty_summary.csv", index=False)
    write_json(
        "parameter_uncertainty_assumptions.json",
        {
            "random_seed": 42,
            "method": "Bootstrap/response-summary of existing deterministic sensitivity outputs, not a new field measurement.",
            "safe_interpretation": "Effective-parameter identifiability and robustness analysis.",
        },
    )
    write_latex_table(
        summary,
        TABLES_DIR / "table_parameter_uncertainty_summary.tex",
        "Parameter uncertainty summary. Calibrated parameters are effective parameters for matching operating data, not independently measured constants.",
        "tab:parameter_uncertainty_summary",
    )
    plot_tornado(ranked, sensitivity)
    plot_boxplots(mc)


def plot_tornado(ranked: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    if not ranked.empty:
        score_col = "total_sensitivity_index" if "total_sensitivity_index" in ranked else ranked.select_dtypes("number").columns[-1]
        data = ranked.sort_values(score_col, ascending=False).head(10).copy()
        if "parameter" not in data:
            data["parameter"] = data.get("case", pd.Series(range(len(data)))).astype(str)
    else:
        data = pd.DataFrame()
        score_col = "value"
    save_grouped_bar_figure(
        "fig_parameter_uncertainty_tornado",
        [{"title": "Dominant effective-parameter sensitivities", "data": data, "category": "parameter", "value": score_col, "ylabel": "Sensitivity index"}],
        title="Parameter-identifiability tornado: effective calibrated parameters",
        ncols=1,
        width=1250,
        height=760,
    )


def plot_boxplots(mc: pd.DataFrame) -> None:
    metrics = ["heat_loss_error_percent", "energy_balance_residual_percent", "pressure_drop_error_percent", "supply_RMSE_C", "return_RMSE_C"]
    rows = []
    for metric in metrics:
        if metric not in mc:
            continue
        vals = pd.to_numeric(mc[metric], errors="coerce").dropna()
        if not vals.empty:
            rows.extend([
                {"metric": metric, "stat": "p05", "value": vals.quantile(0.05)},
                {"metric": metric, "stat": "median", "value": vals.quantile(0.50)},
                {"metric": metric, "stat": "p95", "value": vals.quantile(0.95)},
            ])
    data = pd.DataFrame(rows)
    save_grouped_bar_figure(
        "fig_parameter_uncertainty_boxplots",
        [{"title": "Output intervals under parameter uncertainty", "data": data, "category": "metric", "series": "stat", "value": "value", "ylabel": "Metric value"}],
        title="Output uncertainty from effective-parameter perturbations",
        ncols=1,
        width=1500,
        height=760,
    )


if __name__ == "__main__":
    run()
