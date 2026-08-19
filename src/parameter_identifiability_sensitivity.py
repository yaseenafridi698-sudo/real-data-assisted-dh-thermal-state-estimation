from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.sensor_layouts import apply_sensor_layout
from src.supplementary_study_utils import (
    boundary_from_sim,
    copy_final_figures_to_root_and_paper,
    evaluate_models_on_sim,
    load_calibrated_params,
    load_models_for_sim,
    load_sonderborg_processed,
    save_figure,
    simulate_from_dataframe,
    write_latex_table,
)
from src.thermo_hydraulic_simulator import _causal_flow_proxy_from_load, simulate_thermo_hydraulics
from src.utils import ensure_dir


SELECTED_MODELS = {
    "GRU-MSE",
    "Transformer-MSE",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
}


def _base_context(config: dict, params: dict) -> tuple[dict, dict, dict, dict]:
    df = load_sonderborg_processed(config)
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    df = df.head(max(220, min(max_steps, 768))).copy()
    sim = simulate_from_dataframe(df, config, params)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config)
    trained, loaders = load_models_for_sim(sim, sensors, config)
    return sim, sensors, trained, loaders


def _case_definitions() -> list[tuple[str, str, float | None, str]]:
    cases: list[tuple[str, str, float | None, str]] = [("baseline", "baseline", None, "baseline calibrated effective parameters")]
    for factor in [0.8, 1.2]:
        cases.append((f"heat_loss_U_{factor:.1f}x", "heat_loss_U_W_m2K", factor, "effective heat-loss coefficient perturbation"))
        cases.append((f"friction_factor_{factor:.1f}x", "friction_factor", factor, "effective friction-factor perturbation"))
    for bias in [-2.0, -1.0, 1.0, 2.0]:
        cases.append((f"return_temperature_bias_{bias:+.0f}C", "return_temperature_bias_C", bias, "return-temperature bias perturbation"))
    for factor in [0.9, 1.1]:
        cases.append((f"flow_proxy_{factor:.1f}x", "flow_proxy", factor, "heat-load-based flow proxy perturbation"))
    for factor in [0.85, 1.15]:
        cases.append((f"effective_velocity_{factor:.2f}x", "effective_velocity_factor", factor, "effective velocity/delay-factor perturbation"))
    cases.extend(
        [
            ("low_loss_low_friction", "combined", 0.8, "combined low heat-loss and low-friction effective parameters"),
            ("high_loss_high_friction", "combined", 1.2, "combined high heat-loss and high-friction effective parameters"),
            ("return_bias_flow_uncertainty", "return_bias_flow", 1.1, "combined return-temperature bias and flow-proxy uncertainty"),
        ]
    )
    return cases


def _simulate_case(base_sim: dict, config: dict, base_params: dict, case_name: str, parameter: str, value: float | None) -> dict:
    p = dict(base_params)
    boundary = boundary_from_sim(base_sim)
    if parameter == "baseline":
        return base_sim
    if parameter in {"heat_loss_U_W_m2K", "friction_factor", "effective_velocity_factor"}:
        default = config["system"].get(parameter, p.get(parameter, 1.0))
        p[parameter] = float(p.get(parameter, default)) * float(value)
    elif parameter == "return_temperature_bias_C":
        # A return-temperature measurement bias matters here only through the
        # causal, lagged-return flow proxy.  The former implementation retained
        # a precomputed q_proxy from the baseline, leaving this test inert.
        boundary["T_return_measured"] = boundary["T_return_measured"].copy() + float(value)
        boundary["q_proxy"] = _causal_flow_proxy_from_load(
            boundary["Q_load_W"], boundary["T_source"], boundary["T_return_measured"], config
        )
        boundary["flow_proxy_mode"] = "causal_lagged_return"
    elif parameter == "flow_proxy":
        boundary["q_proxy"] = boundary["q_proxy"].copy() * float(value)
    elif parameter == "combined":
        p["heat_loss_U_W_m2K"] = float(p.get("heat_loss_U_W_m2K", config["system"]["heat_loss_U_W_m2K"])) * float(value)
        p["friction_factor"] = float(p.get("friction_factor", config["system"]["friction_factor"])) * float(value)
    elif parameter == "return_bias_flow":
        boundary["T_return_measured"] = boundary["T_return_measured"].copy() + 2.0
        boundary["q_proxy"] = _causal_flow_proxy_from_load(
            boundary["Q_load_W"], boundary["T_source"], boundary["T_return_measured"], config
        ) * float(value)
        boundary["flow_proxy_mode"] = "causal_lagged_return_with_scale"
    return simulate_thermo_hydraulics(boundary, config, params=p)


def run_parameter_identifiability_sensitivity() -> None:
    config = load_config()
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    params = load_calibrated_params()
    base_sim, _, trained, loaders = _base_context(config, params)
    stats = loaders["train_ds"].stats

    rows = []
    propagation_rows = []
    base_q_proxy = np.asarray(base_sim["q_proxy"], dtype=float)
    base_return = np.asarray(base_sim["T_return_measured"], dtype=float)
    for case_name, parameter, value, note in _case_definitions():
        sim = _simulate_case(base_sim, config, params, case_name, parameter, value)
        q_proxy = np.asarray(sim["q_proxy"], dtype=float)
        return_boundary = np.asarray(sim["T_return_measured"], dtype=float)
        q_change_percent = float(
            100.0 * np.nanmean(np.abs(q_proxy - base_q_proxy)) / max(np.nanmean(np.abs(base_q_proxy)), 1e-12)
        )
        return_change_c = float(np.nanmean(return_boundary - base_return))
        requires_proxy_propagation = parameter in {"return_temperature_bias_C", "return_bias_flow"}
        propagation_rows.append(
            {
                "case": case_name,
                "parameter": parameter,
                "perturbation": value if value is not None else "baseline",
                "return_boundary_mean_change_C": return_change_c,
                "q_proxy_mean_absolute_change_percent": q_change_percent,
                "simulated_flow_mean_change_percent": float(
                    100.0 * np.nanmean(np.abs(np.asarray(sim["q"], dtype=float) - np.asarray(base_sim["q"], dtype=float)))
                    / max(np.nanmean(np.abs(np.asarray(base_sim["q"], dtype=float))), 1e-12)
                ),
                "propagation_status": (
                    "propagated"
                    if not requires_proxy_propagation or (abs(return_change_c) > 1e-9 and q_change_percent > 1e-9)
                    else "failed"
                ),
                "note": (
                    "Return-temperature bias is propagated through the causal lagged-return flow proxy."
                    if requires_proxy_propagation
                    else "Effective-parameter perturbation evaluated against the baseline trained estimators."
                ),
            }
        )
        result, _ = evaluate_models_on_sim(
            sim,
            config,
            trained,
            stats,
            layout="S4_five_sensors",
            case_label=case_name,
            note=(
                note
                + ". Calibrated parameters are effective parameters for matching plant-level data, not independently measured pipe/hydraulic properties."
            ),
            selected_models=SELECTED_MODELS,
        )
        result["parameter"] = parameter
        result["perturbation"] = value if value is not None else "baseline"
        rows.append(result)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(PROJECT_ROOT / "results" / "parameter_identifiability_sensitivity.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "parameter_identifiability_sensitivity_improved.csv", index=False)
    summary = (
        df.groupby(["case", "parameter"], as_index=False)
        .agg(
            mean_supply_RMSE_C=("supply_RMSE_C", "mean"),
            mean_return_RMSE_C=("return_RMSE_C", "mean"),
            mean_heat_loss_error_percent=("heat_loss_error_percent", "mean"),
            mean_pressure_drop_error_percent=("pressure_drop_error_percent", "mean"),
            mean_flow_RMSE_m3_s=("flow_RMSE_m3_s", "mean"),
            mean_energy_balance_residual_percent=("energy_balance_residual_percent", "mean"),
            mean_thermal_delay_error_min=("thermal_delay_error_min", "mean"),
            mean_boundary_residual_C=("boundary_residual_mean_C", "mean"),
        )
    )
    summary["interpretation"] = "Sensitivity of simulator-assisted hidden-state reconstruction to effective calibrated parameter uncertainty."
    summary.to_csv(PROJECT_ROOT / "results" / "parameter_sensitivity_summary.csv", index=False)
    ranked = _rank_parameter_sensitivity(summary)
    ranked.to_csv(PROJECT_ROOT / "results" / "parameter_sensitivity_ranked.csv", index=False)
    propagation = pd.DataFrame(propagation_rows)
    propagation.to_csv(PROJECT_ROOT / "results" / "parameter_sensitivity_propagation_audit.csv", index=False)
    if (propagation["propagation_status"] == "failed").any():
        failed = propagation.loc[propagation["propagation_status"] == "failed", "case"].tolist()
        raise RuntimeError(f"Parameter-sensitivity propagation failed for: {failed}")
    write_latex_table(
        df[[
            "case",
            "model",
            "parameter",
            "perturbation",
            "supply_RMSE_C",
            "return_RMSE_C",
            "pressure_drop_error_percent",
            "flow_RMSE_m3_s",
            "heat_loss_error_percent",
            "energy_balance_residual_percent",
            "thermal_delay_error_min",
        ]],
        PROJECT_ROOT / "paper" / "tables" / "table_parameter_identifiability_sensitivity.tex",
        "Parameter-identifiability sensitivity of effective thermo-hydraulic parameters. Parameters are interpreted as calibrated effective quantities rather than independently measured physical properties.",
        "tab:parameter_identifiability_sensitivity",
    )
    _plot_heat_loss(df)
    _plot_pressure_flow(df)
    _plot_energy(df)
    _plot_tornado(summary)
    _plot_tornado_improved(ranked)
    _plot_grouped_thermal_hydraulic(ranked)
    copy_final_figures_to_root_and_paper()
    print("Parameter-identifiability sensitivity supplementary study completed.")


def _filter_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"].astype(str).str.contains("PI-GNN-GRU-v3 balanced_mode|GRU-MSE|Transformer-MSE", regex=True)].copy()


def _bar_by_case(df: pd.DataFrame, metric: str, title: str, ylabel: str, stem: str, pattern: str) -> None:
    sub = _filter_for_plot(df)
    sub = sub[sub["case"].astype(str).str.contains(pattern, regex=True)].copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    if sub.empty:
        ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center")
        ax.axis("off")
    else:
        sub["model_short"] = sub["model"].str.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 ", regex=False)
        pivot = sub.pivot_table(index="case", columns="model_short", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=6)
    ax.set_title(title)
    save_figure(fig, stem)


def _plot_heat_loss(df: pd.DataFrame) -> None:
    _bar_by_case(
        df,
        "heat_loss_error_percent",
        "Heat-loss sensitivity to effective thermal parameters",
        "Heat-loss error (%)",
        "fig_parameter_sensitivity_heat_loss",
        "heat_loss|combined|baseline",
    )


def _plot_pressure_flow(df: pd.DataFrame) -> None:
    sub = _filter_for_plot(df)
    sub = sub[sub["case"].astype(str).str.contains("friction|flow_proxy|combined|baseline", regex=True)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, metric, ylabel in [
        (axes[0], "pressure_drop_error_percent", "Pressure-drop error (%)"),
        (axes[1], "flow_RMSE_m3_s", "Flow RMSE (m$^3$/s)"),
    ]:
        if sub.empty:
            ax.text(0.5, 0.5, "No sensitivity data", ha="center", va="center")
            ax.axis("off")
            continue
        sub = sub.copy()
        sub["model_short"] = sub["model"].str.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 ", regex=False)
        pivot = sub.pivot_table(index="case", columns="model_short", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("Pressure/head and flow sensitivity; hydraulic states are simulator-assisted hidden states")
    save_figure(fig, "fig_parameter_sensitivity_pressure_flow")


def _plot_energy(df: pd.DataFrame) -> None:
    _bar_by_case(
        df,
        "energy_balance_residual_percent",
        "Energy-residual sensitivity to effective parameter uncertainty",
        "Energy residual (%)",
        "fig_parameter_sensitivity_energy_residual",
        "heat_loss|friction|velocity|flow|return|combined|baseline",
    )


def _plot_tornado(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    baseline = summary[summary["case"].eq("baseline")]
    base = float(baseline["mean_heat_loss_error_percent"].iloc[0]) if not baseline.empty else 0.0
    plot_df = summary[~summary["case"].eq("baseline")].copy()
    plot_df["delta_heat_loss_error"] = (plot_df["mean_heat_loss_error_percent"] - base).abs()
    plot_df = plot_df.sort_values("delta_heat_loss_error", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh(plot_df["case"], plot_df["delta_heat_loss_error"], color="#577590")
    ax.set_xlabel("Absolute change in mean heat-loss error (percentage points)")
    ax.set_title("Parameter-identifiability tornado plot")
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, "fig_parameter_identifiability_tornado")


def _rank_parameter_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    baseline = summary[summary["case"].eq("baseline")]
    base = baseline.iloc[0] if not baseline.empty else None
    metric_cols = [
        "mean_supply_RMSE_C",
        "mean_return_RMSE_C",
        "mean_heat_loss_error_percent",
        "mean_pressure_drop_error_percent",
        "mean_flow_RMSE_m3_s",
        "mean_energy_balance_residual_percent",
        "mean_thermal_delay_error_min",
        "mean_boundary_residual_C",
    ]
    rows = []
    for _, row in summary[~summary["case"].eq("baseline")].iterrows():
        thermal = 0.0
        hydraulic = 0.0
        total = 0.0
        for col in metric_cols:
            base_val = float(base[col]) if base is not None and col in base else 0.0
            delta = abs(float(row.get(col, np.nan)) - base_val) if np.isfinite(row.get(col, np.nan)) else 0.0
            total += delta
            if col in {"mean_supply_RMSE_C", "mean_return_RMSE_C", "mean_heat_loss_error_percent", "mean_energy_balance_residual_percent", "mean_thermal_delay_error_min"}:
                thermal += delta
            else:
                hydraulic += delta
        rows.append(
            {
                "case": row["case"],
                "parameter": row["parameter"],
                "thermal_sensitivity_index": thermal,
                "hydraulic_sensitivity_index": hydraulic,
                "total_sensitivity_index": total,
                "dominant_group": "thermal" if thermal >= hydraulic else "hydraulic",
                "interpretation": "Calibrated parameters are effective parameters for matching plant-level operating data, not independently measured pipe-material or hydraulic properties.",
            }
        )
    ranked = pd.DataFrame(rows).sort_values("total_sensitivity_index", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _plot_tornado_improved(ranked: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    if ranked.empty:
        ax.text(0.5, 0.5, "No ranked sensitivity data", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = ranked.sort_values("total_sensitivity_index", ascending=True).tail(10)
        ax.barh(plot_df["case"], plot_df["total_sensitivity_index"], color="#7b2cbf")
        ax.set_xlabel("Total sensitivity index (relative metric change)")
        ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Improved parameter-identifiability tornado plot")
    save_figure(fig, "fig_parameter_identifiability_tornado_improved")


def _plot_grouped_thermal_hydraulic(ranked: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    if ranked.empty:
        ax.text(0.5, 0.5, "No grouped sensitivity data", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = ranked.head(10).copy()
        x = np.arange(len(plot_df))
        ax.bar(x - 0.18, plot_df["thermal_sensitivity_index"], width=0.36, label="thermal/energy")
        ax.bar(x + 0.18, plot_df["hydraulic_sensitivity_index"], width=0.36, label="hydraulic")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["case"].str.replace("_", "\n", regex=False), fontsize=7)
        ax.set_ylabel("Sensitivity index")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    ax.set_title("Thermal versus hydraulic sensitivity of effective parameters")
    save_figure(fig, "fig_parameter_sensitivity_grouped_thermal_hydraulic")


if __name__ == "__main__":
    run_parameter_identifiability_sensitivity()
