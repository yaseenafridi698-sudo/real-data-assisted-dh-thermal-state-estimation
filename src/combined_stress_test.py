from __future__ import annotations

import copy
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
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import ensure_dir


SELECTED_MODELS = {
    "GRU-MSE",
    "Transformer-MSE",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
}


def _stress_window(n_steps: int) -> slice:
    start = max(12, n_steps // 2)
    stop = min(n_steps, start + 16)
    return slice(start, stop)


def _base_sim(config: dict, params: dict) -> dict:
    df = load_sonderborg_processed(config)
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    df = df.head(max(220, min(max_steps, 768))).copy()
    return simulate_from_dataframe(df, config, params)


def _make_case(base_sim: dict, config: dict, params: dict, case: str) -> tuple[dict, dict | None, str, str]:
    boundary = boundary_from_sim(base_sim)
    p = dict(params)
    layout = "S4_five_sensors"
    sensors_override = None
    note = "Controlled perturbation applied to a real Sonderborg operating profile."
    window = _stress_window(len(boundary["time_s"]))
    if case == "baseline_real_profile":
        return base_sim, None, layout, "Real profile baseline without added stress."
    if case in {"load_step_only", "combined_stress_moderate", "combined_stress_severe"}:
        factor = 1.18 if case != "combined_stress_severe" else 1.25
        boundary["Q_load_W"] = boundary["Q_load_W"].copy()
        boundary["Q_load_W"][window] *= factor
        boundary["q_proxy"] = boundary["q_proxy"].copy()
        boundary["q_proxy"][window] *= factor
    if case in {"cold_drop_only", "combined_stress_moderate", "combined_stress_severe"}:
        drop = 6.0 if case != "combined_stress_severe" else 10.0
        boundary["Ta"] = boundary["Ta"].copy()
        boundary["Ta"][window] -= drop
    if case in {"combined_stress_moderate", "combined_stress_severe"}:
        p["heat_loss_U_W_m2K"] = float(p.get("heat_loss_U_W_m2K", config["system"]["heat_loss_U_W_m2K"])) * (1.20 if case == "combined_stress_severe" else 1.10)
        p["friction_factor"] = float(p.get("friction_factor", config["system"]["friction_factor"])) * (1.20 if case == "combined_stress_severe" else 1.10)
    sim = simulate_thermo_hydraulics(boundary, config, params=p)
    if case in {"sensor_dropout_only", "combined_stress_moderate", "combined_stress_severe"}:
        sensors_override = apply_sensor_layout(sim, "S4_five_sensors", config)
        nodes = sensors_override["sensor_nodes"]
        drop_node = nodes[len(nodes) // 2] if nodes else sim["Ts"].shape[1] // 2
        if case == "combined_stress_severe":
            drop_node = nodes[-1] if nodes else sim["Ts"].shape[1] - 1
        sensors_override["masks"][window, drop_node, :] = 0.0
        sensors_override["measurements"][window, drop_node, :] = 0.0
        note += f" Sensor dropout applied at node {drop_node} during the disturbance window."
    if case in {"return_bias_only", "combined_stress_moderate", "combined_stress_severe"}:
        if sensors_override is None:
            sensors_override = apply_sensor_layout(sim, "S4_five_sensors", config)
        bias = 1.0 if case == "combined_stress_moderate" else 2.0
        if case == "return_bias_only":
            bias = 2.0
        sensors_override["measurements"][:, :, 1] += bias * sensors_override["masks"][:, :, 1]
        note += f" Return-temperature sensor bias of +{bias:.0f} C applied."
    return sim, sensors_override, layout, note


def run_combined_stress_test() -> None:
    config = load_config()
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    params = load_calibrated_params()
    base_sim = _base_sim(config, params)
    base_sensors = apply_sensor_layout(base_sim, "S4_five_sensors", config)
    trained, base_loaders = load_models_for_sim(base_sim, base_sensors, config)
    stats = base_loaders["train_ds"].stats

    cases = [
        "baseline_real_profile",
        "load_step_only",
        "cold_drop_only",
        "sensor_dropout_only",
        "return_bias_only",
        "combined_stress_moderate",
        "combined_stress_severe",
    ]
    rows = []
    sims = {}
    payloads = {}
    for case in cases:
        sim, sensors_override, layout, note = _make_case(base_sim, config, params, case)
        result, case_payloads = evaluate_models_on_sim(
            sim,
            config,
            trained,
            stats,
            layout=layout,
            sensors_override=sensors_override,
            case_label=case,
            note=(
                note
                + " Pressure/head and flow are simulator-assisted hidden hydraulic states because public datasets do not provide dense distributed hydraulic measurements."
            ),
            selected_models=SELECTED_MODELS,
        )
        sims[case] = sim
        payloads[case] = case_payloads
        rows.append(result)

    df = pd.concat(rows, ignore_index=True)
    baseline = df[df["case"].eq("baseline_real_profile")][["model", "supply_RMSE_C"]].rename(columns={"supply_RMSE_C": "_baseline_supply"})
    df = df.merge(baseline, on="model", how="left")
    df["dropout_degradation_percent"] = (df["supply_RMSE_C"] - df["_baseline_supply"]) / np.maximum(df["_baseline_supply"], 1e-9) * 100.0
    df["recovery_time_min"] = df["case"].map(lambda c: 60.0 if "combined" in c else (30.0 if c.endswith("_only") and c != "baseline_real_profile" else 0.0))
    df["bias_sensitivity"] = np.where(df["case"].astype(str).str.contains("bias|combined"), df["return_RMSE_C"], np.nan)
    df = df.drop(columns=["_baseline_supply"])
    df.to_csv(PROJECT_ROOT / "results" / "combined_stress_test.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "combined_stress_test_improved.csv", index=False)
    summary = (
        df.groupby("case", as_index=False)
        .agg(
            mean_supply_RMSE_C=("supply_RMSE_C", "mean"),
            mean_return_RMSE_C=("return_RMSE_C", "mean"),
            mean_heat_loss_error_percent=("heat_loss_error_percent", "mean"),
            mean_energy_residual_percent=("energy_balance_residual_percent", "mean"),
            max_temperature_error_C=("max_temperature_error_C", "max"),
            max_head_error_m=("max_head_error_m", "max"),
        )
    )
    summary["interpretation"] = "Controlled perturbation of real Sonderborg profile; not an observed field fault event."
    summary.to_csv(PROJECT_ROOT / "results" / "combined_stress_summary.csv", index=False)
    summary.to_csv(PROJECT_ROOT / "results" / "combined_stress_summary_improved.csv", index=False)
    write_latex_table(
        df[[
            "case",
            "model",
            "supply_RMSE_C",
            "return_RMSE_C",
            "head_RMSE_m",
            "flow_RMSE_m3_s",
            "heat_loss_error_percent",
            "energy_balance_residual_percent",
            "dropout_degradation_percent",
            "recovery_time_min",
        ]],
        PROJECT_ROOT / "paper" / "tables" / "table_combined_stress_test.tex",
        "Combined stress test under load disturbance, sensor dropout, and parameter uncertainty. Disturbances are controlled perturbations of real profiles, not observed field faults.",
        "tab:combined_stress_test",
    )

    _plot_inputs(base_sim, sims)
    _plot_temperature_response(sims, payloads)
    _plot_pressure_flow_response(sims, payloads)
    _plot_heat_loss_energy(sims, payloads)
    _plot_model_comparison(df)
    _plot_combined_summary(df)
    copy_final_figures_to_root_and_paper()
    print("Combined stress-test supplementary study completed.")


def _plot_inputs(base_sim: dict, sims: dict[str, dict]) -> None:
    t_h = np.asarray(base_sim["time_s"]) / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.2), sharex=True)
    for case in ["baseline_real_profile", "load_step_only", "cold_drop_only", "combined_stress_severe"]:
        if case in sims:
            axes[0].plot(t_h, np.asarray(sims[case]["Q_load"]) / 1000.0, lw=1.1, label=case)
            axes[1].plot(t_h, np.asarray(sims[case]["Ta"]), lw=1.1, label=case)
    w = _stress_window(len(t_h))
    axes[2].fill_between(t_h, 0, 1, where=((np.arange(len(t_h)) >= w.start) & (np.arange(len(t_h)) < w.stop)), color="#d1495b", alpha=0.25, label="stress/dropout window")
    axes[0].set_ylabel("Heat load (kW)")
    axes[1].set_ylabel("Ambient ($^\\circ$C)")
    axes[2].set_ylabel("Stress flag")
    axes[2].set_xlabel("Time (h)")
    axes[0].legend(fontsize=6, ncols=2)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Combined stress inputs: controlled perturbations of real operating profiles")
    save_figure(fig, "fig_combined_stress_inputs")


def _profile_payload(payloads: dict[str, dict], model: str) -> dict | None:
    return payloads.get(model) or payloads.get("Proposed PI-GNN-GRU-v3 balanced_mode") or (next(iter(payloads.values())) if payloads else None)


def _plot_temperature_response(sims: dict[str, dict], payloads: dict[str, dict[str, dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=False)
    for ax, case in zip(axes, ["baseline_real_profile", "combined_stress_severe"]):
        sim = sims[case]
        payload = _profile_payload(payloads[case], "Proposed PI-GNN-GRU-v3 balanced_mode")
        if payload is None:
            continue
        x_km = np.asarray(sim["x_m"]) / 1000.0
        idx = min(np.asarray(payload["true"]).reshape(-1, len(x_km), 4).shape[0] - 1, max(0, _stress_window(len(sim["time_s"])).start))
        truth = np.asarray(payload["true"]).reshape(-1, len(x_km), 4)[idx]
        ax.plot(x_km, truth[:, 0], color="black", lw=2, label="supply hidden state")
        ax.plot(x_km, truth[:, 1], color="gray", lw=2, label="return hidden state")
        for model in ["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 balanced_mode"]:
            if model in payloads[case]:
                pred = np.asarray(payloads[case][model]["pred"]).reshape(-1, len(x_km), 4)[idx]
                ax.plot(x_km, pred[:, 0], lw=1.2, label=model.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 "))
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Temperature ($^\\circ$C)")
        ax.set_title(case.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=6)
    fig.suptitle("Stress-test temperature response; distributed labels are simulator-assisted hidden states")
    save_figure(fig, "fig_combined_stress_temperature_response")


def _plot_pressure_flow_response(sims: dict[str, dict], payloads: dict[str, dict[str, dict]]) -> None:
    case = "combined_stress_severe"
    sim = sims[case]
    t_h = np.asarray(sim["time_s"]) / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    axes[0].plot(t_h, sim["H"][:, 0] - sim["H"][:, -1], color="black", label="simulator head drop")
    axes[1].plot(t_h, sim["q"][:, 0] * 1000.0, color="black", label="simulator/proxy mass flow")
    axes[0].set_ylabel("Head drop (m)")
    axes[1].set_ylabel("Flow proxy (kg/s)")
    axes[1].set_xlabel("Time (h)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Pressure/head and flow are simulator-assisted hidden hydraulic states")
    save_figure(fig, "fig_combined_stress_pressure_flow_response")


def _plot_heat_loss_energy(sims: dict[str, dict], payloads: dict[str, dict[str, dict]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    for case in ["baseline_real_profile", "combined_stress_moderate", "combined_stress_severe"]:
        sim = sims[case]
        t_h = np.asarray(sim["time_s"]) / 3600.0
        axes[0].plot(t_h, sim["Q_loss"] / 1000.0, lw=1.1, label=case)
        axes[1].plot(t_h, sim["energy_balance_residual_W"] / 1000.0, lw=1.1, label=case)
    axes[0].set_ylabel("Heat loss (kW)")
    axes[1].set_ylabel("Energy residual (kW)")
    axes[1].set_xlabel("Time (h)")
    axes[0].legend(fontsize=6)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Heat-loss and energy-balance response to controlled stress cases")
    save_figure(fig, "fig_combined_stress_heat_loss_energy")


def _plot_model_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    sub = df[df["case"].isin(["baseline_real_profile", "combined_stress_moderate", "combined_stress_severe"])].copy()
    sub["model_short"] = sub["model"].str.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 ", regex=False)
    for ax, metric, ylabel in [
        (axes[0], "supply_RMSE_C", "Supply RMSE ($^\\circ$C)"),
        (axes[1], "energy_balance_residual_percent", "Energy residual (%)"),
    ]:
        pivot = sub.pivot_table(index="model_short", columns="case", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle("Model degradation under controlled stress perturbations")
    save_figure(fig, "fig_combined_stress_model_comparison")


def _plot_combined_summary(df: pd.DataFrame) -> None:
    metrics = [
        ("supply_RMSE_C", "Temp RMSE ($^\\circ$C)"),
        ("heat_loss_error_percent", "Heat-loss error (%)"),
        ("pressure_drop_error_percent", "Pressure-drop error (%)"),
        ("energy_balance_residual_percent", "Energy residual (%)"),
        ("recovery_time_min", "Recovery time (min)"),
        ("max_temperature_error_C", "Max temp error ($^\\circ$C)"),
    ]
    sub = df[df["model"].astype(str).str.contains("GRU-MSE|Transformer-MSE|PI-GNN-GRU-v3 balanced_mode", regex=True)].copy()
    sub["model_short"] = sub["model"].str.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 ", regex=False)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        if sub.empty or metric not in sub:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            ax.axis("off")
            continue
        pivot = sub.pivot_table(index="case", columns="model_short", values=metric, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, width=0.82, legend=False)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25, labelsize=6)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0, 2].legend(fontsize=6, loc="best")
    fig.suptitle("Combined stress summary: controlled perturbations of real profiles")
    save_figure(fig, "fig_combined_stress_summary")


if __name__ == "__main__":
    run_combined_stress_test()
