from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Latin Modern Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 1200,
    }
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.evaluate import evaluate_model, save_thermo_hydraulic_estimation_outputs
from src.effective_physics import apply_calibrated_params_to_config
from src.graph_utils import build_line_graph_adjacency, normalized_adjacency
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import build_loaders, greedy_optimized_three_sensors
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import ensure_dir


MODEL_SPECS = [
    ("GRU-MSE", "gru", "seed_11_GRU-MSE_best.pt"),
    ("Transformer-MSE", "transformer", "seed_11_Transformer-MSE_best.pt"),
    ("Proposed PI-GNN-GRU-v3 accuracy_mode", "pignn_v3", "seed_11_Proposed PI-GNN-GRU-v3 accuracy_mode_best.pt"),
    ("Proposed PI-GNN-GRU-v3 balanced_mode", "pignn_v3", "seed_11_Proposed PI-GNN-GRU-v3 balanced_mode_best.pt"),
]


def _load_state_if_exists(model: torch.nn.Module, path_name: str) -> torch.nn.Module | None:
    path = PROJECT_ROOT / "results" / path_name
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def _save(fig: plt.Figure, name: str) -> None:
    out = ensure_dir(PROJECT_ROOT / "figures" / "final")
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=1200, bbox_inches="tight")
    fig.savefig(PROJECT_ROOT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PROJECT_ROOT / "figures" / f"{name}.png", dpi=1200, bbox_inches="tight")
    paper_out = ensure_dir(PROJECT_ROOT / "paper" / "figures" / "final")
    fig.savefig(paper_out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(paper_out / f"{name}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def _flatten(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).reshape(-1, *np.asarray(arr).shape[2:])


def _profile_indices(payload: dict[str, np.ndarray]) -> tuple[int, int]:
    heat = np.asarray(payload.get("heat_load_kw"))
    if heat.size == 0:
        return 0, 0
    heat_flat = heat.reshape(-1)
    normal_value = np.nanmedian(heat_flat)
    normal = int(np.nanargmin(np.abs(heat_flat - normal_value)))
    difficult = int(np.nanargmax(heat_flat))
    return normal, difficult


def _best_payload(payloads: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
    for name in [
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 balanced_mode",
        "Transformer-MSE",
        "GRU-MSE",
    ]:
        if name in payloads:
            return name, payloads[name]
    name = next(iter(payloads))
    return name, payloads[name]


def _model_colors() -> dict[str, str]:
    return {
        "GRU-MSE": "#f9844a",
        "Transformer-MSE": "#577590",
        "PureGNN-MSE": "#43aa8b",
        "Proposed PI-GNN-GRU-v3 accuracy_mode": "#d1495b",
        "Proposed PI-GNN-GRU-v3 balanced_mode": "#7b2cbf",
    }


def _compact(name: str) -> str:
    return name.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 ").replace("_", " ")


def prepare_context(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, torch.nn.Module], dict[str, dict[str, np.ndarray]]]:
    primary = config["real_data"]["primary_dataset"]
    # The frozen reviewer archive is the canonical input for the paper.  When
    # it is enabled, preprocessing validates and loads that checksum-pinned
    # artifact directly; the unavailable annual raw files are not a prerequisite
    # for reproducing the paper-level checkpoint/evaluation workflow.
    real_cfg = config.get("real_data", {})
    frozen_rel = real_cfg.get("canonical_sonderborg_processed_path")
    frozen_path = PROJECT_ROOT / str(frozen_rel) if frozen_rel else None
    if primary == "sonderborg" and real_cfg.get("freeze_canonical_processed", False) and frozen_path and frozen_path.exists():
        df = preprocess_dataset(pd.DataFrame(), primary, config)
    else:
        df = preprocess_dataset(load_dataset_by_name(primary), primary, config)
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    df = df.head(min(len(df), max_steps)).copy()
    boundary = build_boundary_conditions(df, config)
    params_path = PROJECT_ROOT / "results" / "calibrated_parameters.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    config = apply_calibrated_params_to_config(config, params)
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    sim["optimized_sensor_nodes"] = greedy_optimized_three_sensors(sim)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"]["noise_std_fraction"])
    loaders = build_loaders(sim, sensors, config)
    arrays = loaders["arrays"]
    n_nodes = arrays["target"].shape[1]
    a_norm = torch.tensor(normalized_adjacency(build_line_graph_adjacency(n_nodes)), dtype=torch.float32)

    trained: dict[str, torch.nn.Module] = {}
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for label, model_name, state_name in MODEL_SPECS:
        model = build_model(model_name, arrays["features"].shape[-1], n_nodes, a_norm, config)
        loaded = _load_state_if_exists(model, state_name)
        if loaded is None:
            continue
        trained[label] = loaded
        _, payload = evaluate_model(loaded, loaders["test_loader"], config, loaders["train_ds"].stats, label, return_predictions=True)
        payloads[label] = payload
    return df, sim, sensors, loaders, params, trained, payloads


def run_thermo_hydraulic_metric_package(
    config: dict[str, Any],
    sim: dict[str, Any],
    sensors: dict[str, Any],
    payloads: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    metrics, derived = save_thermo_hydraulic_estimation_outputs(payloads, config, sensor_nodes=sensors.get("sensor_nodes", []))
    # Convenience profile/time-series files for paper and plots.
    proposed_name, proposed = _best_payload(payloads)
    proposed_derived = derived[proposed_name]
    x_m = np.asarray(sim["x_m"], dtype=float)
    x_seg_km = 0.5 * (x_m[:-1] + x_m[1:]) / 1000.0
    seg_true = _flatten(proposed_derived["true_segment_heat_loss_kw"]).mean(axis=0)
    seg_pred = _flatten(proposed_derived["pred_segment_heat_loss_kw"]).mean(axis=0)
    pd.DataFrame(
        {
            "segment_midpoint_km": x_seg_km,
            "simulator_heat_loss_kW": seg_true,
            "pignn_v3_heat_loss_kW": seg_pred,
            "error_kW": seg_pred - seg_true,
            "state_type": "calibrated_simulator",
        }
    ).to_csv(PROJECT_ROOT / "results" / "heat_loss_profile_metrics.csv", index=False)

    heat_load = np.asarray(proposed_derived["heat_load_kw"]).reshape(-1)
    true_delivered = np.asarray(proposed_derived["true_delivered_heat_kw"]).reshape(-1)
    pred_delivered = np.asarray(proposed_derived["pred_delivered_heat_kw"]).reshape(-1)
    residual = np.asarray(proposed_derived["energy_residual_kw"]).reshape(-1)
    pd.DataFrame(
        {
            "step": np.arange(len(heat_load)),
            "measured_boundary_heat_load_kw": heat_load,
            "simulator_delivered_heat_kw": true_delivered,
            "pignn_v3_delivered_heat_kw": pred_delivered,
            "pignn_v3_energy_residual_kw": residual,
            "state_type": "calibrated_simulator",
        }
    ).to_csv(PROJECT_ROOT / "results" / "energy_balance_time_series.csv", index=False)

    pd.DataFrame(
        [
            {
                "model": proposed_name,
                "delivered_heat_MAE_kW": float(np.nanmean(np.abs(pred_delivered - true_delivered))),
                "delivered_heat_error_percent": float(np.nanmean(np.abs(pred_delivered - true_delivered) / np.maximum(np.abs(true_delivered), 1.0)) * 100.0),
                "energy_balance_residual_percent": float(np.nanmean(np.abs(residual) / np.maximum(np.abs(heat_load), 1.0)) * 100.0),
                "state_type": "calibrated_simulator",
            }
        ]
    ).to_csv(PROJECT_ROOT / "results" / "heat_delivery_tracking_metrics.csv", index=False)
    return metrics, derived


def _plot_temperature_profiles(sim: dict[str, Any], payloads: dict[str, dict[str, np.ndarray]], sensors: dict[str, Any], state_idx: int, name: str, ylabel: str) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    _, ref_payload = _best_payload(payloads)
    normal, difficult = _profile_indices(ref_payload)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    colors = _model_colors()
    for ax, idx, title in zip(axes, [normal, difficult], ["normal-load window", "high-demand/difficult window"]):
        true = _flatten(ref_payload["true"])[:, :, state_idx][idx]
        ax.plot(x_km, true, color="black", lw=2.0, label="calibrated simulator hidden state")
        for model in ["GRU-MSE", "Transformer-MSE", "PureGNN-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode"]:
            if model in payloads:
                pred = _flatten(payloads[model]["pred"])[:, :, state_idx][idx]
                ax.plot(x_km, pred, lw=1.4, label=_compact(model), color=colors.get(model))
        nodes = sensors.get("sensor_nodes", [])
        if nodes:
            ax.scatter(x_km[nodes], true[nodes], s=35, facecolor="white", edgecolor="black", zorder=5, label="sensor nodes")
        ax.set_xlabel("Distance (km)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=6, loc="best")
    fig.suptitle("Distributed temperature labels are calibrated-simulator hidden states; real data provide boundary conditions and measured-node validation.")
    _save(fig, name)


def _plot_error_heatmap(sim: dict[str, Any], payload: dict[str, np.ndarray], state_idx: int, name: str, label: str) -> None:
    error = np.abs(_flatten(payload["pred"])[..., state_idx] - _flatten(payload["true"])[..., state_idx])
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    im = ax.imshow(error, aspect="auto", origin="lower", extent=[x_km[0], x_km[-1], 0, error.shape[0]], cmap="magma")
    fig.colorbar(im, ax=ax, label=f"Absolute {label} error")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Windowed time index")
    ax.set_title(f"{label} error spacetime map (simulator-assisted hidden states)")
    _save(fig, name)


def _plot_gradient_error(sim: dict[str, Any], payloads: dict[str, dict[str, np.ndarray]]) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    dx_km = np.nanmean(np.diff(x_km))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colors = _model_colors()
    for model, payload in payloads.items():
        if model not in {"GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode"}:
            continue
        pred = _flatten(payload["pred"])[..., 0]
        true = _flatten(payload["true"])[..., 0]
        err = np.abs(np.gradient(pred, dx_km, axis=1) - np.gradient(true, dx_km, axis=1)).mean(axis=0)
        ax.plot(x_km, err, lw=1.5, label=_compact(model), color=colors.get(model))
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Gradient error (C/km)")
    ax.set_title("Supply-temperature gradient error")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, "fig_temperature_gradient_error")


def _plot_profile(sim: dict[str, Any], payloads: dict[str, dict[str, np.ndarray]], derived: dict[str, dict[str, np.ndarray]], quantity: str, name: str, ylabel: str, state_type_note: str) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    _, ref_payload = _best_payload(payloads)
    _, difficult = _profile_indices(ref_payload)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    colors = _model_colors()
    true_source = _flatten(ref_payload["true"])
    if quantity == "head":
        true = true_source[:, :, 2][difficult]
        getter = lambda p, d: _flatten(p["pred"])[:, :, 2][difficult]
    elif quantity == "pressure":
        model0 = next(iter(derived))
        true = _flatten(derived[model0]["pressure_true_kPa"])[difficult]
        getter = lambda p, d: _flatten(d["pressure_pred_kPa"])[difficult]
    elif quantity == "flow":
        model0 = next(iter(derived))
        true = _flatten(derived[model0]["flow_true_kg_s"])[difficult]
        getter = lambda p, d: _flatten(d["flow_pred_kg_s"])[difficult]
    else:
        raise ValueError(quantity)
    ax.plot(x_km, true, color="black", lw=2.0, label="calibrated simulator hidden state")
    for model in ["GRU-MSE", "Transformer-MSE", "PureGNN-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode"]:
        if model in payloads and model in derived:
            ax.plot(x_km, getter(payloads[model], derived[model]), lw=1.4, label=_compact(model), color=colors.get(model))
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel(ylabel)
    ax.set_title(state_type_note)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, name)


def _plot_pressure_drop(sim: dict[str, Any], derived: dict[str, dict[str, np.ndarray]]) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    colors = _model_colors()
    for model, d in derived.items():
        if model not in {"GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode"}:
            continue
        pred = _flatten(d["pressure_pred_kPa"]).mean(axis=0)
        true = _flatten(d["pressure_true_kPa"]).mean(axis=0)
        ax.plot(x_km, pred[0] - pred, lw=1.5, label=f"{_compact(model)} predicted", color=colors.get(model))
    model0 = next(iter(derived))
    true_mean = _flatten(derived[model0]["pressure_true_kPa"]).mean(axis=0)
    ax.plot(x_km, true_mean[0] - true_mean, color="black", lw=2.0, label="simulator hidden pressure drop")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pressure drop (kPa)")
    ax.set_title("Pressure-drop profile (simulator-assisted hidden hydraulic state)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, "fig_pressure_drop_profile")


def _plot_pump_boundary(payloads: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    colors = _model_colors()
    for model, payload in payloads.items():
        if model not in {"GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode"}:
            continue
        pred = _flatten(payload["pred"])
        true = _flatten(payload["true"])
        pred_drop = pred[:, 0, 2] - pred[:, -1, 2]
        true_drop = true[:, 0, 2] - true[:, -1, 2]
        ax.plot(pred_drop - true_drop, lw=1.0, alpha=0.85, label=_compact(model), color=colors.get(model))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Windowed time index")
    ax.set_ylabel("Head-drop residual (m)")
    ax.set_title("Pump/head boundary consistency (simulator-assisted hydraulic state)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, "fig_pump_boundary_consistency")


def _plot_flow_time_and_balance(derived: dict[str, dict[str, np.ndarray]]) -> None:
    model, d = next((m, d) for m, d in derived.items() if "PI-GNN-GRU-v3 accuracy" in m or "PI-GNN-GRU-v3" in m)
    true = _flatten(d["flow_true_kg_s"])
    pred = _flatten(d["flow_pred_kg_s"])
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    ax.plot(true[:, 0], label="simulator inlet flow", lw=1.5, color="black")
    ax.plot(true[:, -1], label="simulator outlet flow", lw=1.2, color="0.45")
    ax.plot(pred[:, 0], label="PI-GNN-v3 inlet flow", lw=1.2, color="#d1495b")
    ax.plot(pred[:, -1], label="PI-GNN-v3 outlet flow", lw=1.2, color="#7b2cbf")
    ax.set_xlabel("Windowed time index")
    ax.set_ylabel("Mass flow (kg/s)")
    ax.set_title("Flow labels are calibrated-simulator/heat-load-proxy hidden states")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, "fig_flow_time_response")

    fig2, ax2 = plt.subplots(figsize=(7.5, 3.4))
    balance = np.nanmax(pred, axis=1) - np.nanmin(pred, axis=1)
    ax2.plot(balance, color="#bc4749", lw=1.3)
    ax2.set_xlabel("Windowed time index")
    ax2.set_ylabel("Predicted flow spread (kg/s)")
    ax2.set_title("Flow balance error over time")
    ax2.grid(True, alpha=0.25)
    _save(fig2, "fig_flow_balance_error")


def _plot_heat_energy(sim: dict[str, Any], payloads: dict[str, dict[str, np.ndarray]], derived: dict[str, dict[str, np.ndarray]]) -> None:
    model, d = _best_derived(derived)
    x_seg = 0.5 * (np.asarray(sim["x_m"][:-1]) + np.asarray(sim["x_m"][1:])) / 1000.0
    heat_load = d["heat_load_kw"].reshape(-1)
    true_delivered = d["true_delivered_heat_kw"].reshape(-1)
    pred_delivered = d["pred_delivered_heat_kw"].reshape(-1)
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.plot(heat_load, label="measured boundary heat load", lw=1.4)
    ax.plot(true_delivered, label="simulator delivered heat", lw=1.2)
    ax.plot(pred_delivered, label="PI-GNN-v3 delivered heat", lw=1.2)
    ax.set_xlabel("Windowed time index")
    ax.set_ylabel("Heat (kW)")
    ax.set_title("Heat-load delivery tracking")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, "fig_heat_load_delivery_tracking")

    true_seg = _flatten(d["true_segment_heat_loss_kw"]).mean(axis=0)
    pred_seg = _flatten(d["pred_segment_heat_loss_kw"]).mean(axis=0)
    fig2, ax2 = plt.subplots(figsize=(7.8, 3.6))
    ax2.plot(x_seg, true_seg / np.maximum(np.diff(sim["x_m"]) / 1000.0, 1e-9), color="black", lw=2.0, label="simulator")
    ax2.plot(x_seg, pred_seg / np.maximum(np.diff(sim["x_m"]) / 1000.0, 1e-9), color="#d1495b", lw=1.5, label="PI-GNN-v3")
    ax2.set_xlabel("Distance (km)")
    ax2.set_ylabel("Heat loss (kW/km)")
    ax2.set_title("Segment-wise heat-loss profile")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7)
    _save(fig2, "fig_heat_loss_profile")

    fig3, ax3 = plt.subplots(figsize=(7.8, 3.6))
    ax3.plot(x_seg, np.cumsum(true_seg), color="black", lw=2.0, label="simulator cumulative")
    ax3.plot(x_seg, np.cumsum(pred_seg), color="#d1495b", lw=1.5, label="PI-GNN-v3 cumulative")
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Cumulative heat loss (kW)")
    ax3.set_title("Cumulative heat loss along pipeline")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=7)
    _save(fig3, "fig_cumulative_heat_loss")

    error = np.abs(_flatten(d["pred_segment_heat_loss_kw"]) - _flatten(d["true_segment_heat_loss_kw"]))
    fig4, ax4 = plt.subplots(figsize=(7.8, 3.7))
    im = ax4.imshow(error, aspect="auto", origin="lower", extent=[x_seg[0], x_seg[-1], 0, error.shape[0]], cmap="inferno")
    fig4.colorbar(im, ax=ax4, label="Heat-loss error (kW)")
    ax4.set_xlabel("Distance (km)")
    ax4.set_ylabel("Windowed time index")
    ax4.set_title("Space-time heat-loss error")
    _save(fig4, "fig_heat_loss_error_spacetime")

    fig5, ax5 = plt.subplots(figsize=(7.8, 3.5))
    colors = _model_colors()
    for m, dm in derived.items():
        if m in {"GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode"}:
            ax5.plot(dm["energy_residual_kw"].reshape(-1), lw=1.1, label=_compact(m), color=colors.get(m))
    ax5.axhline(0, color="black", lw=0.8)
    ax5.set_xlabel("Windowed time index")
    ax5.set_ylabel("Energy residual (kW)")
    ax5.set_title("Energy-balance residual over time")
    ax5.grid(True, alpha=0.25)
    ax5.legend(fontsize=7)
    _save(fig5, "fig_energy_balance_residual_time")

    fig6, ax6 = plt.subplots(figsize=(6.5, 4.2))
    true_return = _flatten(payloads[model]["true"])[:, -1, 1]
    ax6.scatter(heat_load, true_return, s=12, alpha=0.5, label="simulator return")
    ax6.scatter(heat_load, _flatten(payloads[model]["pred"])[:, -1, 1], s=12, alpha=0.5, label="PI-GNN-v3 return")
    ax6.set_xlabel("Heat load (kW)")
    ax6.set_ylabel("Outlet return temperature (C)")
    ax6.set_title("Heat-load and return-temperature coupling")
    ax6.grid(True, alpha=0.25)
    ax6.legend(fontsize=7)
    _save(fig6, "fig_heat_return_temperature_coupling")


def _best_derived(derived: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
    for name in ["Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"]:
        if name in derived:
            return name, derived[name]
    name = next(iter(derived))
    return name, derived[name]


def compute_coupling_metrics(sim: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    heat = np.asarray(sim["Q_load"], dtype=float) / 1000.0
    flow = np.asarray(sim["q"], dtype=float)[:, 0]
    head_drop = np.asarray(sim["H"], dtype=float)[:, 0] - np.asarray(sim["H"], dtype=float)[:, -1]
    ret = np.asarray(sim["Tr"], dtype=float)[:, 0]
    heat_loss = np.asarray(sim["Q_loss"], dtype=float) / 1000.0
    temp_drop = np.asarray(sim["Ts"], dtype=float)[:, 0] - np.asarray(sim["Ts"], dtype=float)[:, -1]

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    dt_s = float(config["system"]["dt_s"])
    delay_all = _lag_minutes(heat, ret, dt_s)
    quantiles = np.nanquantile(flow, [0.33, 0.66])
    regimes = {
        "low_flow": flow <= quantiles[0],
        "medium_flow": (flow > quantiles[0]) & (flow <= quantiles[1]),
        "high_flow": flow > quantiles[1],
    }
    rows = [
        {"metric": "corr_heat_load_flow_proxy", "value": corr(heat, flow), "unit": "-", "state_type": "calibrated_simulator", "interpretation": "coupling between measured heat-load boundary and simulator flow proxy"},
        {"metric": "corr_heat_load_head_drop", "value": corr(heat, head_drop), "unit": "-", "state_type": "simulator_assisted_hidden_state", "interpretation": "head-drop coupling; dense real pressure data unavailable"},
        {"metric": "corr_heat_load_return_temperature", "value": corr(heat, ret), "unit": "-", "state_type": "real_measured_node/calibrated_simulator", "interpretation": "return-temperature/load relationship at available boundary node"},
        {"metric": "corr_flow_heat_loss", "value": corr(flow, heat_loss), "unit": "-", "state_type": "calibrated_simulator", "interpretation": "flow and heat-loss coupling from simulator"},
        {"metric": "corr_temperature_drop_heat_loss", "value": corr(temp_drop, heat_loss), "unit": "-", "state_type": "calibrated_simulator", "interpretation": "temperature-drop and heat-loss coupling"},
        {"metric": "time_lag_heat_load_to_return_response_min", "value": delay_all, "unit": "min", "state_type": "calibrated_simulator", "interpretation": "lag estimated from heat load and return response"},
    ]
    for name, mask in regimes.items():
        if np.count_nonzero(mask) > 8:
            rows.append({"metric": f"thermal_delay_{name}_min", "value": _lag_minutes(heat[mask], ret[mask], dt_s), "unit": "min", "state_type": "calibrated_simulator", "interpretation": f"thermal-delay estimate in {name.replace('_', ' ')} regime"})
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "thermo_hydraulic_coupling_metrics.csv", index=False)
    _plot_coupling_figures(sim, df)
    return df


def _lag_minutes(a: np.ndarray, b: np.ndarray, dt_s: float, max_lag: int = 24) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = min(a.size, b.size)
    if n < 4:
        return float("nan")
    a = a[:n] - np.nanmean(a[:n])
    b = b[:n] - np.nanmean(b[:n])
    max_lag = min(max_lag, n - 2)
    best_lag = 0
    best = -np.inf
    for lag in range(max_lag + 1):
        x = a[: n - lag]
        y = b[lag:n]
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            continue
        val = np.corrcoef(x, y)[0, 1]
        if np.isfinite(val) and val > best:
            best = val
            best_lag = lag
    return float(best_lag * dt_s / 60.0)


def _plot_coupling_figures(sim: dict[str, Any], coupling_df: pd.DataFrame) -> None:
    t_h = np.asarray(sim["time_s"], dtype=float) / 3600.0
    heat = np.asarray(sim["Q_load"], dtype=float) / 1000.0
    flow = np.asarray(sim["q"], dtype=float)[:, 0]
    head_drop = np.asarray(sim["H"], dtype=float)[:, 0] - np.asarray(sim["H"], dtype=float)[:, -1]
    heat_loss = np.asarray(sim["Q_loss"], dtype=float) / 1000.0
    avg_supply = np.nanmean(sim["Ts"], axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 6.2), sharex=True)
    axes[0].plot(t_h, heat, lw=1.2)
    axes[0].set_ylabel("Heat load (kW)")
    axes[1].plot(t_h, flow, lw=1.2, color="#4d908e")
    axes[1].set_ylabel("Flow proxy (m3/s)")
    axes[2].plot(t_h, head_drop, lw=1.2, color="#bc4749")
    axes[2].set_ylabel("Head drop (m)")
    axes[2].set_xlabel("Time (h)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Heat-flow-pressure coupling from calibrated simulator")
    _save(fig, "fig_heat_flow_pressure_coupling")

    fig2, ax2 = plt.subplots(figsize=(6.8, 4.2))
    sc = ax2.scatter(flow, heat_loss, c=avg_supply, cmap="viridis", s=18, alpha=0.7)
    fig2.colorbar(sc, ax=ax2, label="Average supply temperature (C)")
    ax2.set_xlabel("Flow proxy (m3/s)")
    ax2.set_ylabel("Heat loss (kW)")
    ax2.set_title("Heat loss versus flow and supply temperature")
    ax2.grid(True, alpha=0.25)
    _save(fig2, "fig_heat_loss_vs_flow_temperature")

    delay_rows = coupling_df[coupling_df["metric"].astype(str).str.startswith("thermal_delay_")]
    fig3, ax3 = plt.subplots(figsize=(6.8, 3.6))
    if not delay_rows.empty:
        ax3.bar(delay_rows["metric"].str.replace("thermal_delay_", "").str.replace("_min", "").str.replace("_", "\n"), pd.to_numeric(delay_rows["value"], errors="coerce"), color="#577590")
    else:
        ax3.text(0.5, 0.5, "Thermal-delay regimes unavailable", ha="center", va="center")
    ax3.set_ylabel("Estimated delay (min)")
    ax3.set_title("Thermal delay by flow regime")
    ax3.grid(True, axis="y", alpha=0.25)
    _save(fig3, "fig_thermal_delay_by_flow_regime")


def run_robustness_cases(
    config: dict[str, Any],
    base_sim: dict[str, Any],
    params: dict[str, Any],
    trained: dict[str, torch.nn.Module],
    stats: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def eval_case(case_name: str, sim_case: dict[str, Any], layout: str = "S4_five_sensors", noise: float = 0.0, note: str = "") -> None:
        sensors = apply_sensor_layout(sim_case, layout, config, noise_std_fraction=noise)
        loaders = build_loaders(sim_case, sensors, config, stats=stats)
        for model_name, model in trained.items():
            if model_name not in {"GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"}:
                continue
            metrics = evaluate_model(model, loaders["test_loader"], config, stats, model_name, return_predictions=False)
            rows.append(
                {
                    "condition": case_name,
                    "model": model_name,
                    "sensor_layout": layout,
                    "noise_std_fraction": noise,
                    "RMSE_Ts_supply_C": metrics.get("RMSE_Ts_full"),
                    "RMSE_Tr_return_C": metrics.get("RMSE_Tr_full"),
                    "RMSE_head_m": metrics.get("RMSE_H_full"),
                    "RMSE_flow_m3_s": metrics.get("RMSE_q_full"),
                    "heat_loss_error_percent": metrics.get("heat_loss_error_percent"),
                    "energy_balance_residual": metrics.get("energy_balance_residual"),
                    "boundary_residual_mean": metrics.get("boundary_residual_mean"),
                    "state_type": "real_measured_node + simulator_assisted_hidden_state",
                    "note": note,
                }
            )

    for noise in [0.01, 0.03, 0.05]:
        eval_case(f"temperature_head_sensor_noise_{int(noise*100)}pct", base_sim, "S4_five_sensors", noise, "sensor-value perturbation; hydraulic quantities remain simulator-assisted")
        sim_heat = copy.deepcopy(base_sim)
        factor = 1.0 + noise
        sim_heat["Q_load"] = np.asarray(sim_heat["Q_load"]) * factor
        eval_case(f"heat_load_noise_plus_{int(noise*100)}pct", sim_heat, "S4_five_sensors", 0.0, "heat-load feature perturbation diagnostic")
    eval_case("missing_middle_sensor", base_sim, "S2_inlet_outlet", 0.0, "middle sensor removed")
    eval_case("outlet_sensor_dropout", base_sim, "S1_inlet_only", 0.0, "outlet sensor unavailable")
    sim_bias = copy.deepcopy(base_sim)
    sensors_bias = apply_sensor_layout(sim_bias, "S4_five_sensors", config)
    sensors_bias["measurements"][:, :, 1] += 2.0 * sensors_bias["masks"][:, :, 1]
    loaders_bias = build_loaders(sim_bias, sensors_bias, config, stats=stats)
    for model_name, model in trained.items():
        if model_name in {"GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"}:
            metrics = evaluate_model(model, loaders_bias["test_loader"], config, stats, model_name, return_predictions=False)
            rows.append({"condition": "biased_return_temperature_plus_2C", "model": model_name, "sensor_layout": "S4_five_sensors", "noise_std_fraction": 0.0, "RMSE_Ts_supply_C": metrics.get("RMSE_Ts_full"), "RMSE_Tr_return_C": metrics.get("RMSE_Tr_full"), "RMSE_head_m": metrics.get("RMSE_H_full"), "RMSE_flow_m3_s": metrics.get("RMSE_q_full"), "heat_loss_error_percent": metrics.get("heat_loss_error_percent"), "energy_balance_residual": metrics.get("energy_balance_residual"), "boundary_residual_mean": metrics.get("boundary_residual_mean"), "state_type": "real_measured_node + simulator_assisted_hidden_state", "note": "return-temperature sensor bias diagnostic"})

    # Parameter uncertainty: rerun simulator, reuse trained model without retraining.
    for param, mult in [("heat_loss_U_W_m2K", 0.8), ("heat_loss_U_W_m2K", 1.2), ("friction_factor", 0.8), ("friction_factor", 1.2)]:
        p = dict(params)
        base_value = float(p.get(param, config["system"].get(param, 1.0)))
        p[param] = base_value * mult
        boundary = {
            "time_s": base_sim["time_s"],
            "T_source": base_sim["T_source"],
            "T_return_measured": base_sim["T_return_measured"],
            "Q_load_W": base_sim["Q_load"],
            "Ta": base_sim["Ta"],
            "alpha_estimated": base_sim["alpha"],
            "source_dataset": base_sim.get("source_dataset", "sonderborg"),
        }
        sim_uncertain = simulate_thermo_hydraulics(boundary, config, params=p)
        eval_case(f"{param}_{int(mult*100)}pct", sim_uncertain, "S4_five_sensors", 0.0, f"parameter uncertainty diagnostic: {param} x {mult:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "thermo_hydraulic_robustness.csv", index=False)
    _plot_robustness(df)
    return df


def _plot_robustness(df: pd.DataFrame) -> None:
    if df.empty:
        return
    def barplot(filter_pattern: str, metric: str, name: str, title: str) -> None:
        sub = df[df["condition"].astype(str).str.contains(filter_pattern, regex=True)].copy()
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        if sub.empty:
            ax.text(0.5, 0.5, "No robustness data", ha="center", va="center")
            ax.axis("off")
        else:
            pivot = sub.pivot_table(index="condition", columns="model", values=metric, aggfunc="mean")
            cols = [c for c in ["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"] if c in pivot.columns]
            pivot[cols].plot(kind="bar", ax=ax, width=0.82)
            ax.set_ylabel(metric.replace("_", " "))
            ax.tick_params(axis="x", labelrotation=25, labelsize=7)
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=6)
        ax.set_title(title)
        _save(fig, name)
    barplot("noise|heat_load_noise", "RMSE_Ts_supply_C", "fig_temperature_pressure_noise_robustness", "Temperature/head proxy noise and heat-load perturbation")
    barplot("heat_loss_U|friction_factor", "heat_loss_error_percent", "fig_parameter_uncertainty_robustness", "Parameter uncertainty robustness")
    barplot("missing_middle|outlet_sensor|biased_return", "RMSE_Tr_return_C", "fig_sensor_dropout_thermo_hydraulic", "Sensor dropout and return-bias diagnostics")


def generate_thermo_hydraulic_figures(sim: dict[str, Any], sensors: dict[str, Any], payloads: dict[str, dict[str, np.ndarray]], derived: dict[str, dict[str, np.ndarray]]) -> None:
    model_name, payload = _best_payload(payloads)
    _plot_temperature_profiles(sim, payloads, sensors, 0, "fig_temperature_supply_profile", "Supply temperature (C)")
    _plot_temperature_profiles(sim, payloads, sensors, 1, "fig_temperature_return_profile", "Return temperature (C)")
    _plot_error_heatmap(sim, payload, 0, "fig_temperature_error_spacetime", "supply temperature (C)")
    _plot_error_heatmap(sim, payload, 1, "fig_return_temperature_error_spacetime", "return temperature (C)")
    _plot_gradient_error(sim, payloads)
    _plot_profile(sim, payloads, derived, "head", "fig_head_profile_reconstruction", "Hydraulic head (m)", "Head profile reconstruction: simulator-assisted hidden hydraulic state")
    _plot_profile(sim, payloads, derived, "pressure", "fig_pressure_profile_reconstruction", "Pressure (kPa)", "Pressure profile converted from simulator-assisted head")
    _plot_pressure_drop(sim, derived)
    _plot_error_heatmap(sim, payload, 2, "fig_head_error_spacetime", "head (m)")
    _plot_pump_boundary(payloads)
    _plot_profile(sim, payloads, derived, "flow", "fig_flow_profile_reconstruction", "Mass flow (kg/s)", "Flow labels are calibrated-simulator/heat-load-proxy hidden states")
    _plot_flow_time_and_balance(derived)
    _plot_heat_energy(sim, payloads, derived)


def run_thermo_hydraulic_coupling_analysis() -> None:
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    config = load_config()
    _, sim, sensors, loaders, params, trained, payloads = prepare_context(config)
    if not payloads:
        raise RuntimeError("No trained model payloads were available for thermo-hydraulic analysis.")
    metrics, derived = run_thermo_hydraulic_metric_package(config, sim, sensors, payloads)
    compute_coupling_metrics(sim, config)
    run_robustness_cases(config, sim, params, trained, loaders["train_ds"].stats)
    generate_thermo_hydraulic_figures(sim, sensors, payloads, derived)
    # Copy final figures into paper figures for LaTeX use.
    paper_final = ensure_dir(PROJECT_ROOT / "paper" / "figures" / "final")
    for fig in (PROJECT_ROOT / "figures" / "final").glob("*.*"):
        if fig.suffix.lower() in {".pdf", ".png"}:
            target = paper_final / fig.name
            target.write_bytes(fig.read_bytes())
    print("Thermo-hydraulic coupling and estimation analysis completed.")


if __name__ == "__main__":
    run_thermo_hydraulic_coupling_analysis()
