from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.evaluate import _safe_percent
from src.supplementary_study_utils import copy_final_figures_to_root_and_paper, save_figure
from src.thermo_hydraulic_coupling_analysis import prepare_context
from src.utils import ensure_dir


PREFERRED_MODEL_ORDER = [
    "Proposed PI-GNN-GRU-v3 balanced_mode",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Transformer-MSE",
    "GRU-MSE",
]


def _pick_payload(payloads: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
    for name in PREFERRED_MODEL_ORDER:
        if name in payloads:
            return name, payloads[name]
    name = next(iter(payloads))
    return name, payloads[name]


def _flat(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1, *arr.shape[2:]) if arr.ndim >= 4 else arr.reshape(-1, *arr.shape[2:])


def _state_type(variable: str, measured: bool = False) -> str:
    if measured and variable in {"supply_temperature", "return_temperature"}:
        return "real_measured_node"
    if variable in {"head", "pressure", "flow", "pressure_drop"}:
        return "simulator_assisted_hidden_state"
    if variable in {"heat_loss", "energy_balance"}:
        return "calibrated_simulator"
    return "virtual_sensor_estimate"


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(np.abs(np.asarray(a) - np.asarray(b))))


def _series_for_flat(values: np.ndarray, n_steps: int, n_nodes: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == n_steps:
        return arr.reshape(n_steps)
    if arr.size == n_steps * n_nodes:
        return arr.reshape(n_steps, n_nodes)[:, 0]
    if arr.ndim >= 2 and arr.shape[-1] == n_nodes:
        return arr.reshape(-1, n_nodes)[:n_steps, 0]
    return np.resize(arr.reshape(-1), n_steps)


def _segment_heat_loss_flat(state: np.ndarray, ambient: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sys = config["system"]
    U = float(sys["heat_loss_U_W_m2K"])
    P = float(sys["pipe_perimeter_m"])
    dx = float(sys["dx_m"])
    Ta = np.asarray(ambient, dtype=float).reshape(-1, 1)
    supply_seg = 0.5 * (state[:, :-1, 0] + state[:, 1:, 0])
    return_seg = 0.5 * (state[:, :-1, 1] + state[:, 1:, 1])
    return U * P * ((supply_seg - Ta) + (return_seg - Ta)) * dx / 1000.0


def _delivered_heat_flat(state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sys = config["system"]
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    q = np.clip(state[:, -1, 3], 1e-9, None)
    delta_t = np.maximum(state[:, -1, 0] - state[:, -1, 1], 0.0)
    return rho * cp * q * delta_t / 1000.0


def _sensor_health(mask: np.ndarray, residual: np.ndarray, warn_threshold: float) -> np.ndarray:
    availability = np.nanmean(mask > 0.5, axis=(1, 2))
    residual_ok = np.nanmean(residual <= warn_threshold, axis=(1, 2))
    return np.clip(0.5 * availability + 0.5 * residual_ok, 0.0, 1.0)


def run_digital_twin_monitoring() -> None:
    config = load_config()
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    _, sim, sensors, _, _, _, payloads = prepare_context(config)
    if not payloads:
        raise RuntimeError("No saved model payloads available for digital-twin monitoring.")
    model_name, payload = _pick_payload(payloads)

    pred = _flat(payload["pred"])
    true = _flat(payload["true"])
    mask = _flat(payload["mask"])
    sensor = _flat(payload["sensor"])
    ambient = _series_for_flat(payload["ambient"], pred.shape[0], pred.shape[1])
    heat_load = _series_for_flat(payload["heat_load_kw"], pred.shape[0], pred.shape[1])
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    rho = float(config["system"]["rho"])
    g = float(config["system"]["g"])

    virtual_rows: list[dict[str, Any]] = []
    variables = [
        ("supply_temperature", 0, "C"),
        ("return_temperature", 1, "C"),
        ("head", 2, "m"),
        ("flow", 3, "m3/s"),
    ]
    for variable, idx, unit in variables:
        measured = mask[..., idx] > 0.5
        unmeasured = ~measured
        if np.any(measured):
            virtual_rows.append(
                {
                    "virtual_sensor": variable,
                    "metric": "RMSE_measured_nodes",
                    "value": _rmse(pred[..., idx][measured], sensor[..., idx][measured]),
                    "unit": unit,
                    "state_type": _state_type(variable, measured=True),
                    "model": model_name,
                    "interpretation": "virtual-sensor agreement at sparse measured/sensor nodes",
                    "safe_claim": "Measured-node validation is only for available measured variables; hydraulic sensor nodes are simulator-assisted.",
                }
            )
        virtual_rows.append(
            {
                "virtual_sensor": variable,
                "metric": "RMSE_unmeasured_nodes",
                "value": _rmse(pred[..., idx][unmeasured], true[..., idx][unmeasured]),
                "unit": unit,
                "state_type": "simulator_assisted_hidden_state" if variable in {"head", "flow"} else "virtual_sensor_estimate",
                "model": model_name,
                "interpretation": "virtual-sensor reconstruction away from sparse sensors",
                "safe_claim": "Unmeasured distributed states are evaluated against calibrated-simulator hidden states.",
            }
        )
        virtual_rows.append(
            {
                "virtual_sensor": variable,
                "metric": "MAE_unmeasured_nodes",
                "value": _mae(pred[..., idx][unmeasured], true[..., idx][unmeasured]),
                "unit": unit,
                "state_type": "simulator_assisted_hidden_state" if variable in {"head", "flow"} else "virtual_sensor_estimate",
                "model": model_name,
                "interpretation": "mean absolute virtual-sensor reconstruction error away from sensors",
                "safe_claim": "Unmeasured distributed states are evaluated against calibrated-simulator hidden states.",
            }
        )

    pressure_pred = rho * g * pred[..., 2] / 1000.0
    pressure_true = rho * g * true[..., 2] / 1000.0
    pressure_drop_pred = pressure_pred[..., 0] - pressure_pred[..., -1]
    pressure_drop_true = pressure_true[..., 0] - pressure_true[..., -1]
    pred_seg_loss = _segment_heat_loss_flat(pred, ambient, config)
    true_seg_loss = _segment_heat_loss_flat(true, ambient, config)
    pred_total_loss = np.nansum(pred_seg_loss, axis=1)
    true_total_loss = np.nansum(true_seg_loss, axis=1)
    pred_delivered = _delivered_heat_flat(pred, config)
    energy_residual_percent = np.abs(pred_delivered - heat_load) / np.maximum(np.abs(heat_load), 1.0) * 100.0
    pressure_residual_percent = np.abs(pressure_drop_pred - pressure_drop_true) / np.maximum(np.abs(pressure_drop_true), 1e-6) * 100.0

    virtual_rows.extend(
        [
            {
                "virtual_sensor": "pressure_drop",
                "metric": "pressure_drop_residual_percent",
                "value": float(np.nanmean(pressure_residual_percent)),
                "unit": "%",
                "state_type": "simulator_assisted_hidden_state",
                "model": model_name,
                "interpretation": "virtual pressure-drop consistency from simulator-assisted head",
                "safe_claim": "Pressure/head fields are simulator-assisted hidden states.",
            },
            {
                "virtual_sensor": "segment_heat_loss",
                "metric": "segment_heat_loss_RMSE_kW",
                "value": _rmse(pred_seg_loss, true_seg_loss),
                "unit": "kW",
                "state_type": "calibrated_simulator",
                "model": model_name,
                "interpretation": "segment-wise heat-loss estimate from virtual temperature and flow fields",
                "safe_claim": "Heat-loss labels are calibrated-simulator quantities, not direct pipe heat-loss measurements.",
            },
            {
                "virtual_sensor": "cumulative_heat_loss",
                "metric": "cumulative_heat_loss_error_percent",
                "value": _safe_percent(pred_total_loss - true_total_loss, true_total_loss),
                "unit": "%",
                "state_type": "calibrated_simulator",
                "model": model_name,
                "interpretation": "cumulative heat-loss estimate over monitoring windows",
                "safe_claim": "Heat-loss labels are calibrated-simulator quantities.",
            },
            {
                "virtual_sensor": "energy_balance",
                "metric": "mean_energy_balance_residual_percent",
                "value": float(np.nanmean(energy_residual_percent)),
                "unit": "%",
                "state_type": "calibrated_simulator",
                "model": model_name,
                "interpretation": "delivered heat versus real boundary heat-load input",
                "safe_claim": "Energy residual combines real boundary load with simulator-assisted virtual states.",
            },
        ]
    )
    virtual_df = pd.DataFrame(virtual_rows)
    virtual_df.to_csv(PROJECT_ROOT / "results" / "digital_twin_virtual_sensor_metrics.csv", index=False)

    temp_sensor_residual = np.abs(pred[..., :2] - sensor[..., :2])
    temp_sensor_residual = np.where(mask[..., :2] > 0.5, temp_sensor_residual, np.nan)
    warn_temp = float(np.nanpercentile(temp_sensor_residual, 95)) if np.isfinite(temp_sensor_residual).any() else 1.0
    sensor_health = _sensor_health(mask[..., :2], np.nan_to_num(temp_sensor_residual, nan=0.0), warn_temp)
    warning_score = (
        0.35 * np.nanmean(np.nan_to_num(temp_sensor_residual, nan=0.0), axis=(1, 2)) / max(warn_temp, 1e-9)
        + 0.35 * energy_residual_percent / max(float(np.nanpercentile(energy_residual_percent, 95)), 1e-9)
        + 0.30 * pressure_residual_percent / max(float(np.nanpercentile(pressure_residual_percent, 95)), 1e-9)
    )
    warning_threshold = float(np.nanpercentile(warning_score, 95))
    alarm_threshold = float(np.nanpercentile(warning_score, 99))
    flags = pd.DataFrame(
        {
            "window_index": np.arange(pred.shape[0]),
            "temperature_residual_C": np.nanmean(temp_sensor_residual, axis=(1, 2)),
            "energy_balance_residual_percent": energy_residual_percent,
            "pressure_drop_residual_percent": pressure_residual_percent,
            "sensor_health_indicator": sensor_health,
            "warning_score": warning_score,
            "warning_threshold": warning_threshold,
            "alarm_threshold": alarm_threshold,
            "warning_flag": warning_score > warning_threshold,
            "alarm_flag": warning_score > alarm_threshold,
            "state_type": "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
            "note": "Operational anomaly indicators are residual-based monitoring outputs, not field fault labels.",
        }
    )
    flags.to_csv(PROJECT_ROOT / "results" / "digital_twin_anomaly_flags.csv", index=False)

    kpi_rows = [
        ("virtual_sensor_RMSE_measured_nodes_C", virtual_df.query("virtual_sensor == 'supply_temperature' and metric == 'RMSE_measured_nodes'")["value"].mean(), "C", "real_measured_node", "virtual thermal sensor agreement"),
        ("virtual_sensor_RMSE_unmeasured_nodes_C", virtual_df.query("virtual_sensor == 'supply_temperature' and metric == 'RMSE_unmeasured_nodes'")["value"].mean(), "C", "virtual_sensor_estimate", "hidden-state virtual sensor error"),
        ("heat_loss_estimation_error_percent", _safe_percent(pred_total_loss - true_total_loss, true_total_loss), "%", "calibrated_simulator", "pipe heat-loss KPI"),
        ("energy_balance_residual_percent", float(np.nanmean(energy_residual_percent)), "%", "calibrated_simulator", "energy consistency KPI"),
        ("pressure_drop_residual_percent", float(np.nanmean(pressure_residual_percent)), "%", "simulator_assisted_hidden_state", "hydraulic consistency KPI"),
        ("sensor_health_mean", float(np.nanmean(sensor_health)), "0-1", "real_measured_node", "availability and residual health index"),
        ("warning_flag_rate_percent", float(np.nanmean(flags["warning_flag"].astype(float)) * 100.0), "%", "virtual_sensor_estimate", "residual-warning rate in nominal monitoring stream"),
        ("alarm_flag_rate_percent", float(np.nanmean(flags["alarm_flag"].astype(float)) * 100.0), "%", "virtual_sensor_estimate", "residual-alarm rate in nominal monitoring stream"),
    ]
    kpi_df = pd.DataFrame(
        [
            {
                "kpi": kpi,
                "value": float(value) if np.isfinite(value) else np.nan,
                "unit": unit,
                "state_type": state_type,
                "operational_use": use,
                "safe_claim": "Digital-twin KPIs combine real measured-node evidence with calibrated simulator and virtual-sensor estimates.",
            }
            for kpi, value, unit, state_type, use in kpi_rows
        ]
    )
    kpi_df.to_csv(PROJECT_ROOT / "results" / "digital_twin_kpis.csv", index=False)
    kpi_df.to_csv(PROJECT_ROOT / "results" / "digital_twin_dashboard_kpis.csv", index=False)
    kpi_df.to_csv(PROJECT_ROOT / "results" / "digital_twin_kpis_improved.csv", index=False)

    _plot_dashboard(sim, sensors, payloads, model_name, payload, pred_seg_loss, true_seg_loss, flags, config)
    copy_final_figures_to_root_and_paper()
    print("Digital-twin monitoring layer completed.")


def _plot_dashboard(
    sim: dict[str, Any],
    sensors: dict[str, Any],
    payloads: dict[str, dict[str, np.ndarray]],
    model_name: str,
    payload: dict[str, np.ndarray],
    pred_seg_loss: np.ndarray,
    true_seg_loss: np.ndarray,
    flags: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    pred = _flat(payload["pred"])
    true = _flat(payload["true"])
    ensemble = np.stack([_flat(p["pred"]) for p in payloads.values()], axis=0)
    mean_pred = np.nanmean(ensemble, axis=0)
    residual_floor = np.nanstd(pred - true, axis=(0, 1), keepdims=True)
    std_pred = np.nanstd(ensemble, axis=0) + residual_floor
    time_idx = int(np.nanargmax(np.asarray(payload["heat_load_kw"]).reshape(-1)))
    time_idx = min(time_idx, pred.shape[0] - 1)
    seg_x = 0.5 * (x_km[:-1] + x_km[1:])

    fig, axes = plt.subplots(4, 2, figsize=(11.8, 11.2))
    ax = axes[0, 0]
    ax.plot([0, x_km[-1]], [0, 0], color="#444", lw=3, label="20 km pipe")
    nodes = sensors.get("sensor_nodes", [])
    if nodes:
        ax.scatter(x_km[nodes], np.zeros(len(nodes)), s=80, color="#d1495b", label="sparse sensors")
    virtual_nodes = np.linspace(0, len(x_km) - 1, min(8, len(x_km))).round().astype(int)
    ax.scatter(x_km[virtual_nodes], np.full(len(virtual_nodes), 0.05), s=28, color="#277da1", label="virtual sensors")
    ax.set_xlim(-0.5, x_km[-1] + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Distance (km)")
    ax.set_title("A. Sparse and virtual sensors")
    ax.legend(fontsize=7, loc="upper center", ncols=2)

    ax = axes[0, 1]
    ts_mean = mean_pred[time_idx, :, 0]
    ts_low = ts_mean - 1.645 * std_pred.reshape(-1, pred.shape[1], 4)[time_idx, :, 0]
    ts_high = ts_mean + 1.645 * std_pred.reshape(-1, pred.shape[1], 4)[time_idx, :, 0]
    ax.plot(x_km, true[time_idx, :, 0], color="black", lw=1.7, label="calibrated simulator")
    ax.plot(x_km, pred[time_idx, :, 0], color="#7b2cbf", lw=1.5, label=model_name.replace("Proposed ", ""))
    ax.fill_between(x_km, ts_low, ts_high, color="#7b2cbf", alpha=0.18, label="90% band")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Supply temp ($^\\circ$C)")
    ax.set_title("B. Virtual temperature sensors")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    pressure = float(config["system"]["rho"]) * float(config["system"]["g"]) * pred[time_idx, :, 2] / 1000.0
    ax.plot(x_km, pressure, color="#577590", lw=1.6)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Pressure (kPa)")
    ax.set_title("C. Pressure/head (simulator-assisted)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(seg_x, true_seg_loss[time_idx], color="black", lw=1.5, label="calibrated simulator")
    ax.plot(seg_x, pred_seg_loss[time_idx], color="#43aa8b", lw=1.5, label="virtual estimate")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Heat loss (kW/segment)")
    ax.set_title("D. Segment heat loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[2, 0]
    ax.plot(flags["window_index"], flags["energy_balance_residual_percent"], color="#f9844a", lw=1.2)
    ax.set_xlabel("Monitoring window")
    ax.set_ylabel("Energy residual (%)")
    ax.set_title("E. Energy-balance residual")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 1]
    ax.plot(flags["window_index"], flags["warning_score"], color="#577590", lw=1.1, label="residual score")
    ax.axhline(flags["warning_threshold"].iloc[0], color="#f9c74f", ls="--", label="warning")
    ax.axhline(flags["alarm_threshold"].iloc[0], color="#d1495b", ls="--", label="alarm")
    ax2 = ax.twinx()
    ax2.plot(flags["window_index"], flags["sensor_health_indicator"], color="#43aa8b", lw=1.1, alpha=0.8, label="sensor health")
    ax.set_xlabel("Monitoring window")
    ax.set_ylabel("Residual score")
    ax2.set_ylabel("Health index")
    ax.set_title("F. Anomaly flags and sensor health")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")

    ax = axes[3, 0]
    status = np.where(flags["alarm_flag"], 3, np.where(flags["warning_flag"], 2, np.where(flags["sensor_health_indicator"] < 0.55, 1, 0)))
    colors = ["#43aa8b", "#f9c74f", "#f9844a", "#d1495b"]
    labels = ["normal", "dropout/low health", "warning", "alarm"]
    for value, color, label in zip(range(4), colors, labels):
        idx = status == value
        ax.scatter(flags.loc[idx, "window_index"], np.full(np.sum(idx), value), s=12, color=color, label=label)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Monitoring window")
    ax.set_title("G. Sensor health status")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(fontsize=6, ncols=2, loc="upper right")

    ax = axes[3, 1]
    summary_text = (
        "Operational layer\n"
        f"Model: {model_name.replace('Proposed ', '')}\n"
        "Evidence boundary:\n"
        "- thermal measured nodes: real-data-assisted\n"
        "- heat loss: calibrated-simulator quantity\n"
        "- pressure/head/flow: simulator-assisted hidden states\n"
        "- anomaly flags: residual diagnostics, not field fault labels"
    )
    ax.text(0.02, 0.95, summary_text, va="top", ha="left", fontsize=8)
    ax.axis("off")
    fig.suptitle(
        "Uncertainty-aware operational digital-twin dashboard: virtual sensors, residuals, and KPI flags; hydraulic states are simulator-assisted.",
        fontsize=11,
    )
    save_figure(fig, "fig_digital_twin_monitoring_dashboard")


if __name__ == "__main__":
    run_digital_twin_monitoring()
