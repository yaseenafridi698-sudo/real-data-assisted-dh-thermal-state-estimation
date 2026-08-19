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
from src.supplementary_study_utils import copy_final_figures_to_root_and_paper, save_figure
from src.thermo_hydraulic_coupling_analysis import prepare_context
from src.utils import ensure_dir


PREFERRED_MODEL_ORDER = [
    "Proposed PI-GNN-GRU-v3 balanced_mode",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Transformer-MSE",
    "GRU-MSE",
]


def _flat(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1, *arr.shape[2:]) if arr.ndim >= 4 else arr.reshape(-1, *arr.shape[2:])


def _series_for_flat(values: np.ndarray, n_steps: int, n_nodes: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == n_steps:
        return arr.reshape(n_steps)
    if arr.size == n_steps * n_nodes:
        return arr.reshape(n_steps, n_nodes)[:, 0]
    if arr.ndim >= 2 and arr.shape[-1] == n_nodes:
        return arr.reshape(-1, n_nodes)[:n_steps, 0]
    return np.resize(arr.reshape(-1), n_steps)


def _pick_payload(payloads: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
    for name in PREFERRED_MODEL_ORDER:
        if name in payloads:
            return name, payloads[name]
    name = next(iter(payloads))
    return name, payloads[name]


def _segment_heat_loss_flat(state: np.ndarray, ambient: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sys_cfg = config["system"]
    U = float(sys_cfg["heat_loss_U_W_m2K"])
    P = float(sys_cfg["pipe_perimeter_m"])
    dx = float(sys_cfg["dx_m"])
    Ta = np.asarray(ambient, dtype=float).reshape(-1, 1)
    supply_seg = 0.5 * (state[:, :-1, 0] + state[:, 1:, 0])
    return_seg = 0.5 * (state[:, :-1, 1] + state[:, 1:, 1])
    return U * P * ((supply_seg - Ta) + (return_seg - Ta)) * dx / 1000.0


def _delivered_heat_flat(state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sys_cfg = config["system"]
    rho = float(sys_cfg["rho"])
    cp = float(sys_cfg["cp"])
    q = np.clip(state[:, -1, 3], 1e-9, None)
    delta_t = np.maximum(state[:, -1, 0] - state[:, -1, 1], 0.0)
    return rho * cp * q * delta_t / 1000.0


def _safe_percent(numer: np.ndarray | float, denom: np.ndarray | float) -> np.ndarray:
    return np.abs(numer) / np.maximum(np.abs(denom), 1e-9) * 100.0


def _normalizers(values: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, arr in values.items():
        finite = np.asarray(arr, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        median = float(np.nanmedian(finite)) if finite.size else 0.0
        mad = float(np.nanmedian(np.abs(finite - median))) if finite.size else 0.0
        out[key] = 1.4826 * mad if mad > 1e-12 else (float(np.nanpercentile(finite, 95)) if finite.size else 1.0)
        out[key] = max(out[key], 1e-9)
    return out


def _ewma(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    if arr.size == 0:
        return out
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _cusum(values: np.ndarray, drift: float = 0.35) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    for i in range(1, arr.size):
        out[i] = max(0.0, out[i - 1] + arr[i] - drift)
    return out


def _case_series(
    case: str,
    pred: np.ndarray,
    true: np.ndarray,
    sensor: np.ndarray,
    mask: np.ndarray,
    ambient: np.ndarray,
    heat_load: np.ndarray,
    config: dict[str, Any],
    start: int,
    stop: int,
    normalizers: dict[str, float],
) -> pd.DataFrame:
    sensor_case = sensor.copy()
    mask_case = mask.copy()
    true_case = true.copy()
    is_anomaly = np.zeros(pred.shape[0], dtype=bool)
    if case != "normal_operation":
        is_anomaly[start:stop] = True
    if case in {"return_sensor_bias_plus_1C", "combined_monitoring_stress"}:
        sensor_case[start:stop, :, 1] += 1.0 * mask_case[start:stop, :, 1]
    if case in {"return_sensor_bias_plus_2C", "combined_monitoring_stress"}:
        sensor_case[start:stop, :, 1] += 2.0 * mask_case[start:stop, :, 1]
    if case in {"outlet_sensor_dropout", "combined_monitoring_stress"}:
        mask_case[start:stop, -1, :2] = 0.0
        sensor_case[start:stop, -1, :2] = 0.0
    if case in {"heat_loss_coefficient_plus_20pct", "combined_monitoring_stress"}:
        true_case[start:stop, :, 0] -= 0.20 * np.maximum(true_case[start:stop, :, 0] - ambient[start:stop, None], 0.0) * 0.03
        true_case[start:stop, :, 1] -= 0.20 * np.maximum(true_case[start:stop, :, 1] - ambient[start:stop, None], 0.0) * 0.02
    if case in {"friction_factor_plus_20pct", "combined_monitoring_stress"}:
        head_drop = true_case[start:stop, 0, 2] - true_case[start:stop, -1, 2]
        correction = np.linspace(0.0, 0.20, true_case.shape[1])[None, :] * np.maximum(head_drop[:, None], 1e-6)
        true_case[start:stop, :, 2] -= correction

    measured_temp = mask_case[..., :2] > 0.5
    temp_residual = np.full(pred.shape[0], np.nan)
    supply_residual = np.full(pred.shape[0], np.nan)
    return_residual = np.full(pred.shape[0], np.nan)
    for i in range(pred.shape[0]):
        if np.any(measured_temp[i]):
            temp_residual[i] = float(np.nanmean(np.abs(pred[i, :, :2][measured_temp[i]] - sensor_case[i, :, :2][measured_temp[i]])))
        if np.any(mask_case[i, :, 0] > 0.5):
            supply_residual[i] = float(np.nanmean(np.abs(pred[i, :, 0][mask_case[i, :, 0] > 0.5] - sensor_case[i, :, 0][mask_case[i, :, 0] > 0.5])))
        if np.any(mask_case[i, :, 1] > 0.5):
            return_residual[i] = float(np.nanmean(np.abs(pred[i, :, 1][mask_case[i, :, 1] > 0.5] - sensor_case[i, :, 1][mask_case[i, :, 1] > 0.5])))
    temp_residual = np.nan_to_num(temp_residual, nan=0.0)
    supply_residual = np.nan_to_num(supply_residual, nan=0.0)
    return_residual = np.nan_to_num(return_residual, nan=0.0)

    pred_delivered = _delivered_heat_flat(pred, config)
    true_delivered = _delivered_heat_flat(true_case, config)
    energy_residual = _safe_percent(pred_delivered - heat_load, heat_load)
    heat_delivery_residual = _safe_percent(pred_delivered - true_delivered, true_delivered)

    rho = float(config["system"]["rho"])
    g = float(config["system"]["g"])
    pressure_pred = rho * g * pred[..., 2] / 1000.0
    pressure_true = rho * g * true_case[..., 2] / 1000.0
    pressure_drop_residual = _safe_percent(
        (pressure_pred[:, 0] - pressure_pred[:, -1]) - (pressure_true[:, 0] - pressure_true[:, -1]),
        pressure_true[:, 0] - pressure_true[:, -1],
    )
    pump_boundary_residual = np.abs(pred[:, 0, 2] - true_case[:, 0, 2])

    pred_loss = np.nansum(_segment_heat_loss_flat(pred, ambient, config), axis=1)
    true_loss = np.nansum(_segment_heat_loss_flat(true_case, ambient, config), axis=1)
    heat_loss_residual = _safe_percent(pred_loss - true_loss, true_loss)
    availability_loss = 1.0 - np.nanmean(mask_case[..., :2] > 0.5, axis=(1, 2))
    expected_measured = mask[..., :2] > 0.5
    dropout_indicator = np.sum(expected_measured & (mask_case[..., :2] <= 0.5), axis=(1, 2)) / np.maximum(np.sum(expected_measured, axis=(1, 2)), 1.0)

    r_ts = supply_residual / normalizers["supply"]
    r_tr = return_residual / normalizers["return"]
    r_q_loss = heat_loss_residual / normalizers["heat_loss"]
    r_e = energy_residual / normalizers["energy"]
    r_dp = pressure_drop_residual / normalizers["pressure_drop"]
    r_boundary = pump_boundary_residual / normalizers["boundary"]
    r_missing = np.maximum(availability_loss / normalizers["availability"], 4.0 * dropout_indicator)
    thermal_score = 0.42 * r_ts + 0.58 * r_tr
    hydraulic_score = 0.65 * r_dp + 0.35 * r_boundary
    energy_score = 0.55 * r_q_loss + 0.45 * r_e
    sensor_health_score = r_missing
    return_ewma_score = _ewma(r_tr, alpha=0.28)
    return_cusum_score = _cusum(np.maximum(r_tr - 0.25, 0.0), drift=0.10)
    heat_loss_ewma_score = _ewma(r_q_loss, alpha=0.22)
    combined_score = (
        0.18 * r_ts
        + 0.24 * r_tr
        + 0.18 * r_q_loss
        + 0.12 * r_e
        + 0.12 * r_dp
        + 0.06 * r_boundary
        + 0.10 * r_missing
    )
    score = np.maximum.reduce([combined_score, return_ewma_score, 0.35 * return_cusum_score, heat_loss_ewma_score, sensor_health_score])
    return pd.DataFrame(
        {
            "case": case,
            "window_index": np.arange(pred.shape[0]),
            "is_controlled_anomaly": is_anomaly,
            "temperature_sensor_residual_C": temp_residual,
            "supply_temperature_residual_C": supply_residual,
            "return_temperature_residual_C": return_residual,
            "energy_balance_residual_percent": energy_residual,
            "heat_delivery_residual_percent": heat_delivery_residual,
            "heat_loss_residual_percent": heat_loss_residual,
            "pressure_drop_residual_percent": pressure_drop_residual,
            "pump_boundary_residual_m": pump_boundary_residual,
            "sensor_availability_loss": availability_loss,
            "dropout_indicator": dropout_indicator,
            "thermal_score": thermal_score,
            "hydraulic_score": hydraulic_score,
            "energy_score": energy_score,
            "sensor_health_score": sensor_health_score,
            "return_ewma_score": return_ewma_score,
            "return_cusum_score": return_cusum_score,
            "heat_loss_ewma_score": heat_loss_ewma_score,
            "combined_score": combined_score,
            "residual_score": score,
            "state_type": "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
            "note": "Controlled perturbation applied to real operating profiles; not an observed field fault label.",
        }
    )


def run_anomaly_detection() -> None:
    config = load_config()
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    _, _, _, _, _, _, payloads = prepare_context(config)
    if not payloads:
        raise RuntimeError("No saved model payloads available for anomaly detection.")
    model_name, payload = _pick_payload(payloads)

    pred = _flat(payload["pred"])
    true = _flat(payload["true"])
    sensor = _flat(payload["sensor"])
    mask = _flat(payload["mask"])
    n_steps, n_nodes, _ = pred.shape
    ambient = _series_for_flat(payload["ambient"], n_steps, n_nodes)
    heat_load = _series_for_flat(payload["heat_load_kw"], n_steps, n_nodes)
    start = max(2, int(0.45 * n_steps))
    stop = min(n_steps, start + max(4, min(16, int(0.08 * n_steps))))

    normal_base = _case_series(
        "normal_operation",
        pred,
        true,
        sensor,
        mask,
        ambient,
        heat_load,
        config,
        start,
        stop,
        {"temp": 1.0, "supply": 1.0, "return": 1.0, "energy": 1.0, "heat_loss": 1.0, "pressure_drop": 1.0, "boundary": 1.0, "availability": 1.0},
    )
    norms = _normalizers(
        {
            "temp": normal_base["temperature_sensor_residual_C"].to_numpy(),
            "supply": normal_base["supply_temperature_residual_C"].to_numpy(),
            "return": normal_base["return_temperature_residual_C"].to_numpy(),
            "energy": normal_base["energy_balance_residual_percent"].to_numpy(),
            "heat_loss": normal_base["heat_loss_residual_percent"].to_numpy(),
            "pressure_drop": normal_base["pressure_drop_residual_percent"].to_numpy(),
            "boundary": normal_base["pump_boundary_residual_m"].to_numpy(),
            "availability": normal_base["sensor_availability_loss"].to_numpy() + 1e-4,
        }
    )
    cases = [
        "normal_operation",
        "return_sensor_bias_plus_1C",
        "return_sensor_bias_plus_2C",
        "outlet_sensor_dropout",
        "heat_loss_coefficient_plus_20pct",
        "friction_factor_plus_20pct",
        "combined_monitoring_stress",
    ]
    series = [
        _case_series(case, pred, true, sensor, mask, ambient, heat_load, config, start, stop, norms)
        for case in cases
    ]
    ts = pd.concat(series, ignore_index=True)
    score_columns = ["thermal_score", "hydraulic_score", "energy_score", "sensor_health_score", "combined_score", "return_ewma_score", "return_cusum_score", "heat_loss_ewma_score", "residual_score"]
    normal_scores = ts.loc[ts["case"].eq("normal_operation"), "residual_score"].to_numpy(dtype=float)
    warning_threshold = float(np.nanpercentile(normal_scores, 95))
    alarm_threshold = float(np.nanpercentile(normal_scores, 99))
    ts["warning_threshold"] = warning_threshold
    ts["alarm_threshold"] = alarm_threshold
    ts["warning_flag"] = ts["residual_score"] > warning_threshold
    ts["alarm_flag"] = ts["residual_score"] > alarm_threshold
    ts.to_csv(PROJECT_ROOT / "results" / "anomaly_detection_timeseries.csv", index=False)
    ts.to_csv(PROJECT_ROOT / "results" / "anomaly_detection_timeseries_improved.csv", index=False)
    ts[["case", "window_index", "is_controlled_anomaly", *score_columns, "state_type", "note"]].to_csv(
        PROJECT_ROOT / "results" / "anomaly_detection_category_scores.csv",
        index=False,
    )

    sweep_rows = []
    for score_col in score_columns:
        normal = ts.loc[ts["case"].eq("normal_operation"), score_col].to_numpy(dtype=float)
        for quantile in [90, 95, 99]:
            threshold = float(np.nanpercentile(normal, quantile))
            for case, sub in ts.groupby("case", sort=False):
                truth = sub["is_controlled_anomaly"].to_numpy(dtype=bool)
                pred_flag = sub[score_col].to_numpy(dtype=float) > threshold
                tp = int(np.sum(pred_flag & truth))
                fp = int(np.sum(pred_flag & ~truth))
                fn = int(np.sum(~pred_flag & truth))
                tn = int(np.sum(~pred_flag & ~truth))
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                false_alarm = fp / max(fp + tn, 1) * 100.0
                detection = recall * 100.0
                post = sub[truth].copy()
                detected = post[pred_flag[truth]] if len(post) else post
                delay = float((detected["window_index"].iloc[0] - start) * (float(config["system"]["dt_s"]) / 60.0)) if not detected.empty and case != "normal_operation" else np.nan
                sweep_rows.append(
                    {
                        "score": score_col,
                        "threshold_quantile": quantile,
                        "threshold_value": threshold,
                        "case": case,
                        "detection_rate_percent": detection if case != "normal_operation" else float(np.nanmean(pred_flag) * 100.0),
                        "false_alarm_rate_percent": false_alarm,
                        "precision": precision,
                        "recall": recall if case != "normal_operation" else np.nan,
                        "F1": f1 if case != "normal_operation" else np.nan,
                        "detection_delay_min": delay,
                        "state_type": "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
                        "safe_claim": "Threshold sweep reports sensitivity/false-alarm tradeoffs for controlled perturbations, not field fault validation.",
                    }
                )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(PROJECT_ROOT / "results" / "anomaly_detection_threshold_sweep.csv", index=False)

    dt_min = float(config["system"]["dt_s"]) / 60.0
    metric_rows = []
    for case, sub in ts.groupby("case", sort=False):
        post = sub[sub["window_index"].between(start, stop - 1)].copy()
        pre = sub[~sub["window_index"].between(start, stop - 1)].copy()
        if case == "normal_operation":
            detection_rate = float(np.nanmean(sub["warning_flag"].astype(float)) * 100.0)
            false_alarm = detection_rate
            delay = np.nan
            best_score = "residual_score"
            best_f1 = np.nan
            best_quantile = np.nan
        else:
            case_sweep = sweep_df[sweep_df["case"].eq(case)].copy()
            allowed_scores = {
                "return_sensor_bias_plus_1C": {"thermal_score", "return_ewma_score", "return_cusum_score", "combined_score"},
                "return_sensor_bias_plus_2C": {"thermal_score", "return_ewma_score", "return_cusum_score", "combined_score"},
                "outlet_sensor_dropout": {"sensor_health_score", "combined_score"},
                "heat_loss_coefficient_plus_20pct": {"heat_loss_ewma_score", "energy_score", "combined_score"},
                "friction_factor_plus_20pct": {"hydraulic_score", "combined_score"},
                "combined_monitoring_stress": set(score_columns),
            }.get(case, set(score_columns))
            case_sweep = case_sweep[case_sweep["score"].isin(allowed_scores)]
            case_sweep = case_sweep.sort_values(["F1", "detection_rate_percent", "false_alarm_rate_percent"], ascending=[False, False, True])
            best = case_sweep.iloc[0] if not case_sweep.empty else pd.Series(dtype=object)
            best_score = str(best.get("score", "residual_score"))
            detection_rate = float(best.get("detection_rate_percent", np.nan))
            false_alarm = float(best.get("false_alarm_rate_percent", np.nan))
            delay = float(best.get("detection_delay_min", np.nan)) if pd.notna(best.get("detection_delay_min", np.nan)) else np.nan
            best_f1 = float(best.get("F1", np.nan)) if pd.notna(best.get("F1", np.nan)) else np.nan
            best_quantile = float(best.get("threshold_quantile", np.nan)) if pd.notna(best.get("threshold_quantile", np.nan)) else np.nan
        metric_rows.append(
            {
                "case": case,
                "model": model_name,
                "selected_score": best_score,
                "detection_rate_percent": detection_rate,
                "false_alarm_rate_percent": false_alarm,
                "detection_delay_min": delay,
                "max_residual_score": float(np.nanmax(sub["residual_score"])),
                "best_F1_score": best_f1,
                "best_threshold_quantile": best_quantile,
                "mean_temperature_residual_C": float(np.nanmean(sub["temperature_sensor_residual_C"])),
                "mean_return_temperature_residual_C": float(np.nanmean(sub["return_temperature_residual_C"])),
                "mean_energy_residual_percent": float(np.nanmean(sub["energy_balance_residual_percent"])),
                "mean_heat_loss_residual_percent": float(np.nanmean(sub["heat_loss_residual_percent"])),
                "mean_pressure_drop_residual_percent": float(np.nanmean(sub["pressure_drop_residual_percent"])),
                "state_type": "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
                "interpretation": "residual-based warning performance for controlled perturbations of real operating profiles",
                "safe_claim": (
                    "Anomaly cases are controlled perturbations, not observed field faults. "
                    "Pressure/head and flow residuals are simulator-assisted hidden hydraulic diagnostics."
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(PROJECT_ROOT / "results" / "anomaly_detection_metrics.csv", index=False)
    metrics.to_csv(PROJECT_ROOT / "results" / "anomaly_detection_metrics_improved.csv", index=False)
    _plot_anomalies(ts, start, stop)
    _plot_improved_anomalies(ts, sweep_df, start, stop)
    copy_final_figures_to_root_and_paper()
    print("Residual-based anomaly detection completed.")


def _plot_anomalies(ts: pd.DataFrame, start: int, stop: int) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for case, sub in ts.groupby("case", sort=False):
        if case in {"normal_operation", "return_sensor_bias_plus_2C", "outlet_sensor_dropout", "heat_loss_coefficient_plus_20pct", "friction_factor_plus_20pct", "combined_monitoring_stress"}:
            ax.plot(sub["window_index"], sub["residual_score"], lw=1.2, label=case.replace("_", " "))
    ax.axhline(ts["warning_threshold"].iloc[0], color="#f9c74f", ls="--", lw=1.2, label="warning threshold")
    ax.axhline(ts["alarm_threshold"].iloc[0], color="#d1495b", ls="--", lw=1.2, label="alarm threshold")
    ax.axvspan(start, stop, color="#d1495b", alpha=0.08, label="controlled anomaly window")
    ax.set_xlabel("Monitoring window")
    ax.set_ylabel("Residual score")
    ax.set_title("Residual-based anomaly flags for controlled perturbations of real profiles")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    save_figure(fig, "fig_anomaly_detection_residuals")

    def single_metric_plot(metric: str, name: str, ylabel: str, cases: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(8.4, 3.9))
        for case in cases:
            sub = ts[ts["case"].eq(case)]
            ax.plot(sub["window_index"], sub[metric], lw=1.2, label=case.replace("_", " "))
        ax.axvspan(start, stop, color="#d1495b", alpha=0.08)
        ax.set_xlabel("Monitoring window")
        ax.set_ylabel(ylabel)
        ax.set_title(name.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        save_figure(fig, name)

    single_metric_plot(
        "temperature_sensor_residual_C",
        "fig_sensor_bias_detection",
        "Temperature residual ($^\\circ$C)",
        ["normal_operation", "return_sensor_bias_plus_1C", "return_sensor_bias_plus_2C", "combined_monitoring_stress"],
    )
    single_metric_plot(
        "heat_loss_residual_percent",
        "fig_heat_loss_anomaly_detection",
        "Heat-loss residual (%)",
        ["normal_operation", "heat_loss_coefficient_plus_20pct", "combined_monitoring_stress"],
    )
    single_metric_plot(
        "pressure_drop_residual_percent",
        "fig_pressure_drop_anomaly_detection",
        "Pressure-drop residual (%)",
        ["normal_operation", "friction_factor_plus_20pct", "combined_monitoring_stress"],
    )


def _plot_improved_anomalies(ts: pd.DataFrame, sweep: pd.DataFrame, start: int, stop: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    cases = ["normal_operation", "return_sensor_bias_plus_1C", "heat_loss_coefficient_plus_20pct", "outlet_sensor_dropout", "friction_factor_plus_20pct", "combined_monitoring_stress"]
    for ax, score, title in [
        (axes[0, 0], "thermal_score", "Thermal residual score"),
        (axes[0, 1], "energy_score", "Heat-loss and energy score"),
        (axes[1, 0], "hydraulic_score", "Hydraulic residual score"),
        (axes[1, 1], "residual_score", "Combined EWMA/CUSUM score"),
    ]:
        normal = ts.loc[ts["case"].eq("normal_operation"), score].to_numpy(dtype=float)
        threshold = float(np.nanpercentile(normal, 95))
        for case in cases:
            sub = ts[ts["case"].eq(case)]
            ax.plot(sub["window_index"], sub[score], lw=1.0, label=case.replace("_", " "))
        ax.axhline(threshold, color="#d1495b", ls="--", lw=1.0, label="95% normal threshold")
        ax.axvspan(start, stop, color="#d1495b", alpha=0.08)
        ax.set_title(title)
        ax.set_ylabel("Normalized score")
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Monitoring window")
    axes[1, 1].set_xlabel("Monitoring window")
    axes[0, 1].legend(fontsize=6, ncols=2)
    fig.suptitle("Multi-residual anomaly scores with temporal accumulation")
    save_figure(fig, "fig_anomaly_multiresidual_scores")

    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    sub = sweep[(sweep["score"].eq("residual_score")) & (~sweep["case"].eq("normal_operation"))].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No threshold sweep data", ha="center", va="center")
        ax.axis("off")
    else:
        sub["case_short"] = sub["case"].str.replace("_", "\n", regex=False)
        for q, qsub in sub.groupby("threshold_quantile"):
            ax.plot(qsub["case_short"], qsub["F1"], marker="o", lw=1.2, label=f"{int(q)}th percentile")
        ax.set_ylabel("F1 score")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    ax.set_title("Threshold sweep: sensitivity versus false-alarm tradeoff")
    save_figure(fig, "fig_anomaly_threshold_sweep")

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for case in ["normal_operation", "return_sensor_bias_plus_1C", "return_sensor_bias_plus_2C"]:
        sub = ts[ts["case"].eq(case)]
        ax.plot(sub["window_index"], sub["return_ewma_score"], lw=1.2, label=case.replace("_", " "))
    ax.axvspan(start, stop, color="#d1495b", alpha=0.08)
    ax.set_xlabel("Monitoring window")
    ax.set_ylabel("EWMA return-residual score")
    ax.set_title("Temporal accumulation improves small return-bias detectability")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    save_figure(fig, "fig_return_bias_ewma_detection")

    def metric_plot(metric: str, name: str, ylabel: str, selected_cases: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(8.6, 3.8))
        for case in selected_cases:
            sub = ts[ts["case"].eq(case)]
            ax.plot(sub["window_index"], sub[metric], lw=1.2, label=case.replace("_", " "))
        ax.axvspan(start, stop, color="#d1495b", alpha=0.08)
        ax.set_xlabel("Monitoring window")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        save_figure(fig, name)

    metric_plot(
        "heat_loss_ewma_score",
        "fig_heat_loss_anomaly_detection_improved",
        "EWMA heat-loss score",
        ["normal_operation", "heat_loss_coefficient_plus_20pct", "combined_monitoring_stress"],
    )
    metric_plot(
        "sensor_health_score",
        "fig_sensor_dropout_detection_improved",
        "Sensor health/dropout score",
        ["normal_operation", "outlet_sensor_dropout", "combined_monitoring_stress"],
    )

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    summary = sweep[(sweep["threshold_quantile"].eq(95)) & (sweep["score"].eq("residual_score")) & (~sweep["case"].eq("normal_operation"))].copy()
    if summary.empty:
        ax.text(0.5, 0.5, "No anomaly summary data", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(summary))
        ax.bar(x - 0.18, summary["detection_rate_percent"], width=0.36, label="detection rate")
        ax.bar(x + 0.18, summary["false_alarm_rate_percent"], width=0.36, label="false alarm rate")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["case"].str.replace("_", "\n", regex=False), fontsize=7)
        ax.set_ylabel("%")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    ax.set_title("Improved anomaly detection summary at 95th-percentile threshold")
    save_figure(fig, "fig_anomaly_detection_summary_improved")


if __name__ == "__main__":
    run_anomaly_detection()
