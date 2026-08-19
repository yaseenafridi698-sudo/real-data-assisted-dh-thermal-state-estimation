from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import denormalize_state
from .config import PROJECT_ROOT
from .train import _to_device
from .utils import ensure_dir, get_device


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def evaluate_model(
    model: torch.nn.Module,
    loader,
    config: dict[str, Any],
    stats: dict[str, Any],
    model_name: str,
    return_predictions: bool = False,
) -> tuple[dict[str, float | str], dict[str, np.ndarray]] | dict[str, float | str]:
    device = get_device()
    model = model.to(device)
    model.eval()
    preds = []
    targets = []
    masks = []
    sensors = []
    heat_loads = []
    source_temps = []
    ambient_values = []
    time_values = []
    elapsed = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            start = time.perf_counter()
            pred_norm = model(batch["x"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000.0)
            pred = denormalize_state(pred_norm, stats, device=device)
            preds.append(pred.cpu().numpy())
            targets.append(batch["target_physical"].cpu().numpy())
            masks.append(batch["sensor_masks"].cpu().numpy())
            sensors.append(batch["sensor_values"].cpu().numpy())
            heat_loads.append(batch["heat_load_kw"].cpu().numpy())
            source_temps.append(batch["source_temp"].cpu().numpy())
            # Keep context fields in the returned payload for downstream
            # thermo-hydraulic diagnostics. These are boundary/context signals,
            # not additional hidden-state labels.
            ambient_values.append(batch["ambient"].cpu().numpy())
            time_values.append(batch["time_s"].cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(targets, axis=0)
    mask_arr = np.concatenate(masks, axis=0)
    sensor_arr = np.concatenate(sensors, axis=0)
    heat_load_kw = np.concatenate(heat_loads, axis=0)
    source_temp = np.concatenate(source_temps, axis=0)
    ambient_arr = np.concatenate(ambient_values, axis=0) if "ambient_values" in locals() else np.full_like(heat_load_kw, np.nan)
    time_arr = np.concatenate(time_values, axis=0) if "time_values" in locals() else np.full_like(heat_load_kw, np.nan)

    measured = mask_arr > 0.5
    sys = config["system"]
    U = float(sys["heat_loss_U_W_m2K"])
    P = float(sys["pipe_perimeter_m"])
    dx = float(sys["dx_m"])
    true_loss = np.nansum(_segment_heat_loss_kw(true_arr, ambient_arr, config), axis=2)
    pred_loss = np.nansum(_segment_heat_loss_kw(pred_arr, ambient_arr, config), axis=2)
    heat_loss_error = float(np.nanmean(np.abs(pred_loss - true_loss) / np.maximum(np.abs(true_loss), 1e-6)) * 100.0)
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    delivered_kw = rho * cp * np.clip(pred_arr[:, :, -1, 3], 1e-6, None) * np.maximum(pred_arr[:, :, -1, 0] - pred_arr[:, :, -1, 1], 0.0) / 1000.0
    heat_load_consistency = float(np.nanmean(np.abs(delivered_kw - heat_load_kw) / np.maximum(np.abs(heat_load_kw), 1.0)) * 100.0)
    boundary_residual = float(np.nanmean(np.abs(pred_arr[:, :, 0, 0] - source_temp)))
    measured_ts = _rmse(pred_arr[..., 0][measured[..., 0]], sensor_arr[..., 0][measured[..., 0]]) if measured[..., 0].any() else np.nan
    measured_tr = _rmse(pred_arr[..., 1][measured[..., 1]], sensor_arr[..., 1][measured[..., 1]]) if measured[..., 1].any() else np.nan

    metrics: dict[str, float | str] = {
        "model": model_name,
        "RMSE_Ts_full": _rmse(pred_arr[..., 0], true_arr[..., 0]),
        "RMSE_Tr_full": _rmse(pred_arr[..., 1], true_arr[..., 1]),
        "RMSE_H_full": _rmse(pred_arr[..., 2], true_arr[..., 2]),
        "RMSE_q_full": _rmse(pred_arr[..., 3], true_arr[..., 3]),
        "RMSE_Ts_measured_nodes": measured_ts,
        "RMSE_Tr_measured_nodes": measured_tr,
        "RMSE_supply_measured_C": measured_ts,
        "RMSE_return_measured_C": measured_tr,
        "RMSE_load_or_return_proxy": measured_tr,
        "heat_load_consistency_error_percent": heat_load_consistency,
        "heat_loss_error_percent": heat_loss_error,
        "energy_balance_residual": float(np.nanmean(np.abs(_dynamic_energy_residual_kw(pred_arr, ambient_arr, config)) / np.maximum(np.abs(heat_load_kw), 1.0)) * 100.0),
        "thermal_residual_mean": float(np.nanmean(np.abs(np.diff(pred_arr[..., 0], axis=1)))) if pred_arr.shape[1] > 1 else np.nan,
        "hydraulic_residual_mean": float(np.nanmean(np.abs(pred_arr[:, :, :, 2] - true_arr[:, :, :, 2]))),
        "boundary_residual_mean": boundary_residual,
        "thermal_delay_error": float(np.nanmean(np.abs(pred_arr[:, :, -1, 0] - true_arr[:, :, -1, 0]))),
        "inference_time_ms": float(np.mean(elapsed)),
    }
    payload = {
        "pred": pred_arr,
        "true": true_arr,
        "mask": mask_arr,
        "sensor": sensor_arr,
        "heat_load_kw": heat_load_kw,
        "source_temp": source_temp,
        "ambient": ambient_arr,
        "time_s": time_arr,
    }
    if return_predictions:
        return metrics, payload
    return metrics


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(np.abs(a - b)))


def maxae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmax(np.abs(a - b)))


def _safe_percent(numer: np.ndarray | float, denom: np.ndarray | float) -> float:
    return float(np.nanmean(np.abs(numer) / np.maximum(np.abs(denom), 1e-9)) * 100.0)


def _thermal_delay_minutes(signal: np.ndarray, response: np.ndarray, dt_s: float, max_lag: int = 24) -> float:
    sig = np.asarray(signal, dtype=float).reshape(-1)
    resp = np.asarray(response, dtype=float).reshape(-1)
    n = min(sig.size, resp.size)
    if n < 4:
        return float("nan")
    sig = sig[:n] - np.nanmean(sig[:n])
    resp = resp[:n] - np.nanmean(resp[:n])
    best_lag = 0
    best_corr = -np.inf
    max_lag = int(min(max_lag, n - 2))
    for lag in range(max_lag + 1):
        a = sig[: n - lag]
        b = resp[lag:n]
        if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_lag * dt_s / 60.0)


def _segment_heat_loss_kw(state: np.ndarray, ambient: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Return segment heat loss [B, W, N-1] in kW from supply/return temperatures.

    This is a simulator-consistent diagnostic. It is not a directly measured
    pipe-segment heat-loss field in the public operating datasets.
    """
    sys = config["system"]
    U = float(sys["heat_loss_U_W_m2K"])
    P = float(sys["pipe_perimeter_m"])
    dx = float(sys["dx_m"])
    Ta = np.asarray(ambient, dtype=float)
    if Ta.ndim == 2:
        Ta = Ta[:, :, None]
    supply_seg = 0.5 * (state[..., :-1, 0] + state[..., 1:, 0])
    return_seg = 0.5 * (state[..., :-1, 1] + state[..., 1:, 1])
    return U * P * ((supply_seg - Ta) + (return_seg - Ta)) * dx / 1000.0


def _delivered_heat_kw(state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sys = config["system"]
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    q = np.clip(state[..., -1, 3], 1e-9, None)
    delta_t = np.maximum(state[..., -1, 0] - state[..., -1, 1], 0.0)
    return rho * cp * q * delta_t / 1000.0


def _dynamic_energy_residual_kw(state: np.ndarray, ambient: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Reduced dynamic pipe energy residual [B, W] in kW.

    The diagnostic includes source heat input, consumer extraction, ambient pipe
    loss, and the finite-difference change in pipe thermal storage. It remains a
    calibrated-model consistency quantity, not an independently measured target.
    """
    sys = config["system"]
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    diameter = float(sys["diameter_m"])
    area = np.pi * diameter**2 / 4.0
    dx = float(sys["dx_m"])
    dt = float(sys["dt_s"])
    q_source = np.clip(state[..., 0, 3], 1e-9, None)
    q_outlet = np.clip(state[..., -1, 3], 1e-9, None)
    source_heat = rho * cp * q_source * (state[..., 0, 0] - state[..., 0, 1]) / 1000.0
    delivered = rho * cp * q_outlet * np.maximum(state[..., -1, 0] - state[..., -1, 1], 0.0) / 1000.0
    loss = np.nansum(_segment_heat_loss_kw(state, ambient, config), axis=2)
    total_temperature = state[..., 0] + state[..., 1]
    trapz_temperature = 0.5 * total_temperature[..., 0] + np.nansum(total_temperature[..., 1:-1], axis=2) + 0.5 * total_temperature[..., -1]
    pipe_energy_kj = rho * cp * area * dx * trapz_temperature / 1000.0
    storage = np.zeros_like(pipe_energy_kj)
    if pipe_energy_kj.shape[1] > 1:
        storage[:, 1:] = np.diff(pipe_energy_kj, axis=1) / dt
        storage[:, 0] = storage[:, 1]
    return source_heat - delivered - loss - storage


def compute_thermo_hydraulic_metric_rows(
    payload: dict[str, np.ndarray],
    config: dict[str, Any],
    model_name: str,
    sensor_nodes: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    pred = np.asarray(payload["pred"], dtype=float)
    true = np.asarray(payload["true"], dtype=float)
    mask = np.asarray(payload.get("mask", np.zeros_like(true)), dtype=float)
    heat_load_kw = np.asarray(payload.get("heat_load_kw", np.full(pred.shape[:2], np.nan)), dtype=float)
    ambient = np.asarray(payload.get("ambient", np.full(pred.shape[:2], np.nan)), dtype=float)
    dt_s = float(config["system"]["dt_s"])
    dx_km = float(config["system"]["dx_m"]) / 1000.0
    rho = float(config["system"]["rho"])
    g = float(config["system"]["g"])
    cp = float(config["system"]["cp"])
    n_nodes = pred.shape[2]
    mid = n_nodes // 2
    sensor_nodes = sensor_nodes or []

    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, unit: str, state_type: str, category: str, interpretation: str) -> None:
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "value": float(value) if np.isscalar(value) and np.isfinite(value) else value,
                "unit": unit,
                "state_type": state_type,
                "category": category,
                "interpretation": interpretation,
            }
        )

    # Thermal state metrics.
    ts_err = pred[..., 0] - true[..., 0]
    tr_err = pred[..., 1] - true[..., 1]
    add("RMSE_Ts_supply_C", _rmse(pred[..., 0], true[..., 0]), "C", "simulator_assisted_hidden_state", "thermal_state", "distributed supply-temperature labels from calibrated simulator")
    add("MAE_Ts_supply_C", mae(pred[..., 0], true[..., 0]), "C", "simulator_assisted_hidden_state", "thermal_state", "distributed supply-temperature labels from calibrated simulator")
    add("MaxAE_Ts_supply_C", maxae(pred[..., 0], true[..., 0]), "C", "simulator_assisted_hidden_state", "thermal_state", "maximum distributed supply-temperature error")
    add("RMSE_Tr_return_C", _rmse(pred[..., 1], true[..., 1]), "C", "simulator_assisted_hidden_state", "thermal_state", "distributed return-temperature labels from calibrated simulator")
    add("MAE_Tr_return_C", mae(pred[..., 1], true[..., 1]), "C", "simulator_assisted_hidden_state", "thermal_state", "distributed return-temperature labels from calibrated simulator")
    add("MaxAE_Tr_return_C", maxae(pred[..., 1], true[..., 1]), "C", "simulator_assisted_hidden_state", "thermal_state", "maximum distributed return-temperature error")
    add("outlet_supply_temp_error_C", mae(pred[..., -1, 0], true[..., -1, 0]), "C", "simulator_assisted_hidden_state", "thermal_state", "load-end supply-temperature error")
    add("outlet_return_temp_error_C", mae(pred[..., -1, 1], true[..., -1, 1]), "C", "simulator_assisted_hidden_state", "thermal_state", "load-end return-temperature error")
    true_delay = _thermal_delay_minutes(true[..., 0, 0], true[..., -1, 0], dt_s)
    pred_delay = _thermal_delay_minutes(pred[..., 0, 0], pred[..., -1, 0], dt_s)
    add("thermal_delay_error_min", abs(pred_delay - true_delay), "min", "simulator_assisted_hidden_state", "thermal_state", "thermal-delay estimate from inlet/outlet supply response")
    grad_true = np.gradient(true[..., 0], dx_km, axis=2)
    grad_pred = np.gradient(pred[..., 0], dx_km, axis=2)
    add("temperature_gradient_error_C_per_km", mae(grad_pred, grad_true), "C/km", "simulator_assisted_hidden_state", "thermal_state", "temperature-gradient error relevant to heat-loss inference")

    # Measured-node and unmeasured-node thermal metrics.
    measured_temp_mask = (mask[..., 0] > 0.5) | (mask[..., 1] > 0.5)
    unmeasured_temp_mask = ~measured_temp_mask
    temp_err_combined = np.sqrt(0.5 * (ts_err**2 + tr_err**2))
    if np.any(measured_temp_mask):
        add("measured_node_temperature_RMSE_C", float(np.sqrt(np.nanmean(temp_err_combined[measured_temp_mask] ** 2))), "C", "calibrated_simulator", "sparse_sensor", "temperature consistency at corridor-model sensor nodes; not an independent real measured-node validation")
    else:
        add("measured_node_temperature_RMSE_C", np.nan, "C", "calibrated_simulator", "sparse_sensor", "no corridor-model sensor nodes in this layout")
    if np.any(unmeasured_temp_mask):
        add("unmeasured_node_temperature_RMSE_C", float(np.sqrt(np.nanmean(temp_err_combined[unmeasured_temp_mask] ** 2))), "C", "simulator_assisted_hidden_state", "sparse_sensor", "temperature error away from sensors")
    if mask[..., 2].any():
        add("measured_node_head_RMSE_m", _rmse(pred[..., 2][mask[..., 2] > 0.5], true[..., 2][mask[..., 2] > 0.5]), "m", "simulator_assisted_hidden_state", "sparse_sensor", "head consistency at simulated hydraulic sensor nodes")
    else:
        add("measured_node_head_RMSE_m", np.nan, "m", "simulator_assisted_hidden_state", "sparse_sensor", "real head sensors unavailable in public datasets")
    add("unmeasured_node_head_RMSE_m", _rmse(pred[..., 2][mask[..., 2] <= 0.5], true[..., 2][mask[..., 2] <= 0.5]), "m", "simulator_assisted_hidden_state", "sparse_sensor", "head error at unmeasured/simulator-hidden nodes")
    add("middle_pipe_error_C", mae(pred[..., mid, 0], true[..., mid, 0]), "C", "simulator_assisted_hidden_state", "sparse_sensor", "mid-pipe supply-temperature error")
    add("outlet_error_C", mae(pred[..., -1, 0], true[..., -1, 0]), "C", "simulator_assisted_hidden_state", "sparse_sensor", "outlet supply-temperature error")
    if sensor_nodes:
        far_nodes = [i for i in range(n_nodes) if i not in set(sensor_nodes)]
        if far_nodes:
            dist = np.array([min(abs(i - s) for s in sensor_nodes) for i in far_nodes])
            farthest_node = far_nodes[int(np.argmax(dist))]
            add("farthest_unobserved_segment_error_C", mae(pred[..., farthest_node, 0], true[..., farthest_node, 0]), "C", "simulator_assisted_hidden_state", "sparse_sensor", "supply-temperature error at farthest node from any sensor")

    # Hydraulic state metrics.
    pressure_pred = rho * g * pred[..., 2] / 1000.0
    pressure_true = rho * g * true[..., 2] / 1000.0
    flow_pred_kg_s = rho * pred[..., 3]
    flow_true_kg_s = rho * true[..., 3]
    add("RMSE_head_m", _rmse(pred[..., 2], true[..., 2]), "m", "simulator_assisted_hidden_state", "hydraulic_state", "head field is simulator-assisted hidden hydraulic state")
    add("MAE_head_m", mae(pred[..., 2], true[..., 2]), "m", "simulator_assisted_hidden_state", "hydraulic_state", "mean absolute head error")
    add("MaxAE_head_m", maxae(pred[..., 2], true[..., 2]), "m", "simulator_assisted_hidden_state", "hydraulic_state", "maximum head error")
    add("RMSE_pressure_kPa", _rmse(pressure_pred, pressure_true), "kPa", "simulator_assisted_hidden_state", "hydraulic_state", "pressure converted from simulator-assisted head")
    add("MAE_pressure_kPa", mae(pressure_pred, pressure_true), "kPa", "simulator_assisted_hidden_state", "hydraulic_state", "mean absolute pressure error")
    true_drop = true[..., 0, 2] - true[..., -1, 2]
    pred_drop = pred[..., 0, 2] - pred[..., -1, 2]
    add("pressure_drop_error_percent", _safe_percent(pred_drop - true_drop, true_drop), "%", "simulator_assisted_hidden_state", "hydraulic_state", "head/pressure-drop error relative to simulator")
    add("RMSE_flow_kg_s", _rmse(flow_pred_kg_s, flow_true_kg_s), "kg/s", "simulator_assisted_hidden_state", "hydraulic_state", "flow labels from calibrated simulator/heat-load proxy")
    add("MAE_flow_kg_s", mae(flow_pred_kg_s, flow_true_kg_s), "kg/s", "simulator_assisted_hidden_state", "hydraulic_state", "mean absolute mass-flow error")
    pred_flow_balance = np.nanmax(flow_pred_kg_s, axis=2) - np.nanmin(flow_pred_kg_s, axis=2)
    true_flow_mean = np.nanmean(np.abs(flow_true_kg_s), axis=2)
    add("flow_balance_error_percent", _safe_percent(pred_flow_balance, true_flow_mean), "%", "simulator_assisted_hidden_state", "hydraulic_state", "spread in predicted pipe flow relative to simulator flow magnitude")
    add("pump_head_boundary_error_m", mae(pred_drop, true_drop), "m", "simulator_assisted_hidden_state", "hydraulic_state", "pump/head-drop boundary consistency relative to simulator")

    # Heat and energy metrics.
    pred_delivered = _delivered_heat_kw(pred, config)
    true_delivered = _delivered_heat_kw(true, config)
    pred_seg_loss = _segment_heat_loss_kw(pred, ambient, config)
    true_seg_loss = _segment_heat_loss_kw(true, ambient, config)
    pred_total_loss = np.nansum(pred_seg_loss, axis=2)
    true_total_loss = np.nansum(true_seg_loss, axis=2)
    add("delivered_heat_error_percent", _safe_percent(pred_delivered - true_delivered, true_delivered), "%", "calibrated_simulator", "heat_energy", "delivered-heat error relative to simulator")
    add("heat_loss_error_percent", _safe_percent(pred_total_loss - true_total_loss, true_total_loss), "%", "calibrated_simulator", "heat_energy", "total pipe heat-loss error")
    add("segment_heat_loss_RMSE_kW", _rmse(pred_seg_loss, true_seg_loss), "kW", "calibrated_simulator", "heat_energy", "segment-wise heat-loss reconstruction error")
    add("cumulative_heat_loss_error_percent", _safe_percent(np.nansum(pred_total_loss, axis=1) - np.nansum(true_total_loss, axis=1), np.nansum(true_total_loss, axis=1)), "%", "calibrated_simulator", "heat_energy", "cumulative heat-loss error over windows")
    energy_residual = _dynamic_energy_residual_kw(pred, ambient, config)
    add("energy_balance_residual_percent", _safe_percent(energy_residual, heat_load_kw), "%", "calibrated_simulator", "heat_energy", "dynamic source-delivery-loss-storage closure residual")
    ret_energy = rho * cp * np.clip(pred[..., -1, 3], 1e-9, None) * np.abs(pred[..., -1, 1] - true[..., -1, 1]) / 1000.0
    add("return_temperature_energy_error_percent", _safe_percent(ret_energy, np.maximum(heat_load_kw, 1.0)), "%", "calibrated_simulator", "heat_energy", "heat-equivalent return-temperature error")
    add("heat_delivery_ratio_error_percent", _safe_percent(pred_delivered / np.maximum(heat_load_kw, 1.0) - true_delivered / np.maximum(heat_load_kw, 1.0), true_delivered / np.maximum(heat_load_kw, 1.0)), "%", "calibrated_simulator", "heat_energy", "delivered-heat ratio error")

    derived = {
        "pressure_pred_kPa": pressure_pred,
        "pressure_true_kPa": pressure_true,
        "flow_pred_kg_s": flow_pred_kg_s,
        "flow_true_kg_s": flow_true_kg_s,
        "pred_delivered_heat_kw": pred_delivered,
        "true_delivered_heat_kw": true_delivered,
        "pred_segment_heat_loss_kw": pred_seg_loss,
        "true_segment_heat_loss_kw": true_seg_loss,
        "pred_total_heat_loss_kw": pred_total_loss,
        "true_total_heat_loss_kw": true_total_loss,
        "energy_residual_kw": energy_residual,
        "heat_load_kw": heat_load_kw,
        "ambient": ambient,
    }
    return rows, derived


def save_thermo_hydraulic_estimation_outputs(
    prediction_payloads: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    sensor_nodes: list[int] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    output_path = ensure_dir(output_dir or PROJECT_ROOT / "results")
    all_rows: list[dict[str, Any]] = []
    derived_by_model: dict[str, dict[str, np.ndarray]] = {}
    for model_name, payload in prediction_payloads.items():
        rows, derived = compute_thermo_hydraulic_metric_rows(payload, config, model_name, sensor_nodes=sensor_nodes)
        all_rows.extend(rows)
        derived_by_model[model_name] = derived
    df = pd.DataFrame(all_rows)
    df = _add_thermo_hydraulic_claim_columns(df)
    df.to_csv(output_path / "thermo_hydraulic_estimation_metrics.csv", index=False)
    for category, filename in [
        ("thermal_state", "thermal_state_metrics.csv"),
        ("hydraulic_state", "hydraulic_state_metrics.csv"),
        ("heat_energy", "heat_energy_metrics.csv"),
        ("sparse_sensor", "measured_vs_hidden_state_metrics.csv"),
    ]:
        df[df["category"].eq(category)].to_csv(output_path / filename, index=False)
    return df, derived_by_model


def _add_thermo_hydraulic_claim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived claim-safety columns without changing metric values.

    All thermo-hydraulic metrics stored here are error or residual metrics, so
    lower values are better. The added columns make tables reviewer-safe by
    stating the best model, the best PI-GNN-GRU-v3 mode if present, and a safe
    claim for each metric.
    """
    if df.empty or "metric" not in df.columns or "value" not in df.columns:
        return df
    out = df.copy()
    out["best_model"] = ""
    out["best_value"] = np.nan
    out["pignn_gru_v3_value"] = np.nan
    out["pignn_gru_v3_rank"] = np.nan
    out["safe_claim"] = ""
    numeric_values = pd.to_numeric(out["value"], errors="coerce")
    for metric in out["metric"].dropna().unique():
        mask = out["metric"].eq(metric)
        subset = out.loc[mask].copy()
        subset["_numeric_value"] = numeric_values.loc[mask].to_numpy()
        subset = subset.dropna(subset=["_numeric_value"]).sort_values("_numeric_value").reset_index()
        if subset.empty:
            continue
        best = subset.iloc[0]
        v3_subset = subset[subset["model"].astype(str).str.contains("PI-GNN-GRU-v3", regex=False)]
        state_type = str(best.get("state_type", ""))
        if not v3_subset.empty:
            v3 = v3_subset.iloc[0]
            v3_value = float(v3["_numeric_value"])
            v3_rank = int(v3.name) + 1
            if v3_rank == 1:
                safe_claim = (
                    f"PI-GNN-GRU-v3 achieved the lowest {metric} among evaluated models "
                    f"for {state_type}."
                )
            else:
                safe_claim = (
                    f"{best['model']} achieved the lowest {metric}; PI-GNN-GRU-v3 ranked "
                    f"{v3_rank}, so this metric should be reported as a benchmark result, "
                    "not as a PI-GNN superiority claim."
                )
        else:
            v3_value = np.nan
            v3_rank = np.nan
            safe_claim = f"No PI-GNN-GRU-v3 value was available for {metric}; report the best model only."
        out.loc[mask, "best_model"] = str(best["model"])
        out.loc[mask, "best_value"] = float(best["_numeric_value"])
        out.loc[mask, "pignn_gru_v3_value"] = v3_value
        out.loc[mask, "pignn_gru_v3_rank"] = v3_rank
        out.loc[mask, "safe_claim"] = safe_claim
    return out


def save_evaluation_outputs(metrics_df: pd.DataFrame, output_dir: str | Path, fallback: bool = False) -> None:
    output_dir = ensure_dir(output_dir)
    metrics_df = metrics_df.copy()
    metrics_df["used_fallback_synthetic"] = bool(fallback)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics_df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    if "model" in metrics_df.columns and metrics_df["model"].astype(str).str.contains("PI-GNN-GRU-v2", regex=False).any():
        metrics_df.to_csv(output_dir / "baseline_comparison_improved.csv", index=False)
    layout_df = metrics_df[["model", "RMSE_Ts_full", "RMSE_Tr_full", "RMSE_H_full", "RMSE_q_full"]].copy()
    layout_df["sensor_layout"] = "S4_five_sensors"
    layout_df.to_csv(output_dir / "sensor_layout_comparison.csv", index=False)
    measured_cols = [
        "model",
        "RMSE_Ts_measured_nodes",
        "RMSE_Tr_measured_nodes",
        "RMSE_load_or_return_proxy",
        "used_fallback_synthetic",
    ]
    metrics_df[measured_cols].to_csv(output_dir / "real_measured_node_validation.csv", index=False)
    if fallback:
        (output_dir / "DATA_WARNING_REAL_DATA_NOT_FOUND.txt").write_text(
            "This run used fallback synthetic-realistic data for software testing. "
            "Do not report these outputs as journal real-data results.\n",
            encoding="utf-8",
        )
    (output_dir / "EXTERNAL_VALIDATION_NOT_RUN.txt").write_text(
        "Flensburg external validation was not run in this workflow. "
        "Run run_real_data_study.py with Flensburg raw files available to generate external_validation_flensburg.csv.\n",
        encoding="utf-8",
    )
    external_csv = output_dir / "external_validation_flensburg.csv"
    if fallback and external_csv.exists():
        external_csv.unlink()
