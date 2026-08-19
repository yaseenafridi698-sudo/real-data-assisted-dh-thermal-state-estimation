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


STATE_NAMES = [
    ("supply_temperature", 0, "C", "virtual_sensor_estimate"),
    ("return_temperature", 1, "C", "virtual_sensor_estimate"),
    ("head", 2, "m", "simulator_assisted_hidden_state"),
    ("flow", 3, "m3/s", "simulator_assisted_hidden_state"),
]


def _flat(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1, *arr.shape[2:])


def _pick_reference(payloads: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
    for name in ["Proposed PI-GNN-GRU-v3 balanced_mode", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Transformer-MSE", "GRU-MSE"]:
        if name in payloads:
            return name, payloads[name]
    name = next(iter(payloads))
    return name, payloads[name]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4 or np.nanstd(a[mask]) < 1e-12 or np.nanstd(b[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


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


def run_uncertainty_quantification() -> None:
    config = load_config()
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    _, sim, sensors, _, _, _, payloads = prepare_context(config)
    if not payloads:
        raise RuntimeError("No saved model payloads available for uncertainty quantification.")
    ref_name, ref_payload = _pick_reference(payloads)
    true = _flat(ref_payload["true"])
    mask = _flat(ref_payload["mask"])
    sensor = _flat(ref_payload["sensor"])
    ambient = _series_for_flat(ref_payload["ambient"], true.shape[0], true.shape[1])
    ensemble_names = list(payloads)
    ensemble = np.stack([_flat(payloads[name]["pred"]) for name in ensemble_names], axis=0)
    mean_pred = np.nanmean(ensemble, axis=0)
    ensemble_std = np.nanstd(ensemble, axis=0)
    ref_pred = _flat(ref_payload["pred"])
    residual_floor = np.nanstd(ref_pred - true, axis=(0, 1), keepdims=True)
    total_std = ensemble_std + residual_floor

    rows = []
    for quantity, idx, unit, state_type in STATE_NAMES:
        err = np.abs(mean_pred[..., idx] - true[..., idx])
        sigma = np.maximum(total_std[..., idx], 1e-9)
        measured = mask[..., idx] > 0.5
        unmeasured = ~measured
        for level, z in [(80, 1.2816), (90, 1.6449), (95, 1.96)]:
            covered = err <= z * sigma
            rows.append(
                {
                    "quantity": quantity,
                    "interval": f"{level}%",
                    "mean_interval_width": float(np.nanmean(2 * z * sigma)),
                    "unit": unit,
                    "coverage": float(np.nanmean(covered) * 100.0),
                    "measured_node_coverage": float(np.nanmean(covered[measured]) * 100.0) if np.any(measured) else np.nan,
                    "unmeasured_node_coverage": float(np.nanmean(covered[unmeasured]) * 100.0) if np.any(unmeasured) else np.nan,
                    "uncertainty_error_correlation": _corr(sigma, err),
                    "method": "saved-model ensemble plus residual-based calibration floor",
                    "state_type": "real_measured_node" if quantity in {"supply_temperature", "return_temperature"} else state_type,
                    "safe_claim": (
                        "Uncertainty bands quantify confidence in virtual sensor estimates. "
                        "Distributed pressure/head and flow uncertainty is evaluated for simulator-assisted hidden states because dense hydraulic measurements are unavailable."
                    ),
                }
            )

    heat_loss_stack = []
    for i in range(ensemble.shape[0]):
        heat_loss_stack.append(_segment_heat_loss_flat(ensemble[i], ambient, config).sum(axis=1))
    heat_loss_stack = np.stack(heat_loss_stack, axis=0)
    heat_true = _segment_heat_loss_flat(true, ambient, config).sum(axis=1)
    heat_mean = np.nanmean(heat_loss_stack, axis=0)
    heat_sigma = np.nanstd(heat_loss_stack, axis=0) + np.nanstd(heat_loss_stack[0] - heat_true)
    heat_err = np.abs(heat_mean - heat_true)
    for level, z in [(80, 1.2816), (90, 1.6449), (95, 1.96)]:
        rows.append(
            {
                "quantity": "heat_loss",
                "interval": f"{level}%",
                "mean_interval_width": float(np.nanmean(2 * z * heat_sigma)),
                "unit": "kW",
                "coverage": float(np.nanmean(heat_err <= z * heat_sigma) * 100.0),
                "measured_node_coverage": np.nan,
                "unmeasured_node_coverage": np.nan,
                "uncertainty_error_correlation": _corr(heat_sigma, heat_err),
                "method": "saved-model ensemble plus residual-based calibration floor",
                "state_type": "calibrated_simulator",
                "safe_claim": "Heat-loss uncertainty is calibrated against simulator-derived heat-loss labels.",
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(PROJECT_ROOT / "results" / "uncertainty_quantification_metrics.csv", index=False)
    improved_df, conformal_factors = _write_improved_uncertainty_metrics(
        true,
        mean_pred,
        total_std,
        mask,
        heat_true,
        heat_mean,
        heat_sigma,
    )
    _write_confidence_intervals(sim, mean_pred, total_std, heat_mean, heat_sigma)
    _write_calibrated_confidence_intervals(sim, mean_pred, total_std, heat_mean, heat_sigma, conformal_factors)
    _plot_temperature_bands(sim, true, mean_pred, total_std, ref_payload)
    _plot_heat_loss_bands(heat_true, heat_mean, heat_sigma)
    _plot_coverage(metrics_df)
    _plot_uncertainty_calibration_curve(improved_df)
    _plot_temperature_bands_calibrated(sim, true, mean_pred, total_std, ref_payload, conformal_factors)
    _plot_heat_loss_bands_calibrated(heat_true, heat_mean, heat_sigma, conformal_factors)
    _plot_width_vs_error(improved_df, true, mean_pred, total_std, heat_true, heat_mean, heat_sigma)
    copy_final_figures_to_root_and_paper()
    print("Uncertainty quantification completed.")


def _interval_row(
    quantity: str,
    level: int,
    factor: float,
    sigma: np.ndarray,
    err: np.ndarray,
    measured: np.ndarray | None,
    unit: str,
    method: str,
    state_type: str,
) -> dict[str, Any]:
    covered = err <= factor * np.maximum(sigma, 1e-9)
    measured_cov = np.nan
    unmeasured_cov = np.nan
    if measured is not None and np.any(measured):
        measured_cov = float(np.nanmean(covered[measured]) * 100.0)
    if measured is not None and np.any(~measured):
        unmeasured_cov = float(np.nanmean(covered[~measured]) * 100.0)
    width = 2 * factor * np.maximum(sigma, 1e-9)
    return {
        "quantity": quantity,
        "interval": f"{level}%",
        "method": method,
        "calibration_factor": float(factor),
        "mean_interval_width": float(np.nanmean(width)),
        "normalized_interval_width": float(np.nanmean(width) / max(np.nanmean(np.abs(err)), 1e-9)),
        "unit": unit,
        "coverage": float(np.nanmean(covered) * 100.0),
        "measured_node_coverage": measured_cov,
        "unmeasured_node_coverage": unmeasured_cov,
        "width_error_correlation": _corr(width, err),
        "state_type": state_type,
        "safe_claim": "Intervals are operational confidence bands; conformal calibration reports empirical coverage and sharpness without claiming perfect probabilistic calibration.",
    }


def _write_improved_uncertainty_metrics(
    true: np.ndarray,
    mean_pred: np.ndarray,
    total_std: np.ndarray,
    mask: np.ndarray,
    heat_true: np.ndarray,
    heat_mean: np.ndarray,
    heat_sigma: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[int, float]]]:
    z_by_level = {80: 1.2816, 90: 1.6449, 95: 1.96}
    rows: list[dict[str, Any]] = []
    factors: dict[str, dict[int, float]] = {}
    for quantity, idx, unit, state_type in STATE_NAMES:
        err = np.abs(mean_pred[..., idx] - true[..., idx])
        sigma = np.maximum(total_std[..., idx], 1e-9)
        measured = mask[..., idx] > 0.5
        ratios = (err / sigma).reshape(-1)
        ratios = ratios[np.isfinite(ratios)]
        factors[quantity] = {}
        for level, z in z_by_level.items():
            rows.append(_interval_row(quantity, level, z, sigma, err, measured, unit, "raw_ensemble_residual", state_type))
            q = float(np.nanquantile(ratios, level / 100.0)) if ratios.size else z
            q = max(q, 1e-9)
            factors[quantity][level] = q
            rows.append(_interval_row(quantity, level, q, sigma, err, measured, unit, "conformal_calibrated", state_type))
    heat_err = np.abs(np.asarray(heat_mean).reshape(-1) - np.asarray(heat_true).reshape(-1))
    heat_sig = np.maximum(np.asarray(heat_sigma).reshape(-1), 1e-9)
    heat_ratios = heat_err / heat_sig
    factors["heat_loss"] = {}
    for level, z in z_by_level.items():
        rows.append(_interval_row("heat_loss", level, z, heat_sig, heat_err, None, "kW", "raw_ensemble_residual", "calibrated_simulator"))
        q = float(np.nanquantile(heat_ratios[np.isfinite(heat_ratios)], level / 100.0)) if np.isfinite(heat_ratios).any() else z
        q = max(q, 1e-9)
        factors["heat_loss"][level] = q
        rows.append(_interval_row("heat_loss", level, q, heat_sig, heat_err, None, "kW", "conformal_calibrated", "calibrated_simulator"))
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "uncertainty_quantification_metrics_improved.csv", index=False)
    summary = (
        df.pivot_table(index=["quantity", "interval"], columns="method", values=["coverage", "mean_interval_width"], aggfunc="mean")
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary.columns.to_flat_index()]
    summary["interpretation"] = "Conformal scaling reports empirical coverage and interval sharpness for virtual-sensor confidence bands."
    summary.to_csv(PROJECT_ROOT / "results" / "uncertainty_calibration_summary.csv", index=False)
    return df, factors


def _write_calibrated_confidence_intervals(
    sim: dict[str, Any],
    mean_pred: np.ndarray,
    total_std: np.ndarray,
    heat_mean: np.ndarray,
    heat_sigma: np.ndarray,
    factors: dict[str, dict[int, float]],
) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    nodes = sorted(set(np.linspace(0, len(x_km) - 1, min(7, len(x_km))).round().astype(int).tolist()))
    time_indices = np.linspace(0, mean_pred.shape[0] - 1, min(160, mean_pred.shape[0])).round().astype(int)
    rows = []
    for t in time_indices:
        for node in nodes:
            for quantity, idx, unit, state_type in STATE_NAMES:
                mu = float(mean_pred[t, node, idx])
                sigma = float(total_std[t, node, idx])
                q80 = factors.get(quantity, {}).get(80, 1.2816)
                q90 = factors.get(quantity, {}).get(90, 1.6449)
                q95 = factors.get(quantity, {}).get(95, 1.96)
                rows.append(
                    {
                        "window_index": int(t),
                        "node": int(node),
                        "distance_km": float(x_km[node]),
                        "quantity": quantity,
                        "mean": mu,
                        "lower_80": mu - q80 * sigma,
                        "upper_80": mu + q80 * sigma,
                        "lower_90": mu - q90 * sigma,
                        "upper_90": mu + q90 * sigma,
                        "lower_95": mu - q95 * sigma,
                        "upper_95": mu + q95 * sigma,
                        "unit": unit,
                        "state_type": state_type,
                    }
                )
        q80 = factors.get("heat_loss", {}).get(80, 1.2816)
        q90 = factors.get("heat_loss", {}).get(90, 1.6449)
        q95 = factors.get("heat_loss", {}).get(95, 1.96)
        rows.append(
            {
                "window_index": int(t),
                "node": -1,
                "distance_km": np.nan,
                "quantity": "total_heat_loss",
                "mean": float(heat_mean[t]),
                "lower_80": float(heat_mean[t] - q80 * heat_sigma[t]),
                "upper_80": float(heat_mean[t] + q80 * heat_sigma[t]),
                "lower_90": float(heat_mean[t] - q90 * heat_sigma[t]),
                "upper_90": float(heat_mean[t] + q90 * heat_sigma[t]),
                "lower_95": float(heat_mean[t] - q95 * heat_sigma[t]),
                "upper_95": float(heat_mean[t] + q95 * heat_sigma[t]),
                "unit": "kW",
                "state_type": "calibrated_simulator",
            }
        )
    pd.DataFrame(rows).to_csv(PROJECT_ROOT / "results" / "virtual_sensor_confidence_intervals_calibrated.csv", index=False)


def _write_confidence_intervals(
    sim: dict[str, Any],
    mean_pred: np.ndarray,
    total_std: np.ndarray,
    heat_mean: np.ndarray,
    heat_sigma: np.ndarray,
) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    nodes = sorted(set(np.linspace(0, len(x_km) - 1, min(7, len(x_km))).round().astype(int).tolist()))
    time_indices = np.linspace(0, mean_pred.shape[0] - 1, min(160, mean_pred.shape[0])).round().astype(int)
    rows = []
    for t in time_indices:
        for node in nodes:
            for quantity, idx, unit, state_type in STATE_NAMES:
                mu = float(mean_pred[t, node, idx])
                sigma = float(total_std[t, node, idx])
                rows.append(
                    {
                        "window_index": int(t),
                        "node": int(node),
                        "distance_km": float(x_km[node]),
                        "quantity": quantity,
                        "mean": mu,
                        "lower_80": mu - 1.2816 * sigma,
                        "upper_80": mu + 1.2816 * sigma,
                        "lower_90": mu - 1.6449 * sigma,
                        "upper_90": mu + 1.6449 * sigma,
                        "unit": unit,
                        "state_type": state_type,
                    }
                )
        rows.append(
            {
                "window_index": int(t),
                "node": -1,
                "distance_km": np.nan,
                "quantity": "total_heat_loss",
                "mean": float(np.nanmean(heat_mean[t])),
                "lower_80": float(np.nanmean(heat_mean[t] - 1.2816 * heat_sigma[t])),
                "upper_80": float(np.nanmean(heat_mean[t] + 1.2816 * heat_sigma[t])),
                "lower_90": float(np.nanmean(heat_mean[t] - 1.6449 * heat_sigma[t])),
                "upper_90": float(np.nanmean(heat_mean[t] + 1.6449 * heat_sigma[t])),
                "unit": "kW",
                "state_type": "calibrated_simulator",
            }
        )
    pd.DataFrame(rows).to_csv(PROJECT_ROOT / "results" / "virtual_sensor_confidence_intervals.csv", index=False)


def _plot_temperature_bands(sim: dict[str, Any], true: np.ndarray, mean_pred: np.ndarray, total_std: np.ndarray, payload: dict[str, np.ndarray]) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    heat = np.asarray(payload["heat_load_kw"]).reshape(-1)
    idx = int(np.nanargmax(heat)) if heat.size else 0
    idx = min(idx, mean_pred.shape[0] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, state_idx, title in [(axes[0], 0, "Supply temperature"), (axes[1], 1, "Return temperature")]:
        mu = mean_pred[idx, :, state_idx]
        sigma = total_std[idx, :, state_idx]
        ax.plot(x_km, true[idx, :, state_idx], color="black", lw=1.8, label="calibrated simulator")
        ax.plot(x_km, mu, color="#7b2cbf", lw=1.5, label="virtual sensor mean")
        ax.fill_between(x_km, mu - 1.6449 * sigma, mu + 1.6449 * sigma, color="#7b2cbf", alpha=0.18, label="90% interval")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("$^\\circ$C")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Uncertainty bands quantify confidence in virtual temperature sensor estimates.")
    save_figure(fig, "fig_uncertainty_temperature_bands")


def _plot_heat_loss_bands(heat_true: np.ndarray, heat_mean: np.ndarray, heat_sigma: np.ndarray) -> None:
    heat_true = np.asarray(heat_true, dtype=float).reshape(-1)
    heat_mean = np.asarray(heat_mean, dtype=float).reshape(-1)
    heat_sigma = np.asarray(heat_sigma, dtype=float).reshape(-1)
    x = np.arange(heat_true.size)
    mu = heat_mean
    sigma = heat_sigma
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(x, heat_true, color="black", lw=1.7, label="calibrated simulator")
    ax.plot(x, mu, color="#43aa8b", lw=1.5, label="virtual estimate")
    ax.fill_between(x, mu - 1.6449 * sigma, mu + 1.6449 * sigma, color="#43aa8b", alpha=0.2, label="90% interval")
    ax.set_xlabel("Monitoring window step")
    ax.set_ylabel("Total heat loss (kW)")
    ax.set_title("Uncertainty in heat-loss virtual estimate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    save_figure(fig, "fig_uncertainty_heat_loss_bands")


def _plot_coverage(metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    sub = metrics_df[metrics_df["interval"].eq("90%")].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No uncertainty metrics", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(sub["quantity"], sub["coverage"], color="#577590")
        ax.axhline(90, color="#d1495b", ls="--", lw=1.2, label="nominal 90%")
        ax.set_ylabel("Coverage (%)")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    ax.set_title("Prediction-interval coverage for virtual sensors")
    save_figure(fig, "fig_uncertainty_coverage")


def _plot_uncertainty_calibration_curve(improved_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    if improved_df.empty:
        ax.text(0.5, 0.5, "No uncertainty calibration data", ha="center", va="center")
        ax.axis("off")
    else:
        for method, sub in improved_df.groupby("method"):
            nominal = sub["interval"].str.rstrip("%").astype(float)
            coverage = pd.to_numeric(sub["coverage"], errors="coerce")
            grouped = pd.DataFrame({"nominal": nominal, "coverage": coverage}).groupby("nominal", as_index=False).mean()
            ax.plot(grouped["nominal"], grouped["coverage"], marker="o", lw=1.4, label=method.replace("_", " "))
        ax.plot([75, 100], [75, 100], color="black", ls="--", lw=1.0, label="ideal")
        ax.set_xlim(78, 97)
        ax.set_ylim(75, 101)
        ax.set_xlabel("Nominal interval (%)")
        ax.set_ylabel("Empirical coverage (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    ax.set_title("Uncertainty calibration curve")
    save_figure(fig, "fig_uncertainty_calibration_curve")


def _plot_temperature_bands_calibrated(
    sim: dict[str, Any],
    true: np.ndarray,
    mean_pred: np.ndarray,
    total_std: np.ndarray,
    payload: dict[str, np.ndarray],
    factors: dict[str, dict[int, float]],
) -> None:
    x_km = np.asarray(sim["x_m"], dtype=float) / 1000.0
    heat = np.asarray(payload["heat_load_kw"]).reshape(-1)
    idx = int(np.nanargmax(heat)) if heat.size else 0
    idx = min(idx, mean_pred.shape[0] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, state_idx, quantity, title in [(axes[0], 0, "supply_temperature", "Supply temperature"), (axes[1], 1, "return_temperature", "Return temperature")]:
        mu = mean_pred[idx, :, state_idx]
        sigma = total_std[idx, :, state_idx]
        q = factors.get(quantity, {}).get(90, 1.6449)
        ax.plot(x_km, true[idx, :, state_idx], color="black", lw=1.8, label="calibrated simulator")
        ax.plot(x_km, mu, color="#7b2cbf", lw=1.5, label="virtual sensor mean")
        ax.fill_between(x_km, mu - q * sigma, mu + q * sigma, color="#7b2cbf", alpha=0.18, label="calibrated 90% interval")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("$^\\circ$C")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Conformal-calibrated confidence bands for virtual temperature sensors.")
    save_figure(fig, "fig_uncertainty_temperature_bands_calibrated")


def _plot_heat_loss_bands_calibrated(
    heat_true: np.ndarray,
    heat_mean: np.ndarray,
    heat_sigma: np.ndarray,
    factors: dict[str, dict[int, float]],
) -> None:
    heat_true = np.asarray(heat_true, dtype=float).reshape(-1)
    heat_mean = np.asarray(heat_mean, dtype=float).reshape(-1)
    heat_sigma = np.asarray(heat_sigma, dtype=float).reshape(-1)
    q = factors.get("heat_loss", {}).get(90, 1.6449)
    x = np.arange(heat_true.size)
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(x, heat_true, color="black", lw=1.7, label="calibrated simulator")
    ax.plot(x, heat_mean, color="#43aa8b", lw=1.5, label="virtual estimate")
    ax.fill_between(x, heat_mean - q * heat_sigma, heat_mean + q * heat_sigma, color="#43aa8b", alpha=0.2, label="calibrated 90% interval")
    ax.set_xlabel("Monitoring window step")
    ax.set_ylabel("Total heat loss (kW)")
    ax.set_title("Conformal-calibrated heat-loss interval")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    save_figure(fig, "fig_uncertainty_heat_loss_bands_calibrated")


def _plot_width_vs_error(
    improved_df: pd.DataFrame,
    true: np.ndarray,
    mean_pred: np.ndarray,
    total_std: np.ndarray,
    heat_true: np.ndarray,
    heat_mean: np.ndarray,
    heat_sigma: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    err_t = np.abs(mean_pred[..., 0] - true[..., 0]).reshape(-1)
    width_t = (2 * 1.6449 * total_std[..., 0]).reshape(-1)
    axes[0].scatter(err_t[:: max(1, err_t.size // 2000)], width_t[:: max(1, width_t.size // 2000)], s=8, alpha=0.35, color="#7b2cbf")
    axes[0].set_xlabel("Supply-temperature absolute error ($^\\circ$C)")
    axes[0].set_ylabel("Raw 90% interval width ($^\\circ$C)")
    axes[0].set_title("Temperature width versus error")
    heat_err = np.abs(np.asarray(heat_mean).reshape(-1) - np.asarray(heat_true).reshape(-1))
    heat_width = 2 * 1.6449 * np.asarray(heat_sigma).reshape(-1)
    axes[1].scatter(heat_err, heat_width, s=10, alpha=0.45, color="#43aa8b")
    axes[1].set_xlabel("Heat-loss absolute error (kW)")
    axes[1].set_ylabel("Raw 90% interval width (kW)")
    axes[1].set_title("Heat-loss width versus error")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Uncertainty sharpness diagnostic: interval width versus realized error")
    save_figure(fig, "fig_uncertainty_width_vs_error")


if __name__ == "__main__":
    run_uncertainty_quantification()
