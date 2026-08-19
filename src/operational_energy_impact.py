from __future__ import annotations

import json
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

from src.ate_figure_style import PALETTE, add_panel_label, save_ate_figure, set_ate_style, style_axes
from src.combined_stress_test import _base_sim, _make_case
from src.config import PROJECT_ROOT, load_config
from src.supplementary_study_utils import load_calibrated_params, load_sonderborg_processed, simulate_from_dataframe
from src.thermo_hydraulic_coupling_analysis import prepare_context
from src.utils import ensure_dir


RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures" / "final"
TABLES = PROJECT_ROOT / "paper" / "tables"

ASSUMPTIONS: dict[str, Any] = {
    "pump_efficiency_eta": 0.75,
    "heat_price_EUR_per_MWh": 60.0,
    "electricity_price_EUR_per_kWh": 0.15,
    "electricity_emission_factor_kgCO2_per_kWh": 0.20,
    "heat_emission_factor_kgCO2_per_MWh": 180.0,
    "pump_energy_proxy_note": (
        "P_pump_proxy = rho*g*Q_vol*Delta_H/eta. Flow and head are simulator-assisted "
        "hidden hydraulic states because public district-heating datasets do not provide dense distributed "
        "hydraulic measurements."
    ),
    "horizon_note": (
        "Full-run KPIs use the full evaluation horizon; scenario tables report the stated scenario horizon; "
        "main figures/tables use per-day normalization or kWh/MWh delivered where horizons differ."
    ),
    "normalization_note": "Energy quantities are reported over the stated evaluation horizon unless explicitly normalized to MWh/day, kWh/day, or kWh/MWh delivered.",
    "cost_note": "Cost and CO2 values are proxy indicators, not optimized economic-dispatch results.",
    "co2_note": "Cost and CO2 values are proxy indicators, not optimized economic-dispatch results.",
}


def _num(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(values).reshape(-1) if hasattr(pd.Series(values), "reshape") else values, errors="coerce").to_numpy(dtype=float)


def _flat_state(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim >= 4:
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    raise ValueError(f"Expected state array with node/state dimensions, got shape {arr.shape}")


def _align(values: Any, n: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.full(n, default, dtype=float)
    if arr.size >= n:
        return arr[:n]
    return np.resize(arr, n)


def _safe_percent(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(num, dtype=float) / np.maximum(np.abs(np.asarray(den, dtype=float)), 1e-9) * 100.0


def _state_heat_loss_kw(state: np.ndarray, ambient: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sys_cfg = config["system"]
    U = float(sys_cfg["heat_loss_U_W_m2K"])
    perimeter = float(sys_cfg["pipe_perimeter_m"])
    dx_m = float(sys_cfg["dx_m"])
    ambient = _align(ambient, state.shape[0], float(sys_cfg.get("ambient_base_C", 5.0))).reshape(-1, 1)
    supply_segment = 0.5 * (state[:, :-1, 0] + state[:, 1:, 0])
    return_segment = 0.5 * (state[:, :-1, 1] + state[:, 1:, 1])
    segment_kw = U * perimeter * ((supply_segment - ambient) + (return_segment - ambient)) * dx_m / 1000.0
    return segment_kw, np.nansum(segment_kw, axis=1)


def _state_delivered_heat_kw(state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sys_cfg = config["system"]
    rho = float(sys_cfg["rho"])
    cp = float(sys_cfg["cp"])
    q_m3s = np.clip(state[:, -1, 3], 1e-8, None)
    delta_t = np.maximum(state[:, -1, 0] - state[:, -1, 1], 0.0)
    return rho * cp * q_m3s * delta_t / 1000.0


def _hydraulic_power_from_state(state: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sys_cfg = config["system"]
    rho = float(sys_cfg["rho"])
    g = float(sys_cfg["g"])
    eta = float(ASSUMPTIONS["pump_efficiency_eta"])
    flow_m3s = np.clip(state[:, 0, 3], 1e-8, None)
    head_drop_m = np.maximum(state[:, 0, 2] - state[:, -1, 2], 0.0)
    pressure_drop_kpa = rho * g * head_drop_m / 1000.0
    pressure_drop_power_kw = rho * g * flow_m3s * head_drop_m / 1000.0
    pump_power_kw = pressure_drop_power_kw / eta
    return flow_m3s, head_drop_m, pressure_drop_kpa, pump_power_kw


def _series_from_sim(sim: dict[str, Any], config: dict[str, Any], label: str) -> pd.DataFrame:
    dt_h = float(config["system"]["dt_s"]) / 3600.0
    rho = float(config["system"]["rho"])
    g = float(config["system"]["g"])
    eta = float(ASSUMPTIONS["pump_efficiency_eta"])
    q = np.asarray(sim["q"], dtype=float)
    h = np.asarray(sim["H"], dtype=float)
    ts = np.asarray(sim["Ts"], dtype=float)
    tr = np.asarray(sim["Tr"], dtype=float)
    flow = np.clip(q[:, 0], 1e-8, None)
    head_drop = np.maximum(h[:, 0] - h[:, -1], 0.0)
    pressure_drop_kpa = rho * g * head_drop / 1000.0
    pump_kw = rho * g * flow * head_drop / eta / 1000.0
    pressure_drop_power_kw = rho * g * flow * head_drop / 1000.0
    heat_load_kw = np.asarray(sim["Q_load"], dtype=float) / 1000.0
    delivered_kw = rho * float(config["system"]["cp"]) * flow * np.maximum(ts[:, -1] - tr[:, -1], 0.0) / 1000.0
    heat_loss_kw = np.asarray(sim.get("Q_loss", np.zeros_like(heat_load_kw)), dtype=float) / 1000.0
    residual_kw = delivered_kw - heat_load_kw
    return pd.DataFrame(
        {
            "step": np.arange(len(heat_load_kw)),
            "time_h": np.arange(len(heat_load_kw)) * dt_h,
            "source": label,
            "measured_boundary_heat_load_kw": heat_load_kw,
            "delivered_heat_kw": delivered_kw,
            "heat_loss_kw": heat_loss_kw,
            "flow_m3_s": flow,
            "head_drop_m": head_drop,
            "pressure_drop_kPa": pressure_drop_kpa,
            "pump_energy_proxy_kW": pump_kw,
            "pressure_drop_power_proxy_kW": pressure_drop_power_kw,
            "energy_balance_residual_kw": residual_kw,
            "energy_balance_residual_percent": _safe_percent(residual_kw, heat_load_kw),
            "state_type": "calibrated_simulator + simulator_assisted_hidden_state",
            "safe_claim": (
                "Thermal boundary variables use real operating data; pressure/head and flow-based energy "
                "indicators are simulator-assisted hidden-state proxies. Pressure/head and flow are simulator-assisted hidden "
                "hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements."
            ),
        }
    )


def _series_from_payload(payload: dict[str, Any], config: dict[str, Any], label: str) -> pd.DataFrame:
    pred = _flat_state(payload["pred"])
    n = pred.shape[0]
    dt_h = float(config["system"]["dt_s"]) / 3600.0
    heat_load_kw = _align(payload.get("heat_load_kw", np.nan), n, np.nan)
    ambient = _align(payload.get("ambient", np.nan), n, float(config["system"].get("ambient_base_C", 5.0)))
    delivered_kw = _state_delivered_heat_kw(pred, config)
    _, heat_loss_kw = _state_heat_loss_kw(pred, ambient, config)
    flow, head_drop, pressure_drop_kpa, pump_kw = _hydraulic_power_from_state(pred, config)
    pressure_drop_power_kw = pump_kw * float(ASSUMPTIONS["pump_efficiency_eta"])
    residual_kw = delivered_kw - heat_load_kw
    return pd.DataFrame(
        {
            "step": np.arange(n),
            "time_h": np.arange(n) * dt_h,
            "source": label,
            "measured_boundary_heat_load_kw": heat_load_kw,
            "delivered_heat_kw": delivered_kw,
            "heat_loss_kw": heat_loss_kw,
            "flow_m3_s": flow,
            "head_drop_m": head_drop,
            "pressure_drop_kPa": pressure_drop_kpa,
            "pump_energy_proxy_kW": pump_kw,
            "pressure_drop_power_proxy_kW": pressure_drop_power_kw,
            "energy_balance_residual_kw": residual_kw,
            "energy_balance_residual_percent": _safe_percent(residual_kw, heat_load_kw),
            "state_type": "simulator_assisted_hidden_state",
            "safe_claim": (
                "PI-GNN-GRU-v3 operational indicators are derived from reconstructed hidden states; hydraulic "
                "quantities are simulator-assisted proxies, not real distributed measurements. Pressure/head and flow are "
                "simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements."
            ),
        }
    )


def _summarize_series(df: pd.DataFrame, source: str, config: dict[str, Any]) -> dict[str, float]:
    dt_h = float(config["system"]["dt_s"]) / 3600.0
    horizon_hours = float(len(df) * dt_h)
    horizon_days = max(horizon_hours / 24.0, 1e-9)
    delivered_mwh = float(np.nansum(df["delivered_heat_kw"]) * dt_h / 1000.0)
    boundary_mwh = float(np.nansum(df["measured_boundary_heat_load_kw"]) * dt_h / 1000.0)
    heat_loss_mwh = float(np.nansum(df["heat_loss_kw"]) * dt_h / 1000.0)
    pump_kwh = float(np.nansum(df["pump_energy_proxy_kW"]) * dt_h)
    pressure_energy_kwh = float(np.nansum(df["pressure_drop_power_proxy_kW"]) * dt_h)
    residual_mwh = float(np.nansum(df["energy_balance_residual_kw"]) * dt_h / 1000.0)
    heat_cost = delivered_mwh * float(ASSUMPTIONS["heat_price_EUR_per_MWh"])
    pump_cost = pump_kwh * float(ASSUMPTIONS["electricity_price_EUR_per_kWh"])
    co2 = delivered_mwh * float(ASSUMPTIONS["heat_emission_factor_kgCO2_per_MWh"]) + pump_kwh * float(
        ASSUMPTIONS["electricity_emission_factor_kgCO2_per_kWh"]
    )

    def finite_mean(values: pd.Series) -> float:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    def finite_max(values: pd.Series) -> float:
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.max(arr)) if arr.size else float("nan")

    return {
        "source": source,
        "evaluation_horizon": f"{horizon_hours:.1f} h full evaluation horizon",
        "horizon_hours": horizon_hours,
        "delivered_heat_MWh": delivered_mwh,
        "delivered_heat_MWh_per_day": delivered_mwh / horizon_days,
        "boundary_heat_MWh": boundary_mwh,
        "boundary_heat_MWh_per_day": boundary_mwh / horizon_days,
        "mean_delivered_heat_kW": float(np.nanmean(df["delivered_heat_kw"])),
        "peak_delivered_heat_kW": float(np.nanmax(df["delivered_heat_kw"])),
        "delivered_heat_error_percent": float(abs(delivered_mwh - boundary_mwh) / max(abs(boundary_mwh), 1e-9) * 100.0),
        "heat_loss_MWh": heat_loss_mwh,
        "heat_loss_MWh_per_day": heat_loss_mwh / horizon_days,
        "heat_loss_ratio_percent": float(heat_loss_mwh / max(abs(delivered_mwh), 1e-9) * 100.0),
        "pump_energy_proxy_kWh": pump_kwh,
        "pump_energy_proxy_kWh_per_day": pump_kwh / horizon_days,
        "normalized_pump_energy_proxy_kWh_per_MWh": float(pump_kwh / max(abs(delivered_mwh), 1e-9)),
        "pressure_drop_energy_proxy_kWh": pressure_energy_kwh,
        "pressure_drop_energy_proxy_kWh_per_day": pressure_energy_kwh / horizon_days,
        "mean_pressure_drop_kPa": finite_mean(df["pressure_drop_kPa"]),
        "max_pressure_drop_kPa": finite_max(df["pressure_drop_kPa"]),
        "mean_energy_balance_residual_percent": float(np.nanmean(np.abs(df["energy_balance_residual_percent"]))),
        "max_energy_balance_residual_percent": float(np.nanmax(np.abs(df["energy_balance_residual_percent"]))),
        "cumulative_energy_balance_residual_MWh": residual_mwh,
        "cost_proxy_EUR": heat_cost + pump_cost,
        "cost_proxy_EUR_per_day": (heat_cost + pump_cost) / horizon_days,
        "CO2_proxy_kg": co2,
        "CO2_proxy_kg_per_day": co2 / horizon_days,
    }


def _metric_value(metrics: pd.DataFrame, metric: str, model_contains: str | None = None, col: str = "value") -> float:
    if metrics.empty or "metric" not in metrics.columns:
        return float("nan")
    sub = metrics[metrics["metric"].astype(str).eq(metric)].copy()
    if model_contains and "model" in sub.columns:
        match = sub[sub["model"].astype(str).str.contains(model_contains, regex=False)]
        if not match.empty:
            sub = match
    if sub.empty or col not in sub.columns:
        return float("nan")
    return float(pd.to_numeric(sub[col], errors="coerce").dropna().iloc[0])


def _summary_rows(summaries: list[dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_map = {
        "evaluation_horizon": "metadata",
        "horizon_hours": "metadata",
        "delivered_heat_MWh": "real_measured_node/calibrated_simulator",
        "delivered_heat_MWh_per_day": "real_measured_node/calibrated_simulator",
        "boundary_heat_MWh": "real_measured_node",
        "boundary_heat_MWh_per_day": "real_measured_node",
        "heat_loss_MWh": "calibrated_simulator",
        "heat_loss_MWh_per_day": "calibrated_simulator",
        "pump_energy_proxy_kWh": "simulator_assisted_hidden_state",
        "pump_energy_proxy_kWh_per_day": "simulator_assisted_hidden_state",
        "pressure_drop_energy_proxy_kWh": "simulator_assisted_hidden_state",
        "pressure_drop_energy_proxy_kWh_per_day": "simulator_assisted_hidden_state",
        "mean_pressure_drop_kPa": "simulator_assisted_hidden_state",
        "max_pressure_drop_kPa": "simulator_assisted_hidden_state",
        "cost_proxy_EUR": "proxy_assumption_based",
        "cost_proxy_EUR_per_day": "proxy_assumption_based",
        "CO2_proxy_kg": "proxy_assumption_based",
        "CO2_proxy_kg_per_day": "proxy_assumption_based",
    }
    units = {
        "evaluation_horizon": "text",
        "horizon_hours": "h",
        "delivered_heat_MWh": "MWh",
        "delivered_heat_MWh_per_day": "MWh/day",
        "boundary_heat_MWh": "MWh",
        "boundary_heat_MWh_per_day": "MWh/day",
        "mean_delivered_heat_kW": "kW",
        "peak_delivered_heat_kW": "kW",
        "delivered_heat_error_percent": "%",
        "heat_loss_MWh": "MWh",
        "heat_loss_MWh_per_day": "MWh/day",
        "heat_loss_ratio_percent": "%",
        "pump_energy_proxy_kWh": "kWh",
        "pump_energy_proxy_kWh_per_day": "kWh/day",
        "normalized_pump_energy_proxy_kWh_per_MWh": "kWh/MWh",
        "pressure_drop_energy_proxy_kWh": "kWh",
        "pressure_drop_energy_proxy_kWh_per_day": "kWh/day",
        "mean_pressure_drop_kPa": "kPa",
        "max_pressure_drop_kPa": "kPa",
        "mean_energy_balance_residual_percent": "%",
        "max_energy_balance_residual_percent": "%",
        "cumulative_energy_balance_residual_MWh": "MWh",
        "cost_proxy_EUR": "EUR",
        "cost_proxy_EUR_per_day": "EUR/day",
        "CO2_proxy_kg": "kgCO2",
        "CO2_proxy_kg_per_day": "kgCO2/day",
    }
    for summary in summaries:
        source = summary["source"]
        for metric, value in summary.items():
            if metric == "source":
                continue
            rows.append(
                {
                    "source": source,
                    "kpi": metric,
                    "value": value,
                    "unit": units.get(metric, ""),
                    "state_type": state_map.get(metric, "calibrated_simulator"),
                    "interpretation": _interpret_metric(metric),
                    "safe_claim": _safe_claim_for_metric(metric),
                }
            )
    return pd.DataFrame(rows)


def _interpret_metric(metric: str) -> str:
    if "pump" in metric:
        return "pump-energy proxy from simulator-assisted flow/head and assumed pump efficiency"
    if "pressure" in metric:
        return "pressure/head proxy from simulator-assisted hidden hydraulic state"
    if "cost" in metric:
        return "transparent operational cost proxy under stated tariff assumptions"
    if "CO2" in metric:
        return "assumption-based CO2 proxy for operational interpretation"
    if "heat_loss" in metric:
        return "pipe heat-loss estimate from calibrated simulator/reconstructed state"
    if "energy_balance" in metric:
        return "energy-balance consistency indicator"
    return "operational energy-impact KPI"


def _safe_claim_for_metric(metric: str) -> str:
    if "pump" in metric or "pressure" in metric:
        return "Pressure/head and flow-based indicators are simulator-assisted hidden-state proxies, not real distributed measurements."
    if "cost" in metric:
        return "Cost and CO2 values are proxy indicators, not optimized economic-dispatch results."
    if "CO2" in metric:
        return "Cost and CO2 values are proxy indicators, not optimized economic-dispatch results."
    return "Real data support boundary conditions/calibration; internal distributed quantities remain calibrated-simulator or simulator-assisted estimates."


def _nominal_winter_scenario(config: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    df = load_sonderborg_processed(config)
    winter = df[df["timestamp"].dt.month.isin([12, 1, 2])].copy()
    if winter.empty:
        winter = df.copy()
    winter["date"] = winter["timestamp"].dt.date
    day = winter.groupby("date")["heat_load_kw"].mean().sort_values(ascending=False).index[0]
    day_df = winter[winter["date"].eq(day)].drop(columns=["date"]).head(96).copy()
    sim = simulate_from_dataframe(day_df, config, params)
    series = _series_from_sim(sim, config, "nominal_winter_day")
    summary = _summarize_series(series, "nominal_winter_day", config)
    summary.update(
        {
            "scenario": "nominal winter-day",
            "date": str(day),
            "evaluation_horizon": "24 h winter-day scenario",
            "mean_supply_temp_C": float(day_df["supply_temp_C"].mean()),
            "mean_return_temp_C": float(day_df["return_temp_C"].mean()),
            "mean_heat_load_kW": float(day_df["heat_load_kw"].mean()),
            "peak_heat_load_kW": float(day_df["heat_load_kw"].max()),
            "pressure_drop_residual_percent": 0.0,
            "state_type": "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
            "safe_claim": "Nominal winter-day uses real Sonderborg thermal data. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements. Cost and CO2 values are proxy indicators, not optimized economic-dispatch results.",
        }
    )
    out = pd.DataFrame([summary])
    out.to_csv(RESULTS / "scenario_nominal_winter_energy_impact.csv", index=False)
    return out


def _combined_stress_scenario(config: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    stress_metrics = pd.read_csv(RESULTS / "combined_stress_test_improved.csv") if (RESULTS / "combined_stress_test_improved.csv").exists() else pd.DataFrame()
    base_sim = _base_sim(config, params)
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
    base_summary: dict[str, float] | None = None
    for case in cases:
        sim, _, _, note = _make_case(base_sim, config, params, case)
        series = _series_from_sim(sim, config, case)
        summary = _summarize_series(series, case, config)
        if case == "baseline_real_profile":
            base_summary = summary
        metrics_row = pd.DataFrame()
        if not stress_metrics.empty:
            metrics_row = stress_metrics[
                stress_metrics["case"].astype(str).eq(case)
                & stress_metrics["model"].astype(str).eq("Proposed PI-GNN-GRU-v3 balanced_mode")
            ]
        base_heat_loss = base_summary["heat_loss_MWh"] if base_summary else summary["heat_loss_MWh"]
        base_pump = base_summary["pump_energy_proxy_kWh"] if base_summary else summary["pump_energy_proxy_kWh"]
        rows.append(
            {
                "scenario": case,
                "evaluation_horizon": f"{summary.get('horizon_hours', np.nan):.1f} h controlled-stress scenario",
                "horizon_hours": summary.get("horizon_hours", np.nan),
                "delivered_heat_MWh": summary["delivered_heat_MWh"],
                "delivered_heat_MWh_per_day": summary["delivered_heat_MWh_per_day"],
                "heat_loss_MWh": summary["heat_loss_MWh"],
                "heat_loss_MWh_per_day": summary["heat_loss_MWh_per_day"],
                "heat_loss_ratio_percent": summary["heat_loss_ratio_percent"],
                "heat_loss_MWh_increase": summary["heat_loss_MWh"] - base_heat_loss,
                "pump_energy_proxy_kWh": summary["pump_energy_proxy_kWh"],
                "pump_energy_proxy_kWh_per_day": summary["pump_energy_proxy_kWh_per_day"],
                "normalized_pump_energy_proxy_kWh_per_MWh": summary["normalized_pump_energy_proxy_kWh_per_MWh"],
                "pump_energy_proxy_kWh_increase": summary["pump_energy_proxy_kWh"] - base_pump,
                "maximum_pressure_drop_residual_percent": float(pd.to_numeric(metrics_row.get("pressure_drop_error_percent", pd.Series([0.0])), errors="coerce").iloc[0]) if not metrics_row.empty else 0.0,
                "maximum_energy_balance_residual_percent": float(pd.to_numeric(metrics_row.get("energy_balance_residual_percent", pd.Series([summary["max_energy_balance_residual_percent"]])), errors="coerce").iloc[0]) if not metrics_row.empty else summary["max_energy_balance_residual_percent"],
                "recovery_time_min": float(pd.to_numeric(metrics_row.get("recovery_time_min", pd.Series([0.0])), errors="coerce").iloc[0]) if not metrics_row.empty else 0.0,
                "maximum_temperature_error_C": float(pd.to_numeric(metrics_row.get("max_temperature_error_C", pd.Series([np.nan])), errors="coerce").iloc[0]) if not metrics_row.empty else np.nan,
                "maximum_head_error_m": float(pd.to_numeric(metrics_row.get("max_head_error_m", pd.Series([np.nan])), errors="coerce").iloc[0]) if not metrics_row.empty else np.nan,
                "cost_proxy_EUR": summary["cost_proxy_EUR"],
                "cost_proxy_EUR_per_day": summary["cost_proxy_EUR_per_day"],
                "CO2_proxy_kg": summary["CO2_proxy_kg"],
                "CO2_proxy_kg_per_day": summary["CO2_proxy_kg_per_day"],
                "state_type": "calibrated_simulator + simulator_assisted_hidden_state",
                "safe_claim": "Disturbances are controlled perturbations applied to real operating profiles, not documented field fault events. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements.",
                "note": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "scenario_combined_stress_energy_impact.csv", index=False)
    return out


def _sensor_layout_scenario(config: dict[str, Any], reference_summary: dict[str, float]) -> pd.DataFrame:
    path = RESULTS / "sensor_layout_comparison_final.csv"
    layouts = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if layouts.empty:
        out = pd.DataFrame([{"scenario": "not available", "safe_claim": "Sensor-layout energy-impact comparison not available."}])
        out.to_csv(RESULTS / "scenario_sensor_layout_energy_impact.csv", index=False)
        return out
    wanted = {
        "S1_inlet_only": "inlet only",
        "S2_inlet_outlet": "inlet + outlet",
        "S4_five_sensors": "five sensors",
        "S9_optimized_three_sensors": "optimized three sensors",
        "S10_optimized_five_sensors": "optimized five sensors",
    }
    mean_flow = float(pd.to_numeric(_series_from_sim(_base_sim(config, load_calibrated_params()), config, "reference")["flow_m3_s"], errors="coerce").mean())
    mean_head_drop = float(pd.to_numeric(_series_from_sim(_base_sim(config, load_calibrated_params()), config, "reference")["head_drop_m"], errors="coerce").mean())
    uncertainty = pd.read_csv(RESULTS / "uncertainty_calibration_summary.csv") if (RESULTS / "uncertainty_calibration_summary.csv").exists() else pd.DataFrame()
    interval_width = float(pd.to_numeric(uncertainty.get("mean_interval_width_conformal_calibrated", pd.Series([np.nan])), errors="coerce").mean()) if not uncertainty.empty else np.nan
    rows = []
    for layout, label in wanted.items():
        sub = layouts[layouts["sensor_layout"].astype(str).eq(layout)].head(1)
        if sub.empty:
            continue
        row = sub.iloc[0]
        heat_loss_error = float(row.get("heat_loss_error_percent", np.nan))
        delivered_error = float(row.get("heat_load_consistency_error_percent", row.get("delivered_heat_error_percent", np.nan)))
        energy_residual = float(row.get("energy_balance_residual", np.nan))
        pressure_residual = float(row.get("RMSE_H_full", np.nan)) / max(abs(mean_head_drop), 1e-9) * 100.0
        flow_residual = float(row.get("RMSE_q_full", np.nan)) / max(abs(mean_flow), 1e-9) * 100.0
        pump_error = pressure_residual + flow_residual
        delivered_mwh = reference_summary["delivered_heat_MWh"]
        heat_loss_mwh = reference_summary["heat_loss_MWh"] * (1.0 + heat_loss_error / 100.0)
        heat_loss_ratio = heat_loss_mwh / max(abs(delivered_mwh), 1e-9) * 100.0
        pump_kwh = reference_summary["pump_energy_proxy_kWh"] * (1.0 + pump_error / 100.0)
        cost_proxy = delivered_mwh * float(ASSUMPTIONS["heat_price_EUR_per_MWh"]) + pump_kwh * float(ASSUMPTIONS["electricity_price_EUR_per_kWh"])
        co2_proxy = delivered_mwh * float(ASSUMPTIONS["heat_emission_factor_kgCO2_per_MWh"]) + pump_kwh * float(
            ASSUMPTIONS["electricity_emission_factor_kgCO2_per_kWh"]
        )
        rows.append(
            {
                "scenario": label,
                "sensor_layout": layout,
                "evaluation_horizon": reference_summary.get("evaluation_horizon", "full evaluation horizon"),
                "horizon_hours": reference_summary.get("horizon_hours", np.nan),
                "delivered_heat_MWh": delivered_mwh,
                "delivered_heat_MWh_per_day": reference_summary.get("delivered_heat_MWh_per_day", np.nan),
                "heat_loss_MWh": heat_loss_mwh,
                "heat_loss_MWh_per_day": heat_loss_mwh / max(float(reference_summary.get("horizon_hours", 24.0)) / 24.0, 1e-9),
                "heat_loss_ratio_percent": heat_loss_ratio,
                "pump_energy_proxy_kWh": pump_kwh,
                "pump_energy_proxy_kWh_per_day": pump_kwh / max(float(reference_summary.get("horizon_hours", 24.0)) / 24.0, 1e-9),
                "normalized_pump_energy_proxy_kWh_per_MWh": pump_kwh / max(abs(delivered_mwh), 1e-9),
                "delivered_heat_error_percent": delivered_error,
                "heat_loss_error_percent": heat_loss_error,
                "pump_energy_proxy_error_percent": pump_error,
                "energy_balance_residual_percent": energy_residual,
                "pressure_drop_residual_percent": pressure_residual,
                "uncertainty_interval_width": interval_width,
                "cost_proxy_EUR": cost_proxy,
                "cost_proxy_EUR_per_day": cost_proxy / max(float(reference_summary.get("horizon_hours", 24.0)) / 24.0, 1e-9),
                "CO2_proxy_kg": co2_proxy,
                "CO2_proxy_kg_per_day": co2_proxy / max(float(reference_summary.get("horizon_hours", 24.0)) / 24.0, 1e-9),
                "state_type": "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
                "safe_claim": "Sparse-sensor layout affects operational energy interpretation. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements. Cost and CO2 values are proxy indicators, not optimized economic-dispatch results.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "scenario_sensor_layout_energy_impact.csv", index=False)
    return out


def _write_assumptions() -> None:
    ensure_dir(RESULTS)
    (RESULTS / "operational_energy_impact_assumptions.json").write_text(json.dumps(ASSUMPTIONS, indent=2), encoding="utf-8")


def _write_table(nominal: pd.DataFrame, stress: pd.DataFrame, layouts: pd.DataFrame, summary_lookup: dict[str, float]) -> None:
    rows: list[dict[str, Any]] = []

    def row_from(label: str, values: dict[str, Any]) -> None:
        rows.append(
            {
                "Scenario": label,
                "Evaluation horizon": str(values.get("evaluation_horizon", "stated scenario horizon")),
                "Delivered heat (MWh/day)": _fmt(values.get("delivered_heat_MWh_per_day", values.get("delivered_heat_MWh"))),
                "Heat loss (MWh/day)": _fmt(values.get("heat_loss_MWh_per_day", values.get("heat_loss_MWh"))),
                "Heat-loss ratio (%)": _fmt(values.get("heat_loss_ratio_percent", summary_lookup.get("heat_loss_ratio_percent"))),
                "Pump-energy intensity (kWh/MWh)": _fmt(values.get("normalized_pump_energy_proxy_kWh_per_MWh", values.get("pump_energy_proxy_error_percent"))),
                "Pressure-drop residual (%)": _fmt(values.get("pressure_drop_residual_percent", values.get("maximum_pressure_drop_residual_percent"))),
                "Energy-balance residual (%)": _fmt(
                    values.get(
                        "energy_balance_residual_percent",
                        values.get("maximum_energy_balance_residual_percent", values.get("mean_energy_balance_residual_percent")),
                    )
                ),
                "Cost proxy (EUR/day)": _fmt(values.get("cost_proxy_EUR_per_day", summary_lookup.get("cost_proxy_EUR_per_day"))),
                "CO2 proxy (kg/day)": _fmt(values.get("CO2_proxy_kg_per_day", summary_lookup.get("CO2_proxy_kg_per_day"))),
            }
        )

    if not nominal.empty:
        row_from("nominal winter-day", nominal.iloc[0].to_dict())
    severe = stress[stress["scenario"].astype(str).eq("combined_stress_severe")]
    if not severe.empty:
        row_from("combined stress", severe.iloc[0].to_dict())
    for label in ["inlet only", "inlet + outlet", "five sensors"]:
        sub = layouts[layouts["scenario"].astype(str).eq(label)]
        if not sub.empty:
            row_from(f"{label} layout", sub.iloc[0].to_dict())
    opt = layouts[layouts["scenario"].astype(str).str.contains("optimized", regex=False)].tail(1)
    if not opt.empty:
        row_from("optimized-sensor layout", opt.iloc[0].to_dict())

    table = pd.DataFrame(rows)
    ensure_dir(TABLES)
    latex = table.to_latex(
        index=False,
        escape=True,
        caption=(
            "Operational energy-impact scenario summary. Values are reported over the stated evaluation horizon unless normalized; "
            "cost and CO2 values are proxy indicators, not optimized economic-dispatch results. Pressure/head and flow-based "
            "entries are simulator-assisted hidden-state proxies."
        ),
        label="tab:scenario_energy_impact_summary",
    )
    latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
    latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    (TABLES / "table_scenario_energy_impact_summary.tex").write_text(latex, encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        value = float(value)
        if not np.isfinite(value):
            return "not reported"
        return f"{value:.3f}"
    except Exception:
        return "not reported"


def _plot_operational_summary(summary: pd.DataFrame, stress: pd.DataFrame, layouts: pd.DataFrame) -> None:
    set_ate_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
    sources = ["measured_boundary", "calibrated_simulator", "pignn_v3_balanced"]

    def get(source: str, kpi: str) -> float:
        sub = summary[summary["source"].eq(source) & summary["kpi"].eq(kpi)]
        return float(sub["value"].iloc[0]) if not sub.empty else np.nan

    colors = [PALETTE["measured"], PALETTE["safe"], PALETTE["proposed"]]
    labels = ["Measured", "Simulator", "PI-GNN-v3"]
    axes[0, 0].bar(labels, [get(s, "delivered_heat_MWh_per_day") if s != "measured_boundary" else get(s, "boundary_heat_MWh_per_day") for s in sources], color=colors, edgecolor=PALETTE["edge"], linewidth=0.6)
    axes[0, 0].set_ylabel("MWh/day")
    axes[0, 0].set_title("Delivered heat intensity")

    axes[0, 1].bar(["Simulator", "PI-GNN-v3"], [get("calibrated_simulator", "heat_loss_MWh_per_day"), get("pignn_v3_balanced", "heat_loss_MWh_per_day")], color=[PALETTE["safe"], PALETTE["proposed"]], edgecolor=PALETTE["edge"], linewidth=0.6)
    axes[0, 1].set_ylabel("MWh/day")
    axes[0, 1].set_title("Heat-loss intensity")

    axes[0, 2].bar(["Simulator", "PI-GNN-v3"], [get("calibrated_simulator", "normalized_pump_energy_proxy_kWh_per_MWh"), get("pignn_v3_balanced", "normalized_pump_energy_proxy_kWh_per_MWh")], color=[PALETTE["warning"], PALETTE["proposed"]], edgecolor=PALETTE["edge"], linewidth=0.6)
    axes[0, 2].set_ylabel("kWh/MWh delivered")
    axes[0, 2].set_title("Pump-energy intensity")

    stress_show = stress[stress["scenario"].isin(["baseline_real_profile", "combined_stress_moderate", "combined_stress_severe"])].copy()
    axes[1, 0].bar(
        stress_show["scenario"].astype(str).str.replace("_", "\n", regex=False),
        pd.to_numeric(stress_show["maximum_pressure_drop_residual_percent"], errors="coerce"),
        color=[PALETTE["baseline"], PALETTE["pilstm"], PALETTE["alarm"]][: len(stress_show)],
        edgecolor=PALETTE["edge"],
        linewidth=0.6,
    )
    axes[1, 0].set_ylabel("%")
    axes[1, 0].set_title("Stress pressure-drop residual")

    layout_show = layouts[layouts["scenario"].isin(["inlet only", "inlet + outlet", "five sensors", "optimized five sensors", "optimized three sensors"])].copy()
    axes[1, 1].bar(
        layout_show["scenario"].astype(str).str.replace(" ", "\n"),
        pd.to_numeric(layout_show["energy_balance_residual_percent"], errors="coerce"),
        color=[PALETTE["baseline"], PALETTE["pilstm"], PALETTE["safe"], PALETTE["proposed"], PALETTE["transformer"]][: len(layout_show)],
        edgecolor=PALETTE["edge"],
        linewidth=0.6,
    )
    axes[1, 1].set_ylabel("%")
    axes[1, 1].set_title("Layout energy residual")

    axes[1, 2].bar(
        ["Cost\nproxy", "CO2\nproxy"],
        [get("calibrated_simulator", "cost_proxy_EUR_per_day") / 1000.0, get("calibrated_simulator", "CO2_proxy_kg_per_day") / 1000.0],
        color=[PALETTE["pilstm"], PALETTE["transformer"]],
        edgecolor=PALETTE["edge"],
        linewidth=0.6,
    )
    axes[1, 2].set_ylabel("kEUR/day or tCO2/day")
    axes[1, 2].set_title("Proxy cost and CO2 intensity")

    for ax, panel in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        add_panel_label(ax, panel)
        style_axes(ax)
        ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    for out_dir in [FIGURES, PAPER_FIGURES]:
        save_ate_figure(fig, out_dir, "fig_operational_energy_impact_summary")
        break
    # save_ate_figure closes the figure, so copy to paper folder after final-directory save.
    ensure_dir(PAPER_FIGURES)
    for suffix in [".pdf", ".png"]:
        src = FIGURES / f"fig_operational_energy_impact_summary{suffix}"
        if src.exists():
            (PAPER_FIGURES / src.name).write_bytes(src.read_bytes())


def run_operational_energy_impact() -> None:
    ensure_dir(RESULTS)
    ensure_dir(FIGURES)
    ensure_dir(PAPER_FIGURES)
    config = load_config()
    params = load_calibrated_params()
    _write_assumptions()

    _, sim, _, _, _, _, payloads = prepare_context(config)
    sim_series = _series_from_sim(sim, config, "calibrated_simulator")
    summaries = [_summarize_series(sim_series, "calibrated_simulator", config)]

    measured = sim_series.copy()
    measured["delivered_heat_kw"] = measured["measured_boundary_heat_load_kw"]
    measured["heat_loss_kw"] = 0.0
    measured["flow_m3_s"] = np.nan
    measured["head_drop_m"] = np.nan
    measured["pressure_drop_kPa"] = np.nan
    measured["pump_energy_proxy_kW"] = 0.0
    measured["pressure_drop_power_proxy_kW"] = 0.0
    measured["energy_balance_residual_kw"] = 0.0
    measured["energy_balance_residual_percent"] = 0.0
    summaries.insert(0, _summarize_series(measured, "measured_boundary", config))

    pignn_payload = payloads.get("Proposed PI-GNN-GRU-v3 balanced_mode") or payloads.get("Proposed PI-GNN-GRU-v3 accuracy_mode")
    if pignn_payload is not None:
        pignn_series = _series_from_payload(pignn_payload, config, "pignn_v3_balanced")
        pignn_series = pignn_series.head(len(sim_series)).copy()
        pignn_series["step"] = np.arange(len(pignn_series))
        pignn_series["time_h"] = pignn_series["step"] * float(config["system"]["dt_s"]) / 3600.0
        summaries.append(_summarize_series(pignn_series, "pignn_v3_balanced", config))
        timeseries = pd.concat([sim_series, pignn_series], ignore_index=True)
    else:
        timeseries = sim_series
    timeseries.to_csv(RESULTS / "operational_energy_impact_timeseries.csv", index=False)

    summary_df = _summary_rows(summaries)
    heat_metrics = pd.read_csv(RESULTS / "heat_energy_metrics.csv") if (RESULTS / "heat_energy_metrics.csv").exists() else pd.DataFrame()
    hydraulic = pd.read_csv(RESULTS / "hydraulic_state_metrics.csv") if (RESULTS / "hydraulic_state_metrics.csv").exists() else pd.DataFrame()
    extra_rows = [
        {
            "source": "benchmark_metrics",
            "kpi": "cumulative_heat_loss_error_percent",
            "value": _metric_value(heat_metrics, "cumulative_heat_loss_error_percent", "PI-GNN-GRU-v3", "pignn_gru_v3_value"),
            "unit": "%",
            "state_type": "calibrated_simulator",
            "interpretation": "PI-GNN-GRU-v3 cumulative heat-loss reconstruction error from benchmark metrics",
            "safe_claim": "Heat-loss error is evaluated against calibrated-simulator heat-loss states.",
        },
        {
            "source": "benchmark_metrics",
            "kpi": "pressure_drop_residual_percent",
            "value": _metric_value(hydraulic, "pressure_drop_error_percent", "PI-GNN-GRU-v3", "pignn_gru_v3_value"),
            "unit": "%",
            "state_type": "simulator_assisted_hidden_state",
            "interpretation": "pressure-drop residual for PI-GNN-GRU-v3 from benchmark metrics",
            "safe_claim": "Pressure-drop residual is simulator-assisted because dense real pressure measurements are unavailable.",
        },
    ]
    summary_df = pd.concat([summary_df, pd.DataFrame(extra_rows)], ignore_index=True)
    summary_df.to_csv(RESULTS / "operational_energy_impact_summary.csv", index=False)

    nominal = _nominal_winter_scenario(config, params)
    stress = _combined_stress_scenario(config, params)
    reference_summary = summaries[1]
    layouts = _sensor_layout_scenario(config, reference_summary)
    _write_table(nominal, stress, layouts, reference_summary)
    _plot_operational_summary(summary_df, stress, layouts)
    print("Operational energy-impact package completed.")


if __name__ == "__main__":
    run_operational_energy_impact()
