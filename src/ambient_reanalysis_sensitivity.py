"""Audit the configured ambient boundary against independent weather reanalysis.

The main locked benchmark originally used a constant 5 degC ambient boundary
because the public Sonderborg operating files do not contain weather.  This
script creates a second, provenance-locked boundary from hourly Open-Meteo
historical reanalysis and compares recalibrated thermal-model behaviour.  It
does not overwrite the canonical manuscript input or any trained checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.real_data_mapper import build_boundary_conditions
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics

plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


WEATHER_FILE = PROJECT_ROOT / "data" / "external_weather" / "sonderborg_era5_land_2016_2019_hourly.csv"
CANONICAL_FILE = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv"
MERGED_FILE = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703_era5_land.csv"
RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
FIGURES = PROJECT_ROOT / "figures" / "final"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_weather() -> tuple[pd.DataFrame, dict[str, float]]:
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"Missing {WEATHER_FILE}. Export hourly temperature_2m for 2016-2019 "
            "from the Open-Meteo Historical Weather API before running this audit."
        )
    metadata = pd.read_csv(WEATHER_FILE, nrows=1)
    weather = pd.read_csv(WEATHER_FILE, skiprows=3)
    if weather.shape[1] != 2:
        raise ValueError(f"Expected time and temperature columns in {WEATHER_FILE}, found {weather.columns.tolist()}")
    weather = weather.iloc[:, :2].copy()
    weather.columns = ["weather_timestamp", "ambient_temp_C"]
    weather["weather_timestamp"] = pd.to_datetime(weather["weather_timestamp"], utc=True, errors="raise")
    weather["ambient_temp_C"] = pd.to_numeric(weather["ambient_temp_C"], errors="raise")
    meta = {
        "requested_latitude": 54.90896,
        "requested_longitude": 9.78917,
        "grid_latitude": float(metadata.iloc[0]["latitude"]),
        "grid_longitude": float(metadata.iloc[0]["longitude"]),
        "grid_elevation_m": float(metadata.iloc[0]["elevation"]),
    }
    return weather, meta


def _merge_weather() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    canonical = pd.read_csv(CANONICAL_FILE)
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True, errors="raise")
    weather, meta = _load_weather()
    merged = pd.merge_asof(
        canonical.sort_values("timestamp"),
        weather.sort_values("weather_timestamp"),
        left_on="timestamp",
        right_on="weather_timestamp",
        direction="backward",
        tolerance=pd.Timedelta("75min"),
        suffixes=("_constant", ""),
    )
    if merged["ambient_temp_C"].isna().any():
        missing = int(merged["ambient_temp_C"].isna().sum())
        raise RuntimeError(f"Historical reanalysis merge left {missing} ambient values missing.")
    merged = merged.drop(columns=["weather_timestamp", "ambient_temp_C_constant"])
    merged["ambient_temp_assumed"] = False
    merged["ambient_temp_provenance"] = "open_meteo_era5_land_hourly_past_value"
    merged.to_csv(MERGED_FILE, index=False)
    return canonical, merged, meta


def _rmse(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    error = np.asarray(pred, dtype=float)[mask] - np.asarray(ref, dtype=float)[mask]
    return float(np.sqrt(np.nanmean(error**2)))


def _fit_effective_parameters(frame: pd.DataFrame, config: dict) -> tuple[dict[str, float], float]:
    boundary = build_boundary_conditions(frame, config)
    n_train = max(24, int(0.70 * len(frame)))
    train = {key: (value[:n_train] if hasattr(value, "__len__") and not isinstance(value, str) else value) for key, value in boundary.items()}
    measured_return = np.asarray(train["T_return_measured"], dtype=float)
    mask = ~np.asarray(train.get("trajectory_start", np.zeros(n_train, dtype=bool)), dtype=bool)
    nominal_u = float(config["system"]["heat_loss_U_W_m2K"])
    nominal_f = float(config["system"]["friction_factor"])

    def params_from(theta: np.ndarray) -> dict[str, float]:
        return {
            "heat_loss_U_W_m2K": float(theta[0]),
            "effective_velocity_factor": float(theta[1]),
            "flow_proxy_blend": float(theta[2]),
            "friction_factor": nominal_f,
            "return_temperature_offset": 0.0,
        }

    def objective(theta: np.ndarray) -> float:
        sim = simulate_thermo_hydraulics(train, config, params=params_from(theta))
        return_rmse = _rmse(sim["Tr"][:, 0], measured_return, mask)
        residual = np.asarray(sim["energy_balance_residual_W"], dtype=float)
        load = np.asarray(sim["Q_load"], dtype=float)
        energy_fraction = float(np.nanmean(np.abs(residual)) / max(np.nanmean(np.abs(load)), 1.0))
        transition = np.asarray(sim.get("valid_transition", np.ones(n_train, dtype=bool)), dtype=bool)[1:]
        outlet_diff = np.abs(np.diff(sim["Ts"][:, -1]))
        smoothness = float(np.nanmean(outlet_diff[transition])) if transition.any() else 0.0
        regularization = 0.02 * (theta[0] - nominal_u) ** 2 + 0.01 * (theta[1] - 1.0) ** 2
        return float(return_rmse + 8.0 * energy_fraction + 0.02 * smoothness + regularization)

    bounds = [(0.1, 3.5), (0.55, 1.55), (0.35, 1.0)]
    global_fit = differential_evolution(objective, bounds, seed=42, maxiter=24, popsize=6, polish=False)
    local_fit = minimize(objective, global_fit.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 120, "ftol": 1e-5})
    theta = local_fit.x if local_fit.success else global_fit.x
    return params_from(theta), float(objective(theta))


def _score(case: str, frame: pd.DataFrame, config: dict, params: dict[str, float], objective: float) -> tuple[dict, pd.DataFrame]:
    boundary = build_boundary_conditions(frame, config)
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    n_cal = int(0.70 * 768)
    starts = np.asarray(sim["trajectory_start"], dtype=bool)
    measured_return = np.asarray(boundary["T_return_measured"], dtype=float)
    masks = {
        "calibration_prefix": (np.arange(len(frame)) < n_cal) & ~starts,
        "locked_primary_suffix": (np.arange(len(frame)) >= n_cal) & (np.arange(len(frame)) < 768) & ~starts,
        "all_retained_timestamps": ~starts,
    }
    rows = []
    for period, mask in masks.items():
        error = np.asarray(sim["Tr"][:, 0], dtype=float)[mask] - measured_return[mask]
        residual = np.asarray(sim["energy_balance_residual_W"], dtype=float)[mask]
        load = np.asarray(sim["Q_load"], dtype=float)[mask]
        rows.append(
            {
                "ambient_case": case,
                "period": period,
                "n_scored": int(mask.sum()),
                "return_RMSE_C": float(np.sqrt(np.nanmean(error**2))),
                "return_MAE_C": float(np.nanmean(np.abs(error))),
                "return_signed_bias_C": float(np.nanmean(error)),
                "dynamic_energy_residual_percent": float(100.0 * np.nansum(np.abs(residual)) / max(np.nansum(np.abs(load)), 1.0)),
                "mean_heat_loss_kW": float(np.nanmean(sim["Q_loss"][mask]) / 1000.0),
                "mean_outlet_supply_C": float(np.nanmean(sim["Ts"][mask, -1])),
                "mean_head_drop_m": float(np.nanmean(sim["pressure_drop_m"][mask])),
                "thermal_delay_h": float(sim["thermal_delay_h"]),
                "state_type": "measured source-return score plus calibrated-simulator internal quantities",
            }
        )
    summary = {
        "ambient_case": case,
        "stage1_objective": objective,
        **params,
        "ambient_mean_C": float(frame["ambient_temp_C"].mean()),
        "ambient_std_C": float(frame["ambient_temp_C"].std()),
        "ambient_min_C": float(frame["ambient_temp_C"].min()),
        "ambient_max_C": float(frame["ambient_temp_C"].max()),
        "mean_heat_loss_kW": float(np.nanmean(sim["Q_loss"]) / 1000.0),
        "mean_outlet_supply_C": float(np.nanmean(sim["Ts"][:, -1])),
        "thermal_delay_h": float(sim["thermal_delay_h"]),
        "evidence_class": "configured boundary sensitivity" if case == "constant_5C" else "independent historical reanalysis boundary sensitivity",
    }
    return summary, pd.DataFrame(rows)


def _write_table(summary: pd.DataFrame) -> None:
    display = summary[[
        "ambient_case", "ambient_mean_C", "ambient_std_C", "heat_loss_U_W_m2K",
        "effective_velocity_factor", "flow_proxy_blend", "mean_heat_loss_kW",
        "mean_outlet_supply_C", "thermal_delay_h",
    ]].copy()
    display.columns = ["Ambient boundary", "Mean", "SD", "$U$", "$\\eta_v$", "$\\beta_q$", "Heat loss", "Outlet $T_s$", "Delay"]
    display["Ambient boundary"] = display["Ambient boundary"].replace(
        {
            "constant_5C": r"Configured 5~$^\circ$C",
            "era5_land_reanalysis": "ERA5-Land reanalysis",
        }
    )
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Ambient-boundary sensitivity with identical calibration objective and bounds. Temperature columns are in $^\circ$C, $U$ in W m$^{-2}$ K$^{-1}$, heat loss in kW, and delay in h. Open-Meteo values are historical reanalysis, not local weather-station measurements.}",
        r"\label{tab:ambient_reanalysis_sensitivity}", r"\small",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        " & ".join(display.columns) + r" \\", r"\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"{row.iloc[0]} & " + " & ".join(f"{float(value):.3f}" for value in row.iloc[1:]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_ambient_reanalysis_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(summary: pd.DataFrame, period: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"constant_5C": "#555555", "era5_land_reanalysis": "#0000E6"}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), constrained_layout=True)
    for case, group in period.groupby("ambient_case"):
        suffix = group[group["period"].eq("locked_primary_suffix")].iloc[0]
        axes[0].bar(case, suffix["return_RMSE_C"], color=colors[case], edgecolor="#111111")
    axes[0].set_ylabel(r"Locked return RMSE ($^\circ$C)")
    axes[0].set_xticks([])
    axes[0].set_title("(a) Measured-return replay")
    axes[1].bar(summary["ambient_case"], summary["mean_heat_loss_kW"], color=[colors[c] for c in summary["ambient_case"]], edgecolor="#111111")
    axes[1].set_ylabel("Mean simulated heat loss (kW)")
    axes[1].set_xticks([])
    axes[1].set_title("(b) Internal heat loss")
    axes[2].bar(summary["ambient_case"], summary["heat_loss_U_W_m2K"], color=[colors[c] for c in summary["ambient_case"]], edgecolor="#111111")
    axes[2].set_ylabel("Fitted $U$ (W m$^{-2}$ K$^{-1}$)")
    axes[2].set_xticks([])
    axes[2].set_title("(c) Effective calibration")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[c], edgecolor="#111111") for c in colors]
    fig.legend(handles, [r"Configured 5 $^\circ$C", "Historical reanalysis"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.0)
    fig.savefig(FIGURES / "fig_ambient_reanalysis_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_ambient_reanalysis_sensitivity.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_ambient_reanalysis_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    canonical, reanalysis, meta = _merge_weather()
    config = load_config()
    fixed = canonical.copy()
    fixed["timestamp"] = pd.to_datetime(fixed["timestamp"], utc=True)
    cases = {"constant_5C": fixed, "era5_land_reanalysis": reanalysis}
    summaries = []
    periods = []
    for name, frame in cases.items():
        params, objective = _fit_effective_parameters(frame.iloc[:768].copy(), deepcopy(config))
        summary, scored = _score(name, frame, deepcopy(config), params, objective)
        summaries.append(summary)
        periods.append(scored)
    summary_df = pd.DataFrame(summaries)
    period_df = pd.concat(periods, ignore_index=True)
    summary_df.to_csv(RESULTS / "ambient_boundary_reanalysis_sensitivity.csv", index=False)
    period_df.to_csv(RESULTS / "ambient_reanalysis_period_metrics.csv", index=False)
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Open-Meteo Historical Weather API, hourly ERA5-Land temperature_2m",
        "api_url": "https://archive-api.open-meteo.com/v1/archive?latitude=54.90896&longitude=9.78917&start_date=2016-01-01&end_date=2019-12-31&hourly=temperature_2m&models=era5_land&timezone=UTC",
        "reanalysis_model": "ERA5-Land",
        "source_file": str(WEATHER_FILE.relative_to(PROJECT_ROOT)),
        "source_sha256": _sha256(WEATHER_FILE),
        "canonical_source_sha256": _sha256(CANONICAL_FILE),
        "merged_file": str(MERGED_FILE.relative_to(PROJECT_ROOT)),
        "merged_sha256": _sha256(MERGED_FILE),
        "weather_rows": 35064,
        "merge_rule": "past-only hourly value carried forward to retained 15-minute timestamp; tolerance 75 min",
        "evidence_boundary": "historical reanalysis boundary, not a local weather-station measurement",
        **meta,
    }
    (RESULTS / "ambient_reanalysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    _write_table(summary_df)
    _plot(summary_df, period_df)
    print(summary_df.to_string(index=False))
    print(period_df.to_string(index=False))


if __name__ == "__main__":
    main()
