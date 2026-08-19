"""Quantify equifinality of effective calibration parameters and hidden fields."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from matplotlib.colors import BoundaryNorm, ListedColormap

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.real_data_mapper import build_boundary_conditions
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics

plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
FIGURES = PROJECT_ROOT / "figures" / "final"
CANONICAL = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv"


def _objective_and_outputs(theta: np.ndarray, boundary: dict, config: dict) -> dict[str, float]:
    params = {
        "heat_loss_U_W_m2K": float(theta[0]),
        "effective_velocity_factor": float(theta[1]),
        "flow_proxy_blend": float(theta[2]),
        "friction_factor": float(config["system"]["friction_factor"]),
    }
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    mask = ~np.asarray(sim["trajectory_start"], dtype=bool)
    error = np.asarray(sim["Tr"][:, 0]) - np.asarray(boundary["T_return_measured"])
    return_rmse = float(np.sqrt(np.nanmean(error[mask] ** 2)))
    residual = np.asarray(sim["energy_balance_residual_W"], dtype=float)
    load = np.asarray(sim["Q_load"], dtype=float)
    energy_fraction = float(np.nanmean(np.abs(residual)) / max(np.nanmean(np.abs(load)), 1.0))
    transition = np.asarray(sim["valid_transition"], dtype=bool)[1:]
    outlet_diff = np.abs(np.diff(sim["Ts"][:, -1]))
    smoothness = float(np.nanmean(outlet_diff[transition])) if transition.any() else 0.0
    nominal_u = float(config["system"]["heat_loss_U_W_m2K"])
    regularization = 0.02 * (theta[0] - nominal_u) ** 2 + 0.01 * (theta[1] - 1.0) ** 2
    objective = return_rmse + 8.0 * energy_fraction + 0.02 * smoothness + regularization
    return {
        **params,
        "objective": objective,
        "return_RMSE_C": return_rmse,
        "energy_residual_percent": 100.0 * energy_fraction,
        "mean_outlet_supply_C": float(np.nanmean(sim["Ts"][:, -1])),
        "mean_heat_loss_kW": float(np.nanmean(sim["Q_loss"]) / 1000.0),
        "thermal_delay_h": float(sim["thermal_delay_h"]),
        "mean_head_drop_m": float(np.nanmean(sim["pressure_drop_m"])),
        "mean_flow_m3_s": float(np.nanmean(sim["q"][:, 0])),
    }


def _write_table(spread: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Equifinality among near-optimal effective calibrations. Parameter combinations are retained within the stated objective tolerance; internal temperature, heat-loss, head, and flow quantities are calibrated-simulator or simulator-assisted outputs rather than field measurements.}",
        r"\label{tab:calibration_equifinality}", r"\small",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Quantity & Minimum & Median & Maximum & Range & Relative range (\%) \\", r"\midrule",
    ]
    for _, row in spread.iterrows():
        lines.append(
            f"{row['quantity']} & {row['minimum']:.4f} & {row['median']:.4f} & {row['maximum']:.4f} & {row['range']:.4f} & {row['relative_range_percent']:.2f}" + " \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_calibration_equifinality.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(ensemble: pd.DataFrame, near: pd.DataFrame, spread: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
    objective = pd.to_numeric(ensemble["objective"], errors="coerce").to_numpy(float)
    lo, hi = float(np.nanmin(objective)), float(np.nanmax(objective))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    boundaries = np.linspace(lo, hi, 9)
    objective_cmap = ListedColormap(plt.get_cmap("viridis_r")(np.linspace(0.05, 0.95, 8)))
    objective_norm = BoundaryNorm(boundaries, objective_cmap.N)
    scatter = axes[0].scatter(
        ensemble["heat_loss_U_W_m2K"], ensemble["effective_velocity_factor"],
        c=objective, cmap=objective_cmap, norm=objective_norm, s=18, alpha=0.75, edgecolors="none",
    )
    axes[0].scatter(near["heat_loss_U_W_m2K"], near["effective_velocity_factor"], facecolors="none", edgecolors="#FF6626", s=42, linewidth=1.0)
    axes[0].set_xlabel("$U$ (W m$^{-2}$ K$^{-1}$)")
    axes[0].set_ylabel("Effective velocity factor $\\eta_v$ (-)")
    axes[0].set_title("(a) Calibration objective surface")
    fig.colorbar(scatter, ax=axes[0], label="Calibration objective", boundaries=boundaries, drawedges=True)

    blend = pd.to_numeric(near["flow_proxy_blend"], errors="coerce").to_numpy(float)
    blend_lo, blend_hi = float(np.nanmin(blend)), float(np.nanmax(blend))
    if np.isclose(blend_lo, blend_hi):
        blend_hi = blend_lo + 1.0
    blend_bounds = np.linspace(blend_lo, blend_hi, 7)
    blend_cmap = ListedColormap(plt.get_cmap("plasma")(np.linspace(0.05, 0.95, 6)))
    axes[1].scatter(near["return_RMSE_C"], near["mean_heat_loss_kW"], c=blend, cmap=blend_cmap, norm=BoundaryNorm(blend_bounds, blend_cmap.N), s=34, edgecolors="#111111", linewidth=0.35)
    axes[1].set_xlabel(r"Measured-return RMSE ($^\circ$C)")
    axes[1].set_ylabel("Mean simulated heat loss (kW)")
    axes[1].set_title("(b) Similar fit, different fields")

    ranked = spread.sort_values("relative_range_percent").tail(6)
    axes[2].barh(ranked["quantity"], ranked["relative_range_percent"], color="#0000E6", edgecolor="#111111")
    axes[2].set_xlabel("Near-optimal relative range (%)")
    axes[2].set_title("(c) Internal-field spread")
    axes[2].grid(axis="x", alpha=0.2)
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.0)
    fig.savefig(FIGURES / "fig_calibration_equifinality.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_calibration_equifinality.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_calibration_equifinality.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config()
    frame = pd.read_csv(CANONICAL).iloc[:768].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    n_train = int(0.70 * len(frame))
    boundary_full = build_boundary_conditions(frame.iloc[:n_train].copy(), config)
    sampler = qmc.LatinHypercube(d=3, seed=20260807)
    bounds_low = np.array([0.1, 0.55, 0.35])
    bounds_high = np.array([3.5, 1.55, 1.0])
    theta_global = qmc.scale(sampler.random(384), bounds_low, bounds_high)
    calibrated = json.loads((RESULTS / "calibrated_parameters.json").read_text(encoding="utf-8"))
    # Add a dense local design near the bound-active optimum. A global Latin
    # hypercube alone sparsely samples this corner and can make a 5% objective
    # tolerance look artificially empty.
    local_sampler = qmc.LatinHypercube(d=3, seed=20260808)
    local_low = np.array([0.1, 1.15, 0.78])
    local_high = np.array([0.9, 1.55, 1.0])
    theta_local = qmc.scale(local_sampler.random(384), local_low, local_high)
    theta = np.vstack([
        theta_global,
        theta_local,
        [calibrated["heat_loss_U_W_m2K"], calibrated["effective_velocity_factor"], calibrated["flow_proxy_blend"]],
    ])
    rows = [_objective_and_outputs(value, boundary_full, config) for value in theta]
    ensemble = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
    minimum = float(ensemble["objective"].min())
    threshold = max(minimum * 1.05, minimum + 0.05)
    near = ensemble[ensemble["objective"] <= threshold].copy()
    if len(near) < 8:
        raise RuntimeError(
            f"Only {len(near)} parameter sets fall within the prespecified 5%/0.05 objective tolerance; "
            "increase the local design rather than relaxing the definition after seeing results."
        )
    ensemble["near_optimal"] = ensemble.index.isin(near.index)
    ensemble.to_csv(RESULTS / "calibration_equifinality_ensemble.csv", index=False)

    labels = {
        "heat_loss_U_W_m2K": "$U$ (W m$^{-2}$ K$^{-1}$)",
        "effective_velocity_factor": "velocity factor (-)",
        "flow_proxy_blend": "flow-proxy blend (-)",
        "return_RMSE_C": "return RMSE ($^\\circ$C)",
        "mean_outlet_supply_C": "outlet supply ($^\\circ$C)",
        "mean_heat_loss_kW": "heat loss (kW)",
        "thermal_delay_h": "thermal delay (h)",
        "mean_head_drop_m": "head drop (m)",
        "mean_flow_m3_s": "flow (m$^3$ s$^{-1}$)",
    }
    spread_rows = []
    for key, label in labels.items():
        values = pd.to_numeric(near[key], errors="coerce").dropna()
        lo, med, hi = float(values.min()), float(values.median()), float(values.max())
        spread_rows.append({
            "quantity": label,
            "minimum": lo,
            "median": med,
            "maximum": hi,
            "range": hi - lo,
            "relative_range_percent": 100.0 * (hi - lo) / max(abs(med), 1e-12),
            "evidence_class": "effective calibration" if key in {"heat_loss_U_W_m2K", "effective_velocity_factor", "flow_proxy_blend"} else "calibrated-simulator/simulator-assisted internal output",
        })
    spread = pd.DataFrame(spread_rows)
    spread.to_csv(RESULTS / "calibration_equifinality_internal_field_spread.csv", index=False)
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(ensemble),
        "near_optimal_count": len(near),
        "minimum_objective": minimum,
        "near_optimal_threshold": threshold,
        "tolerance_rule": "objective <= max(minimum*1.05, minimum+0.05); fixed 5%/0.05 rule declared before inspecting retained sets",
        "parameter_bounds": {"U": [0.1, 3.5], "eta_v": [0.55, 1.55], "beta_q": [0.35, 1.0]},
        "objective": "return_RMSE + 8*energy_fraction + 0.02*outlet_smoothness + 0.02*(U-U0)^2 + 0.01*(eta_v-1)^2",
        "interpretation": "Near-equivalent plant-level fit does not identify a unique internal distributed field.",
    }
    (RESULTS / "calibration_equifinality_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    _write_table(spread)
    _plot(ensemble, near, spread)
    print(json.dumps(protocol, indent=2))
    print(spread.to_string(index=False))


if __name__ == "__main__":
    main()
