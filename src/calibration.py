from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .config import PROJECT_ROOT
from .thermo_hydraulic_simulator import simulate_thermo_hydraulics
from .utils import ensure_dir


HYDRAULIC_IDENTIFIABILITY_NOTE = (
    "Hydraulic parameters are weakly identifiable without measured pressure/flow data. "
    "The flow trajectory is therefore treated as a heat-load-derived proxy, not a measured truth; "
    "friction_factor is kept near a literature/default value unless pressure or flow data are supplied."
)


def _slice_boundary(boundary: dict[str, Any], end: int) -> dict[str, Any]:
    out = {}
    for key, value in boundary.items():
        if hasattr(value, "__len__") and not isinstance(value, str):
            out[key] = value[:end]
        else:
            out[key] = value
    return out


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a = np.asarray(a)[mask]
        b = np.asarray(b)[mask]
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def _error_summary(predicted: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float)[mask] - np.asarray(measured, dtype=float)[mask]
    return {
        "RMSE_C": float(np.sqrt(np.nanmean(error**2))),
        "MAE_C": float(np.nanmean(np.abs(error))),
        "signed_bias_C": float(np.nanmean(error)),
    }


def _block_metrics(sim: dict[str, Any], measured_return: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    summary = _error_summary(sim["Tr"][:, 0], measured_return, mask)
    delivered = np.asarray(sim["delivered_heat_W"], dtype=float)
    load = np.asarray(sim["Q_load"], dtype=float)
    residual = np.asarray(sim["energy_balance_residual_W"], dtype=float)
    summary.update(
        {
            "boundary_closure_percent": float(
                np.nanmean(np.abs(delivered[mask] - load[mask]) / np.maximum(np.abs(load[mask]), 1.0)) * 100.0
            ),
            "dynamic_energy_residual_percent": float(
                100.0 * np.nansum(np.abs(residual[mask])) / max(np.nansum(np.abs(load[mask])), 1.0)
            ),
        }
    )
    return summary


def _energy_residual_fraction(sim: dict[str, Any]) -> float:
    residual = np.asarray(sim["energy_balance_residual_W"], dtype=float)
    load = np.asarray(sim["Q_load"], dtype=float)
    return float(np.nanmean(np.abs(residual)) / max(np.nanmean(np.abs(load)), 1.0))


def calibrate_simulator(boundary: dict[str, Any], config: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    n = len(boundary["time_s"])
    n_train = max(24, int(0.7 * n))
    train_boundary = _slice_boundary(boundary, n_train)
    measured_return = np.asarray(train_boundary["T_return_measured"], dtype=float)
    measured_supply = np.asarray(train_boundary["T_source"], dtype=float)
    fit_mask = ~np.asarray(train_boundary.get("trajectory_start", np.zeros(n_train, dtype=bool)), dtype=bool)
    if not fit_mask.any():
        fit_mask[:] = True
    return_is_assumed = bool(train_boundary.get("return_temperature_assumed", False))
    sys = config["system"]

    nominal = {
        "heat_loss_U_W_m2K": float(sys["heat_loss_U_W_m2K"]),
        "friction_factor": float(sys["friction_factor"]),
        "effective_velocity_factor": 1.0,
        "flow_proxy_blend": 0.75,
    }

    def stage1_params(theta: np.ndarray) -> dict[str, float]:
        return {
            **nominal,
            "heat_loss_U_W_m2K": float(theta[0]),
            "effective_velocity_factor": float(theta[1]),
            "return_temperature_offset": 0.0,
            "flow_proxy_blend": float(theta[2]),
        }

    def stage1_objective(theta: np.ndarray) -> float:
        params = stage1_params(theta)
        sim = simulate_thermo_hydraulics(train_boundary, config, params=params)
        return_rmse = _rmse(sim["Tr"][:, 0], measured_return, fit_mask) if not return_is_assumed else 0.25 * _rmse(sim["Tr"][:, 0], measured_return, fit_mask)
        energy_frac = _energy_residual_fraction(sim)
        transition_mask = np.asarray(sim.get("valid_transition", np.ones(len(sim["time_s"]), dtype=bool)), dtype=bool)[1:]
        outlet_diff = np.abs(np.diff(sim["Ts"][:, -1]))
        smooth_penalty = np.nanmean(outlet_diff[transition_mask]) if outlet_diff.size and transition_mask.any() else 0.0
        regularization = 0.02 * (theta[0] - nominal["heat_loss_U_W_m2K"]) ** 2 + 0.01 * (theta[1] - 1.0) ** 2
        return float(return_rmse + 8.0 * energy_frac + 0.02 * smooth_penalty + regularization)

    bounds_stage1 = [(0.1, 3.5), (0.55, 1.55), (0.35, 1.0)]
    maxiter = 4 if quick else 24
    polish_options = {"maxiter": 20 if quick else 120, "ftol": 1e-5}
    de = differential_evolution(stage1_objective, bounds_stage1, seed=config["dataset"].get("seed", 42), maxiter=maxiter, popsize=6, polish=False)
    local = minimize(stage1_objective, de.x, method="L-BFGS-B", bounds=bounds_stage1, options=polish_options)
    stage1 = stage1_params(local.x if local.success else de.x)

    q_proxy = np.asarray(train_boundary.get("q_proxy", []), dtype=float)
    if q_proxy.size and np.isfinite(q_proxy).any():
        mean_q = float(np.nanmean(q_proxy))

        def stage2_objective(theta: np.ndarray) -> float:
            friction = float(theta[0])
            params = {**stage1, "friction_factor": friction}
            sim = simulate_thermo_hydraulics(train_boundary, config, params=params)
            q_error = _rmse(sim["q"][:, 0], q_proxy) / max(mean_q, 1e-6)
            regularization = ((friction - nominal["friction_factor"]) / nominal["friction_factor"]) ** 2
            return float(q_error + 0.4 * regularization)

        fr_bounds = [(0.006, 0.06)]
        de2 = differential_evolution(stage2_objective, fr_bounds, seed=config["dataset"].get("seed", 42), maxiter=3 if quick else 12, popsize=5, polish=False)
        local2 = minimize(stage2_objective, de2.x, method="L-BFGS-B", bounds=fr_bounds, options={"maxiter": 30})
        friction = float((local2.x if local2.success else de2.x)[0])
    else:
        friction = nominal["friction_factor"]

    params = {**stage1, "friction_factor": friction}
    sim_train = simulate_thermo_hydraulics(train_boundary, config, params=params)
    rmse_supply = _rmse(sim_train["Ts"][:, 0], measured_supply, fit_mask)
    rmse_return = _rmse(sim_train["Tr"][:, 0], measured_return, fit_mask)
    rmse_outlet_supply_proxy = _rmse(sim_train["Ts"][:, -1], measured_supply, fit_mask)
    energy_frac = _energy_residual_fraction(sim_train)
    mean_abs_energy_residual_kW = float(np.nanmean(np.abs(sim_train["energy_balance_residual_W"])) / 1000.0)
    delivered = np.asarray(sim_train["delivered_heat_W"], dtype=float)
    load = np.asarray(sim_train["Q_load"], dtype=float)
    heat_load_consistency_error_percent = float(np.nanmean(np.abs(delivered[fit_mask] - load[fit_mask]) / np.maximum(np.abs(load[fit_mask]), 1.0)) * 100.0)
    train_error = _block_metrics(sim_train, measured_return, fit_mask)
    bound_flags = []
    stage1_bounds_by_name = {
        "heat_loss_U_W_m2K": bounds_stage1[0],
        "effective_velocity_factor": bounds_stage1[1],
        "flow_proxy_blend": bounds_stage1[2],
    }
    for name, (lower, upper) in stage1_bounds_by_name.items():
        value = float(params[name])
        tolerance = max(1e-6, 0.005 * (upper - lower))
        if abs(value - lower) <= tolerance:
            bound_flags.append(f"{name}=lower_bound")
        elif abs(value - upper) <= tolerance:
            bound_flags.append(f"{name}=upper_bound")
    calibration_identifiability = "limited; one or more effective parameters reached a search bound" if bound_flags else "locally identified within prescribed bounds"
    calibration_success = bool(
        np.isfinite(energy_frac)
        and energy_frac < 0.10
        and np.isfinite(heat_load_consistency_error_percent)
        and heat_load_consistency_error_percent < 2.0
    )
    if not return_is_assumed:
        calibration_success = calibration_success and rmse_return <= 3.0

    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures")
    payload = {
        **params,
        "calibration_success": calibration_success,
        "return_temperature_assumed": return_is_assumed,
        "hydraulic_identifiability_note": HYDRAULIC_IDENTIFIABILITY_NOTE,
    }
    (PROJECT_ROOT / "results" / "calibrated_parameters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "results" / "calibration_identifiability_note.txt").write_text(HYDRAULIC_IDENTIFIABILITY_NOTE + "\n", encoding="utf-8")
    metrics = pd.DataFrame(
        [
            {
                "dataset": boundary.get("source_dataset", "unknown"),
                "stage1_objective": float(stage1_objective(np.array([params["heat_loss_U_W_m2K"], params["effective_velocity_factor"], params["flow_proxy_blend"]]))),
                "calibration_success": calibration_success,
                "return_temperature_assumed": return_is_assumed,
                "RMSE_supply_C": rmse_supply,
                "supply_metric_status": "imposed Dirichlet boundary consistency; not an independent calibration target",
                "RMSE_return_C": rmse_return,
                "MAE_return_C": train_error["MAE_C"],
                "signed_return_bias_C": train_error["signed_bias_C"],
                "RMSE_outlet_supply_proxy_C": rmse_outlet_supply_proxy,
                "heat_delivery_error_percent": heat_load_consistency_error_percent,
                "heat_load_consistency_error_percent": heat_load_consistency_error_percent,
                "heat_delivery_metric_status": "consumer heat-load boundary closure; not independent predictive validation",
                "energy_balance_residual_fraction": energy_frac,
                "energy_balance_residual": mean_abs_energy_residual_kW,
                "mean_abs_energy_balance_residual_kW": mean_abs_energy_residual_kW,
                "calibration_data_source": boundary.get("source_dataset", "unknown"),
                "train_period": f"first {n_train} of {n} samples; {int(fit_mask.sum())} scored after trajectory-start exclusion",
                "validation_period": f"first {n_train} of {n} samples used for calibration fit; trajectory starts excluded from scored fit; held-out period available for downstream validation",
                "hydraulic_identifiability": "weak without measured pressure/flow",
                "thermal_parameter_identifiability": calibration_identifiability,
                "parameter_bound_flags": "; ".join(bound_flags) if bound_flags else "none",
                "note": HYDRAULIC_IDENTIFIABILITY_NOTE,
            }
        ]
    )
    metrics.to_csv(PROJECT_ROOT / "results" / "calibration_metrics.csv", index=False)

    # Locked later-time replay: apply the calibrated parameters to the untouched
    # chronological suffix without retuning.  The complete trajectory is run so
    # state propagation at the train/replay boundary remains physical; only the
    # later suffix is scored.
    full_sim = simulate_thermo_hydraulics(boundary, config, params=params)
    later_mask = np.zeros(n, dtype=bool)
    later_mask[n_train:] = True
    later_mask &= ~np.asarray(full_sim.get("trajectory_start", np.zeros(n, dtype=bool)), dtype=bool)
    if not later_mask.any():
        raise RuntimeError("Locked later-time replay has no scoreable timestamps.")
    later_error = _block_metrics(full_sim, np.asarray(boundary["T_return_measured"], dtype=float), later_mask)
    later_rows = pd.DataFrame(
        [
            {
                "dataset": boundary.get("source_dataset", "unknown"),
                "calibration_samples": n_train,
                "later_replay_samples": int(later_mask.sum()),
                "later_replay_start_index": n_train,
                "RMSE_return_C": later_error["RMSE_C"],
                "MAE_return_C": later_error["MAE_C"],
                "signed_return_bias_C": later_error["signed_bias_C"],
                "boundary_closure_percent": later_error["boundary_closure_percent"],
                "dynamic_energy_residual_percent": later_error["dynamic_energy_residual_percent"],
                "retuned_on_later_period": False,
                "state_type": "measured_return_and_calibrated_simulator_consistency",
                "note": "Parameters were fitted on the chronological prefix and applied unchanged to the later suffix; distributed fields remain calibrated-simulator quantities.",
            }
        ]
    )
    later_rows.to_csv(PROJECT_ROOT / "results" / "locked_later_replay_metrics.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    t_h = np.asarray(train_boundary["time_s"]) / 3600.0
    axes[0].plot(t_h, measured_supply, label="Measured feed/source temp.", lw=1.5)
    axes[0].plot(t_h, sim_train["Ts"][:, 0], "--", label="Simulated source boundary", lw=1.2)
    axes[0].plot(t_h, sim_train["Ts"][:, -1], label="Simulated load-side supply", lw=1.1)
    axes[0].set_ylabel("Supply temp. (C)")
    axes[1].plot(t_h, measured_return, label="Measured/assumed return", lw=1.5)
    axes[1].plot(t_h, sim_train["Tr"][:, 0], "--", label="Simulated source return", lw=1.2)
    axes[1].set_ylabel("Return temp. (C)")
    axes[2].plot(t_h, sim_train["Q_load"] / 1000.0, label="Measured heat load boundary", lw=1.4)
    axes[2].plot(t_h, sim_train["delivered_heat_W"] / 1000.0, "--", label="Simulated delivered heat", lw=1.2)
    axes[2].set_ylabel("Heat (kW)")
    axes[2].set_xlabel("Time (h)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    title_suffix = "weak hydraulic identification" if return_is_assumed else f"success={calibration_success}"
    axes[0].set_title(f"Calibration fit ({title_suffix})")
    fig.tight_layout()
    for name in ["fig4_calibration_fit", "fig_calibration_fit"]:
        fig.savefig(PROJECT_ROOT / "figures" / f"{name}.pdf", dpi=300)
        fig.savefig(PROJECT_ROOT / "figures" / f"{name}.png", dpi=300)
    plt.close(fig)
    return {"params": params, "metrics": metrics, "calibration_success": calibration_success}
