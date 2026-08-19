from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # Plotting is optional for numerical/reviewer audits.
    plt = None
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT


def ensure_dir(path: str | Path) -> Path:
    """Create an output directory without importing the optional Torch stack."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _hydraulic_resistance(config: dict[str, Any], friction_factor: float) -> float:
    sys = config["system"]
    length = float(sys["length_m"])
    diameter = float(sys["diameter_m"])
    area = np.pi * diameter**2 / 4.0
    g = float(sys["g"])
    return friction_factor * length / max(diameter * 2.0 * g * area**2, 1e-12)


def _pump_head(alpha: np.ndarray | float, config: dict[str, Any]) -> np.ndarray | float:
    sys = config["system"]
    return sys["pump_c1"] * np.asarray(alpha) ** 2 + sys["pump_c2"] * np.asarray(alpha) + sys["pump_c3"]


def _flow_proxy_from_load(
    Q_load: np.ndarray,
    T_source: np.ndarray,
    T_return: np.ndarray,
    config: dict[str, Any],
    delta_t_min: float = 8.0,
) -> np.ndarray:
    sys = config["system"]
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    delta_t = np.maximum(T_source - T_return, delta_t_min)
    return np.clip(Q_load / np.maximum(rho * cp * delta_t, 1e-9), 1e-4, None)


def _causal_flow_proxy_from_load(
    Q_load: np.ndarray,
    T_source: np.ndarray,
    T_return: np.ndarray,
    config: dict[str, Any],
    delta_t_min: float = 8.0,
) -> np.ndarray:
    """Heat-load flow proxy using only return temperatures available before k.

    The first value uses the first available return boundary.  For k > 0 the
    proxy uses T_return[k-1], so the current return target cannot be reproduced
    algebraically through q = Q/(rho cp DeltaT).  This is the default mode for
    replay and newly generated benchmark states.  The result remains a proxy,
    not a flow-meter measurement.
    """
    lagged_return = np.empty_like(np.asarray(T_return, dtype=float))
    lagged_return[0] = float(T_return[0])
    lagged_return[1:] = np.asarray(T_return[:-1], dtype=float)
    return _flow_proxy_from_load(Q_load, T_source, lagged_return, config, delta_t_min=delta_t_min)


def simulate_thermo_hydraulics(
    boundary_conditions: dict[str, Any],
    config: dict[str, Any],
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    params = params or {}
    sys = config["system"]
    length = float(sys["length_m"])
    dx = float(sys["dx_m"])
    dt = float(sys["dt_s"])
    n_nodes = int(round(length / dx)) + 1
    n_steps = len(boundary_conditions["time_s"])
    x = np.linspace(0.0, length, n_nodes)

    rho = float(sys["rho"])
    cp = float(sys["cp"])
    diameter = float(sys["diameter_m"])
    area = np.pi * diameter**2 / 4.0
    perimeter = float(sys["pipe_perimeter_m"])
    outlet_head = float(sys["outlet_head_m"])
    U = float(params.get("heat_loss_U_W_m2K", sys["heat_loss_U_W_m2K"]))
    friction = float(params.get("friction_factor", sys["friction_factor"]))
    velocity_factor = float(params.get("effective_velocity_factor", params.get("effective_delay_factor", 1.0)))
    flow_blend = float(params.get("flow_proxy_blend", 0.75))
    tau_q = float(params.get("hydraulic_time_constant_s", 3.0 * dt))
    R_total = _hydraulic_resistance(config, friction)
    K_loss = U * perimeter / max(rho * cp * area, 1e-12)

    T_source = np.asarray(boundary_conditions["T_source"], dtype=float)
    T_return_measured = np.asarray(boundary_conditions["T_return_measured"], dtype=float)
    Q_load = np.asarray(boundary_conditions["Q_load_W"], dtype=float)
    Ta = np.asarray(boundary_conditions["Ta"], dtype=float)
    alpha = np.asarray(boundary_conditions["alpha_estimated"], dtype=float)
    time_s = np.asarray(boundary_conditions["time_s"], dtype=float)
    trajectory_start = np.asarray(boundary_conditions.get("trajectory_start", np.zeros(n_steps, dtype=bool)), dtype=bool)
    if trajectory_start.shape != (n_steps,):
        raise ValueError("trajectory_start must have one flag per boundary timestamp")
    if n_steps:
        trajectory_start[0] = True
    # ``valid_transition[k]`` denotes the physically simulated transition from
    # k-1 to k. It is false at each segment restart and is used to exclude
    # artificial cross-gap storage and smoothness terms.
    valid_transition = ~trajectory_start
    if n_steps:
        valid_transition[0] = False
    flow_proxy_mode = str(
        boundary_conditions.get(
            "flow_proxy_mode",
            params.get("flow_proxy_mode", sys.get("flow_proxy_mode", "causal_lagged_return")),
        )
    )
    # Only a declared causal external proxy may override the internal causal
    # construction.  This prevents legacy same-timestamp return proxies from
    # silently entering a replay-labelled trajectory.
    if "q_proxy" in boundary_conditions and flow_proxy_mode.startswith("causal_lagged_return"):
        q_proxy = np.asarray(boundary_conditions["q_proxy"], dtype=float)
    elif flow_proxy_mode == "direct_current_return":
        raise ValueError("direct_current_return flow proxies are forbidden for causal benchmark generation")
    else:
        q_proxy = _causal_flow_proxy_from_load(Q_load, T_source, T_return_measured, config)
        flow_proxy_mode = "causal_lagged_return_recomputed"

    Ts = np.zeros((n_steps, n_nodes), dtype=float)
    Tr = np.zeros((n_steps, n_nodes), dtype=float)
    H = np.zeros((n_steps, n_nodes), dtype=float)
    q = np.zeros((n_steps, n_nodes), dtype=float)
    Q_loss = np.zeros(n_steps, dtype=float)
    Q_loss_supply = np.zeros(n_steps, dtype=float)
    Q_loss_return = np.zeros(n_steps, dtype=float)
    Q_loss_segments = np.zeros((n_steps, n_nodes - 1), dtype=float)
    delivered_heat_W = np.zeros(n_steps, dtype=float)
    energy_balance_residual_W = np.zeros(n_steps, dtype=float)
    pump_boundary_residual_m = np.zeros(n_steps, dtype=float)
    pressure_drop_m = np.zeros(n_steps, dtype=float)

    def initialize_segment(k: int) -> float:
        """Initialise a new observed segment without propagating across a gap."""
        # Segment initialisation must be causal.  A former implementation used
        # the mean supply/return difference over the complete trajectory, which
        # allowed observations after k to alter the state initialised at k.
        # Both quantities below are boundary values available at the segment
        # start; no later timestamp contributes to the initial spatial profile.
        segment_start_delta = max(float(T_source[k] - T_return_measured[k]), 0.0)
        initial_supply_drop = np.linspace(
            0,
            max(2.0, segment_start_delta * 0.20),
            n_nodes,
        )
        Ts[k] = T_source[k] - initial_supply_drop
        Tr[k] = np.linspace(
            T_return_measured[k],
            min(Ts[k, -1] - 3.0, T_return_measured[k] + 6.0),
            n_nodes,
        )
        pump = float(_pump_head(alpha[k], config))
        q_hyd = np.sqrt(max((pump - outlet_head) / max(R_total, 1e-12), 1e-6))
        return max(flow_blend * q_proxy[k] + (1.0 - flow_blend) * q_hyd, 1e-4)

    previous_q = initialize_segment(0) if n_steps else 1e-4

    for k in range(n_steps):
        if trajectory_start[k] and k > 0:
            previous_q = initialize_segment(k)
        # The measured source temperature is a Dirichlet boundary at the same
        # timestamp. Applying it here avoids an artificial one-step lag.
        Ts[k, 0] = T_source[k]
        pump = float(_pump_head(alpha[k], config))
        q_ss_hyd = np.sqrt(max((pump - outlet_head) / max(R_total, 1e-12), 1e-6))
        q_ss = flow_blend * q_proxy[k] + (1.0 - flow_blend) * q_ss_hyd
        if valid_transition[k]:
            q_mean = previous_q + dt / max(tau_q, dt) * (q_ss - previous_q)
        else:
            q_mean = max(q_ss, 1e-4)
        q_mean = max(q_mean, 1e-4)
        q[k, :] = q_mean
        dq_dt = 0.0 if not valid_transition[k] else (q_mean - previous_q) / dt
        drop = R_total * q_mean**2
        transient = 0.05 * float(sys["wave_speed_mps"]) * dq_dt
        H[k, :] = outlet_head + drop * (1.0 - x / length) + transient * (1.0 - x / length)
        H[k, -1] = outlet_head
        pressure_drop_m[k] = H[k, 0] - H[k, -1]
        pump_boundary_residual_m[k] = pump - (outlet_head + pressure_drop_m[k])

        seg_supply_temp = 0.5 * (Ts[k, :-1] + Ts[k, 1:])
        seg_return_temp = 0.5 * (Tr[k, :-1] + Tr[k, 1:])
        Q_loss_supply[k] = np.sum(U * perimeter * (seg_supply_temp - Ta[k]) * dx)
        Q_loss_return[k] = np.sum(U * perimeter * (seg_return_temp - Ta[k]) * dx)
        Q_loss_segments[k] = U * perimeter * ((seg_supply_temp - Ta[k]) + (seg_return_temp - Ta[k])) * dx
        Q_loss[k] = Q_loss_supply[k] + Q_loss_return[k]
        delivered_heat_W[k] = rho * cp * q_mean * max(Ts[k, -1] - Tr[k, -1], 0.0)
        # Dynamic energy closure is evaluated after the full trajectory is known,
        # because it includes the finite-difference pipe-storage term.

        if k == n_steps - 1:
            break
        if trajectory_start[k + 1]:
            # The next retained sample starts a separate observed trajectory.
            # Do not numerically advance a 15-minute state through the gap.
            previous_q = q_mean
            continue

        v = q_mean / max(area, 1e-12)
        raw_cfl = max(v * dt / dx * velocity_factor, 0.0)
        n_substeps = max(1, int(np.ceil(raw_cfl / 0.8)))
        sub_dt = dt / n_substeps
        cfl = np.clip(v * sub_dt / dx * velocity_factor, 0.0, 0.8)
        loss_coeff = np.clip(sub_dt * K_loss, 0.0, 0.12)
        Ts_sub = Ts[k].copy()
        Tr_sub = Tr[k].copy()
        for _ in range(n_substeps):
            Ts_next = Ts_sub.copy()
            Tr_next = Tr_sub.copy()
            Ts_next[1:] = Ts_sub[1:] - cfl * (Ts_sub[1:] - Ts_sub[:-1]) - loss_coeff * (Ts_sub[1:] - Ta[k])
            Ts_next[0] = T_source[min(k + 1, n_steps - 1)]
            Tr_next[:-1] = Tr_sub[:-1] - cfl * (Tr_sub[:-1] - Tr_sub[1:]) - loss_coeff * (Tr_sub[:-1] - Ta[k])
            heat_delta = Q_load[k] / max(rho * cp * q_mean, 1e-9)
            # Consumer boundary: the prescribed load is extracted from the
            # circulating water. An empirical temperature offset must not be
            # inserted here because that would make delivered heat differ from
            # the same measured load used as the boundary condition.
            demand_return = Ts_sub[-1] - heat_delta
            Tr_next[-1] = np.clip(demand_return, 15.0, max(20.0, Ts_sub[-1] - 1.0))
            Ts_sub = Ts_next
            Tr_sub = Tr_next

        Ts[k + 1] = np.clip(Ts_sub, -20.0, 130.0)
        Ts[k + 1, 0] = T_source[k + 1]
        Tr[k + 1] = np.clip(Tr_sub, -20.0, 120.0)
        previous_q = q_mean

    pipe_energy_J = rho * cp * area * np.trapezoid(Ts + Tr, x=x, axis=1)
    storage_rate_W = np.full(n_steps, np.nan, dtype=float)
    for k in range(1, n_steps):
        if valid_transition[k]:
            storage_rate_W[k] = (pipe_energy_J[k] - pipe_energy_J[k - 1]) / dt
    source_heat_W = rho * cp * q[:, 0] * (Ts[:, 0] - Tr[:, 0])
    energy_balance_residual_W[:] = source_heat_W - delivered_heat_W - Q_loss - storage_rate_W

    thermal_delay_h = length / np.maximum(np.nanmean(q[:, 0]) / max(area, 1e-12), 1e-9) / 3600.0
    steady_state_consistency = float(np.nanmean(np.abs(np.diff(Ts[-min(n_steps, 12) :, -1])))) if n_steps > 2 else np.nan

    return {
        "time_s": time_s,
        "x_m": x,
        "Ts": Ts,
        "Tr": Tr,
        "H": H,
        "q": q,
        "Q_loss": Q_loss,
        "Q_loss_supply": Q_loss_supply,
        "Q_loss_return": Q_loss_return,
        "Q_loss_segments": Q_loss_segments,
        "Q_load": Q_load,
        "delivered_heat_W": delivered_heat_W,
        "energy_balance_residual_W": energy_balance_residual_W,
        "pressure_drop_m": pressure_drop_m,
        "pump_boundary_residual_m": pump_boundary_residual_m,
        "thermal_delay_h": thermal_delay_h,
        "steady_state_consistency_C_per_step": steady_state_consistency,
        "q_proxy": q_proxy,
        "flow_proxy_mode": flow_proxy_mode,
        "proxy_causality_version": boundary_conditions.get("proxy_causality_version", "causal_recomputed_in_simulator"),
        "alpha_provenance": boundary_conditions.get("alpha_provenance", "unknown"),
        "q_proxy_provenance": boundary_conditions.get("q_proxy_provenance", "causal proxy recomputed in simulator"),
        "alpha": alpha,
        "Ta": Ta,
        "T_source": T_source,
        "T_return_measured": T_return_measured,
        "trajectory_start": trajectory_start,
        "valid_transition": valid_transition,
        "return_temperature_assumed": bool(boundary_conditions.get("return_temperature_assumed", False)),
        "source_dataset": boundary_conditions.get("source_dataset", "unknown"),
        "params": {
            "heat_loss_U_W_m2K": U,
            "friction_factor": friction,
            "effective_velocity_factor": velocity_factor,
            "return_temperature_offset": 0.0,
            "flow_proxy_blend": flow_blend,
        },
    }


def save_model_verification(sim: dict[str, Any]) -> pd.DataFrame:
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures")
    residual = np.asarray(sim["energy_balance_residual_W"], dtype=float)
    rows = [
        {
            "source_dataset": sim.get("source_dataset", "unknown"),
            "mean_energy_balance_residual_kW": float(np.nanmean(residual) / 1000.0),
            "mean_abs_energy_balance_residual_kW": float(np.nanmean(np.abs(residual)) / 1000.0),
            "mean_heat_loss_kW": float(np.nanmean(sim["Q_loss"]) / 1000.0),
            "thermal_delay_h": float(sim["thermal_delay_h"]),
            "steady_state_consistency_C_per_step": float(sim["steady_state_consistency_C_per_step"]),
            "flow_proxy_mode": sim.get("flow_proxy_mode", "unknown"),
            "mean_abs_pump_boundary_residual_m": float(np.nanmean(np.abs(sim.get("pump_boundary_residual_m", np.nan)))),
        }
    ]
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "model_verification_summary.csv", index=False)

    if plt is None:
        return df
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    t_h = sim["time_s"] / 3600.0
    axes[0].plot(t_h, sim["Q_load"] / 1000.0, label="Boundary heat load", lw=1.2)
    axes[0].plot(t_h, sim["delivered_heat_W"] / 1000.0, label="Simulator delivered heat", lw=1.2)
    axes[0].set_ylabel("Heat (kW)")
    axes[0].legend()
    axes[1].plot(t_h, sim["Q_loss"] / 1000.0, color="#4d908e", lw=1.2)
    axes[1].set_ylabel("Pipe heat loss (kW)")
    axes[2].plot(t_h, residual / 1000.0, color="#d1495b", lw=1.2)
    axes[2].set_ylabel("Energy residual (kW)")
    axes[2].set_xlabel("Time (h)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Model verification: load, delivered heat, and energy residual")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "figures" / "fig3_model_verification.pdf", dpi=300)
    fig.savefig(PROJECT_ROOT / "figures" / "fig3_model_verification.png", dpi=300)
    plt.close(fig)
    return df


def run_discretization_study(boundary: dict[str, Any], config: dict[str, Any], params: dict[str, float] | None = None) -> pd.DataFrame:
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures")
    rows = []
    sims = {}
    for dx in [2000, 1000, 500]:
        cfg = deepcopy(config)
        cfg["system"]["dx_m"] = dx
        sim = simulate_thermo_hydraulics(boundary, cfg, params=params)
        sims[dx] = sim
        rows.append(
            {
                "dx_m": dx,
                "n_nodes": len(sim["x_m"]),
                "mean_outlet_supply_C": float(np.nanmean(sim["Ts"][:, -1])),
                "mean_return_source_C": float(np.nanmean(sim["Tr"][:, 0])),
                "mean_heat_loss_kW": float(np.nanmean(sim["Q_loss"]) / 1000.0),
                "mean_abs_energy_balance_residual_kW": float(np.nanmean(np.abs(sim["energy_balance_residual_W"])) / 1000.0),
            }
        )
    df = pd.DataFrame(rows)
    baseline = df[df["dx_m"].eq(1000)].iloc[0]
    df["outlet_supply_delta_vs_1000m_C"] = df["mean_outlet_supply_C"] - baseline["mean_outlet_supply_C"]
    df["heat_loss_delta_vs_1000m_kW"] = df["mean_heat_loss_kW"] - baseline["mean_heat_loss_kW"]
    df.to_csv(PROJECT_ROOT / "results" / "discretization_study.csv", index=False)

    if plt is None:
        return df
    fig, ax1 = plt.subplots(figsize=(7.0, 3.4))
    ax1.plot(df["dx_m"], df["mean_outlet_supply_C"], marker="o", label="Outlet supply temp.")
    ax1.set_xlabel("Grid spacing dx (m)")
    ax1.set_ylabel("Mean outlet supply (C)")
    ax1.invert_xaxis()
    ax2 = ax1.twinx()
    ax2.plot(df["dx_m"], df["mean_heat_loss_kW"], marker="s", color="#d1495b", label="Heat loss")
    ax2.set_ylabel("Mean heat loss (kW)")
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Discretization check")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    for name in ["fig3b_discretization_check", "fig14_discretization_check"]:
        fig.savefig(PROJECT_ROOT / "figures" / f"{name}.pdf", dpi=300)
        fig.savefig(PROJECT_ROOT / "figures" / f"{name}.png", dpi=300)
    plt.close(fig)
    return df
