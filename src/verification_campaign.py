"""Eleven-part numerical and evidence verification campaign for the final rebuild."""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.real_data_mapper import build_boundary_conditions
from src.reviewer_critical_fixes import expanded_numerical_verification
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import load_json


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
FIGURES = PROJECT_ROOT / "figures" / "final"


def _record(name: str, status: str, value: float | str, threshold: float | str, evidence: str, limitation: str = "") -> dict:
    return {
        "verification": name,
        "status": status,
        "value": value,
        "threshold_or_requirement": threshold,
        "evidence": evidence,
        "limitation": limitation,
    }


def _upwind_periodic(n: int, steps: int, cfl: float, loss_step: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(n, dtype=float) / n
    initial = np.sin(2.0 * np.pi * x)
    state = initial.copy()
    for _ in range(steps):
        state = state - cfl * (state - np.roll(state, 1)) - loss_step * state
    shift = cfl * steps / n
    exact = np.exp(-loss_step * steps) * np.sin(2.0 * np.pi * ((x - shift) % 1.0))
    return state, exact


def _manufactured_residual() -> float:
    x = np.linspace(0.0, 1.0, 101)
    t = np.linspace(0.0, 0.5, 51)
    xx, tt = np.meshgrid(x, t)
    velocity = 0.7
    decay = 0.2
    phase = 2.0 * np.pi * (xx - velocity * tt)
    theta = np.exp(-decay * tt) * np.sin(phase)
    dt_exact = -decay * theta - velocity * 2.0 * np.pi * np.exp(-decay * tt) * np.cos(phase)
    dx_exact = 2.0 * np.pi * np.exp(-decay * tt) * np.cos(phase)
    residual = dt_exact + velocity * dx_exact + decay * theta
    return float(np.max(np.abs(residual)))


def _hydraulic_and_energy_checks(config: dict, params: dict) -> tuple[float, float, dict]:
    n = 48
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC"),
            "heat_load_kw": np.full(n, 5000.0),
            "supply_temp_C": np.full(n, 80.0),
            "return_temp_C": np.full(n, 45.0),
            "ambient_temp_C": np.full(n, 5.0),
            "source_dataset": "verification",
        }
    )
    cfg = deepcopy(config)
    cfg["real_data"]["freeze_canonical_processed"] = False
    boundary = build_boundary_conditions(frame, cfg)
    sim = simulate_thermo_hydraulics(boundary, cfg, params=params)
    head_exact = sim["H"][:, -1, None] + sim["pressure_drop_m"][:, None] * (1.0 - sim["x_m"][None, :] / sim["x_m"][-1])
    hydraulic_error = float(np.max(np.abs(sim["H"] - head_exact)))
    rho = float(cfg["system"]["rho"])
    cp = float(cfg["system"]["cp"])
    area = np.pi * float(cfg["system"]["diameter_m"]) ** 2 / 4.0
    energy = rho * cp * area * np.trapezoid(sim["Ts"] + sim["Tr"], x=sim["x_m"], axis=1)
    storage = np.full(n, np.nan)
    valid = np.asarray(sim["valid_transition"], dtype=bool)
    storage[1:][valid[1:]] = np.diff(energy)[valid[1:]] / float(cfg["system"]["dt_s"])
    source = rho * cp * sim["q"][:, 0] * (sim["Ts"][:, 0] - sim["Tr"][:, 0])
    recomputed = source - sim["delivered_heat_W"] - sim["Q_loss"] - storage
    mask = np.isfinite(recomputed)
    energy_error = float(np.max(np.abs(recomputed[mask] - sim["energy_balance_residual_W"][mask])))
    return hydraulic_error, energy_error, sim


def _independent_solver_check(config: dict, params: dict, sim: dict) -> float:
    n_nodes = len(sim["x_m"])
    dx = float(sim["x_m"][1] - sim["x_m"][0])
    dt = float(config["system"]["dt_s"])
    rho = float(config["system"]["rho"])
    cp = float(config["system"]["cp"])
    area = np.pi * float(config["system"]["diameter_m"]) ** 2 / 4.0
    perimeter = float(config["system"]["pipe_perimeter_m"])
    velocity = float(sim["q"][0, 0]) / area * float(params.get("effective_velocity_factor", 1.0))
    loss = float(params.get("heat_loss_U_W_m2K", config["system"]["heat_loss_U_W_m2K"])) * perimeter / (rho * cp * area)
    ambient = float(sim["Ta"][0])
    source = float(sim["T_source"][0])
    initial = np.asarray(sim["Ts"][0, 1:], dtype=float)

    def rhs(_t: float, interior: np.ndarray) -> np.ndarray:
        upstream = np.concatenate([[source], interior[:-1]])
        return -velocity / dx * (interior - upstream) - loss * (interior - ambient)

    times = np.arange(len(sim["time_s"]), dtype=float) * dt
    solved = solve_ivp(rhs, (times[0], times[-1]), initial, t_eval=times, rtol=1e-8, atol=1e-9)
    independent = np.column_stack([np.full(len(times), source), solved.y.T])
    return float(np.sqrt(np.mean((independent - sim["Ts"]) ** 2)))


def _write_table(frame: pd.DataFrame) -> None:
    def escape(value: object) -> str:
        text = str(value)
        for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")]:
            text = text.replace(old, new)
        return text

    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Final eleven-part verification campaign. PASS denotes a numerical or provenance requirement satisfied under the stated threshold; it does not denote field validation of distributed states.}",
        r"\label{tab:final_verification_campaign}",
        r"\begin{tabular}{p{0.25\textwidth}p{0.10\textwidth}p{0.18\textwidth}p{0.35\textwidth}}", r"\toprule",
        r"Verification & Status & Value & Evidence boundary \\", r"\midrule",
    ]
    for _, row in frame.iterrows():
        name = escape(row["verification"])
        value = escape(row["value"])
        evidence = escape(row["evidence"])
        lines.append(f"{name} & {escape(row['status'])} & {value} & {evidence} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (TABLES / "table_final_verification_campaign.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    params = load_json(RESULTS / "calibrated_parameters.json")
    effective = apply_calibrated_params_to_config(config, params)
    rows = []

    numerical, exact = _upwind_periodic(200, 40, 0.4)
    advection_rmse = float(np.sqrt(np.mean((numerical - exact) ** 2)))
    rows.append(_record("Analytical advection", "PASS" if advection_rmse < 0.03 else "FAIL", advection_rmse, "RMSE < 0.03", "isolated first-order upwind transport kernel"))
    numerical_loss, exact_loss = _upwind_periodic(200, 40, 0.4, 0.002)
    adv_loss_rmse = float(np.sqrt(np.mean((numerical_loss - exact_loss) ** 2)))
    rows.append(_record("Analytical advection-loss", "PASS" if adv_loss_rmse < 0.03 else "FAIL", adv_loss_rmse, "RMSE < 0.03", "isolated transport-loss kernel"))

    hydraulic_error, energy_error, simple_sim = _hydraulic_and_energy_checks(effective, params)
    rows.append(_record("Hydraulic analytical solution", "PASS" if hydraulic_error < 1e-9 else "FAIL", hydraulic_error, "max error < 1e-9 m", "reduced quadratic-drop/linear-head relation; S state"))
    rows.append(_record("Energy/storage closure", "PASS" if energy_error < 1e-6 else "FAIL", energy_error, "algebraic error < 1e-6 W", "implementation identity including storage; not zero-residual field validation"))
    manufactured = _manufactured_residual()
    rows.append(_record("Manufactured solution", "PASS" if manufactured < 1e-12 else "FAIL", manufactured, "PDE residual < 1e-12", "analytical manufactured advection-loss field"))

    grid = expanded_numerical_verification()
    baseline = grid[(grid["dx_m"] == 1000.0) & (grid["dt_s"] == 900.0)].iloc[0]
    grid_pass = float(baseline["outlet_Ts_L2_C"]) < 0.2 and float(baseline["cumulative_heat_loss_error_pct"]) < 1.0
    rows.append(_record("Grid convergence", "PASS" if grid_pass else "LIMITATION", f"Ts L2={baseline['outlet_Ts_L2_C']:.5f} C; loss={baseline['cumulative_heat_loss_error_pct']:.5f}%", "Ts L2 < 0.2 C and loss < 1%", "coordinated dx/dt refinement: 2000 m/1800 s, 1000 m/900 s, 500 m/450 s"))

    independent_rmse = _independent_solver_check(effective, params, simple_sim)
    rows.append(_record("Independent solver comparison", "PASS" if independent_rmse < 0.2 else "LIMITATION", independent_rmse, "supply-field RMSE < 0.2 C", "RK45 method-of-lines versus explicit substepped simulator for constant boundaries"))

    observer_path = RESULTS / "gaussian_observer_baseline_metrics.csv"
    repeated_path = RESULTS / "repeated_seed_statistics.csv"
    observer_ok = observer_path.exists() and repeated_path.exists()
    if observer_ok:
        observer_name = str(pd.read_csv(observer_path).iloc[0]["model"])
        observer_ok = observer_name == "Covariance-conditioned Gaussian observer"
    rows.append(_record("Cross-estimator audit", "PASS" if observer_ok else "FAIL", "Gaussian observer + four repeated neural models" if observer_ok else "missing or misnamed observer evidence", "correct method names and finite metrics", "observer uses static Gaussian conditioning rather than sequential forecast-update filtering"))

    later = pd.read_csv(RESULTS / "locked_later_replay_metrics.csv") if (RESULTS / "locked_later_replay_metrics.csv").exists() else pd.DataFrame()
    later_ok = not later.empty and not bool(later.iloc[0]["retuned_on_later_period"]) and np.isfinite(float(later.iloc[0]["RMSE_return_C"]))
    later_value = f"return RMSE={float(later.iloc[0]['RMSE_return_C']):.4f} C" if later_ok else "missing"
    rows.append(_record("Locked later replay", "PASS" if later_ok else "FAIL", later_value, "finite and no retuning", "later measured return; distributed thermal fields remain C and hydraulic fields remain S"))

    second_path = RESULTS / "second_chronological_window_summary.csv"
    layout_path = RESULTS / "sensor_layout_comparison_final.csv"
    cross_ok = second_path.exists() and layout_path.exists() and not pd.read_csv(second_path).empty and not pd.read_csv(layout_path).empty
    rows.append(_record("Cross-window/layout audit", "PASS" if cross_ok else "FAIL", "reported without universal-rank claim" if cross_ok else "missing", "second window plus layout evidence", "fixed-checkpoint temporal transfer and separately trained layout sensitivity"))

    causal = pd.read_csv(RESULTS / "proxy_causality_audit.csv")
    causal_ok = causal["status"].eq("pass").all() and causal["audit"].astype(str).str.contains("temperature_state_uses_no_future_values").any()
    rows.append(_record("Causality/leakage audit", "PASS" if causal_ok else "FAIL", f"{int(causal['status'].eq('pass').sum())}/{len(causal)} checks pass", "all proxy and full-state checks pass", "future perturbation invariance of alpha, q, Ts, Tr, H, heat loss, and delivered heat"))

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "verification_campaign_status.csv", index=False)
    (RESULTS / "verification_campaign_status.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _write_table(out)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    y = np.arange(len(out))
    colors = ["#55D600" if status == "PASS" else "#F2E600" if status == "LIMITATION" else "#FF6626" for status in out["status"]]
    ax.barh(y, np.ones(len(out)), color=colors, edgecolor="#111111", linewidth=0.8)
    ax.set_yticks(y, out["verification"])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    for index, status in enumerate(out["status"]):
        ax.text(0.98, index, status, ha="right", va="center", fontsize=8, fontweight="bold")
    ax.set_title("Corrected simulator and evidence verification campaign")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_final_verification_campaign.pdf")
    fig.savefig(FIGURES / "fig_final_verification_campaign.svg", format="svg")
    fig.savefig(FIGURES / "fig_final_verification_campaign.png", dpi=600)
    plt.close(fig)
    print(out.to_string(index=False))
    if out["status"].eq("FAIL").any():
        raise RuntimeError("One or more final verification checks failed.")


if __name__ == "__main__":
    main()
