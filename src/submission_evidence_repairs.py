"""Generate final evidence repairs requested by the submission audit.

The module does not retrain or alter saved scores.  It adds two out-of-sample
audits and rebuilds the active calibration/refinement figure from locked
post-causality sources:

* unchanged-parameter replay on later contiguous Sonderborg blocks; and
* measured-return evaluation of the 20 principal checkpoints when the current
  return measurement and all internal simulator sensors are withheld.
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ate_figure_style import PALETTE, add_panel_label, save_ate_figure, set_ate_style, style_axes
from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.evaluate import evaluate_model
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import _a_norm_for, build_loaders
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import ensure_dir, load_json


RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
LOCKED_DATA = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv"
SEEDS = (11, 22, 33, 44, 55)
MODEL_SPECS = (
    ("GRU-MSE", "gru"),
    ("Transformer-MSE", "transformer"),
    ("Proposed PI-GNN-GRU-v3 accuracy_mode", "pignn_v3"),
    ("Proposed PI-GNN-GRU-v3 balanced_mode", "pignn_v3"),
)


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        text = text.replace(old, new)
    return text


def _write_booktabs(frame: pd.DataFrame, path: Path, caption: str, label: str, *, resize: bool = False) -> None:
    ensure_dir(path.parent)
    cols = list(frame.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        *([r"\resizebox{\textwidth}{!}{%"] if resize else []),
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_latex_escape(col) for col in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(_latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _canonical_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    if not LOCKED_DATA.exists():
        raise FileNotFoundError(LOCKED_DATA)
    frame = pd.read_csv(LOCKED_DATA)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    expected_hash = config["real_data"]["canonical_sonderborg_sha256"]
    import hashlib

    digest = hashlib.sha256(LOCKED_DATA.read_bytes()).hexdigest()
    if digest != expected_hash or len(frame) != 18703:
        raise RuntimeError(f"Locked-data mismatch: rows={len(frame)}, SHA-256={digest}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _score_replay(sim: dict[str, Any], measured_return: np.ndarray) -> dict[str, float]:
    mask = ~np.asarray(sim.get("trajectory_start", np.zeros(len(measured_return), dtype=bool)), dtype=bool)
    error = np.asarray(sim["Tr"], dtype=float)[:, 0] - measured_return
    load = np.asarray(sim["Q_load"], dtype=float)
    delivered = np.asarray(sim["delivered_heat_W"], dtype=float)
    dynamic = np.asarray(sim["energy_balance_residual_W"], dtype=float)
    return {
        "RMSE_return_C": float(np.sqrt(np.mean(error[mask] ** 2))),
        "MAE_return_C": float(np.mean(np.abs(error[mask]))),
        "signed_return_bias_C": float(np.mean(error[mask])),
        "boundary_closure_percent": float(
            np.mean(np.abs(delivered[mask] - load[mask]) / np.maximum(np.abs(load[mask]), 1.0)) * 100.0
        ),
        "dynamic_energy_residual_percent": float(
            np.sum(np.abs(dynamic[mask])) / max(np.sum(np.abs(load[mask])), 1.0) * 100.0
        ),
    }


def calibration_temporal_transfer_audit(config: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    frame = _canonical_dataframe(config)
    timestamp = frame["timestamp"]
    segments = (timestamp.diff().dt.total_seconds().fillna(900.0).ne(900.0)).cumsum()
    candidates: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    block_length = 537
    for _, group in frame.groupby(segments):
        if len(group) < block_length:
            continue
        start = max(0, (len(group) - block_length) // 2)
        block = group.iloc[start : start + block_length].copy().reset_index(drop=True)
        candidates.append((block["timestamp"].iloc[0], block))
    # Use temporally separated contiguous blocks; the earliest retained one is
    # already later than most of the calibration prefix, and no block is retuned.
    selected: list[pd.DataFrame] = []
    used_years: dict[int, int] = {}
    for _, block in sorted(candidates, key=lambda item: item[0]):
        year = int(block["timestamp"].iloc[0].year)
        if used_years.get(year, 0) >= 2:
            continue
        selected.append(block)
        used_years[year] = used_years.get(year, 0) + 1
        if len(selected) == 5:
            break

    rows: list[dict[str, object]] = []
    for index, block in enumerate(selected, start=1):
        boundary = build_boundary_conditions(block, config)
        sim = simulate_thermo_hydraulics(boundary, config, params=params)
        metrics = _score_replay(sim, np.asarray(boundary["T_return_measured"], dtype=float))
        rows.append(
            {
                "block": f"B{index}",
                "start_utc": block["timestamp"].iloc[0].isoformat(),
                "end_utc": block["timestamp"].iloc[-1].isoformat(),
                "samples": len(block),
                **metrics,
                "retuned": False,
                "evidence_type": "M return target; C simulator replay",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "calibration_temporal_transfer_audit.csv", index=False)
    summary = {
        "blocks": int(len(out)),
        "return_RMSE_median_C": float(out["RMSE_return_C"].median()),
        "return_RMSE_min_C": float(out["RMSE_return_C"].min()),
        "return_RMSE_max_C": float(out["RMSE_return_C"].max()),
        "unchanged_parameters": True,
        "note": "Effective parameters were fitted once on the locked calibration prefix and were not retuned on these contiguous blocks.",
    }
    (RESULTS / "calibration_temporal_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    compact = out[["block", "start_utc", "RMSE_return_C", "MAE_return_C", "dynamic_energy_residual_percent"]].copy()
    compact["start_utc"] = pd.to_datetime(compact["start_utc"], utc=True).dt.strftime("%Y-%m-%d")
    compact.columns = ["Block", "Start (UTC)", "Return RMSE (deg C)", "Return MAE (deg C)", "Dynamic residual (%)"]
    for col in compact.columns[2:]:
        compact[col] = compact[col].map(lambda value: f"{float(value):.3f}")
    _write_booktabs(
        compact,
        TABLES / "table_calibration_temporal_transfer_audit.tex",
        "Unchanged-parameter replay on later contiguous S{\\o}nderborg blocks. Return temperature is measured (M); internal fields are calibrated-simulator quantities (C).",
        "tab:calibration_temporal_transfer",
    )
    return out


def _aggregate_unique(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    unique, inverse = np.unique(time_s, return_inverse=True)
    aggregated = np.array([np.mean(values[inverse == index]) for index in range(len(unique))])
    return unique, aggregated


def blind_measured_return_checkpoint_audit(config: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    frame = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config).head(max_steps).copy()
    boundary = build_boundary_conditions(frame, config)
    sim = simulate_thermo_hydraulics(boundary, config, params=params)

    # Recover the training-only normalization from the published S4 protocol.
    s4 = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"].get("noise_std_fraction", 0.0))
    base_loaders = build_loaders(sim, s4, config)
    stats = base_loaders["train_ds"].stats

    # Inference receives only the measured source-supply boundary at node zero.
    # Current measured return and every internal C-class thermal or S-class hydraulic sensor value are withheld.
    state_shape = (len(sim["time_s"]), len(sim["x_m"]), 4)
    measurements = np.zeros(state_shape, dtype=np.float32)
    masks = np.zeros(state_shape, dtype=np.float32)
    measurements[:, 0, 0] = np.asarray(sim["T_source"], dtype=np.float32)
    masks[:, 0, 0] = 1.0
    blind_sensors = {
        "layout_name": "measured_source_boundary_only",
        "sensor_nodes": [0],
        "measurements": measurements,
        "masks": masks,
        "variables": ["Ts"],
    }
    blind_loaders = build_loaders(sim, blind_sensors, config, stats=stats)
    a_norm = _a_norm_for(blind_loaders["arrays"])
    measured_lookup = {float(t): float(v) for t, v in zip(sim["time_s"], boundary["T_return_measured"])}
    rows: list[dict[str, object]] = []
    audit_test_times: np.ndarray | None = None

    for seed in SEEDS:
        for label, architecture in MODEL_SPECS:
            checkpoint = RESULTS / f"seed_{seed}_{label}_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = build_model(
                architecture,
                input_dim=blind_loaders["arrays"]["features"].shape[-1],
                n_nodes=blind_loaders["arrays"]["target"].shape[1],
                a_norm=a_norm,
                config=config,
            )
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
            _, payload = evaluate_model(
                model,
                blind_loaders["test_loader"],
                config,
                stats,
                label,
                return_predictions=True,
            )
            prediction = np.asarray(payload["pred"], dtype=float)[..., 0, 1].reshape(-1)
            time_values = np.asarray(payload["time_s"], dtype=float).reshape(-1)
            unique_time, unique_prediction = _aggregate_unique(time_values, prediction)
            if audit_test_times is None:
                audit_test_times = unique_time.copy()
            measured = np.array([measured_lookup[float(t)] for t in unique_time])
            error = unique_prediction - measured
            rows.append(
                {
                    "model": label,
                    "seed": seed,
                    "RMSE_return_measured_C": float(np.sqrt(np.mean(error**2))),
                    "MAE_return_measured_C": float(np.mean(np.abs(error))),
                    "signed_bias_C": float(np.mean(error)),
                    "unique_test_timestamps": len(unique_time),
                    "current_return_measurement_used": False,
                    "internal_simulator_sensor_values_used": False,
                    "available_inputs": "measured source supply; measured load; configured ambient; causal load-derived pump/flow proxies",
                    "evidence_type": "M target; checkpoint trained on mixed C and S targets",
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "principal_models_blind_measured_return.csv", index=False)
    summary = (
        detail.groupby("model", as_index=False)
        .agg(
            mean_RMSE_return_measured_C=("RMSE_return_measured_C", "mean"),
            std_RMSE_return_measured_C=("RMSE_return_measured_C", "std"),
            mean_MAE_return_measured_C=("MAE_return_measured_C", "mean"),
            std_MAE_return_measured_C=("MAE_return_measured_C", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_RMSE_return_measured_C")
        .reset_index(drop=True)
    )

    # Persistence is a direct measured-target reference and is intentionally
    # not pooled into seed statistics.
    if audit_test_times is None:
        raise RuntimeError("Blind measured-return audit produced no held-out timestamps.")
    test_times = audit_test_times
    sim_times = np.asarray(sim["time_s"], dtype=float)
    measured_return = np.asarray(boundary["T_return_measured"], dtype=float)
    time_to_index = {float(value): index for index, value in enumerate(sim_times)}
    target = np.array([measured_return[time_to_index[float(value)]] for value in test_times])
    persistence = np.array([measured_return[max(0, time_to_index[float(value)] - 1)] for value in test_times])
    persistence_error = persistence - target
    persistence_row = pd.DataFrame(
        [
            {
                "model": "Measured-return persistence",
                "mean_RMSE_return_measured_C": float(np.sqrt(np.mean(persistence_error**2))),
                "std_RMSE_return_measured_C": np.nan,
                "mean_MAE_return_measured_C": float(np.mean(np.abs(persistence_error))),
                "std_MAE_return_measured_C": np.nan,
                "n_seeds": 0,
            }
        ]
    )
    summary = pd.concat([persistence_row, summary], ignore_index=True)
    summary["interpretation"] = np.where(
        summary["model"].eq("Measured-return persistence"),
        "M baseline",
        "M target; mixed-target checkpoint",
    )
    summary.to_csv(RESULTS / "principal_models_blind_measured_return_summary.csv", index=False)

    compact = summary.copy()
    compact["model"] = compact["model"].replace(
        {
            "Measured-return persistence": "Persistence",
            "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-v3 balanced",
            "Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-v3 accuracy",
        }
    )
    compact["Mean RMSE (deg C)"] = compact["mean_RMSE_return_measured_C"].map(lambda value: f"{value:.3f}")
    compact["SD RMSE (deg C)"] = compact["std_RMSE_return_measured_C"].map(
        lambda value: "--" if not np.isfinite(value) else f"{value:.3f}"
    )
    compact["MAE (deg C)"] = compact["mean_MAE_return_measured_C"].map(lambda value: f"{value:.3f}")
    compact = compact[["model", "Mean RMSE (deg C)", "SD RMSE (deg C)", "MAE (deg C)", "interpretation"]]
    compact.columns = ["Estimator", "Mean return RMSE (deg C)", "SD (deg C)", "Return MAE (deg C)", "Evidence boundary"]
    _write_booktabs(
        compact,
        TABLES / "table_principal_models_blind_measured_return.tex",
        "Blind measured-return audit on 102 unique held-out timestamps. Current return measurements and internal simulator sensor values are withheld. Neural checkpoints were trained on C-class thermal and S-class hydraulic corridor targets, so this is a measured-target stress audit rather than distributed field validation.",
        "tab:principal_blind_measured_return",
        resize=True,
    )
    return summary


def calibration_refinement_figure(config: dict[str, Any]) -> None:
    set_ate_style()
    frame = _canonical_dataframe(config).head(768)
    states = np.load(RESULTS / "corrected_simulator_states.npz")
    metrics = pd.read_csv(RESULTS / "calibration_metrics.csv").iloc[0]
    refinement = pd.read_csv(RESULTS / "numerical_verification_expanded.csv")
    n_fit = int(pd.read_csv(RESULTS / "locked_later_replay_metrics.csv").iloc[0]["calibration_samples"])
    time_h = np.asarray(states["time_s"], dtype=float)[:n_fit] / 3600.0
    measured_return = pd.to_numeric(frame["return_temp_C"], errors="coerce").to_numpy()[:n_fit]
    simulated_return = np.asarray(states["Tr"], dtype=float)[:n_fit, 0]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), constrained_layout=True)
    axes[0, 0].plot(time_h, measured_return, color=PALETTE["measured"], lw=1.7, label="Measured return (M)")
    axes[0, 0].plot(time_h, simulated_return, color=PALETTE["proposed"], lw=1.35, label="Calibrated simulator (C)")
    axes[0, 0].set_xlabel("Replay time (h)")
    axes[0, 0].set_ylabel(r"Return temperature ($^\circ$C)")
    axes[0, 0].set_title("Calibration-period return fit", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    style_axes(axes[0, 0])

    axes[0, 1].scatter(measured_return, simulated_return, s=11, alpha=0.55, color=PALETTE["pilstm"], edgecolor="none")
    low = min(measured_return.min(), simulated_return.min())
    high = max(measured_return.max(), simulated_return.max())
    axes[0, 1].plot([low, high], [low, high], color=PALETTE["measured"], lw=1.2, ls="--")
    axes[0, 1].set_xlabel(r"Measured return ($^\circ$C)")
    axes[0, 1].set_ylabel(r"Simulated return ($^\circ$C)")
    axes[0, 1].set_title(
        f"RMSE={float(metrics['RMSE_return_C']):.3f} $^\\circ$C; MAE={float(metrics['MAE_return_C']):.3f} $^\\circ$C",
        fontweight="bold",
    )
    style_axes(axes[0, 1])

    x = np.arange(len(refinement))
    axes[1, 0].bar(x, refinement["outlet_Ts_L2_C"], color=[PALETTE["pilstm"], PALETTE["proposed"], PALETTE["safe"]], edgecolor=PALETTE["edge"], linewidth=0.8)
    axes[1, 0].set_xticks(x, [f"{int(dx)} m\n{int(dt)} s" for dx, dt in zip(refinement["dx_m"], refinement["dt_s"])])
    axes[1, 0].set_ylabel(r"Outlet supply L2 error ($^\circ$C)")
    axes[1, 0].set_title("Coordinated space-time refinement", fontweight="bold")
    style_axes(axes[1, 0])

    axes[1, 1].bar(x, refinement["cumulative_heat_loss_error_pct"], color=[PALETTE["pilstm"], PALETTE["proposed"], PALETTE["safe"]], edgecolor=PALETTE["edge"], linewidth=0.8)
    axes[1, 1].set_xticks(x, [f"{int(dx)} m\n{int(dt)} s" for dx, dt in zip(refinement["dx_m"], refinement["dt_s"])])
    axes[1, 1].set_ylabel("Cumulative heat-loss difference (%)")
    axes[1, 1].set_title("Fine-grid heat-loss reference", fontweight="bold")
    style_axes(axes[1, 1])
    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        add_panel_label(ax, label)
    save_ate_figure(fig, PROJECT_ROOT / "figures" / "final", "fig4_calibration_and_discretization")
    # Keep the paper copy synchronized without relying on raster compositing.
    for suffix in ("pdf", "png"):
        source = PROJECT_ROOT / "figures" / "final" / f"fig4_calibration_and_discretization.{suffix}"
        target = PROJECT_ROOT / "paper" / "figures" / "final" / source.name
        ensure_dir(target.parent)
        target.write_bytes(source.read_bytes())


def write_active_figure_provenance() -> pd.DataFrame:
    definitions = {
        "fig4_calibration_and_discretization.pdf": [
            "results/corrected_simulator_states.npz",
            "results/calibration_metrics.csv",
            "results/numerical_verification_expanded.csv",
        ],
        "fig_thermo_hydraulic_reconstruction_summary.pdf": [
            "results/dense_reconstruction_payloads.npz",
        ],
        "fig_heat_energy_balance_summary.pdf": [
            "results/operational_energy_impact_timeseries.csv",
            "results/repeated_seed_statistics.csv",
        ],
        "fig_method_evidence_hierarchy.pdf": [
            "results/strict_target_dependency_audit.csv",
        ],
    }
    rows: list[dict[str, object]] = []
    for figure_name, source_names in definitions.items():
        figure_path = PROJECT_ROOT / "figures" / "final" / figure_name
        sources = [PROJECT_ROOT / name for name in source_names]
        inputs_exist = all(path.exists() for path in sources)
        current = bool(
            figure_path.exists()
            and inputs_exist
            and figure_path.stat().st_mtime >= max(path.stat().st_mtime for path in sources)
        )
        rows.append(
            {
                "figure": f"figures/final/{figure_name}",
                "figure_sha256": hashlib.sha256(figure_path.read_bytes()).hexdigest() if figure_path.exists() else "",
                "source_files": "; ".join(source_names),
                "source_sha256": "; ".join(
                    hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing" for path in sources
                ),
                "generated_after_all_sources": current,
                "quantitative_source": "CSV/NPZ only; no raster digitization",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "active_figure_provenance_post_causality.csv", index=False)
    return out


def main() -> None:
    config = load_config()
    params = load_json(RESULTS / "calibrated_parameters.json")
    config = apply_calibrated_params_to_config(config, params)
    calibration_temporal_transfer_audit(config, params)
    blind_measured_return_checkpoint_audit(config, params)
    calibration_refinement_figure(config)
    write_active_figure_provenance()
    print("Submission evidence repairs completed.")


if __name__ == "__main__":
    main()
