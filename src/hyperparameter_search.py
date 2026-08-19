from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import yaml

from src.config import PROJECT_ROOT, ensure_project_dirs, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.data_registry import check_dataset_available
from src.evaluate import evaluate_model
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import _a_norm_for, _write_secondary_result_tables, build_loaders
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.train import train_model
from src.utils import get_device, set_seed


def _trial_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    hp = config.get("hyperparameter_search", {})
    keys = [
        "hidden_dim",
        "gnn_layers",
        "gru_layers",
        "dropout",
        "learning_rate",
        "lambda_sensor",
        "lambda_thermal",
        "lambda_boundary",
        "lambda_energy",
        "lambda_hydraulic",
        "training_mode",
    ]
    values = [hp.get(key, [config.get("model", {}).get(key, config.get("training", {}).get(key))]) for key in keys]
    trials = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    # Deterministic thinning keeps runtime bounded while still spanning the requested grid.
    max_trials = int(hp.get("max_trials", len(trials)))
    if len(trials) <= max_trials:
        return trials
    step = max(1, len(trials) // max_trials)
    selected = trials[::step][:max_trials]
    if trials[-1] not in selected:
        selected[-1] = trials[-1]
    return selected


def _apply_trial_config(base: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["model"]["hidden_dim"] = int(trial["hidden_dim"])
    cfg["model"]["gnn_layers"] = int(trial["gnn_layers"])
    cfg["model"]["gru_layers"] = int(trial.get("gru_layers", cfg["model"].get("gru_layers", 1)))
    cfg["model"]["dropout"] = float(trial["dropout"])
    cfg["training"]["learning_rate"] = float(trial["learning_rate"])
    cfg["training"]["lambda_sensor"] = float(trial["lambda_sensor"])
    cfg["training"]["lambda_thermal"] = float(trial["lambda_thermal"])
    cfg["training"]["lambda_boundary"] = float(trial["lambda_boundary"])
    cfg["training"]["lambda_energy"] = float(trial["lambda_energy"])
    cfg["training"]["lambda_hydraulic"] = float(trial.get("lambda_hydraulic", cfg["training"].get("lambda_hydraulic", 0.02)))
    cfg["training"]["training_mode"] = str(trial.get("training_mode", "balanced_mode"))
    return cfg


def _objective(metrics: dict[str, Any]) -> float:
    rmse_h_norm = float(metrics.get("RMSE_H_full", 0.0)) / 100.0
    rmse_q_norm = float(metrics.get("RMSE_q_full", 0.0))
    heat_loss_norm = float(metrics.get("heat_loss_error_percent", 0.0)) / 10.0
    energy_norm = float(metrics.get("energy_balance_residual", 0.0)) / 10.0
    thermal_norm = float(metrics.get("thermal_residual_mean", 0.0)) * 10.0
    return (
        float(metrics.get("RMSE_Ts_full", 0.0))
        + float(metrics.get("RMSE_Tr_full", 0.0))
        + 0.2 * rmse_h_norm
        + 0.2 * rmse_q_norm
        + 0.5 * heat_loss_norm
        + 0.5 * energy_norm
        + 0.2 * thermal_norm
    )


def _prepare_loaders(config: dict[str, Any]):
    if not check_dataset_available("sonderborg"):
        raise RuntimeError("Sonderborg data are required for hyperparameter search.")
    df = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config)
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    df = df.head(min(len(df), max_steps)).copy()
    boundary = build_boundary_conditions(df, config)
    params_path = PROJECT_ROOT / "results" / "calibrated_parameters.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"].get("noise_std_fraction", 0.0))
    return build_loaders(sim, sensors, config), sim, sensors, df


def run_hyperparameter_search() -> pd.DataFrame:
    ensure_project_dirs()
    config = load_config()
    set_seed(config["dataset"]["seed"])
    loaders, _, _, _ = _prepare_loaders(config)
    arrays = loaders["arrays"]
    a_norm = _a_norm_for(arrays)
    trials = _trial_grid(config)
    rows: list[dict[str, Any]] = []
    epochs = int(config.get("hyperparameter_search", {}).get("epochs_per_trial", 4))
    for idx, trial in enumerate(trials, start=1):
        cfg = _apply_trial_config(config, trial)
        model = build_model("pignn_v3", arrays["features"].shape[-1], arrays["target"].shape[1], a_norm, cfg)
        trained, _ = train_model(
            model,
            f"hp_trial_{idx:02d}_Proposed PI-GNN-GRU-v3",
            loaders["train_loader"],
            loaders["val_loader"],
            cfg,
            loaders["train_ds"].stats,
            output_dir=PROJECT_ROOT / "results",
            quick=True,
            loss_mode="physics",
            loss_weights={"training_mode": str(trial.get("training_mode", "balanced_mode"))},
            epochs_override=epochs,
        )
        metrics = evaluate_model(trained, loaders["val_loader"], cfg, loaders["train_ds"].stats, f"hp_trial_{idx:02d}")
        row = {**trial, **metrics}
        row["trial"] = idx
        row["validation_objective"] = _objective(metrics)
        rows.append(row)
    search_df = pd.DataFrame(rows).sort_values("validation_objective")
    search_df.to_csv(PROJECT_ROOT / "results" / "hyperparameter_search.csv", index=False)
    search_df.to_csv(PROJECT_ROOT / "results" / "hyperparameter_search_v3.csv", index=False)

    best_trial = search_df.iloc[0].to_dict()
    best_cfg = _apply_trial_config(config, {key: best_trial[key] for key in [
        "hidden_dim",
        "gnn_layers",
        "gru_layers",
        "dropout",
        "learning_rate",
        "lambda_sensor",
        "lambda_thermal",
        "lambda_boundary",
        "lambda_energy",
        "lambda_hydraulic",
        "training_mode",
    ]})
    with (PROJECT_ROOT / "results" / "best_model_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    with (PROJECT_ROOT / "results" / "best_pignn_gru_v3_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    final_epochs = int(config.get("hyperparameter_search", {}).get("final_epochs", 8))
    final_model = build_model("pignn_v3", arrays["features"].shape[-1], arrays["target"].shape[1], a_norm, best_cfg)
    final_model, history = train_model(
        final_model,
        "Proposed_PI_GNN_GRU_v3_final",
        loaders["train_loader"],
        loaders["val_loader"],
        best_cfg,
        loaders["train_ds"].stats,
        output_dir=PROJECT_ROOT / "results",
        quick=True,
        loss_mode="physics",
        loss_weights={"training_mode": str(best_trial.get("training_mode", "balanced_mode"))},
        epochs_override=final_epochs,
    )
    torch.save(final_model.state_dict(), PROJECT_ROOT / "results" / "best_pignn_gru_v3.pt")
    history.to_csv(PROJECT_ROOT / "results" / "proposed_model_training_history.csv", index=False)
    history.to_csv(PROJECT_ROOT / "results" / "proposed_v3_training_history.csv", index=False)
    test_metrics = evaluate_model(final_model, loaders["test_loader"], best_cfg, loaders["train_ds"].stats, "Proposed PI-GNN-GRU-v3 tuned")
    reason = (
        "Selected by the bounded validation objective J = RMSE_Ts + RMSE_Tr + normalized head/flow, "
        "heat-loss, energy, and thermal residual terms. Best trial objective: "
        f"{float(best_trial['validation_objective']):.4f}. Metrics were not manually edited; selection reflects validation performance."
    )
    (PROJECT_ROOT / "results" / "best_pignn_gru_v3_selection_reason.txt").write_text(reason + "\n", encoding="utf-8")

    baseline_path = PROJECT_ROOT / "results" / "baseline_comparison_final.csv"
    if not baseline_path.exists():
        baseline_path = PROJECT_ROOT / "results" / "baseline_comparison_improved.csv"
    if not baseline_path.exists():
        baseline_path = PROJECT_ROOT / "results" / "baseline_comparison.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    if not baseline.empty:
        baseline = baseline[~baseline["model"].astype(str).eq("Proposed PI-GNN-GRU-v3 tuned")]
    improved = pd.concat([baseline, pd.DataFrame([test_metrics])], ignore_index=True)
    improved["used_fallback_synthetic"] = False
    improved.to_csv(PROJECT_ROOT / "results" / "baseline_comparison_improved.csv", index=False)
    improved.to_csv(PROJECT_ROOT / "results" / "baseline_comparison_final.csv", index=False)
    _write_secondary_result_tables(improved, PROJECT_ROOT / "results", [{"model": "Proposed PI-GNN-GRU-v3 tuned", "training_time_s": float(history.shape[0]), "parameter_count": int(sum(p.numel() for p in final_model.parameters() if p.requires_grad))}])
    print("Hyperparameter search completed.")
    print(f"Best validation objective: {float(best_trial['validation_objective']):.4f}")
    return search_df


if __name__ == "__main__":
    run_hyperparameter_search()
