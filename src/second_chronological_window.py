"""Fixed-checkpoint transfer to a disjoint later Sonderborg window."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.evaluate import evaluate_model
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.repeated_seed_statistics import CORE_MODELS, _write_table
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import _a_norm_for, build_loaders
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import load_json


WINDOW_LENGTH = 768
SECOND_WINDOW_START = 768
SEEDS = [11, 22, 33, 44, 55]


def main() -> None:
    config = load_config()
    processed = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config)
    if len(processed) < SECOND_WINDOW_START + WINDOW_LENGTH:
        raise RuntimeError("Canonical series is too short for the prespecified second window.")
    params = load_json(PROJECT_ROOT / "results" / "calibrated_parameters.json")
    config = apply_calibrated_params_to_config(config, params)

    primary_frame = processed.iloc[:WINDOW_LENGTH].copy()
    primary_sim = simulate_thermo_hydraulics(build_boundary_conditions(primary_frame, config), config, params=params)
    primary_sensors = apply_sensor_layout(primary_sim, "S4_five_sensors", config)
    primary_loaders = build_loaders(primary_sim, primary_sensors, config)
    training_stats = primary_loaders["train_ds"].stats

    second_frame = processed.iloc[SECOND_WINDOW_START : SECOND_WINDOW_START + WINDOW_LENGTH].copy()
    second_boundary = build_boundary_conditions(second_frame, config)
    second_sim = simulate_thermo_hydraulics(second_boundary, config, params=params)
    second_sensors = apply_sensor_layout(second_sim, "S4_five_sensors", config)
    second_loaders = build_loaders(second_sim, second_sensors, config, stats=training_stats)
    arrays = second_loaders["arrays"]
    a_norm = _a_norm_for(arrays)

    rows = []
    for seed in SEEDS:
        for spec in CORE_MODELS:
            label = spec["label"]
            checkpoint = PROJECT_ROOT / "results" / f"seed_{seed}_{label}_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = build_model(
                spec["model"],
                input_dim=arrays["features"].shape[-1],
                n_nodes=arrays["target"].shape[1],
                a_norm=a_norm,
                config=config,
            )
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(state)
            metrics = evaluate_model(
                model,
                second_loaders["test_loader"],
                config,
                training_stats,
                label,
            )
            metrics.update(
                {
                    "seed": seed,
                    "window": "second_chronological_fixed_checkpoint",
                    "checkpoint_retrained_on_second_window": False,
                    "normalization_source": "primary_window_training_only",
                    "sensor_layout": "S4_five_sensors",
                }
            )
            rows.append(metrics)
    raw = pd.DataFrame(rows)
    raw.to_csv(PROJECT_ROOT / "results" / "second_chronological_window_metrics.csv", index=False)
    metric_cols = [
        "RMSE_Ts_full",
        "RMSE_Tr_full",
        "heat_loss_error_percent",
        "energy_balance_residual",
        "boundary_residual_mean",
    ]
    summary_rows = []
    for model, group in raw.groupby("model"):
        row = {"model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROJECT_ROOT / "results" / "second_chronological_window_summary.csv", index=False)
    display = summary.rename(
        columns={
            "model": "Model",
            "n_seeds": "Seeds",
            "RMSE_Ts_full_mean": "Supply RMSE mean (C)",
            "RMSE_Ts_full_std": "Supply RMSE SD (C)",
            "RMSE_Tr_full_mean": "Return RMSE mean (C)",
            "RMSE_Tr_full_std": "Return RMSE SD (C)",
            "heat_loss_error_percent_mean": "Heat-loss error mean (%)",
            "energy_balance_residual_mean": "Energy residual mean (%)",
        }
    )
    keep = [column for column in ["Model", "Seeds", "Supply RMSE mean (C)", "Supply RMSE SD (C)", "Return RMSE mean (C)", "Return RMSE SD (C)", "Heat-loss error mean (%)", "Energy residual mean (%)"] if column in display]
    for column in keep:
        if column not in {"Model", "Seeds"}:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(lambda value: f"{value:.3f}")
    _write_table(
        display[keep],
        PROJECT_ROOT / "paper" / "tables" / "table_second_chronological_window.tex",
        "Fixed-checkpoint transfer to the disjoint second chronological window. Temperature RMSEs are in degrees Celsius and residuals are percentages. The five seed-indexed checkpoints and primary-window normalization are reused without fitting on the second window; the benchmark contains C-class thermal and S-class hydraulic metrics, with dynamic energy reported as a mixed C+S dependency.",
        "tab:second_chronological_window",
    )
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "fixed-checkpoint transfer; no second-window fitting or normalization",
        "primary_window_indices": [0, WINDOW_LENGTH - 1],
        "second_window_indices": [SECOND_WINDOW_START, SECOND_WINDOW_START + WINDOW_LENGTH - 1],
        "primary_window_utc": [str(primary_frame["timestamp"].iloc[0]), str(primary_frame["timestamp"].iloc[-1])],
        "second_window_utc": [str(second_frame["timestamp"].iloc[0]), str(second_frame["timestamp"].iloc[-1])],
        "test_window_count": len(second_loaders["test_ds"]),
        "seeds": SEEDS,
        "models": [item["label"] for item in CORE_MODELS],
        "sensor_layout": "S4_five_sensors",
        "normalization_source": "primary-window training partition only",
        "note": "This evaluates chronological-window sensitivity. It is not repeated retraining on the second period and does not establish annual generalization.",
    }
    (PROJECT_ROOT / "results" / "second_chronological_window_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
