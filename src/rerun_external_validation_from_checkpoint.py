"""Regenerate Flensburg transfer artifacts from the corrected saved PI-GNN checkpoint.

This helper is intentionally inference-only. It rebuilds the corrected primary
normalization statistics and native-hourly Flensburg simulator trajectory, then
rewrites only the external-validation result files. It does not retrain a
model or alter any input observation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import _a_norm_for, build_loaders, run_external_validation
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import load_json


def main() -> None:
    config = load_config()
    max_steps = int(
        config["dataset"]["n_scenarios_full"]
        * config["system"]["horizon_h"]
        * 3600
        / config["system"]["dt_s"]
    )
    # The paper archive uses the checksum-pinned processed Sønderborg input.
    primary = preprocess_dataset(__import__("pandas").DataFrame(), "sonderborg", config).head(max_steps).copy()
    flensburg_processed = PROJECT_ROOT / "data" / "processed" / "flensburg_processed.csv"
    if flensburg_processed.exists():
        import pandas as pd
        flensburg = pd.read_csv(flensburg_processed)
        flensburg["timestamp"] = pd.to_datetime(flensburg["timestamp"], errors="raise", utc=True)
    else:
        flensburg = preprocess_dataset(load_dataset_by_name("flensburg"), "flensburg", config)
    params = load_json(PROJECT_ROOT / "results" / "calibrated_parameters.json")
    config = apply_calibrated_params_to_config(config, params)
    boundary = build_boundary_conditions(primary, config)
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"].get("noise_std_fraction", 0.0))
    loaders = build_loaders(sim, sensors, config)
    model = build_model(
        "pignn_v3",
        input_dim=loaders["arrays"]["features"].shape[-1],
        n_nodes=loaders["arrays"]["target"].shape[1],
        a_norm=_a_norm_for(loaders["arrays"]),
        config=config,
    )
    checkpoint = PROJECT_ROOT / "results" / "seed_11_Proposed PI-GNN-GRU-v3 balanced_mode_best.pt"
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    run_external_validation(flensburg, config, params, model, loaders["train_ds"].stats, primary_df=primary)
    print(PROJECT_ROOT / "results" / "flensburg_measured_only_validation.csv")


if __name__ == "__main__":
    main()
