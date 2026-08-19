"""Export paired dense test arrays from the corrected real-data checkpoints.

The export is deliberately checkpoint-based: it neither retrains a model nor
alters a metric.  It rebuilds the corrected 768-timestamp Sonderborg
trajectory, applies the same contiguous-window rule as the primary benchmark,
and evaluates the saved core models on the resulting test windows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.evaluate import evaluate_model, save_thermo_hydraulic_estimation_outputs
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import _a_norm_for, build_loaders
from src.models import build_model
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import load_json


MODEL_SPECS = (
    ("GRU-MSE", "gru", "seed_11_GRU-MSE_best.pt", "gru_mse"),
    ("Transformer-MSE", "transformer", "seed_11_Transformer-MSE_best.pt", "transformer_mse"),
    (
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "pignn_v3",
        "seed_11_Proposed PI-GNN-GRU-v3 accuracy_mode_best.pt",
        "proposed_pi_gnn_gru_v3_accuracy_mode",
    ),
    (
        "Proposed PI-GNN-GRU-v3 balanced_mode",
        "pignn_v3",
        "seed_11_Proposed PI-GNN-GRU-v3 balanced_mode_best.pt",
        "proposed_pi_gnn_gru_v3_balanced_mode",
    ),
)


def _flatten(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    array = np.asarray(payload[key])
    if array.ndim < 2:
        raise ValueError(f"Expected batched sequence array for {key}, found {array.shape}.")
    return array.reshape((-1, *array.shape[2:]))


def main() -> None:
    config = load_config()
    max_steps = int(
        config["dataset"]["n_scenarios_full"]
        * config["system"]["horizon_h"]
        * 3600
        / config["system"]["dt_s"]
    )
    df = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config)
    df = df.head(min(len(df), max_steps)).copy()
    params = load_json(PROJECT_ROOT / "results" / "calibrated_parameters.json")
    config = apply_calibrated_params_to_config(config, params)
    boundary = build_boundary_conditions(df, config)
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"].get("noise_std_fraction", 0.0))
    loaders = build_loaders(sim, sensors, config)
    arrays = loaders["arrays"]
    a_norm = _a_norm_for(arrays)
    prediction_payloads: dict[str, dict[str, np.ndarray]] = {}

    for label, architecture, checkpoint_name, payload_key in MODEL_SPECS:
        checkpoint = PROJECT_ROOT / "results" / checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing corrected benchmark checkpoint: {checkpoint}")
        model = build_model(
            architecture,
            input_dim=arrays["features"].shape[-1],
            n_nodes=arrays["target"].shape[1],
            a_norm=a_norm,
            config=config,
        )
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # Compatibility with older Torch runtimes.
            state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        _, payload = evaluate_model(model, loaders["test_loader"], config, loaders["train_ds"].stats, label, return_predictions=True)
        prediction_payloads[label] = payload

    anchor = prediction_payloads["Proposed PI-GNN-GRU-v3 balanced_mode"]
    true = _flatten(anchor, "true")
    time_s = _flatten(anchor, "time_s")
    heat_load_kw = _flatten(anchor, "heat_load_kw")
    output: dict[str, np.ndarray] = {
        "Ts_reference": true[..., 0].astype(np.float32),
        "Tr_reference": true[..., 1].astype(np.float32),
        "H_reference": true[..., 2].astype(np.float32),
        "q_reference": true[..., 3].astype(np.float32),
        "time_index": time_s.astype(np.float64),
        "distance_km": (np.asarray(sim["x_m"], dtype=float) / 1000.0).astype(np.float32),
        "sensor_nodes": np.asarray(sensors["sensor_nodes"], dtype=int),
        "chosen_model": np.asarray(["Proposed PI-GNN-GRU-v3 balanced_mode"]),
        "heat_load_kw": heat_load_kw.astype(np.float32),
    }
    for label, _, _, payload_key in MODEL_SPECS:
        pred = _flatten(prediction_payloads[label], "pred")
        output[f"Ts_prediction_{payload_key}"] = pred[..., 0].astype(np.float32)
        output[f"Tr_prediction_{payload_key}"] = pred[..., 1].astype(np.float32)
        output[f"H_prediction_{payload_key}"] = pred[..., 2].astype(np.float32)
        output[f"q_prediction_{payload_key}"] = pred[..., 3].astype(np.float32)
    balanced = _flatten(prediction_payloads["Proposed PI-GNN-GRU-v3 balanced_mode"], "pred")
    output["Ts_prediction"] = balanced[..., 0].astype(np.float32)
    output["Tr_prediction"] = balanced[..., 1].astype(np.float32)
    output["H_prediction"] = balanced[..., 2].astype(np.float32)
    output["q_prediction"] = balanced[..., 3].astype(np.float32)
    target = PROJECT_ROOT / "results" / "dense_reconstruction_payloads.npz"
    np.savez_compressed(target, **output)

    save_thermo_hydraulic_estimation_outputs(
        prediction_payloads,
        config,
        sensor_nodes=sensors["sensor_nodes"],
        output_dir=PROJECT_ROOT / "results",
    )
    protocol = json.loads((PROJECT_ROOT / "results" / "repeated_seed_protocol.json").read_text(encoding="utf-8"))
    provenance = {
        "source": "prespecified seed-11 checkpoints from the corrected five-seed benchmark",
        "dataset": "sonderborg",
        "retained_timestamp_count": int(len(df)),
        "test_window_count": int(len(loaders["test_ds"])),
        "window_steps": int(config["model"]["window_steps"]),
        "flattened_test_time_node_rows": int(true.shape[0]),
        "trajectory_start_indices": np.flatnonzero(boundary["trajectory_start"]).astype(int).tolist(),
        "gap_handling": "test windows were constructed only from continuous retained 15-min segments",
        "window_split_audit": protocol.get("window_split_audit", {}),
        "models": [item[0] for item in MODEL_SPECS],
    }
    (PROJECT_ROOT / "results" / "dense_reconstruction_payload_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(target)


if __name__ == "__main__":
    main()
