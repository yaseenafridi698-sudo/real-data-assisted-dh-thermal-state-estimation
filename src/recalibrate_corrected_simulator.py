"""Recalibrate the full-state-causal simulator and export C thermal and S hydraulic states."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import calibrate_simulator
from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.proxy_causality_audit import FULL_STATE_CAUSALITY_VERSION
from src.real_data_mapper import PROXY_CAUSALITY_VERSION, build_boundary_conditions
from src.thermo_hydraulic_simulator import run_discretization_study, save_model_verification, simulate_thermo_hydraulics


def main() -> None:
    config = load_config()
    processed = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config)
    max_steps = int(
        config["dataset"]["n_scenarios_full"]
        * config["system"]["horizon_h"]
        * 3600
        / config["system"]["dt_s"]
    )
    frame = processed.iloc[:max_steps].copy()
    boundary = build_boundary_conditions(frame, config)
    calibrated = calibrate_simulator(boundary, config, quick=False)
    effective = apply_calibrated_params_to_config(config, calibrated["params"])
    sim = simulate_thermo_hydraulics(boundary, effective, params=calibrated["params"])
    save_model_verification(sim)
    run_discretization_study(boundary, effective, calibrated["params"])

    results = PROJECT_ROOT / "results"
    np.savez_compressed(
        results / "corrected_simulator_states.npz",
        time_s=np.asarray(sim["time_s"], dtype=np.float64),
        x_m=np.asarray(sim["x_m"], dtype=np.float64),
        Ts=np.asarray(sim["Ts"], dtype=np.float32),
        Tr=np.asarray(sim["Tr"], dtype=np.float32),
        H=np.asarray(sim["H"], dtype=np.float32),
        q=np.asarray(sim["q"], dtype=np.float32),
        Q_loss=np.asarray(sim["Q_loss"], dtype=np.float32),
        Q_loss_segments=np.asarray(sim["Q_loss_segments"], dtype=np.float32),
        delivered_heat_W=np.asarray(sim["delivered_heat_W"], dtype=np.float32),
        energy_balance_residual_W=np.asarray(sim["energy_balance_residual_W"], dtype=np.float32),
        pressure_drop_m=np.asarray(sim["pressure_drop_m"], dtype=np.float32),
        q_proxy=np.asarray(sim["q_proxy"], dtype=np.float32),
        trajectory_start=np.asarray(sim["trajectory_start"], dtype=bool),
        valid_transition=np.asarray(sim["valid_transition"], dtype=bool),
    )
    canonical = json.loads((results / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": canonical["canonical_file"],
        "source_sha256": canonical["sha256"],
        "canonical_dataset_path": canonical["canonical_file"],
        "canonical_dataset_sha256": canonical["sha256"],
        "source_rows_total": canonical["retained_timestamps"],
        "source_rows_used": len(frame),
        "proxy_causality_version": PROXY_CAUSALITY_VERSION,
        "full_state_causality_version": FULL_STATE_CAUSALITY_VERSION,
        "parameters": calibrated["params"],
        "state_evidence": {
            "Ts": "calibrated_simulator",
            "Tr": "calibrated_simulator",
            "H": "simulator_assisted_hidden_state",
            "q": "simulator_assisted_hidden_state_heat_load_proxy",
            "Q_loss": "calibrated_simulator_postprocessing",
        },
        "superseded_archive": "superseded_pre_full_state_causality_20260807",
        "note": "These states were generated only after full-state future-perturbation invariance passed. They are not distributed field measurements.",
    }
    (results / "corrected_simulator_states_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(results / "corrected_simulator_states.npz")


if __name__ == "__main__":
    main()
