"""Refresh audit metadata for locked calibration and repeated-seed protocols.

This script reads the existing processed timestamps and locked protocol/result
files. It adds only descriptive metadata; it never recalibrates or retrains a
model and never modifies scientific metric values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import contiguous_window_starts, split_window_indices


RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed" / "sonderborg_processed.csv"


def main() -> None:
    protocol_path = RESULTS / "repeated_seed_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    series = pd.read_csv(PROCESSED, parse_dates=["timestamp"])
    timestamps = pd.to_datetime(series["timestamp"], utc=True, errors="raise")
    retained = int(protocol["maximum_time_steps"])
    window_steps = 12
    embargo_steps = 11
    capped = timestamps.iloc[:retained].reset_index(drop=True)
    gaps = capped.diff().dropna()
    trajectory_start = np.zeros(retained, dtype=bool)
    trajectory_start[0] = True
    trajectory_start[1:] = (capped.diff().iloc[1:] > pd.Timedelta(minutes=15) * 1.5).to_numpy()
    valid_starts = contiguous_window_starts(trajectory_start, window_steps)
    train, validation, test = split_window_indices(
        retained,
        window_steps,
        train_fraction=0.70,
        val_fraction=0.15,
        embargo_steps=embargo_steps,
        valid_window_starts=valid_starts,
    )
    protocol["timestamp_window_audit"] = {
        "retained_timestamp_count": retained,
        "first_timestamp_utc": capped.iloc[0].isoformat(),
        "last_timestamp_utc": capped.iloc[-1].isoformat(),
        "within_window_15min_interval_count": int((gaps == pd.Timedelta(minutes=15)).sum()),
        "within_window_non_15min_intervals": [
            {"ending_timestamp_utc": capped.iloc[index].isoformat(), "duration": str(delta)}
            for index, delta in gaps[gaps != pd.Timedelta(minutes=15)].items()
        ],
        "gap_handling": "split trajectory at every interval longer than 1.5 nominal steps; no 15-minute simulator propagation, sequence window, storage derivative, or smoothness difference crosses a segment start",
        "trajectory_start_indices": np.flatnonzero(trajectory_start).astype(int).tolist(),
    }
    protocol["window_split_audit"] = {
        "window_steps": window_steps,
        "embargo_steps_between_partitions": embargo_steps,
        "candidate_window_count": retained - window_steps + 1,
        "excluded_cross_gap_window_count": int((retained - window_steps + 1) - len(valid_starts)),
        "eligible_contiguous_window_count": len(valid_starts),
        "training_window_starts": len(train),
        "validation_window_starts": len(validation),
        "test_window_starts": len(test),
        "train_start_index": train[0],
        "train_end_index": train[-1],
        "validation_start_index": validation[0],
        "validation_end_index": validation[-1],
        "test_start_index": test[0],
        "test_end_index": test[-1],
        "normalization_source": "training windows only",
    }
    protocol["full_processed_series_audit"] = {
        "retained_timestamp_count": int(len(timestamps)),
        "first_timestamp_utc": timestamps.iloc[0].isoformat(),
        "last_timestamp_utc": timestamps.iloc[-1].isoformat(),
        "intervals_longer_than_15min": int((timestamps.diff().dropna() > pd.Timedelta(minutes=15)).sum()),
        "scope": "used by separately stated replay, seasonal, and transfer studies; not the calibration input for the locked reported fit",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    metrics = pd.read_csv(RESULTS / "calibration_metrics.csv").iloc[0]
    calibration_samples = 537
    calibration = capped.iloc[:calibration_samples]
    calibration_scored_samples = int((~trajectory_start[:calibration_samples]).sum())
    audit = pd.DataFrame(
        [
            {
                "item": "calibration input",
                "value": f"first {calibration_samples} retained timestamps of {retained}; {calibration_scored_samples} scored after trajectory-start exclusion",
                "start_utc": calibration.iloc[0].isoformat(),
                "end_utc": calibration.iloc[-1].isoformat(),
                "independent_simulator_fit_validation": "not performed/reported",
                "source": "results/calibration_metrics.csv; src/calibration.py",
            },
            {
                "item": "calibration fit metrics",
                "value": f"return RMSE={float(metrics['RMSE_return_C']):.12f} C; dynamic ratio={100.0 * float(metrics['energy_balance_residual_fraction']):.12f}%",
                "start_utc": calibration.iloc[0].isoformat(),
                "end_utc": calibration.iloc[-1].isoformat(),
                "independent_simulator_fit_validation": "not performed/reported",
                "source": "results/calibration_metrics.csv",
            },
        ]
    )
    audit.to_csv(RESULTS / "calibration_scope_audit.csv", index=False)


if __name__ == "__main__":
    main()
