"""Synchronize non-numerical provenance after the canonical path is frozen."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def main() -> None:
    canonical = json.loads((RESULTS / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))
    for name in ["corrected_simulator_states_provenance.json", "repeated_seed_protocol.json"]:
        path = RESULTS / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["canonical_dataset_path"] = canonical["canonical_file"]
        payload["canonical_dataset_sha256"] = canonical["sha256"]
        if name.startswith("corrected"):
            payload["source_dataset"] = canonical["canonical_file"]
            payload["source_sha256"] = canonical["sha256"]
        elif name == "repeated_seed_protocol.json":
            frame = pd.read_csv(PROJECT_ROOT / canonical["canonical_file"], usecols=["timestamp"], nrows=int(payload.get("maximum_time_steps", 768)))
            timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
            minutes = timestamp.diff().dt.total_seconds().div(60.0).dropna()
            payload["timestamp_window_audit"] = {
                "window_start_utc": timestamp.iloc[0].isoformat(),
                "window_end_utc": timestamp.iloc[-1].isoformat(),
                "retained_timestamp_count": int(len(timestamp)),
                "within_window_interval_count": int(len(minutes)),
                "within_window_15min_interval_count": int(minutes.eq(15.0).sum()),
                "non_nominal_interval_count": int(minutes.ne(15.0).sum()),
            }
            gap_path = RESULTS / "gap_handling_audit.json"
            if gap_path.is_file():
                gap = json.loads(gap_path.read_text(encoding="utf-8"))
                payload["window_split_audit"] = {
                    key: gap[key]
                    for key in [
                        "candidate_window_count",
                        "excluded_cross_gap_window_count",
                        "eligible_contiguous_window_count",
                        "training_window_starts",
                        "validation_window_starts",
                        "test_window_starts",
                    ]
                    if key in gap
                }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
