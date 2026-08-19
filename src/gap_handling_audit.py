"""Audit the locked retained-timestamp gap without altering scientific results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT
from src.dataset import contiguous_window_starts, split_window_indices


RESULTS = PROJECT_ROOT / "results"
PAPER_TABLES = PROJECT_ROOT / "paper" / "tables"


def _latex_table(rows: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Handling of the observed retained-timestamp discontinuity in the five-seed window. The correction prevents a 15-min simulator advance, temporal loss, storage derivative, smoothness difference, or neural input window from crossing the 17.25-h gap.}",
        r"\label{tab:gap_handling}",
        r"\small",
        r"\begin{tabular}{p{0.26\textwidth}p{0.66\textwidth}}",
        r"\toprule",
        r"Item & Value \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        item = str(row["item"]).replace("_", "\\_")
        value = str(row["value"]).replace("_", "\\_")
        lines.append(f"{item} & {value} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    protocol = json.loads((RESULTS / "repeated_seed_protocol.json").read_text(encoding="utf-8"))
    retained = int(protocol["maximum_time_steps"])
    window_steps = 12
    embargo_steps = 11
    source = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv", parse_dates=["timestamp"]).head(retained)
    timestamps = pd.to_datetime(source["timestamp"], utc=True)
    interval = timestamps.diff()
    gap_rows = np.flatnonzero((interval > pd.Timedelta(minutes=15) * 1.5).to_numpy())
    if len(gap_rows) != 1:
        raise RuntimeError(f"Expected one retained-timestamp gap; found {len(gap_rows)}.")
    gap_index = int(gap_rows[0])
    starts = np.zeros(retained, dtype=bool)
    starts[0] = True
    starts[gap_rows] = True
    eligible = contiguous_window_starts(starts, window_steps)
    train, validation, test = split_window_indices(
        retained,
        window_steps,
        train_fraction=0.70,
        val_fraction=0.15,
        embargo_steps=embargo_steps,
        valid_window_starts=eligible,
    )
    excluded = sorted(set(range(retained - window_steps + 1)) - set(eligible))
    payload = {
        "retained_timestamp_count": retained,
        "nominal_interval_minutes": 15,
        "gap_previous_index": gap_index - 1,
        "gap_start_index": gap_index,
        "gap_previous_timestamp_utc": timestamps.iloc[gap_index - 1].isoformat(),
        "gap_start_timestamp_utc": timestamps.iloc[gap_index].isoformat(),
        "gap_duration_hours": float(interval.iloc[gap_index].total_seconds() / 3600.0),
        "handling_method": "trajectory split and state reinitialization at the first observation after the gap",
        "simulator": "does not propagate a nominal 900-s step across the gap; hydraulic state and thermal fields are reinitialized at segment start",
        "calibration": "first 537 retained samples; segment-start observations excluded from scored return-fit, energy, and smoothness terms",
        "calibration_scored_samples": int((~starts[:537]).sum()),
        "neural_sequences": "windows spanning a trajectory start are excluded before chronological train/validation/test partitioning",
        "candidate_window_count": retained - window_steps + 1,
        "excluded_cross_gap_window_count": len(excluded),
        "excluded_window_start_indices": excluded,
        "eligible_contiguous_window_count": len(eligible),
        "training_window_starts": len(train),
        "validation_window_starts": len(validation),
        "test_window_starts": len(test),
        "moving_block_bootstrap": "overlapping eligible test-window predictions are collapsed to unique chronological timestamps before within-segment block resampling; no bootstrap block crosses the excluded gap",
    }
    (RESULTS / "gap_handling_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    rows = pd.DataFrame(
        [
            {"item": "Observed discontinuity", "value": f"index {gap_index - 1} to {gap_index}: {payload['gap_previous_timestamp_utc']} to {payload['gap_start_timestamp_utc']} ({payload['gap_duration_hours']:.2f} h)"},
            {"item": "Simulator/calibration treatment", "value": "trajectory split; reset at segment start; segment-start fit and dynamic terms excluded"},
            {"item": "Neural sequence treatment", "value": f"{len(excluded)} of {payload['candidate_window_count']} candidate 12-step windows excluded (starts {excluded[0]}--{excluded[-1]})"},
            {"item": "Eligible chronological windows", "value": f"{len(eligible)} total; {len(train)}/{len(validation)}/{len(test)} train/validation/test after 11-step embargo"},
            {"item": "Bootstrap treatment", "value": "regenerated from eligible test windows; no resampled block crosses the retained-data discontinuity"},
        ]
    )
    rows.to_csv(RESULTS / "gap_handling_audit.csv", index=False)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLES / "table_gap_handling_audit.tex").write_text(_latex_table(rows), encoding="utf-8")
    print(RESULTS / "gap_handling_audit.csv")


if __name__ == "__main__":
    main()
