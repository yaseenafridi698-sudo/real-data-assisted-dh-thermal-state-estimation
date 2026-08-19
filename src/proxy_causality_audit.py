"""Public verifier for the frozen proxy/full-state causality audit.

The historical audit-generation implementation was not present in the supplied
public archive.  This module exposes the recorded version tag required by
downstream code and verifies the preserved machine-readable audit instead of
pretending to regenerate it.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PROJECT_ROOT
else:
    from .config import PROJECT_ROOT

FULL_STATE_CAUSALITY_VERSION = "full_state_causality_v1_segment_start_boundaries"

REQUIRED_AUDITS = {
    "alpha_uses_current_or_past_load_only",
    "q_proxy_uses_lagged_return_only",
    "pump_proxy_scaling_uses_expanding_past_only_range",
    "supply_temperature_state_uses_no_future_values",
    "return_temperature_state_uses_no_future_values",
    "head_state_uses_no_future_values",
    "flow_state_uses_no_future_values",
    "heat_loss_state_uses_no_future_values",
    "delivered_heat_state_uses_no_future_values",
    "full_state_causality_version_recorded",
}

def verify_preserved_audit(path: Path | None = None) -> None:
    path = path or PROJECT_ROOT / "results" / "proxy_causality_audit.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_name = {row.get("audit", ""): row for row in rows}
    missing = sorted(REQUIRED_AUDITS - by_name.keys())
    failed = sorted(name for name in REQUIRED_AUDITS if by_name.get(name, {}).get("status") != "pass")
    version = by_name.get("full_state_causality_version_recorded", {}).get("value")
    if missing or failed or version != FULL_STATE_CAUSALITY_VERSION:
        raise RuntimeError(f"Preserved causality audit failed: missing={missing}, failed={failed}, version={version!r}")

def main() -> None:
    verify_preserved_audit()
    print("PASS: preserved proxy/full-state causality audit is internally consistent.")

if __name__ == "__main__":
    main()
