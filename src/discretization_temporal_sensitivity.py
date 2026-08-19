from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, first_existing, save_grouped_bar_figure, write_latex_table


def run() -> None:
    disc = first_existing(["discretization_study.csv"])
    verification = first_existing(["model_verification_summary.csv"])
    if disc.empty:
        out = pd.DataFrame([{"status": "not run", "note": "Discretization source results were not available."}])
    else:
        out = disc.copy()
        out["sensitivity_type"] = "spatial_dx"
        out["temporal_horizon"] = "not rerun; reported in supplementary as numerical sensitivity only"
        out["safe_claim"] = "Discretization sensitivity checks numerical consistency of calibrated simulator outputs, not independent field validation."
        if not verification.empty:
            out["verification_note"] = "Model verification summary available."
    out.to_csv(PROJECT_ROOT / "results" / "discretization_temporal_sensitivity.csv", index=False)
    write_latex_table(
        out,
        TABLES_DIR / "table_discretization_temporal_sensitivity.tex",
        "Discretization/temporal sensitivity. Kept in supplementary; it supports numerical consistency rather than field validation.",
        "tab:discretization_temporal_sensitivity",
    )
    plot(out)


def plot(df: pd.DataFrame) -> None:
    data = df.copy()
    if "dx_m" in data:
        data["dx_label"] = data["dx_m"].astype(str) + " m"
    save_grouped_bar_figure(
        "fig_discretization_temporal_sensitivity",
        [
            {"title": "Outlet supply delta", "data": data, "category": "dx_label", "value": "outlet_supply_delta_vs_1000m_C", "ylabel": "Delta (C)"},
            {"title": "Heat-loss delta", "data": data, "category": "dx_label", "value": "heat_loss_delta_vs_1000m_kW", "ylabel": "Delta (kW)"},
        ],
        title="Discretization sensitivity of calibrated simulator",
        ncols=2,
        width=1300,
        height=680,
    )


if __name__ == "__main__":
    run()
