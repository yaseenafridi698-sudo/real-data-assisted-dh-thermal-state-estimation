from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.robustness_operational_utils import PROJECT_ROOT, TABLES_DIR, first_existing, read_csv, save_grouped_bar_figure, write_latex_table


def run() -> None:
    shift = first_existing(["flensburg_domain_shift_analysis.csv", "flensburg_domain_shift_analysis_improved.csv", "flensburg_transfer_diagnostics.csv"])
    modes = first_existing(["external_validation_flensburg_modes_final.csv", "external_validation_flensburg_modes.csv"])
    ret = read_csv("flensburg_return_temperature_assumption_sensitivity.csv")
    rows = []
    if not shift.empty:
        for _, row in shift.iterrows():
            rows.append({"diagnostic": row.get("metric", row.get("diagnostic", "")), "value": row.get("value", ""), "interpretation": row.get("interpretation", "")})
    if not modes.empty:
        for _, row in modes.iterrows():
            rows.append({"diagnostic": f"transfer_mode_{row.get('mode', row.get('transfer_mode', 'mode'))}", "value": row.get("RMSE_supply_measured_C", row.get("supply_RMSE_C", "")), "interpretation": "Flensburg transfer mode metric; interpreted as domain-shift stress test."})
    if not ret.empty:
        for _, row in ret.iterrows():
            rows.append({"diagnostic": f"return_assumption_{row.get('assumed_return_temp_C', row.get('return_temp_C', ''))}", "value": row.get("RMSE_return_C", row.get("supply_RMSE_C", "")), "interpretation": "Sensitivity to fallback return-temperature assumption."})
    out = pd.DataFrame(rows) if rows else pd.DataFrame([{"diagnostic": "not run", "value": "", "interpretation": "Domain-transfer diagnostic source files were not available."}])
    out["safe_claim"] = "Flensburg is treated as an external domain-shift stress test, not proof of universal transfer."
    out.to_csv(PROJECT_ROOT / "results" / "domain_transfer_diagnostics.csv", index=False)
    write_latex_table(
        out,
        TABLES_DIR / "table_domain_transfer_diagnostics.tex",
        "Domain-transfer diagnostics for Flensburg. Flensburg is treated as an external domain-shift stress test, not proof of universal transfer.",
        "tab:domain_transfer_diagnostics",
    )
    plot(shift, modes)


def plot(shift: pd.DataFrame, modes: pd.DataFrame) -> None:
    s = shift.head(8).copy() if not shift.empty else shift
    if not s.empty and "metric" not in s:
        s["metric"] = s.iloc[:, 0].astype(str)
    m = modes.copy()
    if not m.empty:
        if "mode" not in m:
            m["mode"] = m.iloc[:, 0].astype(str)
        if "RMSE_supply_measured_C" not in m:
            num = m.select_dtypes("number").columns
            m["RMSE_supply_measured_C"] = m[num[0]] if len(num) else 0.0
    save_grouped_bar_figure(
        "fig_domain_transfer_diagnostics",
        [
            {"title": "Domain-shift variables", "data": s, "category": "metric", "value": "value", "ylabel": "Value"},
            {"title": "Transfer modes", "data": m, "category": "mode", "value": "RMSE_supply_measured_C", "ylabel": "Supply RMSE (C)"},
        ],
        title="Flensburg domain-transfer diagnostics",
        ncols=2,
        width=1350,
        height=700,
    )


if __name__ == "__main__":
    run()
