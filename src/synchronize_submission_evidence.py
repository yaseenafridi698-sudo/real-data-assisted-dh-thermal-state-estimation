"""Regenerate publication-facing evidence tables from current result files.

The script never changes scientific CSV or JSON values.  It creates compact
LaTeX tables from the final all-valid-range XAI4HEAT and measured-only
Flensburg files so active publication assets cannot silently retain an older
filtering protocol.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"


def esc(value: object) -> str:
    text = str(value)
    degree = "@@DEGREE@@"
    text = text.replace(r"$^\circ$C", degree).replace(r"$^\circ$C", degree)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text.replace(degree, r"$^\circ$C")


def fmt(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_table(name: str, frame: pd.DataFrame, caption: str, label: str) -> None:
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small", r"\resizebox{\textwidth}{!}{%"]
    lines += [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\toprule"]
    lines.append(" & ".join(esc(column) for column in frame.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in frame.iterrows():
        lines.append(" & ".join(esc(row[column]) for column in frame.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    return frame.loc[frame[column].astype(str).eq(value)].iloc[0]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    xai = pd.read_csv(RESULTS / "xai4heat_sparse_substation_validation_final.csv")
    station = pd.read_csv(RESULTS / "xai4heat_sparse_substation_validation_by_substation.csv")
    quality = pd.read_csv(RESULTS / "xai4heat_station_quality_audit.csv")
    ordered = pd.read_csv(RESULTS / "xai4heat_sparse_substation_validation_physically_ordered_sensitivity.csv")
    flensburg = pd.read_csv(RESULTS / "flensburg_measured_only_validation.csv")
    protocol = pd.read_json(RESULTS / "repeated_seed_protocol.json", typ="series")

    targets = []
    for variable in ("t_sup_prim", "t_ret_prim"):
        result = pick(xai, "variable", variable)
        by_station = station.loc[station["variable"].eq(variable)].copy()
        worst = by_station.loc[by_station["RMSE"].idxmax()]
        targets.append(
            {
                "Measured target": result["variable_label"],
                "Stations": int(result["n_substations"]),
                "Observations": int(result["n_total_samples"]),
                "Mean fold RMSE ($^\\circ$C)": fmt(result["mean_RMSE"]),
                "Median station RMSE ($^\\circ$C)": fmt(by_station["RMSE"].median()),
                "Worst station": worst["substation_id"],
                "Worst RMSE ($^\\circ$C)": fmt(worst["RMSE"]),
                "Evidence": "M",
            }
        )
    write_table(
        "table_xai4heat_withholding_diagnostics.tex",
        pd.DataFrame(targets),
        "Primary XAI4HEAT all-valid-range leave-one-substation-out diagnostics after file-content deduplication. This is the headline measured-node protocol; no distributed pipe or hydraulic field is validated.",
        "tab:xai_withholding_diagnostics",
    )

    display = station.loc[station["variable"].isin(["t_sup_prim", "t_ret_prim"]), ["substation_id", "variable_label", "n_samples", "RMSE", "MAE", "bias"]].copy()
    display.columns = ["Station", "Measured target", "Observations", "RMSE ($^\\circ$C)", "MAE ($^\\circ$C)", "Bias ($^\\circ$C)"]
    for name in display.columns[3:]:
        display[name] = display[name].map(fmt)
    write_table(
        "table_xai4heat_substation_variability.tex",
        display,
        "Station-level XAI4HEAT all-valid-range measured-node errors. L22 remains in the headline protocol and is reported rather than filtered out.",
        "tab:xai_variability",
    )

    audit = quality[["substation_id", "raw_rows_after_file_deduplication", "valid_primary_temperature_pairs", "primary_supply_below_return_percent", "primary_ordered_percent", "primary_supply_median_C", "primary_return_median_C"]].copy()
    audit.columns = ["Station", "Raw rows", "Valid primary pairs", "Supply < return (%)", "Ordered sensitivity retained (%)", "Median supply ($^\\circ$C)", "Median return ($^\\circ$C)"]
    for name in audit.columns[3:]:
        audit[name] = audit[name].map(fmt)
    write_table(
        "table_xai4heat_station_quality_audit.tex",
        audit,
        "XAI4HEAT station-quality audit after SHA-256 file-content deduplication. Temperature reversals are retained in the all-valid-range analysis; the ordered subset is a target-conditioned sensitivity analysis, not a data correction.",
        "tab:xai_quality_audit",
    )

    ordered_display = ordered[["variable_label", "n_substations", "n_total_samples", "mean_RMSE", "mean_MAE", "mean_nRMSE_percent"]].copy()
    ordered_display.columns = ["Measured target", "Stations", "Observations", "Mean fold RMSE ($^\\circ$C)", "Mean MAE ($^\\circ$C)", "Mean nRMSE (%)"]
    for name in ordered_display.columns[3:]:
        ordered_display[name] = ordered_display[name].map(fmt)
    write_table(
        "table_xai4heat_physically_ordered_sensitivity.tex",
        ordered_display,
        "Physically ordered XAI4HEAT sensitivity subset. The supply-at-least-return rule uses withheld target channels; therefore it is not the headline validation protocol.",
        "tab:xai_ordered_sensitivity",
    )

    direct = pick(flensburg, "mode", "direct_transfer")
    offset = pick(flensburg, "mode", "calibration_only_offset_adaptation")
    few = pick(flensburg, "mode", "few_shot_decoder_bias_adaptation")
    reduction = 100 * (float(direct["measured supply RMSE_C"]) - float(few["measured supply RMSE_C"])) / float(direct["measured supply RMSE_C"])
    l22_supply = station.loc[(station["substation_id"].eq("L22")) & (station["variable"].eq("t_sup_prim")), "RMSE"].iloc[0]
    risk = pd.DataFrame(
        [
            ["No independent hydraulic field measurements", "Pressure-drop error spans 8.60--16.60% under proxy/friction perturbations; flow RMSE changes by up to 181.3%.", "Hydraulic identifiability and parameter sensitivity are reported separately from measured thermal validation.", "Pressure/head and flow remain simulator-assisted hidden states."],
            ["Reduced corridor is not utility topology", "Changing 20 km to 10/30 km changes heat loss by -49.9%/+49.9% and delay by -50.0%/+50.0%.", "Length is treated as a structural sensitivity parameter; branches, elevation, and valve states are excluded explicitly.", "Results cannot identify the topology of any source network."],
            ["Repeated-seed scope is limited", f"Five seeds are complete for four neural models under one fixed split and S4 layout on {int(protocol['maximum_time_steps'])} 15-min timestamps.", "Seed variation and moving-block temporal uncertainty are reported separately.", "No seed claim is extended to every baseline, layout, or transfer experiment."],
            ["High XAI4HEAT supply withholding error", f"All-valid mean fold RMSE is {fmt(targets[0]['Mean fold RMSE ($^\\circ$C)'])}/{fmt(targets[1]['Mean fold RMSE ($^\\circ$C)'])} $^\\circ$C (supply/return); L22 supply RMSE is {fmt(l22_supply)} $^\\circ$C.", "All valid-range folds and the L22 station-quality audit are retained; the ordered subset is sensitivity only.", "XAI4HEAT supports measured-node thermal withholding, not full-field validation."],
            ["Weak Flensburg transfer", f"Measured-supply RMSE is {fmt(direct['measured supply RMSE_C'])} $^\\circ$C direct, {fmt(offset['measured supply RMSE_C'])} $^\\circ$C with offset adaptation, and {fmt(few['measured supply RMSE_C'])} $^\\circ$C few-shot ({fmt(reduction, 1)}% reduction versus direct).", "Assumed 50 $^\\circ$C return values are excluded from measured claims; domain-shift variables are reported.", "The experiment supports local adaptation, not broad generalization."],
        ],
        columns=["Risk", "Quantified evidence", "Control", "Residual boundary"],
    )
    write_table(
        "table_residual_submission_risk_mitigation.tex",
        risk,
        "Quantified residual risks and evidence boundaries retained in the submission package.",
        "tab:submission_risk_mitigation",
    )

    resolution = "\n".join(
        [
            "# XAI4HEAT publication-protocol resolution",
            "",
            "The active manuscript and tables use the all-valid-range primary-temperature protocol in `xai4heat_sparse_substation_validation_final.csv`.",
            f"- Primary supply: {float(pick(xai, 'variable', 't_sup_prim')['mean_RMSE']):.12f} C over {int(pick(xai, 'variable', 't_sup_prim')['n_total_samples'])} observations.",
            f"- Primary return: {float(pick(xai, 'variable', 't_ret_prim')['mean_RMSE']):.12f} C over {int(pick(xai, 'variable', 't_ret_prim')['n_total_samples'])} observations.",
            "",
            "The 8,074-sample diagnostics CSV records a different legacy filtering/aggregation path. It is retained in `results/` for provenance but excluded from active publication tables and the reviewer archive.",
            "The physically ordered subset is reported only as a target-conditioned sensitivity analysis.",
        ]
    )
    (RESULTS / "xai4heat_publication_protocol_resolution.md").write_text(resolution + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
