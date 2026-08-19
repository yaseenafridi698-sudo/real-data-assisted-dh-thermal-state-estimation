"""Separate nominal sensor-placement geometry from sensor-condition scenarios.

The locked layout-result CSV contains one saved PI-GNN-GRU-v3 balanced-mode
run for each named scenario.  Several scenario labels share exactly the same
nodes.  This utility preserves every raw result but prevents those rows from
being interpreted as independent placement geometries in publication-facing
tables and figures.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"


# First nominal representative for each unique node set.  The alternatives
# named below are retained as protocol/selection-label duplicates, not merged
# or averaged because they are not statistical replicates.
CANONICAL_LAYOUT_FOR_NODES = {
    "0": "S1_inlet_only",
    "0;20": "S2_inlet_outlet",
    "0;10;20": "S3_inlet_middle_outlet",
    "0;5;10;15;20": "S4_five_sensors",
    "1;6;10;14;19": "S7_xai4heat_substations",
    "0;2;20": "S8_random_three_sensors",
    "10": "S11_middle_only",
    "0;7;14;20": "S12_inlet_two_middle_outlet",
    "20": "S14_outlet_only",
}


SCENARIO_METADATA = {
    "S1_inlet_only": ("nominal", "fixed inlet placement", "eligible_unique_nominal_geometry", ""),
    "S2_inlet_outlet": ("nominal", "fixed inlet/outlet placement", "eligible_unique_nominal_geometry", ""),
    "S3_inlet_middle_outlet": ("nominal", "fixed inlet/middle/outlet placement", "eligible_unique_nominal_geometry", ""),
    "S4_five_sensors": ("nominal", "fixed five-sensor placement", "eligible_unique_nominal_geometry", ""),
    "S5_noisy_inlet_outlet": ("sensor noise", "fixed inlet/outlet placement", "robustness_scenario", "S2_inlet_outlet"),
    "S6_dropout_five_sensors": ("sensor dropout", "fixed five-sensor placement", "robustness_scenario", "S4_five_sensors"),
    "S7_xai4heat_substations": ("nominal", "XAI4HEAT-style positional analogue on corridor", "eligible_unique_nominal_geometry", ""),
    "S8_random_three_sensors": ("nominal", "randomly selected unique three-node geometry", "eligible_unique_nominal_geometry", ""),
    "S9_optimized_three_sensors": ("nominal", "optimization-labelled duplicate geometry", "duplicate_geometry_excluded", "S3_inlet_middle_outlet"),
    "S10_optimized_five_sensors": ("nominal", "optimization-labelled duplicate geometry", "duplicate_geometry_excluded", "S4_five_sensors"),
    "S11_middle_only": ("nominal", "fixed middle placement", "eligible_unique_nominal_geometry", ""),
    "S12_inlet_two_middle_outlet": ("nominal", "fixed inlet/two-middle/outlet placement", "eligible_unique_nominal_geometry", ""),
    "S13_noisy_inlet_only": ("sensor noise", "fixed inlet placement", "robustness_scenario", "S1_inlet_only"),
    "S14_outlet_only": ("nominal", "fixed outlet placement", "eligible_unique_nominal_geometry", ""),
    "S15_noisy_inlet_outlet_5pct": ("5% sensor noise", "fixed inlet/outlet placement", "robustness_scenario", "S2_inlet_outlet"),
    "S16_peak_dropout_five_sensors": ("peak-period sensor dropout", "fixed five-sensor placement", "robustness_scenario", "S4_five_sensors"),
    "S17_optimized_two_sensors": ("nominal", "optimization-labelled duplicate geometry", "duplicate_geometry_excluded", "S2_inlet_outlet"),
}


def canonical_nodes(value: object) -> str:
    return ";".join(str(int(float(item.strip()))) for item in str(value).split(";") if item.strip())


def esc(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_table(frame: pd.DataFrame, name: str, caption: str, label: str, resize: bool = True) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\toprule"]
    lines.append(" & ".join(esc(column) for column in frame.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in frame.iterrows():
        lines.append(" & ".join(esc(row[column]) for column in frame.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines += [r"%", r"}"]
    lines += [r"\end{table}"]
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    source = pd.read_csv(RESULTS / "sensor_layout_comparison_final.csv").copy()
    interpretation = pd.read_csv(RESULTS / "sensor_layout_interpretation_final.csv")
    distance_columns = [
        "sensor_layout",
        "max_unobserved_distance_km",
        "nearest_sensor_distance_mean_km",
        "contains_middle_sensor",
        "contains_outlet_sensor",
    ]
    available_distance_columns = [column for column in distance_columns if column in interpretation.columns]
    source = source.merge(
        interpretation[available_distance_columns],
        on="sensor_layout",
        how="left",
        validate="one_to_one",
    )
    source["canonical_node_set"] = source["sensor_nodes"].map(canonical_nodes)
    source["sensor_count"] = source["canonical_node_set"].map(lambda x: len(x.split(";")))

    classification_rows: list[dict[str, object]] = []
    for _, row in source.iterrows():
        layout = str(row["sensor_layout"])
        condition, method, status, comparator = SCENARIO_METADATA.get(
            layout,
            ("unclassified", "not recorded", "exclude_pending_metadata", ""),
        )
        classification_rows.append(
            {
                "layout_id": layout,
                "canonical_node_set": row["canonical_node_set"],
                "sensor_count": int(row["sensor_count"]),
                "sensor_quality_condition": condition,
                "dropout_or_bias_condition": condition if "dropout" in condition or "bias" in condition else "none recorded",
                "noise_or_bias_condition": condition if "noise" in condition or "bias" in condition else "none recorded",
                "selection_or_optimization_method": method,
                "training_seed_or_run": "single saved balanced-mode run; seed not recorded in layout CSV",
                "evaluation_scope": "all final layout metrics; C thermal and S hydraulic corridor benchmark",
                "classification": status,
                "comparable_reference_layout": comparator,
                "classification_reason": (
                    "unique nominal node geometry under the locked corridor protocol"
                    if status == "eligible_unique_nominal_geometry"
                    else "same node set retained for a condition/protocol comparison, not a distinct placement geometry"
                ),
            }
        )
    classification = pd.DataFrame(classification_rows)
    classification.to_csv(RESULTS / "sensor_layout_scenario_classification.csv", index=False)

    geometry = source.merge(classification[["layout_id", "classification"]], left_on="sensor_layout", right_on="layout_id", how="left")
    geometry = geometry[geometry["classification"].eq("eligible_unique_nominal_geometry")].copy()
    geometry["direct_thermal_score"] = geometry["RMSE_Ts_full"] + geometry["RMSE_Tr_full"]
    geometry["hydraulic_score"] = geometry["RMSE_H_full"] + geometry["RMSE_q_full"]
    geometry["physical_consistency_score"] = (
        geometry["heat_loss_error_percent"] + geometry["energy_balance_residual"] + geometry["boundary_residual_mean"]
    )
    geometry["practical_low_sensor_score"] = geometry["direct_thermal_score"] + 0.2 * geometry["physical_consistency_score"]
    geometry.loc[geometry["sensor_count"] > 3, "practical_low_sensor_score"] = float("nan")

    objective_specs = [
        ("Direct thermal accuracy", "direct_thermal_score", "RMSE_Ts + RMSE_Tr"),
        ("Hydraulic reconstruction", "hydraulic_score", "RMSE_H + RMSE_q"),
        ("Physical consistency", "physical_consistency_score", "heat-loss error + energy residual + boundary residual"),
        ("Practical low-sensor layout", "practical_low_sensor_score", "<=3 sensors: direct score + 0.2 x physical score"),
    ]
    ranking_rows: list[dict[str, object]] = []
    for objective, column, definition in objective_specs:
        ranked = geometry.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
        for idx, row in ranked.iterrows():
            ranking_rows.append(
                {
                    "objective": objective,
                    "rank": idx + 1,
                    "sensor_layout": row["sensor_layout"],
                    "score": float(row[column]),
                    "score_definition": definition,
                    "sensor_nodes": row["canonical_node_set"],
                    "sensor_count": int(row["sensor_count"]),
                    "comparison_class": "unique nominal placement geometry",
                    "max_unobserved_distance_km": float(row.get("max_unobserved_distance_km", float("nan"))),
                    "safe_interpretation": "Ranked only among unique nominal corridor geometries; not a field installation recommendation.",
                }
            )
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(RESULTS / "sensor_layout_geometry_ranking.csv", index=False)

    robustness_rows: list[dict[str, object]] = []
    for _, meta in classification[classification["classification"].eq("robustness_scenario")].iterrows():
        scenario = source[source["sensor_layout"].eq(meta["layout_id"])].iloc[0]
        baseline = source[source["sensor_layout"].eq(meta["comparable_reference_layout"])]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        scenario_direct = float(scenario["RMSE_Ts_full"] + scenario["RMSE_Tr_full"])
        base_direct = float(base["RMSE_Ts_full"] + base["RMSE_Tr_full"])
        scenario_physical = float(scenario["heat_loss_error_percent"] + scenario["energy_balance_residual"] + scenario["boundary_residual_mean"])
        base_physical = float(base["heat_loss_error_percent"] + base["energy_balance_residual"] + base["boundary_residual_mean"])
        robustness_rows.append(
            {
                "scenario_layout": meta["layout_id"],
                "reference_nominal_layout": meta["comparable_reference_layout"],
                "canonical_node_set": meta["canonical_node_set"],
                "condition": meta["sensor_quality_condition"],
                "direct_thermal_score": scenario_direct,
                "direct_score_change_vs_reference_percent": 100.0 * (scenario_direct - base_direct) / base_direct,
                "physical_consistency_score": scenario_physical,
                "physical_score_change_vs_reference_percent": 100.0 * (scenario_physical - base_physical) / base_physical,
                "interpretation": "Single-run condition comparison; not a placement rank or a statistical robustness estimate.",
            }
        )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(RESULTS / "sensor_layout_robustness_scenarios.csv", index=False)

    rank_top = ranking[ranking["rank"].le(3)].copy()
    rank_top["score"] = rank_top["score"].map(fmt)
    write_table(
        rank_top[["objective", "rank", "sensor_layout", "sensor_nodes", "sensor_count", "score", "score_definition"]],
        "table_sensor_layout_geometry_comparison.tex",
        "Objective-dependent ranking of unique nominal sensor-placement geometries on the calibrated-corridor benchmark. Noise, dropout, and duplicate-node selection-label scenarios are excluded from this placement comparison and reported separately.",
        "tab:sensor_geometry_ranking",
        True,
    )
    write_table(
        classification,
        "table_sensor_layout_scenario_classification.tex",
        "Sensor-layout scenario classification. Identical node sets are not treated as different placement geometries when labels instead encode noise, dropout, or selection/protocol differences.",
        "tab:sensor_scenario_classification",
        True,
    )
    robustness_display = robustness.copy()
    for column in ["direct_thermal_score", "direct_score_change_vs_reference_percent", "physical_consistency_score", "physical_score_change_vs_reference_percent"]:
        robustness_display[column] = robustness_display[column].map(fmt)
    write_table(
        robustness_display,
        "table_sensor_layout_robustness_scenarios.tex",
        "Condition-specific sensor robustness scenarios, paired with the same-node nominal geometry where available. These rows are not placement rankings.",
        "tab:sensor_robustness_scenarios",
        True,
    )

    main_rows = ranking[ranking["rank"].eq(1)].copy()
    main_rows["Objective"] = main_rows["objective"]
    main_rows["Leading geometry"] = main_rows["sensor_layout"]
    main_rows["Sensors"] = main_rows["sensor_count"].astype(int)
    main_rows["Nodes"] = main_rows["sensor_nodes"]
    main_rows["Score"] = main_rows["score"].map(fmt)
    main_rows["Max gap (km)"] = main_rows["max_unobserved_distance_km"].map(lambda x: fmt(x, 1))
    write_table(
        main_rows[["Objective", "Leading geometry", "Sensors", "Nodes", "Score", "Max gap (km)"]],
        "table_main_6_sensor_layout.tex",
        "Objective-specific ranking of unique nominal sensor-placement geometries on the calibrated-corridor benchmark. Scores are objective-specific composites; condition scenarios are reported separately and are not placement designs.",
        "tab:main_sensor_layout",
        True,
    )

    direct = main_rows[main_rows["Objective"].eq("Direct thermal accuracy")].iloc[0]
    physical = main_rows[main_rows["Objective"].eq("Physical consistency")].iloc[0]
    practical = main_rows[main_rows["Objective"].eq("Practical low-sensor layout")].iloc[0]
    energy = pd.read_csv(RESULTS / "scenario_sensor_layout_energy_impact.csv")
    s2_energy = energy[energy["sensor_layout"].eq("S2_inlet_outlet")].iloc[0]
    s4_energy = energy[energy["sensor_layout"].eq("S4_five_sensors")].iloc[0]
    guidelines = pd.DataFrame(
        [
            ["Direct thermal reconstruction", direct["Leading geometry"], f"score {direct['Score']}; max gap {direct['Max gap (km)']} km", "Use only if four sensors are feasible; C-class thermal placement result."],
            ["Physical consistency / heat-loss", physical["Leading geometry"], f"score {physical['Score']}; max gap {physical['Max gap (km)']} km", "No universal optimum; heat-loss target is C class."],
            ["Low-sensor compromise", practical["Leading geometry"], f"score {practical['Score']}; {int(practical['Sensors'])} sensors", "A benchmark compromise, not a cost-optimized design."],
            ["Minimum boundary monitoring", "S2_inlet_outlet", f"pressure-drop residual {float(s2_energy['pressure_drop_residual_percent']):.2f}%", "Use S1 only for coarse context; pressure is S."],
            ["Energy-impact proxy tracking", "S4_five_sensors", f"pump-energy proxy {float(s4_energy['normalized_pump_energy_proxy_kWh_per_MWh']):.2f} kWh/MWh", "S4 is the canonical five-sensor geometry; cost/CO2 are proxies."],
        ],
        columns=["Operator objective", "Geometry", "Evidence", "Boundary"],
    )
    guidelines.to_csv(RESULTS / "operator_sensor_guidelines.csv", index=False)
    write_table(
        guidelines,
        "table_operator_sensor_guidelines.tex",
        "Objective-specific guidance derived from unique nominal corridor geometries. It does not constitute field installation guidance; hydraulic and energy entries are simulator-assisted or proxy quantities.",
        "tab:operator_sensor_guidelines",
        True,
    )


if __name__ == "__main__":
    main()
