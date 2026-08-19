from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.data_registry import DataRegistry, check_dataset_available, list_available_raw_files
from src.sensor_layouts import layout_table_rows
try:
    from src.utils import ensure_dir
except Exception:  # pragma: no cover - avoids PyTorch dependency for paper-only assets.
    def ensure_dir(path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        return out


def _write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, resize: bool = False) -> None:
    ensure_dir(path.parent)
    if df.empty:
        df = pd.DataFrame([{"status": "Not available in this run"}])
    try:
        latex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    except Exception:
        latex = _manual_latex_table(df, caption, label)
    if resize or len(df.columns) > 5:
        latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    path.write_text(latex, encoding="utf-8")


def _manual_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    def esc(value: Any) -> str:
        text = "" if value is None else str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    columns = [str(c) for c in df.columns]
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{esc(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * max(1, len(columns)) + r"}",
        r"\toprule",
        " & ".join(esc(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(row.get(c, "")) for c in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def _fmt_value(value: Any, decimals: int = 3, suffix: str = "") -> str:
    try:
        if pd.isna(value):
            return "not run"
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        text = str(value)
        return text if text else "not run"


def _compact_model_name(name: Any) -> str:
    return str(name).replace("Proposed PI-GNN-GRU", "PI-GNN-GRU")


def _write_status_table(path: Path, caption: str, label: str, status: str, note: str) -> None:
    df = pd.DataFrame([{"Status": status, "Interpretation": note}])
    _write_latex_table(df, path, caption, label)


def _metric_summary_table(metrics: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for metric in metric_names:
        subset = metrics[metrics["metric"].astype(str).eq(metric)].copy()
        if subset.empty:
            continue
        subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
        subset = subset.dropna(subset=["value"]).sort_values("value").reset_index(drop=True)
        if subset.empty:
            continue
        v3 = subset[subset["model"].astype(str).str.contains("PI-GNN-GRU-v3", regex=False)].head(1)
        best = subset.iloc[0]
        rows.append(
            {
                "Metric": metric.replace("_", " "),
                "Best model": _compact_model_name(best.get("model", "")),
                "Best value": _fmt_value(best.get("value")),
                "PI-GNN-v3 value": _fmt_value(v3.iloc[0]["value"]) if not v3.empty else "not run",
                "PI-GNN-v3 rank": int(v3.index[0]) + 1 if not v3.empty else "not run",
                "State type": best.get("state_type", ""),
                "Interpretation": best.get("interpretation", ""),
            }
        )
    return pd.DataFrame(rows)


def _write_thermo_hydraulic_tables(tables_dir: Path, results_dir: Path) -> None:
    metrics = _read_csv(results_dir / "thermo_hydraulic_estimation_metrics.csv")
    thermal = _read_csv(results_dir / "thermal_state_metrics.csv")
    hydraulic = _read_csv(results_dir / "hydraulic_state_metrics.csv")
    heat = _read_csv(results_dir / "heat_energy_metrics.csv")
    sparse = _correct_corridor_sensor_evidence(_read_csv(results_dir / "measured_vs_hidden_state_metrics.csv"))
    sparse.to_csv(results_dir / "measured_vs_hidden_state_metrics_evidence_corrected.csv", index=False)
    coupling = _read_csv(results_dir / "thermo_hydraulic_coupling_metrics.csv")
    robustness = _read_csv(results_dir / "thermo_hydraulic_robustness.csv")
    heat_profile = _read_csv(results_dir / "heat_loss_profile_metrics.csv")

    all_metrics = [
        "RMSE_Ts_supply_C",
        "RMSE_Tr_return_C",
        "RMSE_head_m",
        "RMSE_pressure_kPa",
        "RMSE_flow_kg_s",
        "delivered_heat_error_percent",
        "heat_loss_error_percent",
        "energy_balance_residual_percent",
        "measured_node_temperature_RMSE_C",
        "unmeasured_node_temperature_RMSE_C",
    ]
    _write_latex_table(
        _metric_summary_table(metrics, all_metrics),
        tables_dir / "table_thermo_hydraulic_estimation_summary.tex",
        "Thermo-hydraulic estimation metric summary. State-type labels distinguish measured-node validation, calibrated simulator quantities, and simulator-assisted hidden states.",
        "tab:thermo_hydraulic_summary",
        )
    _write_latex_table(
        _metric_summary_table(metrics, all_metrics),
        tables_dir / "table3_thermo_hydraulic_summary.tex",
        "Thermo-hydraulic estimation metric summary. State-type labels distinguish measured-node validation, calibrated simulator quantities, and simulator-assisted hidden states.",
        "tab:thermo_hydraulic_summary",
        resize=True,
    )

    _write_latex_table(
        _metric_summary_table(
            thermal,
            [
                "RMSE_Ts_supply_C",
                "MAE_Ts_supply_C",
                "MaxAE_Ts_supply_C",
                "RMSE_Tr_return_C",
                "MAE_Tr_return_C",
                "MaxAE_Tr_return_C",
                "outlet_supply_temp_error_C",
                "outlet_return_temp_error_C",
                "thermal_delay_error_min",
                "temperature_gradient_error_C_per_km",
            ],
        ),
        tables_dir / "table_thermal_estimation.tex",
        "Temperature estimation results. Measured-node validation is available only for real thermal variables; distributed supply and return temperature fields are evaluated against calibrated-simulator-assisted hidden-state labels.",
        "tab:thermal_estimation",
        resize=True,
    )
    _write_latex_table(
        _metric_summary_table(
            hydraulic,
            [
                "RMSE_head_m",
                "MAE_head_m",
                "MaxAE_head_m",
                "RMSE_pressure_kPa",
                "MAE_pressure_kPa",
                "pressure_drop_error_percent",
                "RMSE_flow_kg_s",
                "MAE_flow_kg_s",
                "flow_balance_error_percent",
                "pump_head_boundary_error_m",
            ],
        ),
        tables_dir / "table_hydraulic_estimation.tex",
        "Pressure/head and flow estimation results. Hydraulic states are simulator-assisted hidden states because dense real pressure/flow data are unavailable.",
        "tab:hydraulic_estimation",
        resize=True,
    )
    _write_latex_table(
        _metric_summary_table(
            heat,
            [
                "delivered_heat_error_percent",
                "heat_loss_error_percent",
                "segment_heat_loss_RMSE_kW",
                "cumulative_heat_loss_error_percent",
                "energy_balance_residual_percent",
                "return_temperature_energy_error_percent",
                "heat_delivery_ratio_error_percent",
            ],
        ),
        tables_dir / "table_heat_energy_estimation.tex",
        "Heat delivery, heat loss, and energy-balance estimation results. Delivered heat is tied to real boundary heat-load data where available; heat-loss and energy-balance quantities are calibrated-simulator or reconstructed hidden-state metrics, not direct pipe-level field measurements.",
        "tab:heat_energy_estimation",
        resize=True,
    )
    _write_latex_table(
        coupling,
        tables_dir / "table_coupling_summary.tex",
        "Thermo-hydraulic coupling summary. Heat-load and boundary-temperature terms use real measured operating data where available; flow, pressure/head, internal temperature, heat-loss, and delay quantities are calibrated-simulator-assisted hidden-state or proxy metrics.",
        "tab:coupling_summary",
        resize=True,
    )

    _write_latex_table(thermal, tables_dir / "supplementary_temperature_error_by_node.tex", "Supplementary detailed temperature-error metrics.", "tab:supp_temperature_detail", resize=True)
    _write_latex_table(hydraulic, tables_dir / "supplementary_head_pressure_error_by_node.tex", "Supplementary detailed head/pressure/flow metrics.", "tab:supp_head_pressure_detail", resize=True)
    _write_latex_table(heat_profile, tables_dir / "supplementary_heat_loss_by_segment.tex", "Supplementary heat-loss profile by pipe segment.", "tab:supp_heat_loss_segment", resize=True)
    _write_latex_table(robustness, tables_dir / "supplementary_thermo_hydraulic_robustness.tex", "Supplementary robustness to sensor noise, dropout, and parameter uncertainty.", "tab:supp_thermo_robustness", resize=True)
    _write_latex_table(
        sparse,
        tables_dir / "supplementary_measured_vs_hidden_metrics.tex",
        "Supplementary corridor sensor-mask versus hidden-state metrics. Simulated sensor-node quantities are calibrated-simulator benchmark quantities, not real measured-node validation.",
        "tab:supp_measured_hidden",
        resize=True,
    )


def _correct_corridor_sensor_evidence(metrics: pd.DataFrame) -> pd.DataFrame:
    """Correct legacy labels without changing a single numerical result.

    The corridor benchmark has a sensor mask, but not real pipe-node labels.
    Keeping the old word 'measured' in its metric names would incorrectly merge
    model-assimilation positions with the separate Sønderborg/XAI4HEAT measured
    validation experiments.
    """
    if metrics.empty:
        return metrics
    out = metrics.copy()
    mask = out.get("metric", pd.Series(index=out.index, dtype=str)).astype(str).eq("measured_node_temperature_RMSE_C")
    if mask.any():
        out.loc[mask, "metric"] = "simulated_sensor_node_temperature_RMSE_C"
        out.loc[mask, "state_type"] = "calibrated_simulator"
        out.loc[mask, "interpretation"] = "temperature error at simulated corridor sensor nodes"
        out.loc[mask, "safe_claim"] = (
            "This is a calibrated-simulator corridor benchmark at sensor-mask nodes, not a real measured-node validation claim."
        )
    return out


def _write_supplementary_robustness_tables(tables_dir: Path, results_dir: Path) -> None:
    seasonal = _read_csv(results_dir / "seasonal_generalization.csv")
    seasonal_cols = [
        c
        for c in [
            "regime",
            "model",
            "supply_RMSE_C",
            "return_RMSE_C",
            "heat_loss_error_percent",
            "energy_balance_residual_percent",
            "thermal_delay_error_min",
            "pressure_drop_error_percent",
        ]
        if c in seasonal.columns
    ]
    _write_latex_table(
        seasonal[seasonal_cols] if seasonal_cols else seasonal,
        tables_dir / "table_seasonal_generalization.tex",
        "Seasonal generalization under heat-load regimes. Seasonal regimes are derived from real Sønderborg operating data; distributed temperature, pressure/head, flow, and heat-loss states are generated by the calibrated simulator and evaluated as simulator-assisted hidden states.",
        "tab:seasonal_generalization",
        resize=True,
    )

    combined = _read_csv(results_dir / "combined_stress_summary.csv")
    combined_rows = []
    for _, row in combined.iterrows():
        combined_rows.append(
            {
                "Case": str(row.get("case", "")).replace("_", " "),
                "Ts RMSE (C)": _fmt_value(row.get("mean_supply_RMSE_C")),
                "Tr RMSE (C)": _fmt_value(row.get("mean_return_RMSE_C")),
                "Loss error [percent]": _fmt_value(row.get("mean_heat_loss_error_percent")),
                "Energy residual [percent]": _fmt_value(row.get("mean_energy_residual_percent")),
                "Max. T error (C)": _fmt_value(row.get("max_temperature_error_C")),
                "Max. head error (m)": _fmt_value(row.get("max_head_error_m")),
            }
        )
    _write_latex_table(
        pd.DataFrame(combined_rows),
        tables_dir / "table_combined_stress_test.tex",
        "Combined stress test, averaged across the compared estimators. Temperature errors are in degrees Celsius. Disturbances are controlled perturbations of real profiles, not observed field faults; full model-level values are retained in the locked reviewer archive.",
        "tab:combined_stress_test",
        resize=True,
    )

    sensitivity = _read_csv(results_dir / "parameter_identifiability_sensitivity.csv")
    sensitivity_cols = [
        c
        for c in [
            "case",
            "model",
            "parameter",
            "perturbation",
            "supply_RMSE_C",
            "return_RMSE_C",
            "pressure_drop_error_percent",
            "flow_RMSE_m3_s",
            "heat_loss_error_percent",
            "energy_balance_residual_percent",
            "thermal_delay_error_min",
        ]
        if c in sensitivity.columns
    ]
    _write_latex_table(
        sensitivity[sensitivity_cols] if sensitivity_cols else sensitivity,
        tables_dir / "table_parameter_identifiability_sensitivity.tex",
        "Parameter-identifiability sensitivity of effective thermo-hydraulic parameters. Parameters are calibrated effective quantities rather than independently measured pipe or hydraulic properties.",
        "tab:parameter_identifiability_sensitivity",
        resize=True,
    )


def _write_digital_twin_tables(tables_dir: Path, results_dir: Path) -> None:
    kpis = _read_csv(results_dir / "digital_twin_kpis.csv")
    kpi_cols = [c for c in ["kpi", "value", "unit", "state_type", "operational_use"] if c in kpis.columns]
    _write_latex_table(
        kpis[kpi_cols] if kpi_cols else kpis,
        tables_dir / "table_digital_twin_kpis.tex",
        "Operational digital-twin KPIs. KPIs combine real measured-node evidence, calibrated-simulator quantities, and simulator-assisted hidden hydraulic states.",
        "tab:digital_twin_kpis",
        resize=True,
    )

    uncertainty = _read_csv(results_dir / "uncertainty_conformal_evaluation_locked.csv")
    if uncertainty.empty:
        uncertainty = pd.DataFrame([{"status": "not run; locked split-conformal evaluation is required"}])
    uncertainty_cols = [
        c
        for c in [
            "quantity",
            "interval",
            "method",
            "mean_interval_width",
            "unit",
            "coverage",
            "measured_node_coverage",
            "unmeasured_node_coverage",
            "state_type",
            "evaluation_protocol",
        ]
        if c in uncertainty.columns
    ]
    uncertainty_display = (uncertainty[uncertainty_cols] if uncertainty_cols else uncertainty).copy()
    # Heat-loss intervals are scalar totals, so node-wise coverage is not applicable.
    # Render this explicitly instead of allowing a misleading NaN in the paper.
    uncertainty_display = uncertainty_display.where(uncertainty_display.notna(), "not applicable")
    _write_latex_table(
        uncertainty_display,
        tables_dir / "table_uncertainty_metrics.tex",
        "Held-out 90 percent split-conformal virtual-sensor uncertainty evaluation. Raw ensemble-residual diagnostics are not pooled with these values. Pressure/head and flow intervals concern simulator-assisted hidden hydraulic states.",
        "tab:uncertainty_metrics",
        resize=True,
    )
    _write_latex_table(
        uncertainty_display,
        tables_dir / "table_uncertainty_calibration.tex",
        "Held-out 90 percent split-conformal uncertainty evaluation. Intervals are empirical chronological confidence diagnostics, not perfect probabilistic forecasts.",
        "tab:uncertainty_calibration",
        resize=True,
    )
    _write_latex_table(
        uncertainty_display,
        tables_dir / "table_uncertainty_by_variable.tex",
        "Per-variable held-out 90 percent split-conformal coverage and interval width. Widths are not averaged across units.",
        "tab:uncertainty_by_variable",
        resize=True,
    )

    anomaly = _read_csv(results_dir / "anomaly_detection_metrics_improved.csv")
    if anomaly.empty:
        anomaly = _read_csv(results_dir / "anomaly_detection_metrics.csv")
    anomaly_cols = [
        c
        for c in [
            "case",
            "selected_score",
            "detection_rate_percent",
            "false_alarm_rate_percent",
            "detection_delay_min",
            "best_F1_score",
            "best_threshold_quantile",
            "max_residual_score",
            "state_type",
        ]
        if c in anomaly.columns
    ]
    _write_latex_table(
        anomaly[anomaly_cols] if anomaly_cols else anomaly,
        tables_dir / "table_anomaly_detection_metrics.tex",
        "Residual-based anomaly detection metrics. Cases are controlled perturbations of real operating profiles, not documented field faults.",
        "tab:anomaly_detection_metrics",
        resize=True,
    )
    _write_latex_table(
        anomaly[anomaly_cols] if anomaly_cols else anomaly,
        tables_dir / "table_anomaly_detection_improved.tex",
        "Improved residual-based anomaly detection metrics with multi-residual scores and temporal accumulation.",
        "tab:anomaly_detection_improved",
        resize=True,
    )

    flensburg = _read_csv(results_dir / "flensburg_domain_shift_analysis_improved.csv")
    flensburg_cols = [c for c in flensburg.columns if c in {
        "analysis",
        "sampling_interval_sonderborg_min",
        "sampling_interval_flensburg_min",
        "return_temperature_assumed",
        "heat_load_kw_wasserstein_distance",
        "supply_temp_C_wasserstein_distance",
        "heat_load_kw_mean_difference",
        "supply_temp_C_mean_difference",
        "note",
    }]
    _write_latex_table(
        flensburg[flensburg_cols] if flensburg_cols else flensburg,
        tables_dir / "table_flensburg_domain_shift_improved.tex",
        "Improved Flensburg domain-shift diagnostics. Flensburg is interpreted as a domain-shift stress test rather than universal transfer evidence.",
        "tab:flensburg_domain_shift_improved",
        resize=True,
    )
    _write_latex_table(
        flensburg[flensburg_cols] if flensburg_cols else flensburg,
        tables_dir / "table6_flensburg_domain_shift.tex",
        "Flensburg domain-shift summary. The external dataset is interpreted as a transfer stress test rather than proof of universal generalization.",
        "tab:flensburg_domain_shift",
        resize=True,
    )

    param = _read_csv(results_dir / "parameter_sensitivity_summary.csv")
    param_rows = []
    for _, row in param.iterrows():
        param_rows.append(
            {
                "Case": str(row.get("case", "")).replace("_", " "),
                "Effective parameter": str(row.get("parameter", "")).replace("_", " "),
                "Tr RMSE (C)": _fmt_value(row.get("mean_return_RMSE_C")),
                "Loss error [percent]": _fmt_value(row.get("mean_heat_loss_error_percent")),
                "Pressure-drop error [percent]": _fmt_value(row.get("mean_pressure_drop_error_percent")),
                "Energy residual [percent]": _fmt_value(row.get("mean_energy_balance_residual_percent")),
            }
        )
    _write_latex_table(
        pd.DataFrame(param_rows),
        tables_dir / "table_parameter_identifiability_sensitivity_improved.tex",
        "Effective-parameter sensitivity averaged across the compared estimators. These are not independently measured pipe-material or hydraulic properties; full model-level values are retained in the locked reviewer archive.",
        "tab:parameter_identifiability_sensitivity_improved",
        resize=True,
    )


def _write_compact_supplementary_ablation_table(tables_dir: Path, results_dir: Path) -> None:
    """Render the readable ablation subset; retain the full CSV in the archive."""
    ablation = _read_csv(results_dir / "ablation_study_final.csv")
    if ablation.empty:
        ablation = _read_csv(results_dir / "ablation_study.csv")
    rows = []
    for _, row in ablation.iterrows():
        rows.append(
            {
                "Ablation": str(row.get("ablation", "")).replace("_", " "),
                "Ts RMSE (C)": _fmt_value(row.get("RMSE_Ts_full")),
                "Loss error [percent]": _fmt_value(row.get("heat_loss_error_percent")),
                "Energy residual [percent]": _fmt_value(row.get("energy_balance_residual")),
                "Thermal residual": _fmt_value(row.get("thermal_residual_mean"), decimals=4),
                "Boundary residual": _fmt_value(row.get("boundary_residual_mean"), decimals=4),
            }
        )
    _write_latex_table(
        pd.DataFrame(rows),
        tables_dir / "supplementary_ablation_study_full.tex",
        "Single-run physics-informed component ablation. Temperature errors are in degrees Celsius; the full metric matrix is retained in the locked reviewer archive.",
        "tab:supp_ablation_full",
        resize=True,
    )


def _write_operational_energy_impact_tables(tables_dir: Path, results_dir: Path) -> None:
    nominal = _read_csv(results_dir / "scenario_nominal_winter_energy_impact.csv")
    stress = _read_csv(results_dir / "scenario_combined_stress_energy_impact.csv")
    layouts = _read_csv(results_dir / "scenario_sensor_layout_energy_impact.csv")
    assumptions_path = results_dir / "operational_energy_impact_assumptions.json"
    rows: list[dict[str, Any]] = []

    def fmt(value: Any) -> str:
        try:
            value = float(value)
            if pd.isna(value):
                return "not reported"
            return f"{value:.3f}"
        except Exception:
            return "not reported"

    def add_row(label: str, row: dict[str, Any]) -> None:
        rows.append(
            {
                "Scenario": label,
                "Evaluation horizon": str(row.get("evaluation_horizon", "stated scenario horizon")),
                "Delivered heat (MWh/day)": fmt(row.get("delivered_heat_MWh_per_day", row.get("delivered_heat_MWh"))),
                "Heat loss (MWh/day)": fmt(row.get("heat_loss_MWh_per_day", row.get("heat_loss_MWh"))),
                "Heat-loss ratio (%)": fmt(row.get("heat_loss_ratio_percent")),
                "Pump-energy intensity (kWh/MWh)": fmt(row.get("normalized_pump_energy_proxy_kWh_per_MWh", row.get("pump_energy_proxy_kWh"))),
                "Pressure-drop residual (%)": fmt(row.get("pressure_drop_residual_percent", row.get("maximum_pressure_drop_residual_percent"))),
                "Energy-balance residual (%)": fmt(
                    row.get(
                        "energy_balance_residual_percent",
                        row.get("maximum_energy_balance_residual_percent", row.get("mean_energy_balance_residual_percent")),
                    )
                ),
                "Cost proxy (EUR/day)": fmt(row.get("cost_proxy_EUR_per_day", row.get("cost_proxy_EUR"))),
                "CO2 proxy (kg/day)": fmt(row.get("CO2_proxy_kg_per_day", row.get("CO2_proxy_kg"))),
            }
        )

    if not nominal.empty:
        add_row("nominal winter-day", nominal.iloc[0].to_dict())
    severe = stress[stress.get("scenario", pd.Series(dtype=str)).astype(str).eq("combined_stress_severe")] if not stress.empty else pd.DataFrame()
    if not severe.empty:
        add_row("combined stress", severe.iloc[0].to_dict())
    for label in ["inlet only", "inlet + outlet", "five sensors"]:
        sub = layouts[layouts.get("scenario", pd.Series(dtype=str)).astype(str).eq(label)] if not layouts.empty else pd.DataFrame()
        if not sub.empty:
            add_row(f"{label} layout", sub.iloc[0].to_dict())
    opt = layouts[layouts.get("scenario", pd.Series(dtype=str)).astype(str).str.contains("optimized", regex=False)].tail(1) if not layouts.empty else pd.DataFrame()
    if not opt.empty:
        add_row("optimized-sensor layout", opt.iloc[0].to_dict())

    table = pd.DataFrame(rows)
    if table.empty:
        note = "Operational energy-impact scenarios were not generated in this run."
        if assumptions_path.exists():
            note += " Assumptions file exists but scenario CSVs are missing."
        table = pd.DataFrame([{"Scenario": "not run", "Interpretation": note}])
    _write_latex_table(
        table,
        tables_dir / "table_scenario_energy_impact_summary.tex",
        (
            "Operational energy-impact scenario summary. Values are reported over the stated evaluation horizon unless normalized; "
            "cost and CO2 values are proxy indicators, not optimized economic-dispatch results. Pressure/head and flow-based entries "
            "are simulator-assisted hidden-state proxies."
        ),
        "tab:scenario_energy_impact_summary",
        resize=True,
    )


def _write_operator_sensor_guidelines_table(tables_dir: Path, results_dir: Path) -> None:
    geometry_ranking = results_dir / "sensor_layout_geometry_ranking.csv"
    ranking = _read_csv(geometry_ranking if geometry_ranking.exists() else results_dir / "sensor_layout_ranking_by_objective.csv")
    layouts = _read_csv(results_dir / "sensor_layout_comparison_final.csv")
    energy = _read_csv(results_dir / "scenario_sensor_layout_energy_impact.csv")

    def fmt(value: Any, decimals: int = 3, suffix: str = "") -> str:
        try:
            numeric = float(value)
            if pd.isna(numeric):
                return "not reported"
            return f"{numeric:.{decimals}f}{suffix}"
        except Exception:
            return "not reported"

    def layout_name(text: Any) -> str:
        return str(text).replace("_", " ")

    def ranking_row(objective: str, preferred: list[str] | None = None) -> pd.Series | None:
        if ranking.empty:
            return None
        sub = ranking[ranking["objective"].astype(str).eq(objective)].copy()
        if preferred:
            preferred_rows = sub[sub["sensor_layout"].astype(str).isin(preferred)].copy()
            if not preferred_rows.empty:
                sub = preferred_rows
        if sub.empty:
            return None
        if "rank" in sub.columns:
            sub["rank"] = pd.to_numeric(sub["rank"], errors="coerce")
            sub = sub.sort_values(["rank", "score"], ascending=[True, True])
        elif "score" in sub.columns:
            sub["score"] = pd.to_numeric(sub["score"], errors="coerce")
            sub = sub.sort_values("score")
        return sub.iloc[0]

    def energy_row(layout: str | None = None, scenario_contains: str | None = None) -> pd.Series | None:
        if energy.empty:
            return None
        sub = energy.copy()
        if layout:
            sub = sub[sub["sensor_layout"].astype(str).eq(layout)]
        if scenario_contains:
            sub = sub[sub["scenario"].astype(str).str.contains(scenario_contains, case=False, regex=False)]
        if sub.empty:
            return None
        return sub.iloc[0]

    rows: list[dict[str, str]] = []

    direct = ranking_row("Direct thermal accuracy", ["S12_inlet_two_middle_outlet"])
    direct_layout = str(direct.get("sensor_layout", "S12/S10 layouts")) if direct is not None else "S12/S10 layouts"
    rows.append(
        {
            "Operator objective": "Direct thermal accuracy",
            "Recommended layout": layout_name(direct_layout),
            "Key evidence": (
                f"Direct thermal score {fmt(direct.get('score') if direct is not None else None)} "
                f"({direct.get('score_definition', 'RMSE_Ts + RMSE_Tr') if direct is not None else 'RMSE_Ts + RMSE_Tr'}); "
                f"maximum unobserved length {fmt(direct.get('max_unobserved_distance_km') if direct is not None else None, 2, ' km')}."
            ),
            "Operator guidance": "Use the S12 nominal geometry when direct supply/return-temperature reconstruction is the priority and four sensors are feasible.",
            "Claim boundary": "Not universally optimal; ranking is objective-specific and uses calibrated-simulator hidden-state labels away from measured nodes.",
        }
    )

    physical = ranking_row("Physical consistency", ["S8_random_three_sensors"])
    physical_layout = str(physical.get("sensor_layout", "S8_random_three_sensors")) if physical is not None else "S8_random_three_sensors"
    physical_energy = energy_row(layout=physical_layout)
    rows.append(
        {
            "Operator objective": "Physical consistency and heat-loss monitoring",
            "Recommended layout": layout_name(physical_layout),
            "Key evidence": (
                f"Physical-consistency score {fmt(physical.get('score') if physical is not None else None)}; "
                f"heat-loss error {fmt(physical_energy.get('heat_loss_error_percent') if physical_energy is not None else None, 3, '%')}; "
                f"energy residual {fmt(physical_energy.get('energy_balance_residual_percent') if physical_energy is not None else None, 3, '%')}."
            ),
            "Operator guidance": "The S8 nominal geometry leads this mixed physical-consistency composite; choose a five-sensor geometry only when redundancy and coverage are the engineering priority.",
            "Claim boundary": "Heat-loss and pressure/head quantities are calibrated-simulator or simulator-assisted indicators, not dense field measurements.",
        }
    )

    s2_energy = energy_row(layout="S2_inlet_outlet")
    if s2_energy is None:
        s2_energy = energy_row(scenario_contains="inlet + outlet")
    s1_energy = energy_row(layout="S1_inlet_only")
    if s1_energy is None:
        s1_energy = energy_row(scenario_contains="inlet only")
    rows.append(
        {
            "Operator objective": "Low-cost monitoring",
            "Recommended layout": "S2 inlet + outlet; S1 inlet-only only for coarse monitoring",
            "Key evidence": (
                f"S2 pump-energy proxy error {fmt(s2_energy.get('pump_energy_proxy_error_percent') if s2_energy is not None else None, 2, '%')} "
                f"and pressure-drop residual {fmt(s2_energy.get('pressure_drop_residual_percent') if s2_energy is not None else None, 2, '%')}; "
                f"S1 pressure-drop residual {fmt(s1_energy.get('pressure_drop_residual_percent') if s1_energy is not None else None, 2, '%')}."
            ),
            "Operator guidance": "Boundary-only layouts reduce sensor cost, but inlet-only monitoring should be reserved for cases where hidden-state and hydraulic uncertainty are acceptable.",
            "Claim boundary": "Low-cost does not mean best reconstruction; it is a cost/coverage compromise under sparse sensing.",
        }
    )

    opt_energy = energy_row(layout="S4_five_sensors")
    if opt_energy is None:
        opt_energy = energy_row(scenario_contains="five sensors")
    opt_layout = str(opt_energy.get("sensor_layout", "S4_five_sensors")) if opt_energy is not None else "S4_five_sensors"
    rows.append(
        {
            "Operator objective": "Energy-impact monitoring",
            "Recommended layout": layout_name(opt_layout),
            "Key evidence": (
                f"Pump-energy intensity {fmt(opt_energy.get('normalized_pump_energy_proxy_kWh_per_MWh') if opt_energy is not None else None, 2, ' kWh/MWh')}; "
                f"cost proxy {fmt(opt_energy.get('cost_proxy_EUR_per_day') if opt_energy is not None else None, 0, ' EUR/day')}; "
                f"CO2 proxy {fmt(opt_energy.get('CO2_proxy_kg_per_day') if opt_energy is not None else None, 0, ' kg/day')}."
            ),
            "Operator guidance": "Use the canonical S4 five-sensor geometry for the reported energy-impact proxy scenario; its duplicate optimized label is not a separate placement geometry.",
            "Claim boundary": "Cost and CO2 are proxy indicators under stated assumptions; they are not optimized dispatch or verified utility billing results.",
        }
    )

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "operator_sensor_guidelines.csv", index=False)
    _write_latex_table(
        table,
        tables_dir / "table_operator_sensor_guidelines.tex",
        "Operator-facing sensor-placement guidelines by monitoring objective. The recommendations are objective-specific; pressure/head, flow, pump-energy, cost, and CO2 entries are simulator-assisted or proxy indicators.",
        "tab:operator_sensor_guidelines",
        resize=True,
    )


def _copy_figures() -> None:
    out = ensure_dir(PROJECT_ROOT / "paper" / "figures")
    for fig in (PROJECT_ROOT / "figures").glob("*.*"):
        if fig.suffix.lower() in {".pdf", ".png"}:
            shutil.copy2(fig, out / fig.name)


def _regenerate_final_robustness_operational_assets() -> None:
    """Regenerate final scenario tables/figures after the table cleanup step.

    The main asset script removes old ``table*.tex`` files before rebuilding.
    These compact scenario scripts are CSV-driven and avoid optional plotting
    dependencies, so running them here keeps the final ATE supplement intact.
    """
    modules = [
        "src.seasonal_load_regime_analysis",
        "src.stress_fault_scenarios",
        "src.parameter_uncertainty_identifiability",
        "src.sensor_noise_dropout_robustness",
        "src.operational_kpi_quantification",
        "src.sensor_placement_operator_guidelines",
        "src.domain_transfer_diagnostics",
        "src.discretization_temporal_sensitivity",
    ]
    warnings: list[str] = []
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=["run"])
            module.run()
        except Exception as exc:
            warnings.append(f"{module_name}: {exc}")
    if warnings:
        (PROJECT_ROOT / "results" / "final_robustness_asset_generation_warning.txt").write_text(
            "\n".join(warnings) + "\n",
            encoding="utf-8",
        )


def make_paper_assets(config: dict[str, Any]) -> None:
    tables_dir = ensure_dir(PROJECT_ROOT / "paper" / "tables")
    results_dir = PROJECT_ROOT / "results"
    registry = DataRegistry()
    for old in tables_dir.glob("table*.tex"):
        old.unlink()

    dataset_rows = []
    for row in registry.as_rows():
        name = row["dataset"]
        dataset_rows.append(
            {
                "Dataset": name,
                "Role": row["role"],
                "Measured variables": row["measured_variables"],
                "Local raw files": len(list_available_raw_files(name)),
                "Used in current run": "yes" if check_dataset_available(name) else "no",
                "Study use": "boundary/calibration/measured-node validation" if name != "aalborg" else "optional demand enrichment",
            }
        )
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(results_dir / "dataset_roles_table.csv", index=False)
    _write_latex_table(dataset_df, tables_dir / "table1_dataset_roles.tex", "Dataset roles and measured variables.", "tab:dataset_roles")

    gap_rows = [
        {"Approach": "Physics-only dynamic model", "Real operating data": "sometimes", "Sparse sensors": "limited", "Graph topology": "limited", "Thermo-hydraulic physics": "yes", "Heat-loss reconstruction": "yes", "Measured-node validation": "sometimes", "External validation": "rare", "Noise test": "rare"},
        {"Approach": "LSTM/GRU forecasting", "Real operating data": "yes", "Sparse sensors": "sometimes", "Graph topology": "no", "Thermo-hydraulic physics": "no", "Heat-loss reconstruction": "no", "Measured-node validation": "yes", "External validation": "sometimes", "Noise test": "limited"},
        {"Approach": "Transformer forecasting", "Real operating data": "yes", "Sparse sensors": "sometimes", "Graph topology": "no", "Thermo-hydraulic physics": "no", "Heat-loss reconstruction": "no", "Measured-node validation": "yes", "External validation": "sometimes", "Noise test": "limited"},
        {"Approach": "Graph neural estimator", "Real operating data": "sometimes", "Sparse sensors": "yes", "Graph topology": "yes", "Thermo-hydraulic physics": "limited", "Heat-loss reconstruction": "limited", "Measured-node validation": "sometimes", "External validation": "rare", "Noise test": "limited"},
        {"Approach": "Physics-informed neural model", "Real operating data": "sometimes", "Sparse sensors": "limited", "Graph topology": "limited", "Thermo-hydraulic physics": "yes", "Heat-loss reconstruction": "sometimes", "Measured-node validation": "sometimes", "External validation": "rare", "Noise test": "limited"},
        {"Approach": "Energy-system digital twin", "Real operating data": "sometimes", "Sparse sensors": "sometimes", "Graph topology": "sometimes", "Thermo-hydraulic physics": "yes", "Heat-loss reconstruction": "sometimes", "Measured-node validation": "sometimes", "External validation": "rare", "Noise test": "limited"},
        {"Approach": "This study", "Real operating data": "yes, when local data are available", "Sparse sensors": "yes", "Graph topology": "yes", "Thermo-hydraulic physics": "yes", "Heat-loss reconstruction": "simulation-assisted", "Measured-node validation": "yes", "External validation": "Flensburg when available", "Noise test": "yes"},
    ]
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(results_dir / "literature_gap_table.csv", index=False)
    _write_latex_table(gap_df, tables_dir / "table2_literature_gap.tex", "Literature gap and positioning.", "tab:literature_gap")

    calibrated_path = results_dir / "calibrated_parameters.json"
    calibrated = json.loads(calibrated_path.read_text(encoding="utf-8")) if calibrated_path.exists() else {}
    params_rows = []
    for key, value in config["system"].items():
        if isinstance(value, (int, float)) and key in {
            "length_m",
            "dx_m",
            "dt_s",
            "diameter_m",
            "friction_factor",
            "heat_loss_U_W_m2K",
            "pipe_perimeter_m",
            "outlet_head_m",
        }:
            params_rows.append({"Parameter": key, "Default": value, "Calibrated/proxy": calibrated.get(key, "")})
    for key in ["effective_velocity_factor", "return_temperature_offset", "flow_proxy_blend"]:
        params_rows.append({"Parameter": key, "Default": "", "Calibrated/proxy": calibrated.get(key, "")})
    params_df = pd.DataFrame(params_rows)
    params_df.to_csv(results_dir / "thermo_hydraulic_parameters_table.csv", index=False)
    _write_latex_table(params_df, tables_dir / "table3_parameters.tex", "Thermo-hydraulic parameters.", "tab:parameters")
    _write_latex_table(params_df, tables_dir / "table2_main_parameters.tex", "Thermo-hydraulic parameters and calibrated effective parameters.", "tab:main_parameters")

    calibration = _read_csv(results_dir / "calibration_metrics.csv")
    _write_latex_table(calibration, tables_dir / "table4_calibration_metrics.tex", "Calibration metrics and identifiability notes.", "tab:calibration_metrics")
    verification = _read_csv(results_dir / "model_verification_summary.csv")
    cal_ver = pd.concat(
        [
            calibration.assign(source_table="calibration"),
            verification.assign(source_table="model verification"),
        ],
        ignore_index=True,
        sort=False,
    )
    _write_latex_table(cal_ver, tables_dir / "table2_calibration_model_verification.tex", "Calibration and model-verification metrics.", "tab:calibration_verification", resize=True)

    n_nodes = int(round(config["system"]["length_m"] / config["system"]["dx_m"])) + 1
    sensor_layout_def_df = pd.DataFrame(layout_table_rows(n_nodes))
    sensor_layout_def_df.to_csv(results_dir / "sensor_layout_definitions_table.csv", index=False)
    _write_latex_table(sensor_layout_def_df, tables_dir / "table5_sensor_layouts.tex", "Sparse sensor layouts.", "tab:sensor_layouts")

    baseline = _read_csv(results_dir / "baseline_comparison_final.csv")
    if baseline.empty:
        baseline = _read_csv(results_dir / "baseline_comparison_improved.csv")
    if baseline.empty:
        baseline = _read_csv(results_dir / "baseline_comparison.csv")
    _write_latex_table(baseline, tables_dir / "supplementary_baseline_comparison_full.tex", "Full baseline comparison.", "tab:supp_baseline_full")
    baseline_cols = [c for c in ["model", "loss_mode", "RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "thermal_residual_mean", "energy_balance_residual"] if c in baseline.columns]
    _write_latex_table(baseline[baseline_cols] if baseline_cols else baseline, tables_dir / "table6_baseline_comparison.tex", "Baseline comparison.", "tab:baseline_comparison")
    if not baseline.empty:
        direct_rows = []
        for _, row in baseline.iterrows():
            direct_rows.append(
                {
                    "Model": _compact_model_name(row.get("model", "")),
                    "Ts RMSE (C)": _fmt_value(row.get("RMSE_Ts_full")),
                    "Tr RMSE (C)": _fmt_value(row.get("RMSE_Tr_full")),
                    "H RMSE": _fmt_value(row.get("RMSE_H_full")),
                    "q RMSE": _fmt_value(row.get("RMSE_q_full")),
                }
            )
        _write_latex_table(pd.DataFrame(direct_rows), tables_dir / "table6a_direct_rmse.tex", "Direct simulator-hidden-state reconstruction RMSE. Lower is better; these metrics use calibrated-simulator hidden states rather than real dense pipe measurements.", "tab:direct_rmse")
        phys_rows = []
        for _, row in baseline.iterrows():
            phys_rows.append(
                {
                    "Model": _compact_model_name(row.get("model", "")),
                    "Heat-loss error (%)": _fmt_value(row.get("heat_loss_error_percent"), suffix="%"),
                    "Energy residual": _fmt_value(row.get("energy_balance_residual")),
                    "Thermal residual": _fmt_value(row.get("thermal_residual_mean"), decimals=4),
                    "Boundary residual": _fmt_value(row.get("boundary_residual_mean"), decimals=4),
                }
            )
        _write_latex_table(pd.DataFrame(phys_rows), tables_dir / "table6b_physics_consistency.tex", "Physical-consistency metrics. The table demonstrates that heat-loss, energy, thermal, and boundary diagnostics can rank models differently from direct RMSE.", "tab:physics_compact")

    physics = _read_csv(results_dir / "physics_consistency_comparison_final.csv")
    if physics.empty:
        physics = _read_csv(results_dir / "physics_consistency_comparison_improved.csv")
    if physics.empty:
        physics = _read_csv(results_dir / "physics_consistency_comparison.csv")
    _write_latex_table(physics, tables_dir / "supplementary_physics_consistency_full.tex", "Full physical consistency comparison.", "tab:supp_physics_full")
    physics_cols = [c for c in ["model", "heat_loss_error_percent", "energy_balance_residual", "thermal_residual_mean", "boundary_residual_mean"] if c in physics.columns]
    _write_latex_table(physics[physics_cols] if physics_cols else physics, tables_dir / "table7_physics_consistency.tex", "Physical consistency comparison.", "tab:physics_consistency")

    ablation = _read_csv(results_dir / "ablation_study_final.csv")
    if ablation.empty:
        ablation = _read_csv(results_dir / "ablation_study.csv")
    ablation_cols = [c for c in ["ablation", "RMSE_Ts_full", "heat_loss_error_percent", "energy_balance_residual", "thermal_residual_mean", "boundary_residual_mean"] if c in ablation.columns]
    _write_compact_supplementary_ablation_table(tables_dir, results_dir)
    _write_latex_table(ablation[ablation_cols] if ablation_cols else ablation, tables_dir / "table8_ablation_study.tex", "Physics-informed loss ablation study.", "tab:ablation_study")

    layouts = _read_csv(results_dir / "sensor_layout_comparison_final.csv")
    if layouts.empty:
        layouts = _read_csv(results_dir / "sensor_layout_comparison_improved.csv")
    if layouts.empty:
        layouts = _read_csv(results_dir / "sensor_layout_comparison_detailed.csv")
    if layouts.empty:
        layouts = _read_csv(results_dir / "sensor_layout_comparison.csv")
    _write_latex_table(layouts, tables_dir / "supplementary_sensor_layout_comparison_full.tex", "Full sensor-layout condition and placement scenario results. Use the scenario-classification table to distinguish unique nominal geometries from noise, dropout, and duplicate-node protocol rows.", "tab:supp_sensor_layout_full")
    layout_cols = [c for c in ["sensor_layout", "sensor_nodes", "RMSE_Ts_full", "heat_loss_error_percent", "worst_node_temperature_error_C", "outlet_node_temperature_error_C", "peak_period_error_C"] if c in layouts.columns]
    _write_latex_table(layouts[layout_cols] if layout_cols else layouts, tables_dir / "table9_sensor_layout_comparison.tex", "Sensor-layout comparison.", "tab:sensor_layout_comparison")
    geometry_ranking = results_dir / "sensor_layout_geometry_ranking.csv"
    layout_recommendations = _read_csv(geometry_ranking if geometry_ranking.exists() else results_dir / "sensor_layout_ranking_by_objective.csv")
    rec_cols = [c for c in ["objective", "rank", "sensor_layout", "score", "safe_interpretation"] if c in layout_recommendations.columns]
    rec_table = layout_recommendations[rec_cols] if rec_cols else layout_recommendations
    _write_latex_table(rec_table, tables_dir / "table5_sensor_layout_recommendation.tex", "Unique nominal sensor-placement geometry ranking by objective.", "tab:sensor_layout_recommendation", resize=True)
    _write_latex_table(rec_table, tables_dir / "table_sensor_layout_recommendation_by_objective.tex", "Unique nominal sensor-placement geometry ranking by objective. Noise, dropout, and duplicate-node scenarios are reported separately; no layout is universally optimal.", "tab:sensor_layout_recommendation_by_objective", resize=True)

    robustness = _read_csv(results_dir / "noise_dropout_robustness_final.csv")
    if robustness.empty:
        robustness = _read_csv(results_dir / "noise_dropout_robustness.csv")
    robust_cols = [c for c in ["base_model", "condition", "sensor_layout", "noise_std_fraction", "RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual"] if c in robustness.columns]
    _write_latex_table(robustness[robust_cols] if robust_cols else robustness, tables_dir / "table10_noise_dropout.tex", "Noise/dropout robustness.", "tab:noise_dropout")

    external = _read_csv(results_dir / "external_validation_flensburg_modes_final.csv")
    if external.empty:
        external = _read_csv(results_dir / "external_validation_flensburg_modes.csv")
    if external.empty:
        external = _read_csv(results_dir / "external_validation_flensburg.csv")
    # Flensburg has no measured return series in the current public files. Do
    # not let legacy raw columns make an assumed 50 C value look measured.
    external_for_paper = external.copy()
    misleading_return_cols = [
        c
        for c in external_for_paper.columns
        if c in {"RMSE_Tr_measured_nodes", "RMSE_return_measured_C", "RMSE_load_or_return_proxy"}
        or c.startswith("RMSE_Tr")
        or "return_temperature" in c.lower()
    ]
    external_for_paper = external_for_paper.drop(columns=misleading_return_cols, errors="ignore")
    if not external_for_paper.empty:
        external_for_paper["return_validation"] = "not evaluated; assumed 50 C"
    external_full_rows = []
    for _, row in external_for_paper.iterrows():
        mode = str(row.get("mode", "")).replace("_", " ")
        if "direct" in mode:
            interpretation = "zero-shot domain-shift stress test"
        elif "few" in mode:
            interpretation = "small local decoder adaptation"
        elif "calibration" in mode:
            interpretation = "simulator/output-bias adaptation"
        elif "normalized" in mode:
            interpretation = "normalization diagnostic"
        else:
            interpretation = "transfer diagnostic"
        external_full_rows.append(
            {
                "Transfer mode": mode,
                "Measured supply RMSE (C)": _fmt_value(row.get("RMSE_Ts_measured_nodes")),
                "Load consistency (%)": _fmt_value(row.get("heat_load_consistency_error_percent")),
                "C heat-loss error (%)": _fmt_value(row.get("heat_loss_error_percent")),
                "Mixed C+S energy residual (%)": _fmt_value(row.get("energy_balance_residual")),
                "Return evidence": "not evaluated; assumed 50 C",
                "Interpretation": interpretation,
            }
        )
    _write_latex_table(
        pd.DataFrame(external_full_rows),
        tables_dir / "supplementary_external_validation_modes_full.tex",
        "Flensburg transfer modes. Supply RMSE is measured. Return temperature is unavailable and the assumed 50 C value is excluded from measured validation. Heat loss is a calibrated-simulator thermal diagnostic (C), and dynamic energy has mixed C+S dependency; neither is an additional Flensburg field measurement.",
        "tab:supp_external_modes",
    )
    external_cols = [c for c in ["mode", "mode_status", "RMSE_Ts_measured_nodes", "RMSE_Ts_full", "heat_loss_error_percent", "note"] if c in external.columns]
    _write_latex_table(external[external_cols] if external_cols else external, tables_dir / "table11_external_validation.tex", "External validation on Flensburg.", "tab:external_validation")
    if not external.empty:
        mode_rows = []
        for _, row in external.iterrows():
            mode = str(row.get("mode", ""))
            if "direct" in mode:
                interpretation = "zero-shot stress test"
            elif "few" in mode:
                interpretation = "small local adaptation"
            elif "calibration" in mode:
                interpretation = "simulator/output-bias adaptation"
            elif "normalized" in mode:
                interpretation = "domain-shift diagnostic"
            else:
                interpretation = "transfer mode"
            mode_rows.append(
                {
                    "Mode": mode.replace("_", " "),
                    "Ts RMSE (C)": _fmt_value(row.get("RMSE_Ts_measured_nodes")),
                    "Return evidence": "not evaluated; assumed 50 C",
                    "Heat-load error (%)": _fmt_value(row.get("heat_load_consistency_error_percent"), suffix="%"),
                    "Interpretation": interpretation,
                }
            )
        _write_latex_table(pd.DataFrame(mode_rows), tables_dir / "table11_flensburg_modes.tex", "Flensburg external transfer modes. Supply RMSE is measured; return temperature is unavailable and excluded from measured validation. Direct transfer is a domain-shift stress test; adapted modes use limited local information.", "tab:flensburg_modes")

    xai = _read_csv(results_dir / "xai4heat_sparse_substation_validation.csv")
    _write_latex_table(xai, tables_dir / "table12_xai4heat_validation.tex", "XAI4HEAT measured-node validation.", "tab:xai4heat_validation")
    xai_quality = _read_csv(results_dir / "xai4heat_station_quality_audit.csv")
    if not xai_quality.empty:
        xai_quality_cols = [
            c
            for c in [
                "substation_id",
                "raw_rows_after_file_deduplication",
                "valid_primary_temperature_pairs",
                "primary_supply_below_return_percent",
                "primary_ordered_percent",
                "primary_supply_median_C",
                "primary_return_median_C",
            ]
            if c in xai_quality.columns
        ]
        _write_latex_table(
            xai_quality[xai_quality_cols] if xai_quality_cols else xai_quality,
            tables_dir / "table_xai4heat_station_quality_audit.tex",
            "XAI4HEAT station-level primary-temperature quality audit after SHA-256 file-content deduplication. All valid-range records are retained in the primary analysis. The physically ordered subset is a target-conditioned sensitivity analysis and does not diagnose an instrument fault.",
            "tab:xai_quality_audit",
            resize=True,
        )
    numerical_expanded = _read_csv(results_dir / "numerical_verification_expanded.csv")
    _write_latex_table(
        numerical_expanded,
        tables_dir / "table_numerical_verification_expanded.tex",
        "Coordinated numerical refinement against the 500 m / 450 s reference. These are numerical-consistency diagnostics for the reduced calibrated model, not field-network validation.",
        "tab:numerical_verification_expanded",
        resize=True,
    )
    dependency_audit = _read_csv(results_dir / "strict_target_dependency_audit.csv")
    _write_latex_table(
        dependency_audit,
        tables_dir / "table_strict_target_dependency_audit.tex",
        "Dependency and evidence audit. It distinguishes directly measured targets, imposed boundaries, algebraic proxies, calibrated-simulator quantities, and simulator-assisted hidden states.",
        "tab:metric_dependency_evidence",
        resize=True,
    )
    length_sensitivity = _read_csv(results_dir / "corridor_length_sensitivity.csv")
    _write_latex_table(
        length_sensitivity,
        tables_dir / "table_corridor_length_sensitivity.tex",
        "Structural sensitivity to the assumed reduced-corridor length. It quantifies model dependence and does not identify the topology of the source networks.",
        "tab:corridor_length_sensitivity",
        resize=True,
    )
    hydraulic_identifiability = _read_csv(results_dir / "hydraulic_identifiability_summary.csv")
    _write_latex_table(
        hydraulic_identifiability,
        tables_dir / "table_hydraulic_identifiability_summary.tex",
        "Hydraulic identifiability sensitivity. Pressure/head and flow are simulator-assisted hidden hydraulic states, not field-meter validation.",
        "tab:hydraulic_identifiability",
        resize=True,
    )
    xai_withholding = _read_csv(results_dir / "xai4heat_withholding_diagnostics.csv")
    _write_latex_table(
        xai_withholding,
        tables_dir / "table_xai4heat_withholding_diagnostics.tex",
        "All-valid-range XAI4HEAT withholding diagnostics form the primary measured-node analysis. The physically ordered, target-conditioned subset is reported separately as sensitivity evidence; neither protocol validates internal pipe fields.",
        "tab:xai_withholding_diagnostics",
        resize=True,
    )
    risk_mitigation = _read_csv(results_dir / "residual_submission_risk_mitigation.csv")
    _write_latex_table(
        risk_mitigation,
        tables_dir / "table_residual_submission_risk_mitigation.tex",
        "Quantified residual risks and evidence boundaries retained in the submission package.",
        "tab:submission_risk_mitigation",
        resize=True,
    )
    if xai.empty:
        _write_status_table(
            tables_dir / "table12_xai4heat_status.tex",
            "XAI4HEAT sparse-substation validation status.",
            "tab:xai4heat_status",
            "not run",
            "XAI4HEAT files were not available locally. No sparse-substation result is claimed in this run.",
        )
        _write_status_table(
            tables_dir / "table_xai4heat_status.tex",
            "XAI4HEAT sparse-substation validation status.",
            "tab:xai4heat_status_alias",
            "not run",
            "XAI4HEAT files were not available locally. No sparse-substation result is claimed in this run.",
        )
    else:
        xai_cols = [
            c
            for c in [
                "variable_label",
                "category",
                "unit",
                "n_substations",
                "n_total_samples",
                "mean_RMSE",
                "mean_MAE",
                "mean_nRMSE_percent",
                "state_type",
            ]
            if c in xai.columns
        ]
        _write_latex_table(
            xai[xai_cols] if xai_cols else xai,
            tables_dir / "table12_xai4heat_status.tex",
            "XAI4HEAT sparse-substation measured-node validation. Metrics use real measured thermal and energy variables only; pressure/head and flow are not measured.",
            "tab:xai4heat_status",
        )

    cost = _read_csv(results_dir / "computational_cost.csv")
    _write_latex_table(cost, tables_dir / "table13_computational_cost.tex", "Computational cost and inference time.", "tab:computational_cost")

    qgate = _read_csv(results_dir / "paper_quality_gate_report.csv")
    qgate_cols = [c for c in ["check", "passed", "note", "verdict"] if c in qgate.columns]
    _write_latex_table(qgate[qgate_cols] if qgate_cols else qgate, tables_dir / "table14_quality_gate.tex", "Paper quality-gate summary.", "tab:quality_gate")

    evidence_rows = [
        {"Evidence boundary": "Boundary conditions and calibration", "Supported by": "Sonderborg measured heat load, feed/supply temperature, return temperature, and an explicitly recorded configured ambient boundary where weather data are unavailable"},
        {"Evidence boundary": "Measured-node validation", "Supported by": "Measured plant/substation variables only where present in public data"},
        {"Evidence boundary": "Distributed hidden states", "Supported by": "Calibrated thermo-hydraulic simulator; not real full-field measurements"},
        {"Evidence boundary": "Hydraulic head and flow", "Supported by": "Simulator-assisted proxy because public datasets generally lack distributed pressure/flow"},
        {"Evidence boundary": "External transfer", "Supported by": "Flensburg feed/load transfer diagnostics; return temperature assumed as 50 C if unavailable"},
    ]
    ranking = _read_csv(results_dir / "model_ranking_summary_final.csv")
    if ranking.empty:
        ranking = _read_csv(results_dir / "model_ranking_summary.csv")
    ranking_cols = [c for c in ["metric", "best_model", "best_value", "proposed_model", "proposed_rank", "proposed_value", "interpretation"] if c in ranking.columns]
    # The broader architecture screen is one unreplicated run. It remains a
    # machine-readable result artifact but is intentionally not emitted as a
    # manuscript ranking table once five-seed primary evidence is available.
    if not ranking.empty:
        (PROJECT_ROOT / "results" / "single_run_ranking_not_used_in_manuscript.txt").write_text(
            "model_ranking_summary_final.csv is an exploratory single-run architecture screen. "
            "It is not used for manuscript comparative claims because repeated_seed_statistics.csv is the primary neural comparison.\n",
            encoding="utf-8",
        )
    if not ranking.empty:
        ranking_rows = []
        for _, row in ranking.iterrows():
            ranking_rows.append(
                {
                    "Metric": str(row.get("metric", "")),
                    "Best model": _compact_model_name(row.get("best_model", "")),
                    "Best value": _fmt_value(row.get("best_value")),
                    "V3 best rank": _fmt_value(row.get("pignn_gru_v3_best_rank"), decimals=0),
                    "V3 best value": _fmt_value(row.get("pignn_gru_v3_best_value")),
                    "V3 improves V2": str(row.get("v3_improves_over_v2", "")),
                }
            )
        _write_latex_table(pd.DataFrame(ranking_rows), tables_dir / "table6c_model_ranking.tex", "Model ranking summary. The proposed graph-temporal model is not assumed to be the lowest-RMSE model; the rank is reported metric by metric.", "tab:model_ranking_compact")
        _write_latex_table(pd.DataFrame(ranking_rows), tables_dir / "table4_model_ranking_by_objective.tex", "Model ranking by objective. Direct RMSE and thermo-hydraulic consistency can rank models differently.", "tab:model_ranking_objective", resize=True)

    # Do not regenerate a single-run proposed-model ``value'' table: its
    # ranks conflict with the locked five-seed primary comparison.
    (PROJECT_ROOT / "results" / "proposed_model_value_summary_demoted.txt").write_text(
        "proposed_model_value_summary.csv is retained as an exploratory single-run artifact only. "
        "The manuscript uses the five-seed primary neural comparison and does not claim PI-GNN-GRU-v3 rank 1 from this file.\n",
        encoding="utf-8",
    )

    claim_mapping = pd.DataFrame(
        [
            {
                "Claim": "Real data support calibration and measured-node thermal validation.",
                "Evidence": "Sonderborg processed data, calibration metrics, Table 2.",
                "Strength": "strong",
                "Safe wording": "Real operating data are used for boundary conditions, calibration, and measured-node thermal validation.",
                "Avoid": "full distributed field validation",
            },
            {
                "Claim": "Distributed pressure/head and flow are simulator-assisted hidden states.",
                "Evidence": "thermo-hydraulic metric tables and hydraulic figures.",
                "Strength": "strong evidence-boundary statement",
                "Safe wording": "Pressure/head and flow are simulator-assisted hidden hydraulic states.",
                "Avoid": "real distributed pressure/flow measurements",
            },
            {
                "Claim": "PI-GNN-GRU-v3 adds value for selected physical-consistency metrics.",
                "Evidence": "proposed-model value table and model-ranking summary.",
                "Strength": "metric-specific",
                "Safe wording": "PI-GNN-GRU-v3 improves selected ATE-relevant metrics in this run.",
                "Avoid": "best overall model",
            },
            {
                "Claim": "Flensburg transfer reveals domain shift.",
                "Evidence": "Flensburg transfer diagnostics and external validation modes.",
                "Strength": "moderate",
                "Safe wording": "Flensburg is a domain-shift stress test requiring local calibration/adaptation.",
                "Avoid": "universal generalization",
            },
            {
                "Claim": "Stress/fault and parameter studies are robustness diagnostics.",
                "Evidence": "controlled perturbation and parameter-sensitivity tables.",
                "Strength": "controlled scenario evidence",
                "Safe wording": "Controlled perturbations are applied to real operating profiles.",
                "Avoid": "field-validated fault detection",
            },
        ]
    )
    _write_latex_table(claim_mapping, tables_dir / "table_final_claim_mapping.tex", "Final claim-safety mapping. Each claim is tied to a result source and safe wording.", "tab:final_claim_mapping", resize=True)

    _write_latex_table(pd.DataFrame(evidence_rows), tables_dir / "table16_evidence_boundaries.tex", "Limitations and evidence boundaries.", "tab:evidence_boundaries")
    _write_thermo_hydraulic_tables(tables_dir, results_dir)
    _write_supplementary_robustness_tables(tables_dir, results_dir)
    _write_digital_twin_tables(tables_dir, results_dir)
    _write_operational_energy_impact_tables(tables_dir, results_dir)
    _write_operator_sensor_guidelines_table(tables_dir, results_dir)
    if not os.environ.get("SKIP_EXPENSIVE_FIGURE_REGEN"):
        _regenerate_final_robustness_operational_assets()
    try:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "make_primary_repeated_seed_figure.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "make_real_data_overview_evidence.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "repeated_seed_primary_figure_warning.txt"
        warning.write_text(f"Primary repeated-seed figure was not regenerated: {exc}\n", encoding="utf-8")
    if importlib.util.find_spec("reportlab") is None:
        status = PROJECT_ROOT / "results" / "ordered_ate_main_figure_generation_status.txt"
        status.write_text(
            "Ordered conceptual figures were retained because the optional reportlab dependency is unavailable in this runtime. "
            "Install reportlab to regenerate that optional figure package.\n",
            encoding="utf-8",
        )
    else:
        try:
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "make_ordered_ate_main_figures.py")],
                cwd=PROJECT_ROOT,
                check=True,
            )
        except Exception as exc:
            warning = PROJECT_ROOT / "results" / "ordered_ate_main_figure_generation_warning.txt"
            warning.write_text(f"Ordered 14-figure ATE package was not regenerated by make_paper_assets: {exc}\n", encoding="utf-8")

    fallback = False
    if "used_fallback_synthetic" in baseline.columns:
        fallback = bool(baseline["used_fallback_synthetic"].fillna(False).astype(bool).any())
    if baseline.empty:
        text = (
            "No main numerical results were generated in this run. "
            "The manuscript is a methodological draft until real operating datasets are placed in data/raw/ and the real-data study is rerun.\n"
        )
    elif fallback:
        text = (
            "The current numerical tables are from the software quick demo using fallback synthetic-realistic data. "
            "They are included only to verify the computational pipeline and must not be interpreted as journal real-data results.\n"
        )
    else:
        best = baseline.sort_values("RMSE_Ts_full").iloc[0]
        proposed = baseline[baseline["model"].astype(str).eq("Proposed PI-GNN-GRU")]
        improved = baseline[baseline["model"].astype(str).eq("Proposed PI-GNN-GRU improved")]
        proposed_label = "Proposed PI-GNN-GRU improved" if not improved.empty else "Proposed PI-GNN-GRU"
        proposed_row = improved if not improved.empty else proposed
        proposed_note = ""
        if not proposed_row.empty and "thermal_residual_mean" in baseline.columns:
            thermal_pool = baseline[baseline.get("loss_mode", pd.Series(dtype=str)).astype(str).eq("physics")]
            if thermal_pool.empty:
                thermal_pool = baseline[baseline["model"].astype(str).ne("Interpolation")]
            best_thermal = thermal_pool.sort_values("thermal_residual_mean").iloc[0]
            heat_loss_rank = ""
            if "heat_loss_error_percent" in baseline.columns:
                heat_ranked = baseline[["model", "heat_loss_error_percent"]].dropna().sort_values("heat_loss_error_percent").reset_index(drop=True)
                match = heat_ranked[heat_ranked["model"].astype(str).eq(proposed_label)]
                if not match.empty:
                    heat_loss_rank = f" Its heat-loss error ranking was {int(match.index[0]) + 1} of {len(heat_ranked)}."
            proposed_note = (
                f" Among physics-informed neural models, {proposed_label} thermal residual was "
                f"{float(proposed_row['thermal_residual_mean'].iloc[0]):.4f}; the lowest value in that group was "
                f"{best_thermal['model']} with {float(best_thermal['thermal_residual_mean']):.4f}.{heat_loss_rank}"
            )
        text = (
            f"In the current real-data-assisted run, the lowest full-field supply-temperature RMSE was obtained by "
            f"{best['model']} with RMSE {float(best['RMSE_Ts_full']):.3f} C. "
            "The proposed graph-temporal model is therefore not claimed as the universal RMSE winner unless the table shows that outcome. "
            "Measured-node and physics-consistency metrics are reported separately because full distributed states remain simulator-generated."
            f"{proposed_note}\n"
        )
    (PROJECT_ROOT / "paper" / "key_results_text.tex").write_text(text, encoding="utf-8")
    _copy_figures()
    try:
        from src.final_result_audit_for_ate import run_final_result_audit

        run_final_result_audit()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "final_audit_asset_generation_warning.txt"
        warning.write_text(f"Final ATE audit assets were not regenerated by make_paper_assets: {exc}\n", encoding="utf-8")
    try:
        from src.online_replay_validation import run_online_replay_validation_package

        run_online_replay_validation_package()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "online_replay_validation_warning.txt"
        warning.write_text(f"Online replay/blind-validation assets were not regenerated by make_paper_assets: {exc}\n", encoding="utf-8")
    try:
        from src.replay_robustness_evidence import build_replay_robustness_evidence

        build_replay_robustness_evidence()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "replay_robustness_evidence_warning.txt"
        warning.write_text(
            f"Operating-regime/dropout/fault online-replay evidence was not regenerated by make_paper_assets: {exc}\n",
            encoding="utf-8",
        )
    # The legacy submission-polish generator rewrites the active manuscript from
    # an older broad template. Preserve the focused, reviewed manuscript unless
    # an explicit legacy rebuild is requested for archival purposes.
    if os.environ.get("REBUILD_LEGACY_SUBMISSION_MANUSCRIPT") == "1":
        try:
            from src.submission_polish import polish_submission_package

            polish_submission_package()
        except Exception as exc:
            warning = PROJECT_ROOT / "results" / "submission_polish_warning.txt"
            warning.write_text(f"Submission polish assets were not regenerated by make_paper_assets: {exc}\n", encoding="utf-8")
    # Restore the focused supplementary tables that are intentionally separate
    # from the broad legacy table generator.  Calling the narrow writers keeps
    # the active manuscript untouched while making ``make_paper_assets``
    # reproducible from the locked CSV artifacts.
    try:
        from src.reviewer_submission_additions import (
            write_flensburg_distribution_shift_table,
            write_seed_stability_status,
            write_temporal_split_tables,
            write_uncertainty_coverage_table,
        )

        write_temporal_split_tables()
        write_seed_stability_status()
        write_uncertainty_coverage_table()
        write_flensburg_distribution_shift_table()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "focused_supplementary_table_generation_warning.txt"
        warning.write_text(f"Focused supplementary tables were not regenerated: {exc}\n", encoding="utf-8")
    try:
        from src.xai4heat_measured_node_validation import main as regenerate_xai4heat_tables

        regenerate_xai4heat_tables()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "xai4heat_paper_table_generation_warning.txt"
        warning.write_text(f"XAI4HEAT supplementary tables were not regenerated: {exc}\n", encoding="utf-8")
    # Recreate the focused submission tables after the broad asset generator.
    # This preserves the compact, evidence-separated main-paper table set.
    try:
        from src.make_focused_main_tables import main as make_focused_main_tables

        make_focused_main_tables()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "focused_main_table_generation_warning.txt"
        warning.write_text(f"Focused main tables were not regenerated: {exc}\n", encoding="utf-8")
    try:
        from src.freeze_final_submission_results import freeze_final_submission_results

        freeze_final_submission_results()
    except Exception as exc:
        warning = PROJECT_ROOT / "results" / "final_submission_lock_warning.txt"
        warning.write_text(f"Final submission result lock was not regenerated by make_paper_assets: {exc}\n", encoding="utf-8")


if __name__ == "__main__":
    make_paper_assets(load_config())
