"""Generate the main quantitative prose directly from post-correction CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SECTIONS = ROOT / "paper" / "sections"


def alias(model: str) -> str:
    return {
        "Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-GRU-v3 accuracy mode",
        "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-GRU-v3 balanced mode",
    }.get(model, model)


def latex(text: str) -> str:
    return text.replace("%", r"\%").replace("±", r"$\pm$")


def metric(stats: pd.DataFrame, name: str) -> tuple[str, float, float]:
    rows = stats[stats["metric"].eq(name)].copy()
    if rows.empty or pd.to_numeric(rows["n_seeds"], errors="coerce").min() != 5:
        raise RuntimeError(f"Final five-seed metric is incomplete: {name}")
    row = rows.loc[pd.to_numeric(rows["mean"], errors="coerce").idxmin()]
    return str(row["model"]), float(row["mean"]), float(row["std"])


def main() -> None:
    SECTIONS.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(RESULTS / "calibration_metrics.csv").iloc[0]
    later = pd.read_csv(RESULTS / "locked_later_replay_metrics.csv").iloc[0]
    numerical = pd.read_csv(RESULTS / "numerical_verification_expanded.csv")
    baseline_grid = numerical.loc[pd.to_numeric(numerical["dx_m"], errors="coerce").eq(1000.0)].iloc[0]
    stats = pd.read_csv(RESULTS / "repeated_seed_statistics.csv")
    params = json.loads((RESULTS / "calibrated_parameters.json").read_text(encoding="utf-8"))
    xai = pd.read_csv(RESULTS / "xai4heat_sparse_substation_validation_final.csv")
    flensburg = pd.read_csv(RESULTS / "flensburg_measured_only_validation.csv")

    ts_model, ts_mean, ts_std = metric(stats, "RMSE_Ts_full")
    tr_model, tr_mean, tr_std = metric(stats, "RMSE_Tr_full")
    loss_model, loss_mean, loss_std = metric(stats, "heat_loss_error_percent")
    energy_model, energy_mean, energy_std = metric(stats, "energy_balance_residual")
    boundary_model, boundary_mean, boundary_std = metric(stats, "boundary_residual_mean")
    direct = float(flensburg.loc[flensburg["mode"].eq("direct_transfer"), "measured supply RMSE_C"].iloc[0])
    few = float(flensburg.loc[flensburg["mode"].eq("few_shot_decoder_bias_adaptation"), "measured supply RMSE_C"].iloc[0])
    xai_supply = float(xai.loc[xai["variable"].eq("t_sup_prim"), "mean_RMSE"].iloc[0])
    xai_return = float(xai.loc[xai["variable"].eq("t_ret_prim"), "mean_RMSE"].iloc[0])

    abstract = (
        "Sparse sensing limits observation of transport delay, heat loss, and hydraulic behaviour in district-heating networks. "
        "This study combines public operating data, a calibrated reduced thermo-hydraulic simulator, and sparse-sensor estimators in an evidence-separated benchmark. "
        f"S{{\\o}}nderborg calibration gives {float(calibration['RMSE_return_C']):.3f}~$^\\circ$C return-temperature RMSE, "
        f"{float(calibration['heat_delivery_error_percent']):.3f}\\% consumer-boundary closure, and a "
        f"{100*float(calibration['energy_balance_residual_fraction']):.3f}\\% dynamic-residual ratio; unchanged parameters give "
        f"{float(later['RMSE_return_C']):.3f}~$^\\circ$C on the later replay. "
        f"Across five seeds, {alias(ts_model)} gives the lowest simulator-field supply RMSE "
        f"({ts_mean:.3f}$\\pm${ts_std:.3f}~$^\\circ$C), whereas {alias(tr_model)} gives the lowest return RMSE "
        f"({tr_mean:.3f}$\\pm${tr_std:.3f}~$^\\circ$C). {alias(loss_model)} minimizes heat-loss error "
        f"({loss_mean:.3f}$\\pm${loss_std:.3f}\\%), and {alias(energy_model)} minimizes the model-conditioned energy residual "
        f"({energy_mean:.3f}$\\pm${energy_std:.3f}\\%). XAI4HEAT withholding gives {xai_supply:.3f}/{xai_return:.3f}~$^\\circ$C supply/return RMSE; "
        f"Flensburg few-shot bias adaptation changes measured-supply RMSE from {direct:.2f} to {few:.2f}~$^\\circ$C. "
        "Rankings are objective-dependent. Distributed thermal fields are calibrated-simulator quantities; pressure/head and flow are simulator-assisted hidden states."
    )
    (SECTIONS / "post_causality_abstract.tex").write_text(abstract + "\n", encoding="utf-8")

    calibration_text = (
        f"The corrected measured-boundary calibration fit gives {float(calibration['RMSE_return_C']):.3f}~$^\\circ$C return RMSE, "
        f"{float(calibration['MAE_return_C']):.3f}~$^\\circ$C MAE, and a signed bias of {float(calibration['signed_return_bias_C']):.3f}~$^\\circ$C over the scored calibration samples. "
        f"Consumer-boundary closure is {float(calibration['heat_delivery_error_percent']):.3f}\\%, and the block-level dynamic-residual ratio is "
        f"{100*float(calibration['energy_balance_residual_fraction']):.3f}\\%, computed as total absolute residual divided by total absolute imposed load. Source supply is imposed and is not counted as predictive calibration skill. "
        f"The fitted effective parameters are $U={float(params['heat_loss_U_W_m2K']):.4f}$~W~m$^{{-2}}$~K$^{{-1}}$, "
        f"$\\eta_v={float(params['effective_velocity_factor']):.4f}$, $\\beta_q={float(params['flow_proxy_blend']):.4f}$, and "
        f"$f={float(params['friction_factor']):.5f}$. The bound positions reported in Table~\\ref{{tab:main_calibration_verification}} confirm limited effective-parameter identifiability. "
        f"Without retuning, the locked later replay gives {float(later['RMSE_return_C']):.3f}~$^\\circ$C RMSE, "
        f"{float(later['MAE_return_C']):.3f}~$^\\circ$C MAE, {float(later['boundary_closure_percent']):.3f}\\% closure, and "
        f"{float(later['dynamic_energy_residual_percent']):.3f}\\% dynamic residual. This later suffix is the temporal replay check; it does not validate distributed states.\n\n"
        f"Coordinated refinement compares 2000~m/1800~s, 1000~m/900~s, and 500~m/450~s cases with matched CFL treatment. Relative to the fine reference, the baseline gives "
        f"{float(baseline_grid['outlet_Ts_L2_C']):.4f}~$^\\circ$C outlet-supply L2 difference, "
        f"{float(baseline_grid['outlet_Ts_Linf_C']):.4f}~$^\\circ$C Linf difference, "
        f"{float(baseline_grid['source_Tr_L2_C']):.4f}~$^\\circ$C return-source L2 difference, and "
        f"{float(baseline_grid['cumulative_heat_loss_error_pct']):.4f}\\% integrated heat-loss difference. "
        "These checks support numerical consistency of the reduced discretization, not identification of the physical topology of any source network."
    )
    (SECTIONS / "post_causality_calibration_results.tex").write_text(calibration_text + "\n", encoding="utf-8")

    model_text = (
        "The post-correction five-seed comparison in Table~\\ref{tab:main_benchmark} is the primary neural result; the broader single-run architecture screen remains exploratory. "
        f"{alias(ts_model)} gives the lowest mean simulator-field supply RMSE ({ts_mean:.3f}$\\pm${ts_std:.3f}~$^\\circ$C), whereas "
        f"{alias(tr_model)} gives the lowest return RMSE ({tr_mean:.3f}$\\pm${tr_std:.3f}~$^\\circ$C). "
        f"{alias(loss_model)} gives the lowest heat-loss error ({loss_mean:.3f}$\\pm${loss_std:.3f}\\%), "
        f"{alias(energy_model)} gives the lowest model-conditioned dynamic energy residual ({energy_mean:.3f}$\\pm${energy_std:.3f}\\%), and "
        f"{alias(boundary_model)} gives the lowest decoded-source boundary residual ({boundary_mean:.3f}$\\pm${boundary_std:.3f}~$^\\circ$C). "
        "The energy residual is part of physics-informed optimization, so a reduction demonstrates objective alignment rather than independent validation. "
        "These C-class thermal and S-class hydraulic results use one fixed split and S4 layout; they establish neither universal model superiority nor repeatability for every downstream sensitivity study."
    )
    (SECTIONS / "post_causality_model_results.tex").write_text(model_text + "\n", encoding="utf-8")

    conclusion = (
        f"After correcting segment initialization, the measured-boundary fit gives {float(calibration['RMSE_return_C']):.3f}~$^\\circ$C return RMSE, while the unchanged-parameter later replay gives "
        f"{float(later['RMSE_return_C']):.3f}~$^\\circ$C. The baseline-to-fine outlet-supply L2 difference is {float(baseline_grid['outlet_Ts_L2_C']):.4f}~$^\\circ$C. "
        f"The five-seed benchmark is objective-dependent: {alias(ts_model)} leads simulator-field supply reconstruction, {alias(tr_model)} leads return reconstruction, "
        f"{alias(loss_model)} minimizes heat-loss error, {alias(energy_model)} minimizes the model-conditioned energy residual, and {alias(boundary_model)} gives the lowest decoded-source boundary residual. "
        "The result is a reproducible comparison of estimation objectives, not a universal winner or independent validation of model-conditioned residuals."
    )
    (SECTIONS / "post_causality_conclusion_results.tex").write_text(conclusion + "\n", encoding="utf-8")

    sensitivity = pd.read_csv(RESULTS / "parameter_sensitivity_summary.csv")
    by_case = sensitivity.set_index("case")
    base = by_case.loc["baseline"]
    flow_low = by_case.loc["flow_proxy_0.9x"]
    flow_high = by_case.loc["flow_proxy_1.1x"]
    friction_low = by_case.loc["friction_factor_0.8x"]
    friction_high = by_case.loc["friction_factor_1.2x"]
    bias_candidates = sensitivity[sensitivity["case"].astype(str).str.contains("return", case=False)]
    bias_sentence = ""
    if not bias_candidates.empty:
        worst_bias = bias_candidates.loc[pd.to_numeric(bias_candidates["mean_return_RMSE_C"], errors="coerce").idxmax()]
        bias_sentence = f" The most adverse tested return-bias case gives {float(worst_bias['mean_return_RMSE_C']):.3f}~$^\\circ$C mean return RMSE."
    sensitivity_text = (
        f"Hydraulic sensitivity is material. Relative to the nominal mean return RMSE of {float(base['mean_return_RMSE_C']):.3f}~$^\\circ$C and pressure-drop error of {float(base['mean_pressure_drop_error_percent']):.2f}\\%, "
        f"the $-10\\%$ flow-proxy case gives {float(flow_low['mean_return_RMSE_C']):.3f}~$^\\circ$C and {float(flow_low['mean_pressure_drop_error_percent']):.2f}\\%, while the $+10\\%$ case gives "
        f"{float(flow_high['mean_return_RMSE_C']):.3f}~$^\\circ$C and {float(flow_high['mean_pressure_drop_error_percent']):.2f}\\%. "
        f"The $\\pm20\\%$ friction cases span {min(float(friction_low['mean_pressure_drop_error_percent']), float(friction_high['mean_pressure_drop_error_percent'])):.2f}--"
        f"{max(float(friction_low['mean_pressure_drop_error_percent']), float(friction_high['mean_pressure_drop_error_percent'])):.2f}\\% pressure-drop error."
        + bias_sentence
        + " These are effective-parameter sensitivity diagnostics for S states, not independent hydraulic validation."
    )
    (SECTIONS / "post_causality_hydraulic_sensitivity.tex").write_text(sensitivity_text + "\n", encoding="utf-8")

    layouts = pd.read_csv(RESULTS / "sensor_layout_geometry_ranking.csv")
    leaders = layouts[pd.to_numeric(layouts["rank"], errors="coerce").eq(1)].copy()
    layout_clauses = []
    for _, row in leaders.iterrows():
        layout_name = str(row["sensor_layout"]).replace("_", "\\_")
        layout_clauses.append(
            f"{str(row['objective']).lower()} selects {layout_name} "
            f"(score {float(row['score']):.3f}; nodes {row['sensor_nodes']})"
        )
    sensor_text = (
        "The validation-only layout ranking is objective-dependent: "
        + "; ".join(layout_clauses)
        + ". Scores are normalized composites, and duplicate geometries and noise/dropout scenarios are not counted as independent placements. "
        "These reduced-corridor results guide benchmark interpretation but are not field installation recommendations because branches, access, redundancy, and maintenance are outside the objective."
    )
    (SECTIONS / "post_causality_sensor_layout_results.tex").write_text(sensor_text + "\n", encoding="utf-8")

    shift = pd.read_csv(RESULTS / "flensburg_domain_shift_analysis.csv").set_index("metric")["value"]
    offset = float(flensburg.loc[flensburg["mode"].eq("calibration_only_offset_adaptation"), "measured supply RMSE_C"].iloc[0])
    load_shift = 100.0 * (float(shift["mean_heat_load_flensburg_kw"]) / float(shift["mean_heat_load_sonderborg_kw"]) - 1.0)
    flensburg_text = (
        f"Direct S{{\\o}}nderborg-to-Flensburg transfer gives {direct:.2f}~$^\\circ$C measured-supply RMSE. "
        f"Few-shot decoder-bias adaptation gives {few:.2f}~$^\\circ$C ({100*(direct-few)/direct:.1f}\\% lower), whereas calibration-only offset adaptation gives {offset:.2f}~$^\\circ$C. "
        f"Flensburg mean load is {load_shift:.1f}\\% higher than the S{{\\o}}nderborg training mean, with different supply range and hourly cadence. "
        "Return temperature is unavailable and fixed at 50~$^\\circ$C only for assumption-consistency diagnostics; these diagnostics are excluded from measured external-validation claims. "
        "The experiment is therefore a domain-shift stress test, not evidence of broad zero-shot generalization."
    )
    (SECTIONS / "post_causality_flensburg_results.tex").write_text(flensburg_text + "\n", encoding="utf-8")

    uncertainty = pd.read_csv(RESULTS / "uncertainty_conformal_evaluation_locked.csv")
    uncertainty_parts = []
    for _, row in uncertainty.iterrows():
        unit = {"C": r"$^\circ$C", "m3/s": r"m$^3$~s$^{-1}$"}.get(str(row["unit"]), str(row["unit"]))
        uncertainty_parts.append(
            f"{str(row['quantity']).replace('_', ' ')} {float(row['coverage']):.1f}\\%/{float(row['mean_interval_width']):.4g}~{unit}"
        )
    uncertainty_text = (
        "Under the locked chronological split-conformal protocol, coverage/mean width by quantity is "
        + "; ".join(uncertainty_parts)
        + ". Coverage is empirical under serial dependence and widths are never averaged across units. Undercoverage identifies quantities for which the present intervals should not be used as operational confidence limits."
    )
    (SECTIONS / "post_causality_uncertainty_results.tex").write_text(uncertainty_text + "\n", encoding="utf-8")

    ablation = pd.read_csv(RESULTS / "ablation_study_final.csv")
    full = ablation.loc[ablation["ablation"].eq("full_physics")].iloc[0]
    no_graph = ablation.loc[ablation["ablation"].eq("no_graph_topology")].iloc[0]
    ablation_text = (
        f"The diagnostic ablation does not identify a universally indispensable component. The full formulation gives {float(full['RMSE_Ts_full']):.3f}/{float(full['RMSE_Tr_full']):.3f}~$^\\circ$C supply/return RMSE, "
        f"{float(full['heat_loss_error_percent']):.3f}\\% heat-loss error, and {float(full['energy_balance_residual']):.3f}\\% energy residual; removing graph topology gives "
        f"{float(no_graph['RMSE_Ts_full']):.3f}/{float(no_graph['RMSE_Tr_full']):.3f}~$^\\circ$C, {float(no_graph['heat_loss_error_percent']):.3f}\\%, and {float(no_graph['energy_balance_residual']):.3f}\\%, respectively. "
        "Because this is a single-seed component audit, it diagnoses trade-offs but does not establish graph superiority or statistical component necessity."
    )
    (SECTIONS / "post_causality_ablation_results.tex").write_text(ablation_text + "\n", encoding="utf-8")
    cover_text = (
        "The contribution is intentionally framed as a rigorous benchmark rather than a single-model superiority claim. "
        f"Under the corrected common five-seed protocol, {alias(ts_model)} gives the lowest simulator-field supply RMSE, "
        f"{alias(tr_model)} gives the lowest return-temperature RMSE, {alias(loss_model)} gives the lowest heat-loss error, "
        f"{alias(energy_model)} gives the lowest model-conditioned dynamic energy residual, and {alias(boundary_model)} gives the lowest decoded-source boundary residual. "
        "The XAI4HEAT leave-one-substation-out test provides measured-substation thermal evidence, while Flensburg is treated as a measured-supply domain-shift stress test. Distributed hydraulic fields remain simulator-assisted hidden states."
    )
    (SECTIONS / "post_causality_cover_results.tex").write_text(cover_text + "\n", encoding="utf-8")
    print(SECTIONS)


if __name__ == "__main__":
    main()
