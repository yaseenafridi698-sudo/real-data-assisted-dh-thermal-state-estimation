from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
T = ROOT / "paper" / "tables"


def esc(value: object) -> str:
    s = str(value)
    # Protect project nomenclature and mathematical uncertainty notation before
    # general LaTeX escaping. This also repairs legacy UTF-8 mojibake on output.
    sonder_token = "@@SONDERBORG@@"
    plusminus_token = "@@PLUSMINUS@@"
    s = s.replace(r"S{\o}nderborg", sonder_token).replace("Sonderborg", sonder_token)
    s = s.replace("Sønderborg", sonder_token).replace("SÃ¸nderborg", sonder_token)
    s = s.replace("+/-", plusminus_token).replace("±", plusminus_token)
    if chr(195) in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    if "Ã" in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    if "Ã" in s:
        try:
            s = s.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    degree_token = "@@DEGREEC@@"
    percent_token = "@@PERCENT@@"
    eta_token = "@@ETAV@@"
    beta_token = "@@BETAQ@@"
    s = s.replace(r"$^\circ$C", degree_token)
    s = s.replace(r"\%", percent_token)
    s = s.replace(r"$\eta_v$", eta_token)
    s = s.replace(r"$\beta_q$", beta_token)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")]:
        s = s.replace(a, b)
    return (
        s.replace(degree_token, r"$^\circ$C")
        .replace(percent_token, r"\%")
        .replace(eta_token, r"$\eta_v$")
        .replace(beta_token, r"$\beta_q$")
        .replace(sonder_token, r"S{\o}nderborg")
        .replace(plusminus_token, r"$\pm$")
    )


def table(df: pd.DataFrame, name: str, caption: str, label: str, resize: bool = False) -> None:
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule", " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(row[c]) for c in df.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + ("%" if resize else "")]
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    (T / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(x: object, d: int = 3) -> str:
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def main() -> None:
    T.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame([
        ["Sønderborg (seven plants)", "M", "Plant load, supply and return; ambient provenance reported separately", "Calibration, replay, plant/year withholding"],
        ["XAI4HEAT", "M", "Five-substation supply/return and energy variables", "Leave-one-substation-out validation"],
        ["Flensburg", "M", "Load and measured supply; return unavailable", "External measured-supply domain shift"],
        ["Calibrated pipe model", "C", "Distributed supply/return and heat loss", "Hidden-state benchmark; not dense field data"],
        ["Reduced hydraulic model", "S", "Pressure/head and flow proxy", "Internal consistency only"],
    ], columns=["Source", "Class", "Available quantities", "Role"])
    # Override legacy mojibake in older generated assets with the canonical
    # UTF-8 dataset name and the actual ambient-data evidence boundary.
    data.loc[0, "Source"] = "Sønderborg (seven plants)"
    data.loc[0, "Available quantities"] = r"Plant load, supply and return; primary run uses configured 5 $^\circ$C ambient, sensitivity uses ERA5-Land reanalysis"
    data.loc[1, "Available quantities"] = "Five-substation primary supply/return; energy signals retained as source context"
    data.loc[1, "Role"] = "Primary-temperature leave-one-substation-out validation"
    table(data, "table_main_1_data_evidence.tex", "Datasets and evidence hierarchy. M: measured; C: calibrated-simulator quantity; S: simulator-assisted hidden state.", "tab:main_data_evidence", True)

    params = json.loads((R / "calibrated_parameters.json").read_text(encoding="utf-8"))
    definitions = pd.DataFrame([
        ["Thermal transport", "1-D upwind advection with ambient loss", "C", "CFL <= 0.8 by substepping"],
        ["Heat-loss coefficient U", f(params.get("heat_loss_U_W_m2K", "n/a"), 4) + " W m$^{-2}$ K$^{-1}$", "C", "effective calibrated value"],
        ["Effective transport factor eta_v", f(params.get("effective_velocity_factor", "n/a"), 4), "C", "multiplies only thermal Courant number"],
        ["Flow-proxy blend beta_q", f(params.get("flow_proxy_blend", "n/a"), 4), "S proxy", "algebraic effective closure; not a measured flow fraction"],
        ["Flow proxy", "Q/(rho cp DeltaT), using lagged return", "S", "not flow-meter validation"],
        ["Pressure/head", "pump boundary plus Darcy--Weisbach drop", "S", "reduced one-dimensional diagnostic"],
        ["PI-GNN-GRU", "residual graph encoder + GRU + multi-head decoder", "estimator", "paper-facing name; no universal superiority"],
        ["Physics loss", "scaled thermal, hydraulic, boundary, loss and energy residuals", "training", "curriculum scheduled"],
    ], columns=["Component", "Definition", "Class", "Scope"])
    table(definitions, "table_main_2_method_definition.tex", "Compact model definition and evidence scope. Full discretization, parameter bounds, and loss terms are provided in the Supplementary Material.", "tab:main_method", True)

    cal = pd.read_csv(R / "calibration_metrics.csv").iloc[0]
    later = pd.read_csv(R / "locked_later_replay_metrics.csv").iloc[0]
    nv = pd.read_csv(R / "numerical_verification_expanded.csv")
    base = nv[nv["dx_m"].eq(1000)].iloc[0]
    verification = pd.DataFrame([
        ["Source-supply boundary consistency", f(cal["RMSE_supply_C"]), r"$^\circ$C", "M imposed Dirichlet boundary; not a calibration target"],
        ["Return-temperature calibration RMSE", f(cal["RMSE_return_C"]), r"$^\circ$C", "M measured return fit"],
        ["Return-temperature calibration MAE", f(cal["MAE_return_C"]), r"$^\circ$C", "M measured return fit"],
        ["Return-temperature signed bias", f(cal["signed_return_bias_C"]), r"$^\circ$C", "M measured return fit"],
        ["Consumer heat-load boundary closure", f(cal["heat_delivery_error_percent"]), "%", "C consistency; not independent validation"],
        ["Calibration dynamic-residual ratio", f(100.0 * cal["energy_balance_residual_fraction"]), "%", "C: total absolute residual / total absolute imposed load"],
        ["Locked later replay return RMSE", f(later["RMSE_return_C"]), r"$^\circ$C", "M measured return; parameters unchanged"],
        ["Locked later replay return MAE", f(later["MAE_return_C"]), r"$^\circ$C", "M measured return; parameters unchanged"],
        ["Effective heat-loss coefficient $U$", f(params["heat_loss_U_W_m2K"], 4), r"W m$^{-2}$ K$^{-1}$", "lower calibration bound"],
        [r"Effective transport factor $\eta_v$", f(params["effective_velocity_factor"], 4), "-", "upper calibration bound"],
        [r"Flow-proxy blend $\beta_q$", f(params["flow_proxy_blend"], 4), "-", "effective closure; not measured flow"],
        ["Darcy friction factor $f$", f(params["friction_factor"], 5), "-", "weakly proxy-calibrated; no pressure/flow meter"],
        ["Outlet supply L2: 1000 m/900 s vs fine", f(base["outlet_Ts_L2_C"], 4), r"$^\circ$C", "numerical verification"],
        ["Outlet supply Linf: 1000 m/900 s vs fine", f(base["outlet_Ts_Linf_C"], 4), r"$^\circ$C", "numerical verification"],
        ["Return-source L2: 1000 m/900 s vs fine", f(base["source_Tr_L2_C"], 4), r"$^\circ$C", "numerical verification"],
        ["Integrated heat-loss difference", f(base["cumulative_heat_loss_error_pct"], 4), "%", "numerical verification"],
    ], columns=["Metric", "Value", "Unit", "Evidence"])
    table(verification, "table_main_3_calibration_verification.tex", "Corrected measured thermal calibration, locked later replay without retuning, effective parameters, and coordinated spatial-temporal numerical verification. The fine reference uses 500 m and 450 s.", "tab:main_calibration_verification", True)

    proxy_audit = pd.read_csv(R / "proxy_causality_audit.csv")
    proxy_audit = proxy_audit[["audit", "status", "evidence", "note"]].copy()
    proxy_audit["status"] = proxy_audit["status"].str.upper()
    table(
        proxy_audit,
        "table_causal_proxy_audit.tex",
        "Causal boundary-proxy audit. Future perturbations were applied strictly after each tested prefix; a pass means that no future load or return observation altered an earlier proxy value. This audit verifies chronology, not independent hydraulic field validity.",
        "tab:causal_proxy_audit",
        True,
    )

    display = {"Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-GRU (accuracy)", "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-GRU (balanced)"}
    seed = pd.read_csv(R / "repeated_seed_statistics.csv")
    metrics = ["RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"]
    wanted = ["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"]
    rows = []
    for model in wanted:
        row = [display.get(model, model)]
        for metric in metrics:
            x = seed[(seed["model"].eq(model)) & (seed["metric"].eq(metric))]
            row.append("not available" if x.empty else f"{float(x.iloc[0]['mean']):.3f} +/- {float(x.iloc[0]['std']):.3f}")
        rows.append(row)
    tbl = pd.DataFrame(rows, columns=["Model", "Ts RMSE ($^\\circ$C)", "Tr RMSE ($^\\circ$C)", "Loss error (%)", "Energy residual (%)", "Decoded-source residual ($^\\circ$C)"])
    table(tbl, "table_main_4_benchmark.tex", "Prespecified five-seed primary neural comparison (mean $\\pm$ sample standard deviation; fixed chronological split and S4 layout). Distributed temperature and consistency targets are calibrated-simulator quantities; lower is better. A broader single-run architecture screen is retained in the locked reviewer archive for hypothesis generation and is not pooled with these statistics.", "tab:main_benchmark", True)

    replay = pd.read_csv(R / "measured_node_baseline_comparison.csv")
    replay = replay[replay["model"].isin(["Last observation persistence", "Linear autoregression", "Causal replay estimator"])].copy()
    mrows = [[x["model"], "Sønderborg return", f(x["RMSE_C"]), f(x["MAE_C"]), "M chronological"] for _, x in replay.iterrows()]
    causal = pd.read_csv(R / "causal_heat_load_input_ablation.csv")
    for variant, label in [
        ("past_return_no_load", "Causal one-step, no load"),
        ("past_return_lagged_load", "Causal one-step, lagged load"),
        ("past_return_current_load", "Causal one-step, current load"),
    ]:
        x = causal[causal["variant"].eq(variant)].iloc[0]
        mrows.append([label, "Sønderborg return", f(x["RMSE_C"]), f(x["MAE_C"]), "M chronological"])
    plant = pd.read_csv(R / "sonderborg_blind_plant_validation_summary.csv")
    x = plant[plant["protocol"].eq("combined_plant_and_time_withholding")].iloc[0]
    mrows.append(["Pooled ridge", "Unseen plant + 2019 return", f(x["mean_RMSE_C"]), f(x["mean_MAE_C"]), "M plant/time transfer"])
    xai = pd.read_csv(R / "xai4heat_sparse_substation_validation_final.csv")
    for variable in ["Primary supply temperature", "Primary return temperature"]:
        x = xai[xai["variable_label"].eq(variable)].iloc[0]
        mrows.append(
            [
                "Spatial interpolation baseline",
                "XAI4HEAT " + variable.replace(" temperature", ""),
                f(x["mean_RMSE"]),
                f(x["mean_MAE"]),
                "M spatial withholding; all valid-range primary protocol",
            ]
        )
    mt = pd.DataFrame(mrows, columns=["Estimator", "Held-out variable", "RMSE ($^\\circ$C)", "MAE ($^\\circ$C)", "Evidence"])
    table(mt, "table_main_5_measured_node.tex", "Blind measured-node validation. One-step rows use return history only through k-1; plant/time transfer excludes the target plant and 2019 from fitting. XAI4HEAT headline values use all valid-range observations after file-content deduplication. A physically ordered, target-conditioned subset and reversal frequencies are reported separately as sensitivity evidence.", "tab:main_measured_node", True)

    geometry_ranking = R / "sensor_layout_geometry_ranking.csv"
    rank = pd.read_csv(geometry_ranking if geometry_ranking.exists() else R / "sensor_layout_ranking_by_objective.csv")
    rank = rank[rank["rank"].eq(1)].copy()
    sr = []
    for _, x in rank.iterrows():
        sr.append([x["objective"].replace("Best ", ""), x["sensor_layout"], int(x["sensor_count"]), x["sensor_nodes"], f(x["score"]), f(x["max_unobserved_distance_km"], 1)])
    st = pd.DataFrame(sr, columns=["Objective", "Leading layout", "Sensors", "Nodes", "Score", "Max gap (km)"])
    table(st, "table_main_6_sensor_layout.tex", "Objective-specific ranking of unique nominal sensor-placement geometries on the calibrated-corridor benchmark. Composite scores are defined in the Supplementary Material; noise, dropout, and duplicate-node scenarios are reported separately and are not placement designs.", "tab:main_sensor_layout", True)

    fl = pd.read_csv(R / "flensburg_measured_only_validation.csv")
    mode_labels = {
        "direct_transfer": "Direct transfer",
        "calibration_only_offset_adaptation": "Calibration-only offset",
        "few_shot_decoder_bias_adaptation": "Few-shot bias",
        "normalized_transfer_flensburg_boundary_statistics": "Boundary-stat normalization",
    }
    ft = pd.DataFrame(
        {
            "Mode": fl["mode"].map(mode_labels).fillna(fl["mode"]),
            "Supply RMSE ($^\\circ$C)": fl["measured supply RMSE_C"].map(lambda x: f(x)),
            "Load consistency (%)": fl["heat-load consistency_pct"].map(lambda x: f(x)),
            "Return evidence": "Assumed 50 $^\\circ$C; excluded",
        }
    )
    table(ft, "table_main_7_flensburg.tex", "Flensburg domain-shift test. Supply RMSE uses measured feed temperature; return-temperature values are excluded because the 50 $^\\circ$C return is assumed.", "tab:main_flensburg", True)

    cost = pd.read_csv(R / "computational_cost.csv")
    protocol = json.loads((R / "repeated_seed_protocol.json").read_text(encoding="utf-8"))
    processed = pd.read_csv(ROOT / "data" / "processed" / "sonderborg_processed.csv", parse_dates=["timestamp"])
    cap = int(protocol["maximum_time_steps"])
    timestamp_audit = protocol.get("timestamp_window_audit", {})
    split_audit = protocol.get("window_split_audit", {})
    gap_audit_path = R / "gap_handling_audit.json"
    gap_audit = json.loads(gap_audit_path.read_text(encoding="utf-8")) if gap_audit_path.exists() else {}
    window_start = processed["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M UTC")
    window_end = processed["timestamp"].iloc[cap - 1].strftime("%Y-%m-%d %H:%M UTC")
    names = ["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"]
    cost = cost[cost["model"].isin(names)]
    rr = []
    for _, x in cost.iterrows():
        rr.append([
            display.get(x["model"], x["model"]),
            int(x["parameter_count"]),
            f"{float(x['training_time_s']):.1f} +/- {float(x['training_time_s_seed_std']):.1f}",
            f"{float(x['inference_time_ms']):.2f} +/- {float(x['inference_time_ms_seed_std']):.2f}",
            "five-seed mean +/- SD; CPU-only PyTorch 2.11.0; fixed split and S4 layout",
        ])
    rr.append([
        "Repeated-seed retained-timestamp window",
        "-",
        f"{cap} retained timestamps ({window_start} to {window_end}); {timestamp_audit.get('within_window_15min_interval_count', 'n/a')} 15-min intervals and one observed {gap_audit.get('gap_duration_hours', 17.25):.2f}-h gap",
        "-",
        f"all four models; fixed S4; {gap_audit.get('excluded_cross_gap_window_count', 'n/a')} gap-crossing starts excluded; {split_audit.get('training_window_starts', 'n/a')}/{split_audit.get('validation_window_starts', 'n/a')}/{split_audit.get('test_window_starts', 'n/a')} train/validation/test window starts; 11-step embargo",
    ])
    rr.append([
        "Processed S{\\o}nderborg sequence",
        "-",
        f"{len(processed)} retained timestamps; nominal 15-min source with retained gaps",
        "-",
        "frozen canonical input; full SHA-256 and excluded legacy-artifact explanation are reported in the supplement",
    ])
    rt = pd.DataFrame(rr, columns=["Model/item", "Parameters", "Train time/scope", "Inference (ms)", "Evidence"])
    table(rt, "table_main_8_reproducibility.tex", "Computational feasibility and scope. Timing values are mean $\\pm$ sample standard deviation across the five fixed-window seeds on the recorded CPU-only PyTorch 2.11.0 runtime. They are implementation-specific measurements, not hardware-independent latency guarantees. Replay, seasonal, and transfer analyses use their separately stated data blocks.", "tab:main_reproducibility", True)

    robust = pd.read_csv(R / "noise_dropout_robustness_final.csv")
    keep_models = [
        "GRU-MSE",
        "Transformer-MSE",
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 balanced_mode",
    ]
    keep_conditions = ["nominal_five_sensors", "noise_5_percent", "sensor_dropout"]
    robust = robust[
        robust["base_model"].isin(keep_models) & robust["condition"].isin(keep_conditions)
    ].copy()
    robust["Model"] = robust["base_model"].map(display).fillna(robust["base_model"])
    robust["Condition"] = robust["condition"].str.replace("_", " ")
    compact_robust = robust[[
        "Condition", "Model", "RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"
    ]].copy()
    compact_robust.columns = [
        "Condition", "Model", "Ts RMSE ($^\\circ$C)", "Tr RMSE ($^\\circ$C)", "Loss error (%)", "Energy residual (%)", "Boundary residual"
    ]
    for col in compact_robust.columns[2:]:
        compact_robust[col] = compact_robust[col].map(lambda x: f(x, 3))
    table(
        compact_robust,
        "table_robustness_compact_causal.tex",
        "Causal-proxy noise and dropout sensitivity under the fixed S4 layout. Values are calibrated-simulator/model-consistency diagnostics; the small noise changes should not be interpreted as field-fault robustness evidence.",
        "tab:robustness_compact_causal",
        True,
    )

    methodology = pd.DataFrame([
        ["Supply/return transport", "upwind advection + ambient loss; C=eta_v qv dt/(A dx)", r"$^\circ$C", "C; CFL <= 0.8"],
        ["Consumer boundary", "Qload/(rho cp qv)", r"$^\circ$C", "M boundary -> C"],
        ["Causal flow proxy", "Qload/[rho cp max(Ts-Tr_lag, dTmin)]", "m3/s", "S proxy"],
        ["Head loss", "Darcy-Weisbach f(dx/D)u2/(2g)", "m", "S reduced model"],
        ["Heat loss", "UP dx[(Ts-Ta)+(Tr-Ta)]", "W/segment", "C"],
        ["Dynamic energy closure", "Qsrc-Qdel-Qloss-dEpipe/dt", "kW or % of load", "mixed C+S dependency"],
        ["PI loss", "scaled state, sensor, thermal, hydraulic, boundary, energy, heat-loss, smoothness terms", "-", "training objective"],
    ], columns=["Component", "Definition", "Unit", "Evidence/scope"])
    table(methodology, "table_methodology_equation_summary.tex", "Methodology equation and evidence summary. Symbols and complete equations are defined in the main text.", "tab:method_equations", True)

    assumptions = pd.DataFrame([
        ["Topology", "one unbranched 20 km corridor", "does not represent utility branches"],
        ["Hydraulics", "incompressible reduced flow; effective pump/outlet boundary", "no field hydraulic validation"],
        ["Losses", "effective calibrated U; no independently identified local losses", "heat loss is C"],
        ["Flow", "load/lagged-return proxy blended with pump-friction estimate", "not measured flow"],
        ["Elevation/valves/HX", "not independently identified", "excluded from pressure interpretation"],
    ], columns=["Item", "Assumption", "Consequence"])
    table(assumptions, "table_model_assumptions_scope.tex", "Reduced-model assumptions and their evidence consequences.", "tab:model_assumptions", True)

    workflow = pd.DataFrame([
        [1, "Raw measured data", "timestamped M variables", "quality control before splitting"],
        [2, "Chronological split", "train/validation/test + embargo", "no future-window leakage"],
        [3, "Calibration", "effective thermal parameters", "training period only"],
        [4, "Dynamic simulation", "C thermal and S hydraulic states", "common model-referenced benchmark"],
        [5, "Sparse masking and training", "model predictions", "identical split/layout within comparison"],
        [6, "Evidence-specific scoring", "M, C, and S metrics", "no proxy presented as measurement"],
    ], columns=["Step", "Operation", "Output", "Control"])
    table(workflow, "table_calculation_workflow.tex", "Reproducible calculation workflow from measured operating data to evidence-separated scoring.", "tab:calculation_workflow", True)

    conformal = pd.read_csv(R / "uncertainty_conformal_evaluation_locked.csv")
    conformal["coverage (%)"] = conformal["coverage"].map(lambda x: f"{float(x):.1f}")
    conformal["mean width"] = conformal["mean_interval_width"].map(lambda x: f"{float(x):.4f}")
    table(conformal[["quantity", "interval", "method", "coverage (%)", "mean width", "unit", "state_type"]], "table_uncertainty_by_variable.tex", "Held-out split-conformal coverage and width reported separately by variable; widths are not averaged across units and raw ensemble diagnostics are excluded.", "tab:uncertainty_by_variable", False)

    blind = pd.read_csv(R / "xai4heat_sparse_substation_validation_by_substation.csv")
    xai = blind.groupby("variable", as_index=False).agg(substations=("substation_id", "nunique"), samples=("n_samples", "sum"), mean_RMSE_C=("RMSE", "mean"), worst_RMSE_C=("RMSE", "max"))
    xai["mean_RMSE_C"] = xai["mean_RMSE_C"].map(lambda x: f"{float(x):.3f}")
    xai["worst_RMSE_C"] = xai["worst_RMSE_C"].map(lambda x: f"{float(x):.3f}")
    table(xai, "table_xai4heat_substation_variability.tex", "XAI4HEAT leave-one-substation-out primary-temperature variability after raw-file deduplication using all valid-range observations. References are directly measured substation temperatures; no hydraulic fields are evaluated. The physically ordered subset is reported separately as a target-conditioned sensitivity analysis.", "tab:xai_variability", False)
    xai_status = pd.DataFrame([
        ["Raw files", "available locally", "Mendeley package processed without inventing hydraulic variables"],
        ["Validation", "completed", "leave-one-substation-out measured thermal validation"],
        ["Pressure/head and flow", "not evaluated", "not present as dense measured fields"],
    ], columns=["Item", "Status", "Evidence boundary"])
    table(xai_status, "table_xai4heat_status.tex", "XAI4HEAT status in the present run.", "tab:xai_status", True)

    topology = pd.read_csv(R / "topology_necessity_audit.csv")
    for column in topology.select_dtypes(include="number").columns:
        topology[column] = topology[column].map(lambda x: f"{float(x):.4f}")
    table(topology, "table_topology_necessity_audit.tex", "Graph/no-graph necessity audit for the line-network benchmark. The mixed ranking does not establish graph superiority.", "tab:topology_necessity", True)


if __name__ == "__main__":
    main()
