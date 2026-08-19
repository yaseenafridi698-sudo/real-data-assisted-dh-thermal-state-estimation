from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.real_data_mapper import build_boundary_conditions
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics


RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"


def _escape(value: object) -> str:
    text = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")]:
        text = text.replace(old, new)
    return text


def _table(df: pd.DataFrame, path: Path, caption: str, label: str, resize: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend([r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule", " & ".join(_escape(c) for c in df.columns) + r" \\", r"\midrule"])
    for _, row in df.iterrows():
        lines.append(" & ".join(_escape(row[c]) for c in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}" + ("%" if resize else "")])
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(error, dtype=float) ** 2)))


def _mark_trajectory_starts(df: pd.DataFrame, nominal_minutes: float = 15.0) -> pd.DataFrame:
    """Mark retained discontinuities before constructing causal lag features.

    A retained timestamp after a gap is an observation at a new trajectory
    start, not the next 15-minute state.  This helper is intentionally shared
    by the simple real-temperature baselines so their reported scores use the
    same evidence rule as the gap-aware replay and neural protocols.
    """
    out = df.sort_values("timestamp").copy()
    timestamp = pd.to_datetime(out["timestamp"], utc=True)
    dt_minutes = timestamp.diff().dt.total_seconds().div(60.0)
    out["trajectory_start"] = dt_minutes.isna() | dt_minutes.gt(1.5 * nominal_minutes)
    out["trajectory_id"] = out["trajectory_start"].cumsum().astype(int)
    return out


def dependency_audit() -> pd.DataFrame:
    rows = [
        {
            "quantity": "Sonderborg supply/return calibration RMSE",
            "timestamp-k inputs": "measured supply, return, load, ambient",
            "reference": "measured plant temperature",
            "dependency class": "M calibration",
            "independent validation?": "No - calibration fit",
            "safe interpretation": "thermal calibration credibility",
        },
        {
            "quantity": "Sonderborg chronological return replay",
            "timestamp-k inputs": "current supply/load/ambient; return history through k-1",
            "reference": "withheld measured return at k",
            "dependency class": "M blind replay",
            "independent validation?": "Yes at timestamp k",
            "safe interpretation": "one-step historical online replay",
        },
        {
            "quantity": "Flow proxy",
            "timestamp-k inputs": "load and supply at k; return at k-1",
            "reference": "none",
            "dependency class": "algebraic S proxy",
            "independent validation?": "No",
            "safe interpretation": "causal heat-load-derived flow proxy",
        },
        {
            "quantity": "Delivered-heat error",
            "timestamp-k inputs": "imposed load boundary and proxy flow",
            "reference": "imposed heat load",
            "dependency class": "C consistency",
            "independent validation?": "No",
            "safe interpretation": "boundary heat-delivery consistency error",
        },
        {
            "quantity": "Distributed supply/return RMSE",
            "timestamp-k inputs": "real boundaries and sparse node mask",
            "reference": "calibrated simulator field",
            "dependency class": "mixed C+S dependency",
            "independent validation?": "No dense field measurement",
            "safe interpretation": "simulator-assisted reconstruction benchmark",
        },
        {
            "quantity": "Pressure/head and flow RMSE",
            "timestamp-k inputs": "pump, friction and causal flow-proxy assumptions",
            "reference": "reduced simulator field",
            "dependency class": "S hydraulic hidden state",
            "independent validation?": "No",
            "safe interpretation": "internal hydraulic consistency diagnostic",
        },
        {
            "quantity": "XAI4HEAT withheld-substation RMSE",
            "timestamp-k inputs": "other measured substations only",
            "reference": "withheld measured substation",
            "dependency class": "M spatial withholding",
            "independent validation?": "Yes for measured thermal variables",
            "safe interpretation": "blind measured-node validation",
        },
        {
            "quantity": "Flensburg return comparison",
            "timestamp-k inputs": "assumed 50 degC return",
            "reference": "assumption, not measurement",
            "dependency class": "assumption sensitivity",
            "independent validation?": "No",
            "safe interpretation": "return-assumption consistency only",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "strict_target_dependency_audit.csv", index=False)
    _table(out, TABLES / "table_strict_target_dependency_audit.tex", "Timestamp-level dependency and evidence audit. M denotes measured evidence, C calibrated-simulator quantities, and S simulator-assisted hidden states.", "tab:strict_dependency")
    return out


def measured_node_baselines() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv", parse_dates=["timestamp"])
    cols = ["timestamp", "supply_temp_C", "return_temp_C", "heat_load_kw", "ambient_temp_C"]
    df = _mark_trajectory_starts(df[cols].dropna().reset_index(drop=True))
    n = len(df)
    window = 12
    embargo = window - 1
    train_end = int(0.70 * n)
    val_start = train_end + embargo
    val_end = min(n, val_start + int(0.15 * n))
    test_start = min(n, val_end + embargo)
    train = df.iloc[:train_end].copy()
    test = df.iloc[test_start:].copy()

    train_ar = train.copy()
    train_ar["lag_return"] = train_ar.groupby("trajectory_id", sort=False)["return_temp_C"].shift(1)
    train_ar = train_ar.dropna()
    X = np.column_stack([
        np.ones(len(train_ar)),
        train_ar["supply_temp_C"].to_numpy(float),
        train_ar["heat_load_kw"].to_numpy(float) / 1000.0,
        train_ar["ambient_temp_C"].to_numpy(float),
        train_ar["lag_return"].to_numpy(float),
    ])
    beta = np.linalg.lstsq(X, train_ar["return_temp_C"].to_numpy(float), rcond=None)[0]

    replay_path = RESULTS / "online_replay_timeseries.csv"
    replay = pd.read_csv(replay_path, parse_dates=["timestamp"]) if replay_path.exists() else pd.DataFrame()
    replay_map = replay.set_index("timestamp")["return_temp_predicted_C"] if not replay.empty else pd.Series(dtype=float)
    daily_steps = 96
    rows = []
    # Exclude every retained restart from the scored contiguous-replay set.
    # The preceding observation is separated by an observed gap and cannot be
    # treated as a 15-minute persistence or lagged-autoregression input.
    test = test.loc[~test["trajectory_start"]].copy()
    actual = test["return_temp_C"].to_numpy(float)
    global_indices = test.index.to_numpy(int)
    predictions: dict[str, np.ndarray] = {}
    predictions["Last observation persistence"] = df.loc[global_indices - 1, "return_temp_C"].to_numpy(float)
    daily_idx = np.maximum(global_indices - daily_steps, 0)
    predictions["Daily persistence"] = df.loc[daily_idx, "return_temp_C"].to_numpy(float)
    predictions["Training mean"] = np.full(len(test), float(train["return_temp_C"].mean()))
    ar_x = np.column_stack([
        np.ones(len(test)),
        test["supply_temp_C"].to_numpy(float),
        test["heat_load_kw"].to_numpy(float) / 1000.0,
        test["ambient_temp_C"].to_numpy(float),
        df.loc[global_indices - 1, "return_temp_C"].to_numpy(float),
    ])
    predictions["Linear autoregression"] = ar_x @ beta
    if not replay_map.empty:
        aligned = test["timestamp"].map(replay_map)
        if aligned.notna().all():
            predictions["Causal replay estimator"] = aligned.to_numpy(float)

    for model, pred in predictions.items():
        err = pred - actual
        rows.append({
            "model": model,
            "RMSE_C": _rmse(err),
            "MAE_C": float(np.mean(np.abs(err))),
            "Bias_C": float(np.mean(err)),
            "test_samples": len(err),
            "current target used?": "No",
            "state_type": "real_measured_node",
        })
    out = pd.DataFrame(rows).sort_values("RMSE_C")
    out["gap_handling"] = "trajectory starts excluded; no lag or persistence transition crosses a retained gap"
    out.to_csv(RESULTS / "measured_node_baseline_comparison.csv", index=False)
    tab = out.copy()
    for c in ["RMSE_C", "MAE_C", "Bias_C"]:
        tab[c] = tab[c].map(lambda x: f"{x:.3f}")
    _table(tab, TABLES / "table_measured_node_baseline_comparison.tex", "Leakage-controlled one-step return-temperature replay against simple measured-node baselines. All predictors use only information available before scoring timestamp k; retained-gap restart samples are excluded so no 15-minute transition spans a discontinuity.", "tab:measured_node_baselines", resize=False)
    return out


def xai4heat_protocol() -> pd.DataFrame:
    blind = pd.read_csv(RESULTS / "blind_sensor_validation.csv")
    rows = []
    for (variable, target), group in blind.groupby(["variable", "target_substation"]):
        row = group.iloc[0]
        rows.append({
            "variable": variable,
            "withheld substation": target,
            "samples": int(row["samples"]),
            "RMSE_C": float(row["RMSE_C"]),
            "target used in update?": "No",
            "normalization/fitting": "other substations only",
            "state_type": "real_measured_node",
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "xai4heat_withholding_protocol_audit.csv", index=False)
    summary = out.groupby("variable", as_index=False).agg(folds=("withheld substation", "count"), samples=("samples", "sum"), mean_RMSE_C=("RMSE_C", "mean"), worst_RMSE_C=("RMSE_C", "max"))
    summary["mean_RMSE_C"] = summary["mean_RMSE_C"].map(lambda x: f"{x:.3f}")
    summary["worst_RMSE_C"] = summary["worst_RMSE_C"].map(lambda x: f"{x:.3f}")
    _table(summary, TABLES / "table_xai4heat_withholding_summary.tex", "XAI4HEAT leave-one-substation-out measured-node validation. The target substation is excluded from each estimate; pressure/head and flow are not evaluated.", "tab:xai_withholding", resize=False)
    return out


def flensburg_measured_only() -> pd.DataFrame:
    src = pd.read_csv(RESULTS / "external_validation_flensburg_modes_final.csv")
    rows = []
    for _, row in src.iterrows():
        rows.append({
            "mode": row.get("mode", row.get("model", "unknown")),
            "measured supply RMSE_C": float(row["RMSE_supply_measured_C"]),
            "heat-load consistency_pct": float(row["heat_load_consistency_error_percent"]),
            "return reference": "assumed 50 degC",
            "return metric status": "assumption-consistency; excluded from external validation",
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "flensburg_measured_only_validation.csv", index=False)
    tab = out.copy()
    for c in ["measured supply RMSE_C", "heat-load consistency_pct"]:
        tab[c] = tab[c].map(lambda x: f"{x:.3f}")
    _table(tab, TABLES / "table_flensburg_measured_only.tex", "Flensburg external transfer using measured supply temperature. Return-temperature comparisons are assumption-consistency diagnostics because return temperature is unavailable and set to 50 degC.", "tab:flensburg_measured_only")
    return out


def model_selection_protocol() -> pd.DataFrame:
    config = load_config()
    hp = config.get("hyperparameter_search", {})
    rows = [
        {"item": "Data partition", "setting": "70% train / 15% validation / 15% test", "test access": "test locked until final scoring"},
        {"item": "Sequence leakage control", "setting": f"window={config['model']['window_steps']} steps; embargo={config['dataset']['embargo_steps']} steps", "test access": "none during fitting"},
        {"item": "Search budget", "setting": f"max_trials={hp.get('max_trials')}; epochs_per_trial={hp.get('epochs_per_trial')}", "test access": "validation objective only"},
        {"item": "Accuracy mode", "setting": "state/sensor loss emphasized", "test access": "mode fixed before test"},
        {"item": "Balanced mode", "setting": "state and normalized physics residuals balanced", "test access": "mode fixed before test"},
        {"item": "Physics curriculum", "setting": "MSE warm-up, residual ramp, full-loss fine-tune", "test access": "schedule fixed from training config"},
        {"item": "Repeated seeds", "setting": "11, 22, 33, 44, 55; 20-epoch cap; batch size 8; common normalized state-MSE selection", "test access": "fixed split, normalization, and S4 layout"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "model_selection_fairness_audit.csv", index=False)
    _table(out, TABLES / "table_model_selection_protocol.tex", "Model-selection and leakage-control protocol. Accuracy and balanced modes are predefined training objectives rather than post-test selections.", "tab:model_selection_protocol", resize=False)
    return out


def hydraulic_definition() -> pd.DataFrame:
    rows = [
        {"term": "Pump command alpha", "definition": "dimensionless effective pump-speed/command proxy", "units": "-", "evidence": "assumed boundary input"},
        {"term": "Pump head", "definition": "H_p=c1 alpha^2+c2 alpha+c3", "units": "m", "evidence": "reduced simulator"},
        {"term": "Volumetric flow q_v", "definition": "causal blend of lagged-return heat-load proxy and pump/friction solution", "units": "m3/s", "evidence": "S proxy"},
        {"term": "Mass flow", "definition": "m_dot=rho q_v", "units": "kg/s", "evidence": "S conversion"},
        {"term": "Friction loss", "definition": "Darcy-Weisbach f (dx/D) u^2/(2g)", "units": "m per segment", "evidence": "S reduced model"},
        {"term": "Downstream boundary", "definition": "fixed effective outlet head", "units": "m", "evidence": "assumed boundary"},
        {"term": "Dynamic correction", "definition": "term proportional to dq_v/dt", "units": "m", "evidence": "reduced transient regularization"},
        {"term": "Excluded effects", "definition": "local valves, elevation, branches and heat-exchanger pressure losses not identified", "units": "-", "evidence": "model limitation"},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "hydraulic_model_definition_audit.csv", index=False)
    _table(out, TABLES / "table_hydraulic_model_definition.tex", "Definition and evidence status of the reduced hydraulic model. Hydraulic quantities are simulator-assisted hidden states, not measured field validation.", "tab:hydraulic_definition")
    return out


def _resample_boundary(boundary: dict, new_dt: float) -> dict:
    """Resample only within observed continuous trajectories.

    Numerical-resolution checks require a regular step inside each observed
    segment, but must not linearly interpolate inputs across a retained 17.25-h
    observation gap.  A fresh trajectory flag is emitted at every segment
    start so the simulator reinitializes rather than advancing through a
    fictitious time interval.
    """
    old_t = np.asarray(boundary["time_s"], dtype=float)
    old_starts = np.asarray(boundary.get("trajectory_start", np.zeros(len(old_t), dtype=bool)), dtype=bool)
    if len(old_starts):
        old_starts[0] = True
    start_indices = np.flatnonzero(old_starts)
    new_segments: list[np.ndarray] = []
    for pos, start in enumerate(start_indices):
        end = int(start_indices[pos + 1]) if pos + 1 < len(start_indices) else len(old_t)
        segment_t = old_t[start:end]
        if len(segment_t) == 1:
            new_segments.append(segment_t.copy())
        else:
            new_segments.append(np.arange(segment_t[0], segment_t[-1] + 0.1 * new_dt, new_dt))
    new_t = np.concatenate(new_segments) if new_segments else np.asarray([], dtype=float)
    starts = np.zeros(len(new_t), dtype=bool)
    cursor = 0
    for segment_t in new_segments:
        if len(segment_t):
            starts[cursor] = True
            cursor += len(segment_t)

    out = dict(boundary)
    out["time_s"] = new_t
    out["trajectory_start"] = starts
    out["trajectory_id"] = np.cumsum(starts, dtype=int) - 1
    for key in ["T_source", "T_return_measured", "Q_load_W", "Ta", "alpha_estimated"]:
        values = np.asarray(boundary[key], dtype=float)
        pieces = []
        for pos, start in enumerate(start_indices):
            end = int(start_indices[pos + 1]) if pos + 1 < len(start_indices) else len(old_t)
            pieces.append(np.interp(new_segments[pos], old_t[start:end], values[start:end]))
        out[key] = np.concatenate(pieces) if pieces else np.asarray([], dtype=float)
    out.pop("q_proxy", None)
    out["flow_proxy_mode"] = "causal_lagged_return"
    return out


def _trajectory_integral(values: np.ndarray, time_s: np.ndarray, trajectory_start: np.ndarray) -> float:
    """Integrate without spanning retained observation gaps."""
    starts = np.flatnonzero(np.asarray(trajectory_start, dtype=bool))
    total = 0.0
    for pos, start in enumerate(starts):
        end = int(starts[pos + 1]) if pos + 1 < len(starts) else len(time_s)
        if end - start >= 2:
            total += float(np.trapezoid(np.asarray(values)[start:end], np.asarray(time_s)[start:end]))
    return total


def expanded_numerical_verification() -> pd.DataFrame:
    config = load_config()
    data = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv", parse_dates=["timestamp"]).head(768)
    boundary = build_boundary_conditions(data, config)
    params_path = RESULTS / "calibrated_parameters.json"
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    cases = [(2000.0, 1800.0), (1000.0, 900.0), (500.0, 450.0)]
    sims = {}
    for dx, dt in cases:
        cfg = deepcopy(config)
        cfg["system"]["dx_m"] = dx
        cfg["system"]["dt_s"] = dt
        sims[(dx, dt)] = simulate_thermo_hydraulics(_resample_boundary(boundary, dt), cfg, params=params)
    ref = sims[(500.0, 450.0)]
    ref_t = np.asarray(ref["time_s"], dtype=float)
    ref_ts = np.asarray(ref["Ts"][:, -1], dtype=float)
    ref_tr = np.asarray(ref["Tr"][:, 0], dtype=float)
    ref_loss_J = _trajectory_integral(ref["Q_loss"], ref_t, ref["trajectory_start"])
    rows = []
    for (dx, dt), sim in sims.items():
        t = np.asarray(sim["time_s"], dtype=float)
        ts = np.interp(ref_t, t, np.asarray(sim["Ts"][:, -1], dtype=float))
        tr = np.interp(ref_t, t, np.asarray(sim["Tr"][:, 0], dtype=float))
        ts_err = ts - ref_ts
        tr_err = tr - ref_tr
        corr = np.correlate(ts - ts.mean(), ref_ts - ref_ts.mean(), mode="full")
        lag_steps = int(np.argmax(corr) - (len(ref_ts) - 1))
        loss_J = _trajectory_integral(sim["Q_loss"], t, sim["trajectory_start"])
        rows.append({
            "dx_m": dx,
            "dt_s": dt,
            "max_effective_CFL": "<=0.8 by substepping",
            "outlet_Ts_L2_C": _rmse(ts_err),
            "outlet_Ts_Linf_C": float(np.max(np.abs(ts_err))),
            "source_Tr_L2_C": _rmse(tr_err),
            "source_Tr_Linf_C": float(np.max(np.abs(tr_err))),
            "arrival_time_error_min": abs(lag_steps * 450.0 / 60.0),
            "cumulative_heat_loss_error_pct": float(100.0 * abs(loss_J - ref_loss_J) / max(abs(ref_loss_J), 1.0)),
            "reference": "500 m / 450 s",
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "numerical_verification_expanded.csv", index=False)
    tab = out.copy()
    for c in ["outlet_Ts_L2_C", "outlet_Ts_Linf_C", "source_Tr_L2_C", "source_Tr_Linf_C", "arrival_time_error_min", "cumulative_heat_loss_error_pct"]:
        tab[c] = tab[c].map(lambda x: f"{x:.4f}")
    _table(tab, TABLES / "table_numerical_verification_expanded.tex", "Coordinated spatial-temporal refinement against the 500 m / 450 s solution. The table reports full time-series norms, phase error, and integrated heat-loss error rather than a single mean outlet value.", "tab:numerical_verification_expanded")
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    dependency_audit()
    measured_node_baselines()
    xai4heat_protocol()
    flensburg_measured_only()
    model_selection_protocol()
    hydraulic_definition()
    expanded_numerical_verification()
    print(RESULTS / "strict_target_dependency_audit.csv")


if __name__ == "__main__":
    main()
