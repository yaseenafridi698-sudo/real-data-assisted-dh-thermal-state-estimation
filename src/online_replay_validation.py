from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
FIGURES = PROJECT_ROOT / "figures" / "final"
LOCKED = PROJECT_ROOT / "results_locked"


def _ensure_dirs() -> None:
    for path in [RESULTS, TABLES, FIGURES, LOCKED]:
        path.mkdir(parents=True, exist_ok=True)


def _escape(value: object) -> str:
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
        "±": r"$\pm$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "not available"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_table(df: pd.DataFrame, path: Path, caption: str, label: str, max_rows: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    align = "l" * len(data.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_escape(c) for c in data.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(" & ".join(_escape(row[c]) for c in data.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trajectory_start_flags(df: pd.DataFrame, nominal_dt_s: float = 900.0, include_first: bool = False) -> np.ndarray:
    """Flag retained timestamp discontinuities without imputing a transition."""
    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    intervals = timestamps.diff().dt.total_seconds().to_numpy(dtype=float)
    starts = np.zeros(len(df), dtype=bool)
    if include_first and len(starts):
        starts[0] = True
    if len(starts) > 1:
        starts[1:] = intervals[1:] > 1.5 * nominal_dt_s
    return starts


def leakage_free_split_audit(window_steps: int = 12) -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    embargo = window_steps - 1
    train_end = int(0.70 * n)
    val_start = min(n, train_end + embargo)
    val_len = int(0.15 * n)
    val_end = min(n, val_start + val_len)
    test_start = min(n, val_end + embargo)
    starts = _trajectory_start_flags(df, include_first=False)
    parts = [
        ("train", 0, train_end),
        ("embargo_train_val", train_end, val_start),
        ("validation", val_start, val_end),
        ("embargo_val_test", val_end, test_start),
        ("test", test_start, n),
    ]
    rows = []
    for role, start, end in parts:
        sub = df.iloc[start:end]
        rows.append(
            {
                "role": role,
                "start_index": start,
                "end_index_exclusive": end,
                "samples": len(sub),
                "start_timestamp": "" if sub.empty else str(sub["timestamp"].iloc[0]),
                "end_timestamp": "" if sub.empty else str(sub["timestamp"].iloc[-1]),
                "window_steps": window_steps,
                "embargo_steps": embargo if "embargo" in role else 0,
                "normalization_source": "training block only" if role == "train" else "not fitted here",
                "retained_gap_starts": int(starts[start:end].sum()),
                "leakage_control_note": "raw time-series split before window construction; future samples excluded; retained timestamp gaps reset replay state",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "leakage_free_chronological_split_audit.csv", index=False)
    compact = out[["role", "samples", "start_timestamp", "end_timestamp", "window_steps", "embargo_steps", "leakage_control_note"]]
    _write_table(
        compact,
        TABLES / "table_leakage_free_chronological_split.tex",
        "Leakage-free chronological split protocol. The raw time series is split before window construction, with an 11-step embargo for 12-step windows; normalization and calibration are fitted from the training block only. Retained timestamp discontinuities reset the replay state and are not treated as 15-min transitions.",
        "tab:leakage_free_split",
    )
    return out


def _chronological_blocks(df: pd.DataFrame, window_steps: int = 12) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    embargo = window_steps - 1
    train_end = int(0.70 * n)
    val_start = min(n, train_end + embargo)
    val_end = min(n, val_start + int(0.15 * n))
    test_start = min(n, val_end + embargo)
    return df.iloc[:train_end].copy(), df.iloc[val_start:val_end].copy(), df.iloc[test_start:].copy()


def _fit_linear_return_model(train: pd.DataFrame) -> np.ndarray:
    data = train.copy()
    data["lag_return"] = data["return_temp_C"].shift(1)
    data.loc[_trajectory_start_flags(data, include_first=False), "lag_return"] = np.nan
    data = data.dropna(subset=["supply_temp_C", "heat_load_kw", "ambient_temp_C", "lag_return", "return_temp_C"])
    x = np.column_stack(
        [
            np.ones(len(data)),
            data["supply_temp_C"].to_numpy(float),
            data["heat_load_kw"].to_numpy(float) / 1000.0,
            data["ambient_temp_C"].to_numpy(float),
            data["lag_return"].to_numpy(float),
        ]
    )
    y = data["return_temp_C"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _replay_return(
    df: pd.DataFrame,
    beta: np.ndarray,
    initial_return: float,
    update_gain: float,
    q90: float | None = None,
    exclude_first_from_score: bool = True,
    initial_segment_start: bool = False,
) -> pd.DataFrame:
    rows = []
    lag_est = float(initial_return)
    bias = 0.0
    runtimes = []
    replay_df = df.reset_index(drop=True).copy()
    starts = _trajectory_start_flags(replay_df, include_first=False)
    if initial_segment_start and len(starts):
        starts[0] = True
    trajectory_id = np.cumsum(starts, dtype=int)
    for k, row in replay_df.iterrows():
        if starts[k]:
            # A long retained-data hiatus has no resolved state trajectory.
            # Reset to the prespecified cold-start state rather than advancing a
            # lagged return or bias through an unobserved interval.
            lag_est = float(initial_return)
            bias = 0.0
        start = time.perf_counter()
        x = np.array([1.0, row["supply_temp_C"], row["heat_load_kw"] / 1000.0, row["ambient_temp_C"], lag_est], dtype=float)
        pred = float(x @ beta + bias)
        pred = float(np.clip(pred, 10.0, 95.0))
        actual = float(row["return_temp_C"])
        resid = actual - pred
        if q90 is not None:
            covered = abs(resid) <= q90
        else:
            covered = np.nan
        # Current hidden return measurement is scored first. Only after scoring is
        # it allowed to update a slow bias term for later timestamps.
        bias = float(np.clip((1.0 - update_gain) * bias + update_gain * resid, -6.0, 6.0))
        lag_est = pred
        runtimes.append(time.perf_counter() - start)
        rows.append(
            {
                "timestamp": row["timestamp"],
                "supply_temp_C": row["supply_temp_C"],
                "heat_load_kw": row["heat_load_kw"],
                "return_temp_measured_C": actual,
                "return_temp_predicted_C": pred,
                "residual_C": resid,
                "bias_state_C": bias,
                "covered_by_validation_q90": covered,
                "trajectory_start": bool(starts[k]),
                "trajectory_id": int(trajectory_id[k]),
                "scored_for_contiguous_replay": bool(not starts[k] or not exclude_first_from_score),
                "current_return_used_in_prediction": False,
                "current_return_used_after_scoring_for_next_bias_update": True,
                "runtime_ms": runtimes[-1] * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def online_chronological_replay() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["supply_temp_C", "return_temp_C", "heat_load_kw", "ambient_temp_C"]).reset_index(drop=True)
    train, val, test = _chronological_blocks(df, window_steps=12)
    beta = _fit_linear_return_model(train)
    val_init = float(train["return_temp_C"].iloc[-1])
    val_pred = _replay_return(val, beta, val_init, update_gain=0.0, initial_segment_start=True)
    val_residual = val_pred.loc[val_pred["scored_for_contiguous_replay"], "residual_C"].to_numpy(float)
    q90 = float(np.quantile(np.abs(val_residual), 0.90))
    test_init = float(val["return_temp_C"].iloc[-1]) if not val.empty else val_init
    replay = _replay_return(test, beta, test_init, update_gain=0.03, q90=q90, initial_segment_start=True)
    replay.to_csv(RESULTS / "online_replay_timeseries.csv", index=False)

    scored = replay["scored_for_contiguous_replay"].astype(bool)
    residual = replay.loc[scored, "residual_C"].to_numpy(float)
    dt_s = 900.0
    metrics = pd.DataFrame(
        [
            {
                "experiment": "Sonderborg chronological online replay",
                "held_out_variable": "return_temp_C",
                "available_inputs": "current supply temperature, heat load, ambient temperature, previous estimated return",
                "test_samples": len(replay),
                "scored_contiguous_samples": int(scored.sum()),
                "excluded_gap_restart_samples": int((~scored).sum()),
                "RMSE_C": float(np.sqrt(np.mean(residual**2))),
                "MAE_C": float(np.mean(np.abs(residual))),
                "MaxAE_C": float(np.max(np.abs(residual))),
                "Bias_C": float(np.mean(residual)),
                "empirical_90pct_interval_width_C": float(2 * q90),
                "empirical_coverage_percent": float(100.0 * replay.loc[scored, "covered_by_validation_q90"].mean()),
                "mean_update_time_ms": float(replay["runtime_ms"].mean()),
                "p95_update_time_ms": float(replay["runtime_ms"].quantile(0.95)),
                "real_time_factor": float((replay["runtime_ms"].mean() / 1000.0) / dt_s),
                "failed_updates": 0,
                "state_type": "real_measured_node",
                "safe_claim": "online chronological replay with current return temperature hidden during prediction; retained-gap restart samples excluded from scoring; not field-deployed real-time control",
            }
        ]
    )
    metrics.to_csv(RESULTS / "online_replay_metrics.csv", index=False)

    forecast_rows = []
    for horizon_steps, horizon_min in [(1, 15), (2, 30), (4, 60), (8, 120)]:
        actual = replay["return_temp_measured_C"].shift(-horizon_steps)
        pred = replay["return_temp_predicted_C"]
        same_trajectory = replay["trajectory_id"].eq(replay["trajectory_id"].shift(-horizon_steps))
        valid = actual.notna() & same_trajectory & replay["scored_for_contiguous_replay"].astype(bool)
        err = (actual[valid] - pred[valid]).to_numpy(float)
        forecast_rows.append(
            {
                "horizon_min": horizon_min,
                "RMSE_return_C": float(np.sqrt(np.mean(err**2))),
                "MAE_return_C": float(np.mean(np.abs(err))),
                "samples": int(valid.sum()),
                "forecast_boundary_assumption": "persistence of current boundary inputs",
                "safe_claim": "short-horizon online replay diagnostic, not operational forecast guarantee",
            }
        )
    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(RESULTS / "online_replay_forecast_horizons.csv", index=False)

    gap_audit = pd.DataFrame(
        [
            {
                "processed_rows": int(len(df)),
                "retained_gap_restarts": int(_trajectory_start_flags(df, include_first=False).sum()),
                "nominal_dt_s": 900.0,
                "treatment": "reset lagged state and bias at each retained gap; exclude restart sample from contiguous replay scoring",
                "scored_test_samples": int(scored.sum()),
                "excluded_test_restart_samples": int((~scored).sum()),
            }
        ]
    )
    gap_audit.to_csv(RESULTS / "online_replay_gap_handling_audit.csv", index=False)

    table = metrics.copy()
    for c in ["RMSE_C", "MAE_C", "MaxAE_C", "Bias_C", "empirical_90pct_interval_width_C", "empirical_coverage_percent", "mean_update_time_ms", "p95_update_time_ms", "real_time_factor"]:
        table[c] = table[c].map(lambda x: _fmt(x, 4 if c == "real_time_factor" else 3))
    _write_table(
        table[["experiment", "held_out_variable", "test_samples", "scored_contiguous_samples", "excluded_gap_restart_samples", "RMSE_C", "MAE_C", "MaxAE_C", "empirical_coverage_percent", "p95_update_time_ms", "safe_claim"]],
        TABLES / "table_online_replay_metrics.tex",
        "Online chronological replay validation. The current return-temperature measurement is hidden during prediction and scored as a real measured node; it is used only after scoring to update a slow bias state for future timestamps. Samples immediately following retained timestamp gaps reset the replay state and are excluded from the contiguous-replay score.",
        "tab:online_replay_metrics",
    )
    return metrics, replay


def blind_xai4heat_ensemble_observer_validation(n_ens: int = 40, seed: int = 123, max_rows: int = 6000, max_test_rows: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "xai4heat_processed.csv", parse_dates=["timestamp"])
    variables = [
        ("supply_temp_C", "Primary supply temperature"),
        ("return_temp_C", "Primary return temperature"),
        ("secondary_supply_temp_C", "Secondary supply temperature"),
        ("secondary_return_temp_C", "Secondary return temperature"),
    ]
    rows = []
    idw_rows = []
    by_time_rows = []
    for variable, label in variables:
        clean = df[["timestamp", "substation_id", "ordered_position", variable]].copy()
        clean[variable] = pd.to_numeric(clean[variable], errors="coerce")
        if "supply" in variable:
            clean = clean[(clean[variable] >= 15.0) & (clean[variable] <= 120.0)]
        else:
            clean = clean[(clean[variable] >= 5.0) & (clean[variable] <= 95.0)]
        pivot = clean.pivot_table(index="timestamp", columns="substation_id", values=variable, aggfunc="mean").sort_index()
        positions = clean.groupby("substation_id")["ordered_position"].median().to_dict()
        pivot = pivot.dropna(axis=1, thresh=max(100, int(0.10 * len(pivot))))
        if pivot.shape[1] < 3:
            continue
        pivot = pivot.interpolate(limit=2).ffill(limit=1).bfill(limit=1)
        if len(pivot) > max_rows:
            stride = int(np.ceil(len(pivot) / max_rows))
            pivot = pivot.iloc[::stride].copy()
        stations = list(pivot.columns)
        train_len = max(10, int(0.70 * len(pivot)))
        train = pivot.iloc[:train_len]
        test = pivot.iloc[train_len:].head(max_test_rows)
        mean = train.mean().to_numpy(float)
        diffs = train.diff().dropna()
        proc_std = np.nanstd(diffs.to_numpy(float), axis=0)
        proc_std = np.where(np.isfinite(proc_std) & (proc_std > 0), proc_std, np.nanstd(train.to_numpy(float), axis=0) * 0.02 + 0.05)
        meas_var = np.maximum((np.nanstd(train.to_numpy(float), axis=0) * 0.03) ** 2, 0.05**2)

        for target in stations:
            target_idx = stations.index(target)
            target_pos = float(positions.get(target, target_idx))
            idw_preds = []
            idw_actuals = []
            ens = rng.normal(mean, np.maximum(np.nanstd(train.to_numpy(float), axis=0), 0.1), size=(n_ens, len(stations)))
            preds = []
            actuals = []
            for ts, row in test.iterrows():
                if np.isfinite(row[target]):
                    obs_values = []
                    obs_weights = []
                    for station in stations:
                        if station == target or not np.isfinite(row[station]):
                            continue
                        distance = abs(float(positions.get(station, stations.index(station))) - target_pos)
                        obs_values.append(float(row[station]))
                        obs_weights.append(1.0 / max(distance, 1.0) ** 2)
                    if len(obs_values) >= 2:
                        idw_preds.append(float(np.average(obs_values, weights=obs_weights)))
                        idw_actuals.append(float(row[target]))
                ens = 0.985 * ens + 0.015 * mean + rng.normal(0.0, proc_std, size=ens.shape)
                obs_cols = [i for i, s in enumerate(stations) if s != target and np.isfinite(row[s])]
                if obs_cols:
                    y = row.iloc[obs_cols].to_numpy(float)
                    Y = ens[:, obs_cols]
                    Xc = ens - ens.mean(axis=0, keepdims=True)
                    Yc = Y - Y.mean(axis=0, keepdims=True)
                    Pxy = Xc.T @ Yc / max(n_ens - 1, 1)
                    Pyy = Yc.T @ Yc / max(n_ens - 1, 1) + np.diag(meas_var[obs_cols])
                    K = Pxy @ np.linalg.pinv(Pyy)
                    perturbed = y + rng.normal(0.0, np.sqrt(meas_var[obs_cols]), size=(n_ens, len(obs_cols)))
                    ens = ens + (perturbed - Y) @ K.T
                if np.isfinite(row[target]):
                    pred = float(ens[:, target_idx].mean())
                    actual = float(row[target])
                    preds.append(pred)
                    actuals.append(actual)
                    if len(by_time_rows) < 20000:
                        by_time_rows.append(
                            {
                                "timestamp": ts,
                                "variable": variable,
                                "target_substation": target,
                                "predicted": pred,
                                "actual": actual,
                                "error": actual - pred,
                            }
                        )
            if idw_preds:
                pred_arr = np.asarray(idw_preds)
                actual_arr = np.asarray(idw_actuals)
                err = actual_arr - pred_arr
                idw_rows.append(
                    {
                        "validation_type": "leave_one_substation_out_IDW",
                        "variable": variable,
                        "variable_label": label,
                        "target_substation": target,
                        "samples": len(err),
                        "RMSE_C": float(np.sqrt(np.mean(err**2))),
                        "MAE_C": float(np.mean(np.abs(err))),
                        "MaxAE_C": float(np.max(np.abs(err))),
                        "Bias_C": float(np.mean(err)),
                        "state_type": "real_measured_node",
                        "blind_node_used_in_update": False,
                        "safe_claim": "real measured-node sparse-substation validation only; target substation is hidden from the estimator",
                    }
                )
            if preds:
                pred_arr = np.asarray(preds)
                actual_arr = np.asarray(actuals)
                err = actual_arr - pred_arr
                rows.append(
                    {
                        "validation_type": "leave_one_substation_out_ensemble_observer",
                        "variable": variable,
                        "variable_label": label,
                        "target_substation": target,
                        "samples": len(err),
                        "RMSE_C": float(np.sqrt(np.mean(err**2))),
                        "MAE_C": float(np.mean(np.abs(err))),
                        "MaxAE_C": float(np.max(np.abs(err))),
                        "Bias_C": float(np.mean(err)),
                        "state_type": "real_measured_node",
                        "blind_node_used_in_update": False,
                        "safe_claim": "real measured-node sparse-substation validation only; not pressure/head, flow, or dense pipe-field validation",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "blind_sensor_validation_ensemble_observer_stress.csv", index=False)
    idw_out = pd.DataFrame(idw_rows)
    idw_out.to_csv(RESULTS / "blind_sensor_validation.csv", index=False)
    pd.DataFrame(by_time_rows).to_csv(RESULTS / "blind_sensor_validation_timeseries_sample.csv", index=False)
    summary = (
        idw_out.groupby(["variable_label", "state_type"], as_index=False)
        .agg(samples=("samples", "sum"), mean_RMSE_C=("RMSE_C", "mean"), mean_MAE_C=("MAE_C", "mean"), max_MaxAE_C=("MaxAE_C", "max"))
        .sort_values("variable_label")
    )
    summary.to_csv(RESULTS / "blind_sensor_validation_summary.csv", index=False)
    if not out.empty:
        ensemble_summary = (
            out.groupby(["variable_label", "state_type"], as_index=False)
            .agg(samples=("samples", "sum"), mean_RMSE_C=("RMSE_C", "mean"), mean_MAE_C=("MAE_C", "mean"), max_MaxAE_C=("MaxAE_C", "max"))
            .sort_values("variable_label")
        )
        ensemble_summary.to_csv(RESULTS / "blind_sensor_validation_ensemble_observer_stress_summary.csv", index=False)
    table = summary.copy()
    for c in ["mean_RMSE_C", "mean_MAE_C", "max_MaxAE_C"]:
        table[c] = table[c].map(lambda x: _fmt(x, 3))
    _write_table(
        table,
        TABLES / "table_blind_measured_node_validation.tex",
        "Blind measured-node validation using XAI4HEAT leave-one-substation-out spatial replay. The target substation is not assimilated during the estimate; metrics use real measured substation temperatures only.",
        "tab:blind_measured_node_validation",
    )
    return idw_out


def dense_gaussian_observer(seed: int = 321) -> tuple[pd.DataFrame, np.ndarray]:
    """Training-only static Gaussian conditioning baseline for simulator fields.

    This is a static covariance-conditioned Gaussian observer: it has no
    forecast ensemble, dynamical propagation, or sequential covariance update.
    The state mean and covariance are fitted only on unique timestamps covered
    by the primary training windows. Metrics are evaluated only on unique held-
    out timestamps; the returned dense array repeats those predictions solely
    to remain compatible with the overlapping-window bootstrap routine.
    """
    _ = seed
    payload = np.load(RESULTS / "dense_reconstruction_payloads.npz")
    refs = np.stack([payload["Ts_reference"], payload["Tr_reference"], payload["H_reference"], payload["q_reference"]], axis=-1).astype(float)
    time_s = np.asarray(payload["time_index"], dtype=float)
    _, n_nodes, n_state = refs.shape
    sensor_nodes = payload["sensor_nodes"].astype(int).tolist()

    # Reconstruct the exact chronological training-timestamp set from the
    # corrected state artifact and the same window/embargo protocol used by the
    # neural benchmark. This avoids fitting covariance on any test payload.
    from src.config import load_config
    from src.dataset import contiguous_window_starts, split_window_indices

    config = load_config()
    corrected = np.load(RESULTS / "corrected_simulator_states.npz")
    corrected_states = np.stack(
        [corrected["Ts"], corrected["Tr"], corrected["H"], corrected["q"]], axis=-1
    ).astype(float)
    window_steps = int(config["model"]["window_steps"])
    valid_starts = contiguous_window_starts(corrected["trajectory_start"], window_steps)
    train_starts, _, _ = split_window_indices(
        n_steps=corrected_states.shape[0],
        window_steps=window_steps,
        train_fraction=float(config["dataset"]["train_fraction"]),
        val_fraction=float(config["dataset"]["val_fraction"]),
        embargo_steps=int(config["dataset"].get("embargo_steps", window_steps - 1)),
        valid_window_starts=valid_starts,
    )
    train_indices = sorted(
        {timestamp for start in train_starts for timestamp in range(start, start + window_steps)}
    )
    train_states = corrected_states[np.asarray(train_indices, dtype=int)]
    train_times = np.asarray(corrected["time_s"], dtype=float)[np.asarray(train_indices, dtype=int)]

    # Collapse overlapping test-window rows before scoring. Reference values at
    # duplicate physical timestamps are identical up to floating-point noise.
    unique_test_times, inverse = np.unique(time_s, return_inverse=True)
    refs_unique = np.empty((len(unique_test_times), n_nodes, n_state), dtype=float)
    for index in range(len(unique_test_times)):
        refs_unique[index] = np.mean(refs[inverse == index], axis=0)
    overlap_count = int(np.intersect1d(train_times, unique_test_times).size)
    if overlap_count:
        raise RuntimeError(f"Gaussian-observer train/test timestamp overlap: {overlap_count}")

    obs = np.asarray(sensor_nodes, dtype=int)
    meas_std_by_state = np.asarray([0.05, 0.10, 0.25, 0.001], dtype=float)
    pred_unique = np.zeros_like(refs_unique)
    runtimes = []
    for t in range(len(unique_test_times)):
        start = time.perf_counter()
        for state in range(n_state):
            train_state = train_states[:, :, state]
            mu = train_state.mean(axis=0)
            cov = np.cov(train_state.T)
            if cov.ndim == 0:
                cov = np.eye(n_nodes) * float(cov)
            r = np.eye(len(obs)) * meas_std_by_state[state] ** 2
            gain = cov[:, obs] @ np.linalg.pinv(cov[np.ix_(obs, obs)] + r)
            y = refs_unique[t, obs, state]
            pred_unique[t, :, state] = mu + gain @ (y - mu[obs])
        runtimes.append((time.perf_counter() - start) * 1000.0)
    diff = pred_unique - refs_unique
    pred = pred_unique[inverse]
    metrics = {
        "model": "Covariance-conditioned Gaussian observer",
        "RMSE_Ts_full": float(np.sqrt(np.mean(diff[..., 0] ** 2))),
        "RMSE_Tr_full": float(np.sqrt(np.mean(diff[..., 1] ** 2))),
        "RMSE_H_full": float(np.sqrt(np.mean(diff[..., 2] ** 2))),
        "RMSE_q_full": float(np.sqrt(np.mean(diff[..., 3] ** 2))),
        "RMSE_Ts_measured_nodes": float(np.sqrt(np.mean(diff[:, sensor_nodes, 0] ** 2))),
        "RMSE_Tr_measured_nodes": float(np.sqrt(np.mean(diff[:, sensor_nodes, 1] ** 2))),
        "mean_update_time_ms": float(np.mean(runtimes)),
        "p95_update_time_ms": float(np.quantile(runtimes, 0.95)),
        "training_timestamp_count": int(len(train_indices)),
        "held_out_unique_timestamp_count": int(len(unique_test_times)),
        "train_test_timestamp_overlap_count": overlap_count,
        "observer_type": "static_training_only_covariance_gaussian_conditioning",
        "state_type": "simulator_assisted_hidden_state",
        "safe_claim": "training-only static covariance-conditioned Gaussian observer on unique held-out calibrated-simulator timestamps; not sequential filtering and not real hydraulic field validation",
    }
    out = pd.DataFrame([metrics])
    out.to_csv(RESULTS / "gaussian_observer_baseline_metrics.csv", index=False)
    baseline = pd.read_csv(RESULTS / "baseline_comparison_final.csv")
    compatible = {c: metrics.get(c, np.nan) for c in baseline.columns}
    compatible["model"] = "Covariance-conditioned Gaussian observer"
    combined = pd.concat([baseline, pd.DataFrame([compatible])], ignore_index=True)
    combined.to_csv(RESULTS / "baseline_comparison_with_gaussian_observer.csv", index=False)
    table = out[["model", "RMSE_Ts_full", "RMSE_Tr_full", "RMSE_H_full", "RMSE_q_full", "training_timestamp_count", "held_out_unique_timestamp_count", "train_test_timestamp_overlap_count", "p95_update_time_ms", "state_type"]].copy()
    for c in table.columns:
        if c not in ["model", "state_type"]:
            table[c] = table[c].map(lambda x: _fmt(x, 4))
    _write_table(
        table,
        TABLES / "table_gaussian_observer_baseline.tex",
        "Training-only covariance-conditioned Gaussian observer. Covariance is fitted on unique primary-training timestamps and scored on disjoint unique held-out timestamps. Pressure/head and flow are simulator-assisted hidden states.",
        "tab:gaussian_observer_baseline",
    )
    np.savez_compressed(
        RESULTS / "gaussian_observer_dense_predictions.npz",
        prediction=pred.astype(np.float32),
        reference=refs.astype(np.float32),
        unique_prediction=pred_unique.astype(np.float32),
        unique_reference=refs_unique.astype(np.float32),
        unique_test_time_s=unique_test_times.astype(np.float64),
        training_time_s=train_times.astype(np.float64),
        sensor_nodes=np.asarray(sensor_nodes),
    )

    # Directly comparable unique-timestamp diagnostic for the deterministic
    # observer and the four seed-11 neural checkpoints represented in the dense
    # payload. This table is kept separate from five-seed means.
    comparison_rows = []
    model_keys = {
        "GRU-MSE (seed 11)": "gru_mse",
        "Transformer-MSE (seed 11)": "transformer_mse",
        "PI-GNN-GRU-v3 accuracy (seed 11)": "proposed_pi_gnn_gru_v3_accuracy_mode",
        "PI-GNN-GRU-v3 balanced (seed 11)": "proposed_pi_gnn_gru_v3_balanced_mode",
    }
    for label, key in model_keys.items():
        neural = np.stack(
            [
                payload[f"Ts_prediction_{key}"], payload[f"Tr_prediction_{key}"],
                payload[f"H_prediction_{key}"], payload[f"q_prediction_{key}"],
            ], axis=-1,
        ).astype(float)
        neural_unique = np.empty_like(refs_unique)
        for index in range(len(unique_test_times)):
            neural_unique[index] = np.mean(neural[inverse == index], axis=0)
        err = neural_unique - refs_unique
        comparison_rows.append(
            {"model": label, "RMSE_Ts_C": np.sqrt(np.mean(err[..., 0] ** 2)), "RMSE_Tr_C": np.sqrt(np.mean(err[..., 1] ** 2)), "RMSE_H_m": np.sqrt(np.mean(err[..., 2] ** 2)), "RMSE_q_m3_s": np.sqrt(np.mean(err[..., 3] ** 2)), "protocol": "seed-11 checkpoint; unique held-out timestamps"}
        )
    comparison_rows.append(
        {"model": metrics["model"], "RMSE_Ts_C": metrics["RMSE_Ts_full"], "RMSE_Tr_C": metrics["RMSE_Tr_full"], "RMSE_H_m": metrics["RMSE_H_full"], "RMSE_q_m3_s": metrics["RMSE_q_full"], "protocol": "training-only covariance; unique held-out timestamps"}
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RESULTS / "observer_heldout_unique_timestamp_comparison.csv", index=False)
    comparison_table = comparison.copy()
    for column in ["RMSE_Ts_C", "RMSE_Tr_C", "RMSE_H_m", "RMSE_q_m3_s"]:
        comparison_table[column] = comparison_table[column].map(lambda value: _fmt(value, 4))
    _write_table(
        comparison_table,
        TABLES / "table_observer_heldout_unique_timestamp_comparison.tex",
        "Held-out unique-timestamp comparison. Neural rows use seed-11 checkpoints; the deterministic observer uses covariance fitted only on primary-training timestamps. All targets are calibrated-simulator or simulator-assisted states.",
        "tab:observer_unique_timestamp_comparison",
    )
    return out, pred


def _aggregate_overlapping_windows(
    timestamps: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse overlapping sequence-window rows to unique physical timestamps.

    Dense reconstructions are emitted per 12-step evaluation window.  Adjacent
    windows overlap, so flattening them would duplicate physical timestamps and
    break chronological block-bootstrap assumptions.  References are identical
    for a physical timestamp; predictions are averaged over the window contexts
    that reconstruct that timestamp.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if timestamps.shape[0] != reference.shape[0] or reference.shape != prediction.shape:
        raise ValueError("Timestamp, reference, and prediction arrays must share their first dimension.")
    unique, inverse, counts = np.unique(timestamps, return_inverse=True, return_counts=True)
    ref_sum = np.zeros((len(unique), *reference.shape[1:]), dtype=float)
    pred_sum = np.zeros((len(unique), *prediction.shape[1:]), dtype=float)
    np.add.at(ref_sum, inverse, reference)
    np.add.at(pred_sum, inverse, prediction)
    divisor = counts.reshape((-1,) + (1,) * (reference.ndim - 1))
    return unique, ref_sum / divisor, pred_sum / divisor


def _continuous_timestamp_segments(timestamps: np.ndarray, nominal_dt_s: float = 900.0) -> list[np.ndarray]:
    """Return index segments that do not cross an observed timestamp gap."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if not len(timestamps):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(timestamps) > 1.5 * nominal_dt_s) + 1]
    ends = np.r_[starts[1:], len(timestamps)]
    return [np.arange(start, end, dtype=int) for start, end in zip(starts, ends)]


def _resample_continuous_blocks(
    segments: list[np.ndarray],
    block_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Moving-block resample within each continuous chronological segment."""
    sampled: list[np.ndarray] = []
    for segment in segments:
        length = len(segment)
        if length <= block_len:
            sampled.append(segment.copy())
            continue
        starts = np.arange(0, length - block_len + 1, dtype=int)
        n_blocks = int(np.ceil(length / block_len))
        local = np.concatenate([np.arange(start, start + block_len, dtype=int) for start in rng.choice(starts, size=n_blocks, replace=True)])[:length]
        sampled.append(segment[local])
    return np.concatenate(sampled) if sampled else np.empty(0, dtype=int)


def moving_block_bootstrap(pred_observer: np.ndarray | None = None, n_boot: int = 200, block_len: int = 12, seed: int = 2026) -> pd.DataFrame:
    """Bootstrap unique, chronological held-out timestamps rather than windows.

    The held-out protocol contains 91 overlapping 12-step windows but only 102
    unique 15-minute timestamps.  A 12-step (3-hour) moving block retains local
    temporal dependence while avoiding the false interpretation that flattened
    window rows form a 24-hour chronological sequence.
    """
    rng = np.random.default_rng(seed)
    payload = np.load(RESULTS / "dense_reconstruction_payloads.npz")
    timestamps = payload["time_index"].astype(float)
    refs = {
        "Ts": payload["Ts_reference"].astype(float),
        "Tr": payload["Tr_reference"].astype(float),
        "H": payload["H_reference"].astype(float),
        "q": payload["q_reference"].astype(float),
    }
    models = {
        "GRU-MSE": {
            "Ts": payload["Ts_prediction_gru_mse"].astype(float),
            "Tr": payload["Tr_prediction_gru_mse"].astype(float),
            "H": payload["H_prediction_gru_mse"].astype(float),
            "q": payload["q_prediction_gru_mse"].astype(float),
        },
        "Transformer-MSE": {
            "Ts": payload["Ts_prediction_transformer_mse"].astype(float),
            "Tr": payload["Tr_prediction_transformer_mse"].astype(float),
            "H": payload["H_prediction_transformer_mse"].astype(float),
            "q": payload["q_prediction_transformer_mse"].astype(float),
        },
        "PI-GNN-GRU-v3 accuracy": {
            "Ts": payload["Ts_prediction_proposed_pi_gnn_gru_v3_accuracy_mode"].astype(float),
            "Tr": payload["Tr_prediction_proposed_pi_gnn_gru_v3_accuracy_mode"].astype(float),
            "H": payload["H_prediction_proposed_pi_gnn_gru_v3_accuracy_mode"].astype(float),
            "q": payload["q_prediction_proposed_pi_gnn_gru_v3_accuracy_mode"].astype(float),
        },
        "PI-GNN-GRU-v3 balanced": {
            "Ts": payload["Ts_prediction_proposed_pi_gnn_gru_v3_balanced_mode"].astype(float),
            "Tr": payload["Tr_prediction_proposed_pi_gnn_gru_v3_balanced_mode"].astype(float),
            "H": payload["H_prediction_proposed_pi_gnn_gru_v3_balanced_mode"].astype(float),
            "q": payload["q_prediction_proposed_pi_gnn_gru_v3_balanced_mode"].astype(float),
        },
    }
    if pred_observer is not None:
        models["Covariance-conditioned Gaussian observer"] = {"Ts": pred_observer[..., 0], "Tr": pred_observer[..., 1], "H": pred_observer[..., 2], "q": pred_observer[..., 3]}
    rows = []
    unique_timestamps, _, _ = _aggregate_overlapping_windows(timestamps, refs["Ts"], refs["Ts"])
    segments = _continuous_timestamp_segments(unique_timestamps)
    if not segments:
        raise ValueError("No chronological timestamps are available for moving-block bootstrap.")
    evidence_type = {
        "Ts": "calibrated_simulator",
        "Tr": "calibrated_simulator",
        "H": "simulator_assisted_hidden_state",
        "q": "simulator_assisted_hidden_state",
    }
    for model, pred_map in models.items():
        for state, ref in refs.items():
            state_timestamps, unique_ref, unique_pred = _aggregate_overlapping_windows(timestamps, ref, pred_map[state])
            if not np.array_equal(state_timestamps, unique_timestamps):
                raise ValueError("State timestamps differ across dense reconstruction payloads.")
            per_t = np.sqrt(np.mean((unique_pred - unique_ref) ** 2, axis=1))
            boot = []
            for _ in range(n_boot):
                idx = _resample_continuous_blocks(segments, block_len, rng)
                boot.append(float(np.mean(per_t[idx])))
            boot = np.asarray(boot)
            rows.append(
                {
                    "model": model,
                    "metric": f"RMSE_{state}_unique_timestamp_block_mean",
                    "single_run_mean": float(np.mean(per_t)),
                    "block_length_steps": block_len,
                    "block_length_hours": block_len * 0.25,
                    "bootstrap_resamples": n_boot,
                    "bootstrap_seed": seed,
                    "source_window_rows": int(len(timestamps)),
                    "unique_timestamp_count": int(len(unique_timestamps)),
                    "continuous_segment_count": int(len(segments)),
                    "duplicate_window_rows_collapsed": int(len(timestamps) - len(unique_timestamps)),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "state_type": evidence_type[state],
                    "safe_claim": "paired moving-block bootstrap over unique chronological held-out timestamps; not seed-level repeatability",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "moving_block_bootstrap_ci.csv", index=False)
    (RESULTS / "moving_block_bootstrap_protocol.json").write_text(
        json.dumps(
            {
                "method": "paired moving-block bootstrap after averaging overlapping window predictions at each unique physical timestamp",
                "source_payload": "results/dense_reconstruction_payloads.npz",
                "source_window_rows": int(len(timestamps)),
                "unique_timestamp_count": int(len(unique_timestamps)),
                "duplicate_window_rows_collapsed": int(len(timestamps) - len(unique_timestamps)),
                "continuous_segment_count": int(len(segments)),
                "timestamp_spacing_seconds": 900,
                "block_length_steps": int(block_len),
                "block_length_hours": float(block_len * 0.25),
                "bootstrap_resamples": int(n_boot),
                "random_seed": int(seed),
                "confidence_interval": "percentile 95% [0.025, 0.975]",
                "scope": "limited temporal sampling uncertainty over one 25.5-hour continuous held-out segment; not seed-level repeatability",
                "rationale": "The original 96-step block would be a 24-hour block, but only 102 unique held-out timestamps are available after collapsing overlapping 12-step windows. A 12-step block is used to preserve local dependence without treating duplicated window rows as chronological observations.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    table = out[out["metric"].isin(["RMSE_Ts_unique_timestamp_block_mean", "RMSE_Tr_unique_timestamp_block_mean"])].copy()
    table["mean [95% CI]"] = table.apply(lambda row: f"{_fmt(row['single_run_mean'])} [{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]", axis=1)
    table["block"] = table.apply(lambda row: f"{int(row['block_length_steps'])} steps ({row['block_length_hours']:.1f} h)", axis=1)
    _write_table(
        table[["model", "metric", "mean [95% CI]", "unique_timestamp_count", "block", "bootstrap_resamples", "state_type"]],
        TABLES / "table_moving_block_bootstrap_ci.tex",
        "Paired moving-block bootstrap intervals after collapsing overlapping sequence-window predictions to 102 unique chronological held-out timestamps (12-step/3-h blocks, 200 resamples, seed 2026). The single 25.5-h segment limits inference; intervals quantify within-segment temporal sampling uncertainty and do not replace seed-level repeatability.",
        "tab:moving_block_bootstrap",
    )
    return out


def plot_replay_and_validation() -> None:
    if plt is None:
        (RESULTS / "online_replay_validation_figure_status.txt").write_text(
            "Skipped fig_online_replay_blind_observer_summary because matplotlib is not available in this runtime. "
            "All numerical CSV and LaTeX table artifacts were generated.\n",
            encoding="utf-8",
        )
        return
    replay_path = RESULTS / "online_replay_timeseries.csv"
    blind_path = RESULTS / "blind_sensor_validation_summary.csv"
    observer_path = RESULTS / "gaussian_observer_baseline_metrics.csv"
    if not (replay_path.exists() and blind_path.exists() and observer_path.exists()):
        return
    replay = pd.read_csv(replay_path, parse_dates=["timestamp"])
    blind = pd.read_csv(blind_path)
    observer = pd.read_csv(observer_path)
    colors = {"blue": "#0000E6", "orange": "#FF6626", "green": "#55D600", "magenta": "#E600E6", "black": "#111111", "gray": "#555555"}
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.family": "serif"})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
    sample = replay.iloc[: min(350, len(replay))]
    axes[0, 0].plot(sample["timestamp"], sample["return_temp_measured_C"], color=colors["black"], lw=1.2, label="Measured return")
    axes[0, 0].plot(sample["timestamp"], sample["return_temp_predicted_C"], color=colors["blue"], lw=1.1, label="Online replay estimate")
    axes[0, 0].set_ylabel("Return temperature ($^\\circ$C)")
    axes[0, 0].set_title("(a) Held-out return replay")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 1].plot(sample["timestamp"], sample["residual_C"], color=colors["orange"], lw=1.0)
    axes[0, 1].axhline(0.0, color=colors["gray"], lw=0.8)
    axes[0, 1].set_ylabel("Residual ($^\\circ$C)")
    axes[0, 1].set_title("(b) Online replay residual")
    x = np.arange(len(blind))
    axes[1, 0].bar(x, blind["mean_RMSE_C"], color=[colors["blue"], colors["orange"], colors["green"], colors["magenta"]][: len(blind)], edgecolor=colors["black"], lw=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(blind["variable_label"].str.replace(" temperature", "", regex=False), rotation=20, ha="right")
    axes[1, 0].set_ylabel("Mean blind RMSE ($^\\circ$C)")
    axes[1, 0].set_title("(c) XAI4HEAT leave-one-substation-out")
    vals = observer[["RMSE_Ts_full", "RMSE_Tr_full", "RMSE_H_full", "RMSE_q_full"]].iloc[0].to_numpy(float)
    labels = ["Ts", "Tr", "H", "q"]
    axes[1, 1].bar(labels, vals, color=[colors["blue"], colors["orange"], colors["green"], colors["magenta"]], edgecolor=colors["black"], lw=0.8)
    axes[1, 1].set_ylabel("RMSE (native units)")
    axes[1, 1].set_title("(d) Static Gaussian observer benchmark")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.2)
    fig.suptitle("Blind measured-node validation, online replay, and Gaussian observer", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES / "fig_online_replay_blind_observer_summary.pdf")
    fig.savefig(FIGURES / "fig_online_replay_blind_observer_summary.png", dpi=600)
    plt.close(fig)


def lock_authoritative_results() -> pd.DataFrame:
    files = [
        "calibration_metrics.csv",
        "model_verification_summary.csv",
        "discretization_study.csv",
        "baseline_comparison_final.csv",
        "physics_consistency_comparison_final.csv",
        "model_ranking_summary_final.csv",
        "sensor_layout_comparison_final.csv",
        "external_validation_flensburg_modes_final.csv",
        "xai4heat_sparse_substation_validation_final.csv",
        "blind_sensor_validation.csv",
        "online_replay_metrics.csv",
        "online_replay_forecast_horizons.csv",
        "gaussian_observer_baseline_metrics.csv",
        "observer_heldout_unique_timestamp_comparison.csv",
        "baseline_comparison_with_gaussian_observer.csv",
        "principal_models_blind_measured_return.csv",
        "principal_models_blind_measured_return_summary.csv",
        "calibration_temporal_transfer_audit.csv",
        "active_figure_provenance_post_causality.csv",
        "moving_block_bootstrap_ci.csv",
        "leakage_free_chronological_split_audit.csv",
        "repeated_seed_statistics_status.csv",
        "thermal_state_metrics.csv",
        "hydraulic_state_metrics.csv",
        "heat_energy_metrics.csv",
        "thermo_hydraulic_coupling_metrics.csv",
        "thermo_hydraulic_robustness.csv",
    ]
    rows = []
    manifest_lines = []
    for name in files:
        src = RESULTS / name
        if not src.exists():
            rows.append({"file": name, "status": "missing", "sha256": "", "note": "not copied"})
            continue
        dst = LOCKED / name
        shutil.copy2(src, dst)
        digest = hashlib.sha256(dst.read_bytes()).hexdigest()
        rows.append({"file": name, "status": "locked", "sha256": digest, "note": "copied from results"})
        manifest_lines.append(f"{digest}  {name}")
    (LOCKED / "manifest_sha256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    inventory = pd.DataFrame(rows)
    inventory.to_csv(LOCKED / "locked_result_inventory.csv", index=False)
    inventory.to_csv(RESULTS / "locked_result_inventory.csv", index=False)
    _write_table(
        inventory.head(12),
        TABLES / "table_locked_result_manifest.tex",
        "Locked authoritative result manifest excerpt. Full hashes are stored in results_locked/manifest_sha256.txt and identify the CSV sources used for manuscript numbers.",
        "tab:locked_manifest",
    )
    return inventory


def run_online_replay_validation_package() -> None:
    _ensure_dirs()
    leakage_free_split_audit()
    online_chronological_replay()
    blind_xai4heat_ensemble_observer_validation()
    _, pred_observer = dense_gaussian_observer()
    moving_block_bootstrap(pred_observer=pred_observer)
    plot_replay_and_validation()
    lock_authoritative_results()
    status = "\n".join(
        [
            "Online replay validation package completed.",
            "Generated: blind_sensor_validation.csv, online_replay_metrics.csv, gaussian_observer_baseline_metrics.csv, moving_block_bootstrap_ci.csv.",
            "Five-seed statistics are generated separately in the Torch-enabled corrected benchmark protocol.",
            "Pressure/head and flow remain simulator-assisted hidden hydraulic states.",
        ]
    )
    (RESULTS / "online_replay_validation_status.txt").write_text(status + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_online_replay_validation_package()
