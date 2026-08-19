from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROXY_CAUSALITY_VERSION = "causal_proxy_v2_trailing_mean_running_range"


def _forward_fill_boundary(series: pd.Series, fallback: float) -> pd.Series:
    """Fill operational boundaries without using a future observation."""
    values = pd.to_numeric(series, errors="coerce").ffill()
    return values.fillna(float(fallback))


def _running_load_range(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return expanding load extrema available at each replay timestamp."""
    loads = np.asarray(values, dtype=float)
    return np.minimum.accumulate(loads), np.maximum.accumulate(loads)


def _causal_proxy_arrays(
    heat_load_kw: np.ndarray,
    source_temp_c: np.ndarray,
    return_temp_c: np.ndarray,
    config: dict[str, Any],
    trajectory_start: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Construct causal pump-speed and flow proxies.

    ``alpha[k]`` uses a trailing five-sample mean of load through k.  The flow
    proxy uses source supply at k and aggregate boundary return at k-1; the
    initial value uses the first available return measurement only to initialise
    the replay.  A retained-timestamp discontinuity starts a new causal
    segment, so neither proxy treats observations across a long gap as
    consecutive 15-minute samples. Neither proxy accesses a future timestamp.
    """
    sys = config["system"]
    loads = np.asarray(heat_load_kw, dtype=float)
    source = np.asarray(source_temp_c, dtype=float)
    returns = np.asarray(return_temp_c, dtype=float)
    starts = np.zeros(len(loads), dtype=bool) if trajectory_start is None else np.asarray(trajectory_start, dtype=bool)
    if starts.shape != loads.shape:
        raise ValueError("trajectory_start must have one value per boundary timestamp")
    if len(starts):
        starts[0] = True
    running_min = np.empty_like(loads)
    running_max = np.empty_like(loads)
    segment_start = np.flatnonzero(starts)
    for index, start in enumerate(segment_start):
        end = int(segment_start[index + 1]) if index + 1 < len(segment_start) else len(loads)
        running_min[start:end], running_max[start:end] = _running_load_range(loads[start:end])
    running_range = running_max - running_min
    normalized = np.divide(
        loads - running_min,
        running_range,
        out=np.zeros_like(loads),
        where=running_range > 1e-9,
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    alpha_raw = float(sys["pump_alpha_min"]) + normalized * (float(sys["pump_alpha_max"]) - float(sys["pump_alpha_min"]))
    alpha = np.empty_like(alpha_raw)
    for index, start in enumerate(segment_start):
        end = int(segment_start[index + 1]) if index + 1 < len(segment_start) else len(alpha_raw)
        alpha[start:end] = pd.Series(alpha_raw[start:end]).rolling(window=5, min_periods=1).mean().to_numpy(dtype=float)

    lagged_return = np.empty_like(returns)
    if len(returns):
        lagged_return[0] = returns[0]
    for k in range(1, len(returns)):
        lagged_return[k] = returns[k] if starts[k] else returns[k - 1]
    delta_t = np.maximum(source - lagged_return, 8.0)
    q_proxy = loads * 1000.0 / np.maximum(float(sys["rho"]) * float(sys["cp"]) * delta_t, 1e-9)
    q_proxy = np.clip(q_proxy, 1e-4, None)
    return alpha, q_proxy, {
        "load_reference_min_kw": float(running_min[-1]),
        "load_reference_max_kw": float(running_max[-1]),
        "load_reference_steps": int(len(loads)),
    }


def refresh_causal_boundary_proxies(boundary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Recompute causal proxies after a controlled boundary perturbation."""
    out = dict(boundary)
    alpha, q_proxy, metadata = _causal_proxy_arrays(
        np.asarray(out["Q_load_W"], dtype=float) / 1000.0,
        np.asarray(out["T_source"], dtype=float),
        np.asarray(out["T_return_measured"], dtype=float),
        config,
        trajectory_start=np.asarray(out.get("trajectory_start", []), dtype=bool) if "trajectory_start" in out else None,
    )
    out["alpha_estimated"] = alpha
    out["q_proxy"] = q_proxy
    out["flow_proxy_mode"] = "causal_lagged_return"
    out["proxy_causality_version"] = PROXY_CAUSALITY_VERSION
    out["alpha_provenance"] = (
        "simulator-assisted causal load-derived pump-speed proxy: trailing five-sample load mean "
        "with an expanding past-only load range; no measured SCADA pump-speed record is available"
    )
    out["q_proxy_provenance"] = (
        "causal heat-load flow proxy using aggregate measured supply boundary at k and aggregate measured "
        "return boundary at k-1; the first sample uses the initial return only for replay initialisation"
    )
    out.update(metadata)
    return out


def build_boundary_conditions(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    sys = config["system"]
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    dt = float(sys["dt_s"])
    timestamps = pd.to_datetime(out["timestamp"], utc=True, errors="raise")
    elapsed_s = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    interval_s = timestamps.diff().dt.total_seconds().to_numpy(dtype=float)
    trajectory_start = np.zeros(len(out), dtype=bool)
    if len(trajectory_start):
        trajectory_start[0] = True
        trajectory_start[1:] = interval_s[1:] > 1.5 * dt
    heat_load_kw = _forward_fill_boundary(out["heat_load_kw"], 0.0)
    source_temp = _forward_fill_boundary(out["supply_temp_C"], float(sys["source_temp_base_C"]))
    return_temp = _forward_fill_boundary(out["return_temp_C"], 50.0)
    ambient = _forward_fill_boundary(out["ambient_temp_C"], float(sys["ambient_base_C"]))

    source_dataset = str(out["source_dataset"].iloc[0]) if "source_dataset" in out.columns and len(out) else "unknown"
    boundary: dict[str, Any] = {
        # Real elapsed time is retained for figures and auditability. The
        # simulator resets at trajectory starts rather than advancing a 15-min
        # numerical state across an observed long gap.
        "time_s": elapsed_s,
        "timestamp_utc": timestamps.astype(str).to_numpy(),
        "trajectory_start": trajectory_start,
        "trajectory_id": np.cumsum(trajectory_start, dtype=int) - 1,
        "T_source": source_temp.to_numpy(dtype=float),
        "T_return_measured": return_temp.to_numpy(dtype=float),
        "Q_load_kw": heat_load_kw.to_numpy(dtype=float),
        "Q_load_W": heat_load_kw.to_numpy(dtype=float) * 1000.0,
        "Ta": ambient.to_numpy(dtype=float),
        "return_temperature_assumed": bool(out.get("return_temp_assumed", pd.Series([False])).fillna(False).astype(bool).any()),
        "source_dataset": source_dataset,
    }
    return refresh_causal_boundary_proxies(boundary, config)


def build_sparse_substation_measurements(xai4heat_df: pd.DataFrame | None, graph_nodes: int) -> dict[str, Any]:
    if xai4heat_df is None or xai4heat_df.empty:
        return {"available": False, "measurements": None, "node_map": {}}
    df = xai4heat_df.copy()
    if "substation_id" not in df.columns:
        df["substation_id"] = "substation_0"
    stations = sorted(df["substation_id"].dropna().astype(str).unique())
    if not stations:
        return {"available": False, "measurements": None, "node_map": {}}
    candidate_nodes = np.linspace(1, graph_nodes - 2, len(stations)).round().astype(int)
    node_map = {station: int(node) for station, node in zip(stations, candidate_nodes)}
    rows = []
    for _, row in df.iterrows():
        station = str(row["substation_id"])
        rows.append(
            {
                "timestamp": row["timestamp"],
                "node": node_map[station],
                "substation_id": station,
                "Ts": row.get("supply_temp_C", np.nan),
                "Tr": row.get("return_temp_C", np.nan),
                "H": np.nan,
                "q": np.nan,
                "heat_load_kw": row.get("heat_load_kw", np.nan),
            }
        )
    return {"available": True, "measurements": pd.DataFrame(rows), "node_map": node_map}
