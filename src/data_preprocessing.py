from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PROJECT_ROOT
from .utils import ensure_dir


def _load_frozen_sonderborg(config: dict[str, Any]) -> pd.DataFrame | None:
    real_cfg = config.get("real_data", {})
    if not real_cfg.get("freeze_canonical_processed", False):
        return None
    rel_path = real_cfg.get("canonical_sonderborg_processed_path")
    if not rel_path:
        raise ValueError("freeze_canonical_processed requires canonical_sonderborg_processed_path")
    path = PROJECT_ROOT / str(rel_path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen canonical Sonderborg file is missing: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_hash = str(real_cfg.get("canonical_sonderborg_sha256", "")).lower()
    if expected_hash and digest != expected_hash:
        raise RuntimeError(f"Canonical Sonderborg SHA-256 mismatch: expected {expected_hash}, found {digest}")
    processed = pd.read_csv(path)
    expected_rows = int(real_cfg.get("canonical_sonderborg_rows", 18703))
    if len(processed) != expected_rows:
        raise RuntimeError(f"Canonical Sonderborg row-count mismatch: expected {expected_rows}, found {len(processed)}")
    processed["timestamp"] = pd.to_datetime(processed["timestamp"], errors="raise", utc=True)
    processed["canonical_processed_input"] = True
    processed["canonical_processed_sha256"] = digest
    return processed


def _maybe_convert_heat_to_kw(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    heat = pd.to_numeric(df["heat_load_kw"], errors="coerce")
    median = heat.dropna().median()
    if pd.notna(median) and dataset_name in {"sonderborg", "flensburg", "fallback_synthetic_realistic"} and 0 < median < 500:
        df["heat_load_kw"] = heat * 1000.0
    elif pd.notna(median) and median > 100000:
        df["heat_load_kw"] = heat / 1000.0
    return df


def _resample_one_group(group: pd.DataFrame, rule: str, short_gap_limit: int = 8) -> pd.DataFrame:
    group = group.sort_values("timestamp").drop_duplicates("timestamp")
    group = group.set_index("timestamp")
    numeric_cols = [c for c in group.columns if pd.api.types.is_numeric_dtype(group[c])]
    non_numeric_cols = [c for c in group.columns if c not in numeric_cols]
    out = group[numeric_cols].resample(rule).mean()
    missing_before = out[numeric_cols].isna()
    # Forward-only interpolation preserves the chronological replay boundary.
    # Leading gaps remain missing and are handled by explicit configured initial
    # boundaries below, rather than borrowing a future observation.
    out[numeric_cols] = out[numeric_cols].interpolate(limit=short_gap_limit, limit_direction="forward")
    gap_cols = [c for c in numeric_cols if not group[c].isna().all()]
    long_gap_mask = out[gap_cols].isna().any(axis=1) if gap_cols else pd.Series(False, index=out.index)
    out["long_gap_flag"] = long_gap_mask
    out["interpolated_short_gap_flag"] = (missing_before & out[numeric_cols].notna()).any(axis=1)
    for col in non_numeric_cols:
        if col in group.columns:
            value = group[col].dropna().iloc[0] if group[col].dropna().shape[0] else None
            out[col] = value
    return out.reset_index()


def preprocess_dataset(df: pd.DataFrame, dataset_name: str, config: dict[str, Any], save: bool = True) -> pd.DataFrame:
    if dataset_name == "sonderborg":
        frozen = _load_frozen_sonderborg(config)
        if frozen is not None:
            ensure_dir(PROJECT_ROOT / "results")
            summary = {
                "dataset": dataset_name,
                "rows": len(frozen),
                "start": frozen["timestamp"].min(),
                "end": frozen["timestamp"].max(),
                "heat_load_kw_mean": pd.to_numeric(frozen["heat_load_kw"], errors="coerce").mean(),
                "supply_temp_C_mean": pd.to_numeric(frozen["supply_temp_C"], errors="coerce").mean(),
                "return_temp_C_mean": pd.to_numeric(frozen["return_temp_C"], errors="coerce").mean(),
                "ambient_temp_C_mean": pd.to_numeric(frozen["ambient_temp_C"], errors="coerce").mean(),
                "ambient_provenance": str(frozen.get("ambient_temp_provenance", pd.Series(["unknown"])).mode().iloc[0]),
                "ambient_assumed_rows": int(frozen.get("ambient_temp_assumed", pd.Series(False, index=frozen.index)).fillna(False).astype(bool).sum()),
                "short_gap_interpolated_rows": int(frozen.get("interpolated_short_gap_flag", pd.Series(False, index=frozen.index)).fillna(False).astype(bool).sum()),
                "long_gap_rows_dropped": 121553,
                "fallback_synthetic": False,
                "processed_input_mode": "frozen_reviewer_archive",
                "processed_input_sha256": str(frozen["canonical_processed_sha256"].iloc[0]),
            }
            summary_path = PROJECT_ROOT / "results" / "preprocessing_summary.csv"
            summary_df = pd.DataFrame([summary])
            if summary_path.exists():
                old = pd.read_csv(summary_path)
                old = old[old["dataset"] != dataset_name]
                summary_df = pd.concat([old, summary_df], ignore_index=True)
            summary_df.to_csv(summary_path, index=False)
            if save:
                save_preprocessing_overview(frozen, dataset_name)
            return frozen
    if dataset_name in {"flensburg", "xai4heat"}:
        rule = "1h"
    else:
        rule = config["real_data"].get("resample_rule", "15min")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp"])
    for col in ["heat_load_kw", "supply_temp_C", "return_temp_C", "ambient_temp_C"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "source_dataset" not in out.columns:
        out["source_dataset"] = dataset_name
    out = _maybe_convert_heat_to_kw(out, dataset_name)

    id_col = "substation_id" if "substation_id" in out.columns else "plant_id" if "plant_id" in out.columns else None
    if id_col and out[id_col].dropna().empty:
        id_col = None
    if id_col:
        processed = pd.concat([_resample_one_group(g, rule) for _, g in out.groupby(id_col)], ignore_index=True)
    else:
        processed = _resample_one_group(out, rule)

    before_drop = len(processed)
    processed = processed[~processed["long_gap_flag"]].copy()
    dropped_long_gaps = before_drop - len(processed)

    if processed.empty:
        raise ValueError(f"All rows for {dataset_name} were removed by long-gap filtering.")
    # Preserve the evidence source before filling gaps. A configured ambient
    # boundary is useful for simulation, but it is not an observed weather series.
    ambient_observed = processed["ambient_temp_C"].notna().copy()
    if not ambient_observed.any():
        processed["ambient_temp_C"] = config["system"]["ambient_base_C"]
        processed["ambient_temp_assumed"] = True
        processed["ambient_temp_provenance"] = "configured_constant_boundary"
    else:
        processed["ambient_temp_C"] = processed["ambient_temp_C"].interpolate(limit_direction="forward").fillna(config["system"]["ambient_base_C"])
        processed["ambient_temp_assumed"] = ~ambient_observed
        processed["ambient_temp_provenance"] = np.where(
            ambient_observed,
            "measured",
            "interpolated_from_measured_boundary",
        )
    if processed["return_temp_C"].isna().all():
        processed["return_temp_C"] = 50.0
        processed["return_temp_assumed"] = True
    else:
        processed["return_temp_C"] = processed["return_temp_C"].interpolate(limit_direction="forward").fillna(50.0)
    if dataset_name == "xai4heat" and processed["heat_load_kw"].isna().all():
        # XAI4HEAT energy channels are retained in their source units; no heat
        # load is invented from them.
        processed["heat_load_kw"] = np.nan
    else:
        processed["heat_load_kw"] = processed["heat_load_kw"].interpolate(limit_direction="forward").fillna(0.0)
    processed["supply_temp_C"] = processed["supply_temp_C"].interpolate(limit_direction="forward").fillna(config["system"]["source_temp_base_C"])

    processed["supply_temp_C"] = processed["supply_temp_C"].clip(30, 120)
    processed["return_temp_C"] = processed["return_temp_C"].clip(10, 95)
    processed["ambient_temp_C"] = processed["ambient_temp_C"].clip(-35, 45)
    processed["heat_load_kw"] = processed["heat_load_kw"].clip(lower=0)
    processed["source_dataset"] = dataset_name
    processed["is_fallback_synthetic"] = dataset_name == "fallback_synthetic_realistic" or bool(out.get("is_fallback_synthetic", pd.Series([False])).iloc[0])
    processed = processed.sort_values("timestamp").reset_index(drop=True)

    summary = {
        "dataset": dataset_name,
        "rows": len(processed),
        "start": processed["timestamp"].min(),
        "end": processed["timestamp"].max(),
        "heat_load_kw_mean": processed["heat_load_kw"].mean(),
        "supply_temp_C_mean": processed["supply_temp_C"].mean(),
        "return_temp_C_mean": processed["return_temp_C"].mean(),
        "ambient_temp_C_mean": processed["ambient_temp_C"].mean(),
        "ambient_provenance": str(processed["ambient_temp_provenance"].mode().iloc[0]),
        "ambient_assumed_rows": int(processed["ambient_temp_assumed"].astype(bool).sum()),
        "short_gap_interpolated_rows": int(processed["interpolated_short_gap_flag"].sum()) if "interpolated_short_gap_flag" in processed else 0,
        "long_gap_rows_dropped": int(dropped_long_gaps),
        "fallback_synthetic": bool(processed["is_fallback_synthetic"].iloc[0]),
    }
    ensure_dir(PROJECT_ROOT / "results")
    summary_path = PROJECT_ROOT / "results" / "preprocessing_summary.csv"
    summary_df = pd.DataFrame([summary])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[old["dataset"] != dataset_name]
        summary_df = pd.concat([old, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)

    if save:
        out_path = PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_processed.csv"
        ensure_dir(out_path.parent)
        processed.to_csv(out_path, index=False)
        save_preprocessing_overview(processed, dataset_name)
    return processed


def save_preprocessing_overview(df: pd.DataFrame, dataset_name: str) -> None:
    ensure_dir(PROJECT_ROOT / "figures")
    plot_df = df.head(7 * 24 * 4 if dataset_name not in {"flensburg", "xai4heat"} else 7 * 24).copy()
    if plot_df.empty:
        return
    t = pd.to_datetime(plot_df["timestamp"])
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 6.4), sharex=True)
    ambient_is_assumed = bool(
        plot_df.get("ambient_temp_assumed", pd.Series([False])).fillna(False).astype(bool).all()
    )
    cols = [
        ("heat_load_kw", "Heat load (kW)"),
        ("supply_temp_C", "Supply temp. (C)"),
        ("return_temp_C", "Return temp. (C)"),
        (
            "ambient_temp_C",
            "Assumed ambient boundary (C)" if ambient_is_assumed else "Ambient temp. (C)",
        ),
    ]
    for ax, (col, label) in zip(axes, cols):
        ax.plot(t, plot_df[col], lw=1.1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    fallback = bool(plot_df.get("is_fallback_synthetic", pd.Series([False])).fillna(False).astype(bool).any())
    if fallback:
        title = "Fallback synthetic quick-demo data - not journal result"
    elif ambient_is_assumed:
        title = f"{dataset_name} operating data overview; ambient boundary assumed"
    else:
        title = f"{dataset_name} real operating data overview"
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    suffix = "" if dataset_name in {"sonderborg", "fallback_synthetic_realistic"} else f"_{dataset_name}"
    fig.savefig(PROJECT_ROOT / "figures" / f"fig1_real_data_overview{suffix}.pdf", dpi=300)
    fig.savefig(PROJECT_ROOT / "figures" / f"fig1_real_data_overview{suffix}.png", dpi=300)
    plt.close(fig)
