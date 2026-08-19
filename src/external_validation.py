from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.study_workflow import run_external_validation
from src.utils import ensure_dir


def validate_flensburg_transfer(
    flensburg_df: pd.DataFrame | None,
    config: dict[str, Any],
    calibrated_params: dict[str, float],
    model: torch.nn.Module | None,
    stats: dict[str, Any],
) -> pd.DataFrame | None:
    """Run Flensburg transfer validation without claiming distributed field truth."""
    return run_external_validation(flensburg_df, config, calibrated_params, model, stats)


def _read_processed(name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / f"{name}_processed.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def _dist_stats(df: pd.DataFrame, col: str, prefix: str) -> dict[str, float]:
    vals = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
    if vals.empty:
        return {f"{prefix}_{name}": np.nan for name in ["mean", "std", "p05", "p50", "p95"]}
    return {
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_std": float(vals.std()),
        f"{prefix}_p05": float(vals.quantile(0.05)),
        f"{prefix}_p50": float(vals.quantile(0.50)),
        f"{prefix}_p95": float(vals.quantile(0.95)),
    }


def _wasserstein_1d(a: pd.Series, b: pd.Series) -> float:
    av = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    bv = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if av.size == 0 or bv.size == 0:
        return float("nan")
    q = np.linspace(0.02, 0.98, 200)
    return float(np.nanmean(np.abs(np.quantile(av, q) - np.quantile(bv, q))))


def _sampling_minutes(df: pd.DataFrame) -> float:
    if "timestamp" not in df or df["timestamp"].dropna().shape[0] < 3:
        return float("nan")
    diffs = df["timestamp"].dropna().sort_values().diff().dt.total_seconds().dropna() / 60.0
    return float(diffs.median()) if len(diffs) else float("nan")


def run_improved_flensburg_domain_shift() -> None:
    ensure_dir(PROJECT_ROOT / "results")
    ensure_dir(PROJECT_ROOT / "figures" / "final")
    sonderborg = _read_processed("sonderborg")
    flensburg = _read_processed("flensburg")
    rows = []
    if not sonderborg.empty and not flensburg.empty:
        row: dict[str, Any] = {
            "analysis": "sonderborg_to_flensburg_domain_shift",
            "state_type": "real_measured_node",
            "sampling_interval_sonderborg_min": _sampling_minutes(sonderborg),
            "sampling_interval_flensburg_min": _sampling_minutes(flensburg),
            "return_temperature_assumed": bool(flensburg.get("return_temp_assumed", pd.Series([True])).fillna(True).astype(bool).any()),
            "note": "Flensburg is a domain-shift stress test, not proof of universal transfer.",
        }
        for col, label in [("heat_load_kw", "heat_load_kw"), ("supply_temp_C", "supply_temp_C"), ("return_temp_C", "return_temp_C")]:
            row.update(_dist_stats(sonderborg, col, f"sonderborg_{label}"))
            row.update(_dist_stats(flensburg, col, f"flensburg_{label}"))
            row[f"{label}_wasserstein_distance"] = _wasserstein_1d(sonderborg.get(col, pd.Series(dtype=float)), flensburg.get(col, pd.Series(dtype=float)))
            if np.isfinite(row.get(f"sonderborg_{label}_mean", np.nan)):
                row[f"{label}_mean_difference"] = row.get(f"flensburg_{label}_mean", np.nan) - row.get(f"sonderborg_{label}_mean", np.nan)
        rows.append(row)
    diag = pd.DataFrame(rows)
    diag.to_csv(PROJECT_ROOT / "results" / "flensburg_domain_shift_analysis_improved.csv", index=False)

    sens_rows = []
    if not flensburg.empty:
        supply = pd.to_numeric(flensburg.get("supply_temp_C", pd.Series(dtype=float)), errors="coerce")
        base_return = pd.to_numeric(flensburg.get("return_temp_C", pd.Series(50.0, index=flensburg.index)), errors="coerce").fillna(50.0)
        load = pd.to_numeric(flensburg.get("heat_load_kw", pd.Series(dtype=float)), errors="coerce")
        delta_base = np.maximum(supply - base_return, 1.0)
        for bias in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assumed_return = base_return + bias
            delta = np.maximum(supply - assumed_return, 1.0)
            flow_proxy_ratio = delta_base / delta
            sens_rows.append(
                {
                    "return_assumption_bias_C": bias,
                    "mean_assumed_return_C": float(np.nanmean(assumed_return)),
                    "mean_deltaT_C": float(np.nanmean(delta)),
                    "flow_proxy_change_percent": float((np.nanmean(flow_proxy_ratio) - 1.0) * 100.0),
                    "heat_load_mean_kW": float(np.nanmean(load)),
                    "return_temperature_assumed": True,
                    "state_type": "real_measured_boundary_assumption",
                    "safe_claim": "Return-temperature sensitivity is a diagnostic for the assumed 50 C return value, not real return-temperature validation.",
                }
            )
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(PROJECT_ROOT / "results" / "flensburg_return_temperature_assumption_sensitivity.csv", index=False)
    _plot_domain_shift(sonderborg, flensburg, diag)
    _plot_return_sensitivity(sens)
    _plot_transfer_modes()
    print("Improved Flensburg domain-shift diagnostics completed.")


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for out in [PROJECT_ROOT / "figures" / "final", PROJECT_ROOT / "paper" / "figures" / "final"]:
        ensure_dir(out)
        fig.savefig(out / f"{stem}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_domain_shift(sonderborg: pd.DataFrame, flensburg: pd.DataFrame, diag: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, col, title, unit in [
        (axes[0], "heat_load_kw", "Heat-load distribution", "kW"),
        (axes[1], "supply_temp_C", "Supply-temperature distribution", "$^\\circ$C"),
        (axes[2], "return_temp_C", "Return-temperature assumption", "$^\\circ$C"),
    ]:
        data = [
            pd.to_numeric(sonderborg.get(col, pd.Series(dtype=float)), errors="coerce").dropna(),
            pd.to_numeric(flensburg.get(col, pd.Series(dtype=float)), errors="coerce").dropna(),
        ]
        if data[0].empty or data[1].empty:
            ax.text(0.5, 0.5, "missing variable", ha="center", va="center")
            ax.axis("off")
            continue
        ax.boxplot(data, labels=["Sonderborg", "Flensburg"])
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Flensburg transfer is a domain-shift stress test")
    _save(fig, "fig_flensburg_domain_shift_improved")


def _plot_return_sensitivity(sens: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if sens.empty:
        ax.text(0.5, 0.5, "No return-assumption sensitivity data", ha="center", va="center")
        ax.axis("off")
    else:
        ax.plot(sens["return_assumption_bias_C"], sens["flow_proxy_change_percent"], marker="o", lw=1.4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Return-temperature assumption bias ($^\\circ$C)")
        ax.set_ylabel("Flow-proxy change (%)")
        ax.grid(True, alpha=0.25)
    ax.set_title("Flensburg return-temperature assumption sensitivity")
    _save(fig, "fig_flensburg_return_assumption_sensitivity")


def _plot_transfer_modes() -> None:
    modes = pd.read_csv(PROJECT_ROOT / "results" / "external_validation_flensburg_modes_final.csv") if (PROJECT_ROOT / "results" / "external_validation_flensburg_modes_final.csv").exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    if modes.empty:
        ax.text(0.5, 0.5, "No transfer-mode metrics", ha="center", va="center")
        ax.axis("off")
    else:
        metric = "RMSE_Ts_measured_nodes" if "RMSE_Ts_measured_nodes" in modes.columns else "RMSE_Ts_full"
        modes = modes.copy()
        modes["mode_short"] = modes["mode"].astype(str).str.replace("_", "\n", regex=False)
        ax.bar(modes["mode_short"], pd.to_numeric(modes[metric], errors="coerce"), color="#577590")
        ax.set_ylabel("Supply RMSE ($^\\circ$C)")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Flensburg transfer modes: direct transfer and local adaptation diagnostics")
    _save(fig, "fig_flensburg_transfer_modes_improved")


if __name__ == "__main__":
    run_improved_flensburg_domain_shift()
