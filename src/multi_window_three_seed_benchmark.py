"""Repeated neural training across three separated operating windows.

This audit complements, and does not silently replace, the original five-seed
single-window confirmation.  Each model is trained independently with three
seeds inside each prespecified chronological window using a common S4 sensor
layout and training-only normalization.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.effective_physics import apply_calibrated_params_to_config
from src.real_data_mapper import build_boundary_conditions
from src.repeated_seed_statistics import CORE_MODELS
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import build_loaders, train_and_evaluate_specs
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import load_json, set_seed

plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


WINDOWS = [
    {
        "window": "winter_2016",
        "start": "2016-01-02 08:30:00+00:00",
        "regime": "winter/high-load",
    },
    {
        "window": "shoulder_2016",
        "start": "2016-11-08 19:30:00+00:00",
        "regime": "late-autumn/heating-onset",
    },
    {
        "window": "late_winter_2018",
        "start": "2018-02-22 03:00:00+00:00",
        "regime": "later-year high-load",
    },
]
WINDOW_LENGTH = 512
SEEDS = [11, 22, 33]
METRICS = ["RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"]
RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
FIGURES = PROJECT_ROOT / "figures" / "final"
RAW_PATH = RESULTS / "multi_window_three_seed_raw_metrics.csv"


def _frame_for_window(canonical: pd.DataFrame, spec: dict) -> pd.DataFrame:
    start = pd.Timestamp(spec["start"])
    matches = canonical.index[canonical["timestamp"].eq(start)].tolist()
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one start timestamp for {spec['window']}: {start}")
    frame = canonical.iloc[matches[0] : matches[0] + WINDOW_LENGTH].copy()
    if len(frame) != WINDOW_LENGTH:
        raise RuntimeError(f"Window {spec['window']} is incomplete.")
    gaps = frame["timestamp"].diff().dt.total_seconds().dropna()
    if not (gaps == 900).all():
        raise RuntimeError(f"Window {spec['window']} contains non-15-minute gaps.")
    return frame


def _save_summary(raw: pd.DataFrame) -> None:
    summary_rows = []
    for (window, regime, model), group in raw.groupby(["window", "regime", "model"]):
        row = {"window": window, "regime": regime, "model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "multi_window_three_seed_summary.csv", index=False)

    aggregate_rows = []
    for model, group in raw.groupby("model"):
        row = {"model": model, "windows": int(group["window"].nunique()), "runs": len(group)}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_between_run_std"] = float(values.std(ddof=1))
            ranks = group.groupby("window")[metric].mean()
            row[f"{metric}_window_range"] = float(ranks.max() - ranks.min())
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(RESULTS / "multi_window_three_seed_aggregate.csv", index=False)

    rank_rows = []
    for window, group in summary.groupby("window"):
        for metric in METRICS:
            ordered = group.sort_values(f"{metric}_mean")
            for rank, (_, item) in enumerate(ordered.iterrows(), start=1):
                rank_rows.append({"window": window, "metric": metric, "rank": rank, "model": item["model"], "value": item[f"{metric}_mean"]})
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(RESULTS / "multi_window_rank_stability.csv", index=False)
    _write_table(summary)
    _write_aggregate_table(aggregate, ranks)
    _plot(summary)


def _short(name: str) -> str:
    return name.replace("Proposed PI-GNN-GRU-v3 ", "PI-GNN-v3 ").replace("_mode", "")


def _write_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Three-seed repeated training across three prespecified chronological operating windows. Values are mean $\pm$ sample standard deviation. All distributed targets are calibrated-simulator or simulator-assisted quantities; the audit measures optimization and window sensitivity, not seasonal field validation.}",
        r"\label{tab:multi_window_three_seed}", r"\small",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lllrrrrr}", r"\toprule",
        r"Window & Regime & Model & Supply RMSE ($^\circ$C) & Return RMSE ($^\circ$C) & Heat-loss error (\%) & Energy residual (\%) & Source-boundary residual ($^\circ$C) \\", r"\midrule",
    ]
    for _, row in summary.sort_values(["window", "model"]).iterrows():
        def cell(metric: str) -> str:
            return f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}"
        lines.append(
            f"{row['window'].replace('_', ' ')} & {row['regime']} & {_short(row['model'])} & {cell('RMSE_Ts_full')} & {cell('RMSE_Tr_full')} & {cell('heat_loss_error_percent')} & {cell('energy_balance_residual')} & {cell('boundary_residual_mean')}" + " \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_multi_window_three_seed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aggregate_table(aggregate: pd.DataFrame, ranks: pd.DataFrame) -> None:
    wins = (
        ranks[ranks["rank"].eq(1)]
        .groupby("model", as_index=False)
        .size()
        .rename(columns={"size": "window_metric_wins"})
    )
    display = aggregate.merge(wins, on="model", how="left").fillna({"window_metric_wins": 0})
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Aggregate three-window, three-seed stability audit. Values are mean $\pm$ sample standard deviation over nine independent runs (three seeds in each of three windows). A win is the lowest window-mean value for one of five audited metrics; wins are descriptive and do not establish seasonal field generalization.}",
        r"\label{tab:multi_window_three_seed_aggregate}", r"\small",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Model & Supply RMSE ($^\circ$C) & Return RMSE ($^\circ$C) & Heat-loss error (\%) & Energy residual (\%) & Source-boundary residual ($^\circ$C) & Wins/15 \\", r"\midrule",
    ]
    for _, row in display.sort_values("model").iterrows():
        cell = lambda metric: f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_between_run_std']:.3f}"
        lines.append(
            f"{_short(row['model'])} & {cell('RMSE_Ts_full')} & {cell('RMSE_Tr_full')} & "
            f"{cell('heat_loss_error_percent')} & {cell('energy_balance_residual')} & {cell('boundary_residual_mean')} & "
            f"{int(row['window_metric_wins'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_multi_window_three_seed_aggregate.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = list(summary["model"].drop_duplicates())
    windows = list(summary["window"].drop_duplicates())
    colors = ["#0000E6", "#555555", "#FF6626", "#55D600"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "RMSE_Ts_full", "Supply RMSE (°C)"),
        (axes[1], "RMSE_Tr_full", "Return RMSE (°C)"),
        (axes[2], "energy_balance_residual", "Energy residual (%)"),
    ]:
        x = np.arange(len(windows))
        width = 0.19
        for j, model in enumerate(models):
            values, errors = [], []
            for window in windows:
                row = summary[(summary["window"] == window) & (summary["model"] == model)].iloc[0]
                values.append(row[f"{metric}_mean"])
                errors.append(row[f"{metric}_std"])
            ax.bar(x + (j - 1.5) * width, values, width, yerr=errors, color=colors[j], edgecolor="#111111", linewidth=0.6, capsize=2)
        ax.set_xticks(x, [w.replace("_", "\n") for w in windows], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(1.0)
    axes[0].set_title("(a) Supply reconstruction")
    axes[1].set_title("(b) Return reconstruction")
    axes[2].set_title("(c) Physical consistency")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor="#111111") for i in range(len(models))]
    fig.legend(handles, [_short(name) for name in models], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10), fontsize=8)
    fig.savefig(FIGURES / "fig_multi_window_three_seed.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_multi_window_three_seed.svg", format="svg", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_multi_window_three_seed.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_csv(PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv")
    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True)
    params = load_json(RESULTS / "calibrated_parameters.json")
    base_config = apply_calibrated_params_to_config(load_config(), params)
    epochs = int(os.environ.get("MULTI_WINDOW_EPOCHS", "12"))
    batch_size = int(os.environ.get("MULTI_WINDOW_BATCH_SIZE", "32"))
    base_config["training"]["batch_size"] = batch_size
    existing = pd.read_csv(RAW_PATH) if RAW_PATH.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    complete = set()
    if not existing.empty:
        complete = set(zip(existing["window"], existing["seed"].astype(int), existing["model"]))

    protocol_windows = []
    for window_spec in WINDOWS:
        frame = _frame_for_window(canonical, window_spec)
        protocol_windows.append({
            **window_spec,
            "end": str(frame["timestamp"].iloc[-1]),
            "timestamps": len(frame),
            "mean_heat_load_kW": float(frame["heat_load_kw"].mean()),
            "mean_supply_C": float(frame["supply_temp_C"].mean()),
            "mean_return_C": float(frame["return_temp_C"].mean()),
        })
        sim = simulate_thermo_hydraulics(build_boundary_conditions(frame, base_config), base_config, params=params)
        sensors = apply_sensor_layout(sim, "S4_five_sensors", base_config)
        loaders = build_loaders(sim, sensors, base_config)
        for seed in SEEDS:
            for spec in CORE_MODELS:
                key = (window_spec["window"], seed, spec["label"])
                if key in complete:
                    continue
                set_seed(seed)
                prefix = f"mw_{window_spec['window']}_seed_{seed}_"
                metrics, _, _ = train_and_evaluate_specs(
                    [copy.deepcopy(spec)], loaders, base_config, quick=False,
                    output_prefix=prefix, epochs_override=epochs, write_secondary_tables=False,
                )
                row = metrics.iloc[0].to_dict()
                row.update({
                    "window": window_spec["window"], "regime": window_spec["regime"], "seed": seed,
                    "window_start_utc": str(frame["timestamp"].iloc[0]), "window_end_utc": str(frame["timestamp"].iloc[-1]),
                    "window_timestamps": WINDOW_LENGTH, "epochs_cap": epochs, "batch_size": batch_size,
                    "sensor_layout": "S4_five_sensors", "normalization": "training_partition_only",
                    "state_type": "calibrated-simulator/simulator-assisted benchmark targets",
                })
                rows.append(row)
                pd.DataFrame(rows).to_csv(RAW_PATH, index=False)
                complete.add(key)

    raw = pd.DataFrame(rows)
    expected = len(WINDOWS) * len(SEEDS) * len(CORE_MODELS)
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "windows": protocol_windows,
        "window_length": WINDOW_LENGTH,
        "seeds": SEEDS,
        "models": [item["label"] for item in CORE_MODELS],
        "epochs_cap": epochs,
        "batch_size": batch_size,
        "sensor_layout": "S4_five_sensors",
        "expected_runs": expected,
        "completed_runs": len(raw),
        "complete": len(raw) == expected,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "scope": "three separated heating-season windows; no summer block of sufficient contiguous length exists in the locked retained series",
        "window_selection_rule": "chosen before model fitting from timestamp, continuity, and temporal separation only: one early-winter block, one heating-onset block, and one later-year winter block; no model error or rank entered selection",
    }
    (RESULTS / "multi_window_three_seed_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    if len(raw) != expected:
        raise RuntimeError(f"Multi-window campaign incomplete: {len(raw)} of {expected} runs.")
    _save_summary(raw)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
