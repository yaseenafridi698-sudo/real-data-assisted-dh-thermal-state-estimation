from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, load_config
from src.proxy_causality_audit import FULL_STATE_CAUSALITY_VERSION


CORE_MODELS = [
    {"label": "GRU-MSE", "model": "gru", "loss_mode": "mse"},
    {"label": "Transformer-MSE", "model": "transformer", "loss_mode": "mse"},
    {
        "label": "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "model": "pignn_v3",
        "loss_mode": "physics",
        "loss_weights": {"training_mode": "accuracy_mode"},
    },
    {
        "label": "Proposed PI-GNN-GRU-v3 balanced_mode",
        "model": "pignn_v3",
        "loss_mode": "physics",
        "loss_weights": {"training_mode": "balanced_mode"},
    },
]

METRICS = [
    "RMSE_Ts_full",
    "RMSE_Tr_full",
    "heat_loss_error_percent",
    "energy_balance_residual",
    "boundary_residual_mean",
    "training_time_s",
]


def _latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
        "+/-": r"$\pm$",
    }
    value = value.replace("+/-", "±")
    return "".join(replacements.get(ch, ch) for ch in value).replace("±", r"$\pm$")


def _write_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(df.columns) + r"}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row[c]) for c in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_status_only(reason: str) -> None:
    results = PROJECT_ROOT / "results"
    tables = PROJECT_ROOT / "paper" / "tables"
    results.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    seeds = [11, 22, 33, 44, 55]
    for model in CORE_MODELS:
        plan_rows.append(
            {
                "model": model["label"],
                "planned_seeds": ", ".join(str(s) for s in seeds),
                "planned_metrics": ", ".join(METRICS),
                "status": "not_run_runtime_missing",
                "note": reason,
            }
        )
    pd.DataFrame(plan_rows).to_csv(results / "repeated_seed_training_plan.csv", index=False)

    rows = []
    for model in CORE_MODELS:
        for metric in METRICS:
            rows.append(
                {
                    "model": model["label"],
                    "metric": metric,
                    "mean": "",
                    "std": "",
                    "n_seeds": 0,
                    "status": "not_run",
                    "interpretation": reason,
                }
            )
    stats = pd.DataFrame(rows)
    stats.to_csv(results / "repeated_seed_statistics.csv", index=False)

    single = pd.read_csv(results / "baseline_comparison_final.csv") if (results / "baseline_comparison_final.csv").exists() else pd.DataFrame()
    cost = pd.read_csv(results / "computational_cost.csv") if (results / "computational_cost.csv").exists() else pd.DataFrame()
    if not single.empty:
        single_ref = single[single["model"].isin([m["label"] for m in CORE_MODELS])].copy()
        if not cost.empty and "training_time_s" in cost.columns:
            single_ref = single_ref.merge(cost[["model", "training_time_s"]], on="model", how="left")
        keep = [c for c in ["model"] + METRICS if c in single_ref.columns]
        single_ref[keep].to_csv(results / "repeated_seed_single_run_reference.csv", index=False)

    status = pd.DataFrame(
        [
            {
                "Item": "five-seed mean +/- std",
                "Status": "not available in this runtime",
                "Interpretation": reason,
            },
            {
                "Item": "single-run reference",
                "Status": "available",
                "Interpretation": "Use only as final benchmark point estimates, not statistical repeatability evidence.",
            },
            {
                "Item": "required action",
                "Status": "run src/repeated_seed_statistics.py after installing torch",
                "Interpretation": "The script will generate seed-indexed metrics and mean +/- std tables when Torch is available.",
            },
        ]
    )
    status.to_csv(results / "repeated_seed_statistics_status.csv", index=False)
    _write_table(
        status,
        tables / "table_repeated_seed_statistics.tex",
        "Repeated-seed training statistics status. True mean +/- standard deviation is not claimed unless seed-indexed reruns are generated in a Torch-enabled environment.",
        "tab:repeated_seed_statistics",
    )


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("model"):
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": round(float(values.mean()), 5),
                    "std": round(float(values.std(ddof=1)), 5) if len(values) > 1 else 0.0,
                    "n_seeds": int(len(values)),
                    "status": "completed",
                    "interpretation": "Mean +/- std over repeated random seeds; lower is better for listed errors/residuals.",
                }
            )
    return pd.DataFrame(rows)


def _format_mean_std(summary: pd.DataFrame, model: str, metric: str) -> str:
    row = summary[(summary["model"] == model) & (summary["metric"] == metric)]
    if row.empty:
        return "not available"
    return f"{float(row.iloc[0]['mean']):.3f} +/- {float(row.iloc[0]['std']):.3f}"


def _write_compact_outputs(summary: pd.DataFrame, raw: pd.DataFrame) -> None:
    short_names = {
        "GRU-MSE": "GRU-MSE",
        "Transformer-MSE": "Transformer-MSE",
        "Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-GRU-v3 accuracy",
        "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-GRU-v3 balanced",
    }
    rows = []
    for model in short_names:
        rows.append(
            {
                "Model": short_names[model],
                "Supply RMSE (deg C)": _format_mean_std(summary, model, "RMSE_Ts_full"),
                "Return RMSE (deg C)": _format_mean_std(summary, model, "RMSE_Tr_full"),
                "Heat-loss error (%)": _format_mean_std(summary, model, "heat_loss_error_percent"),
                "Energy residual (%)": _format_mean_std(summary, model, "energy_balance_residual"),
                "Boundary residual": _format_mean_std(summary, model, "boundary_residual_mean"),
            }
        )
    compact = pd.DataFrame(rows)
    compact.to_csv(PROJECT_ROOT / "results" / "repeated_seed_statistics_compact.csv", index=False)
    _write_table(
        compact,
        PROJECT_ROOT / "paper" / "tables" / "table_repeated_seed_statistics.tex",
        "Five-seed confirmation for the four principal neural estimators (mean +/- sample standard deviation; seeds 11, 22, 33, 44, and 55). Distributed states and consistency metrics are evaluated against calibrated-simulator outputs, not dense field measurements.",
        "tab:repeated_seed_statistics",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#0000E6", "#555555", "#FF6626", "#55D600"]
    metrics = ["RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"]
    labels = ["Supply RMSE", "Return RMSE", "Heat-loss error", "Energy residual", "Boundary residual"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.2, 3.2), constrained_layout=True)
    for ax, metric, label in zip(axes, metrics, labels):
        means = []
        stds = []
        for model in short_names:
            row = summary[(summary["model"] == model) & (summary["metric"] == metric)].iloc[0]
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
        x = np.arange(len(means))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="#111111", linewidth=0.8, capsize=2.5)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#111111") for c in colors]
    fig.legend(handles, list(short_names.values()), loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.08))
    fig_dir = PROJECT_ROOT / "figures" / "final"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "fig_repeated_seed_stability.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_repeated_seed_stability.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _run_torch_enabled() -> None:
    import numpy as np
    import torch

    from src.data_loaders import load_dataset_by_name
    from src.data_preprocessing import preprocess_dataset
    from src.effective_physics import apply_calibrated_params_to_config
    from src.real_data_mapper import build_boundary_conditions
    from src.sensor_layouts import apply_sensor_layout
    from src.study_workflow import build_loaders, train_and_evaluate_specs
    from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
    from src.utils import load_json, set_seed

    config = load_config()
    seeds = [int(s) for s in os.environ.get("REPEATED_SEEDS", "11,22,33,44,55").split(",") if s.strip()]
    epochs_override = os.environ.get("REPEATED_SEED_EPOCHS")
    epochs = int(epochs_override) if epochs_override else None
    batch_override = os.environ.get("REPEATED_SEED_BATCH_SIZE")
    if batch_override:
        config["training"]["batch_size"] = int(batch_override)
    results = PROJECT_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    causality_path = results / "proxy_causality_audit.csv"
    if not causality_path.exists():
        raise FileNotFoundError("Full-state causality audit is required before repeated training.")
    causality = pd.read_csv(causality_path)
    required_state_audits = {
        "supply_temperature_state_uses_no_future_values",
        "return_temperature_state_uses_no_future_values",
        "head_state_uses_no_future_values",
        "flow_state_uses_no_future_values",
    }
    passed = set(causality.loc[causality["status"].eq("pass"), "audit"].astype(str))
    if not required_state_audits.issubset(passed):
        raise RuntimeError("Repeated training refused because full-state causality has not passed.")
    canonical = json.loads((results / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))

    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    protocol = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "epochs": epochs if epochs is not None else int(config["training"]["epochs_full"]),
        "batch_size": int(config["training"]["batch_size"]),
        "sensor_layout": "S4_five_sensors",
        "maximum_time_steps": max_steps,
        "models": [model["label"] for model in CORE_MODELS],
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "status": "smoke_test" if epochs is not None and epochs <= 1 else "publication_confirmation_run",
        "canonical_dataset_path": canonical["canonical_file"],
        "canonical_dataset_sha256": canonical["sha256"],
        "full_state_causality_version": FULL_STATE_CAUSALITY_VERSION,
        "simulator_state_artifact": "results/corrected_simulator_states.npz",
        "selection_metric": str(config["training"].get("selection_metric", "normalized_state_mse")),
        "early_stopping_patience": int(config["training"].get("early_stopping_patience", 15)),
        "note": "Repeated-seed confirmation uses the same capped real-data window, causal proxy construction, fixed chronological split, training-only normalization, calibrated effective parameters, S4 layout, and common normalized-state-MSE checkpoint selection as the primary benchmark.",
    }
    (results / "repeated_seed_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    df = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config)
    df = df.head(min(len(df), max_steps)).copy()
    boundary = build_boundary_conditions(df, config)
    params_path = results / "calibrated_parameters.json"
    params = load_json(params_path) if params_path.exists() else {}
    config = apply_calibrated_params_to_config(config, params)
    protocol["calibrated_heat_loss_U_W_m2K"] = float(config["system"]["heat_loss_U_W_m2K"])
    protocol["calibrated_friction_factor"] = float(config["system"]["friction_factor"])
    (results / "repeated_seed_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    sim = simulate_thermo_hydraulics(boundary, config, params=params)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config)

    raw_path = results / "repeated_seed_raw_metrics.csv"
    append_mode = os.environ.get("REPEATED_SEED_APPEND", "0") == "1"
    raw_rows = []
    if append_mode and raw_path.exists():
        existing = pd.read_csv(raw_path)
        # Replace only the requested seed/model rows. This makes individual
        # CPU-bounded seed calls resumable without mixing duplicate metrics.
        existing = existing[~existing.get("seed", pd.Series(dtype=int)).isin(seeds)].copy()
        if not existing.empty:
            raw_rows.append(existing)
    for seed in seeds:
        set_seed(seed)
        config["dataset"]["seed"] = seed
        loaders = build_loaders(sim, sensors, config)
        metrics, _, _ = train_and_evaluate_specs(
            CORE_MODELS,
            loaders,
            config,
            quick=False,
            output_prefix=f"seed_{seed}_",
            epochs_override=epochs,
            write_secondary_tables=False,
        )
        metrics["seed"] = seed
        raw_rows.append(metrics)
        raw = pd.concat(raw_rows, ignore_index=True)
        raw.to_csv(raw_path, index=False)
        summary = _summarize(raw)
        summary.to_csv(results / "repeated_seed_statistics.csv", index=False)
        _write_compact_outputs(summary, raw)
        # In resumable mode the requested seed is only a new increment. Record
        # the full set actually represented by the aggregate statistics.
        protocol["requested_seeds_this_call"] = seeds
        protocol["executed_seeds_in_aggregate"] = sorted(
            int(seed_value) for seed_value in pd.to_numeric(raw["seed"], errors="coerce").dropna().unique()
        )
        protocol["seeds"] = protocol["executed_seeds_in_aggregate"]
        protocol["aggregate_model_seed_rows"] = int(len(raw))
        protocol["aggregate_complete_for_five_seed_protocol"] = (
            set(protocol["executed_seeds_in_aggregate"]) == {11, 22, 33, 44, 55}
        )
        (results / "repeated_seed_protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        _write_status_only("Torch is not installed in this execution environment, so repeated training seeds were not run here.")
        print(PROJECT_ROOT / "results" / "repeated_seed_statistics_status.csv")
        return
    if os.environ.get("REPEATED_SEED_AGGREGATE_ONLY") == "1":
        raw_path = PROJECT_ROOT / "results" / "repeated_seed_raw_metrics.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Cannot aggregate because {raw_path} does not exist.")
        raw = pd.read_csv(raw_path)
        summary = _summarize(raw)
        summary.to_csv(PROJECT_ROOT / "results" / "repeated_seed_statistics.csv", index=False)
        _write_compact_outputs(summary, raw)
        protocol_path = PROJECT_ROOT / "results" / "repeated_seed_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
        protocol["executed_seeds_in_aggregate"] = sorted(
            int(seed_value) for seed_value in pd.to_numeric(raw["seed"], errors="coerce").dropna().unique()
        )
        protocol["seeds"] = protocol["executed_seeds_in_aggregate"]
        protocol["aggregate_model_seed_rows"] = int(len(raw))
        protocol["aggregate_complete_for_five_seed_protocol"] = (
            set(protocol["executed_seeds_in_aggregate"]) == {11, 22, 33, 44, 55}
        )
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    else:
        _run_torch_enabled()
    print(PROJECT_ROOT / "results" / "repeated_seed_statistics.csv")


if __name__ == "__main__":
    main()
