from __future__ import annotations

from typing import Any
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import calibrate_simulator
from .config import PROJECT_ROOT
from .data_download import write_data_availability_report
from .data_loaders import load_dataset_by_name
from .data_preprocessing import preprocess_dataset
from .data_registry import DataRegistry, check_dataset_available
from .dataset import StateWindowDataset, build_state_arrays, contiguous_window_starts, split_window_indices
from .evaluate import evaluate_model, save_evaluation_outputs
from .effective_physics import apply_calibrated_params_to_config
from .graph_utils import build_line_graph_adjacency, normalized_adjacency
from .make_paper_assets import make_paper_assets
from .models import InterpolationBaseline, build_model
from .plots import fig10_external_validation, fig12_xai4heat_sparse_substations, generate_all_figures
from .real_data_mapper import build_boundary_conditions, build_sparse_substation_measurements
from .sensor_layouts import apply_sensor_layout, sensor_nodes_for_layout
from .thermo_hydraulic_simulator import (
    run_discretization_study,
    save_model_verification,
    simulate_thermo_hydraulics,
)
from .train import train_model
from .utils import ensure_dir, set_seed, synthetic_realistic_operating_data


BASELINE_SPECS = [
    {"label": "Interpolation", "model": "interpolation", "loss_mode": "none"},
    {"label": "LSTM-MSE", "model": "lstm", "loss_mode": "mse"},
    {"label": "GRU-MSE", "model": "gru", "loss_mode": "mse"},
    {"label": "Transformer-MSE", "model": "transformer", "loss_mode": "mse"},
    {"label": "PureGNN-MSE", "model": "pure_gnn", "loss_mode": "mse"},
    {"label": "PI-LSTM", "model": "pi_lstm", "loss_mode": "physics"},
    {"label": "PI-GNN-no-temporal", "model": "pignn_no_temporal", "loss_mode": "physics"},
    {"label": "Proposed PI-GNN-GRU", "model": "pignn", "loss_mode": "physics"},
    {"label": "Proposed PI-GNN-GRU improved", "model": "pignn_improved", "loss_mode": "physics"},
    {"label": "Proposed PI-GNN-GRU-v2", "model": "pignn_v2", "loss_mode": "physics"},
    {"label": "Proposed PI-GNN-GRU-v3 accuracy_mode", "model": "pignn_v3", "loss_mode": "physics", "loss_weights": {"training_mode": "accuracy_mode"}},
    {"label": "Proposed PI-GNN-GRU-v3 balanced_mode", "model": "pignn_v3", "loss_mode": "physics", "loss_weights": {"training_mode": "balanced_mode"}},
    {"label": "Proposed PI-GNN-GRU-v3 physics_mode", "model": "pignn_v3", "loss_mode": "physics", "loss_weights": {"training_mode": "physics_mode"}},
]

CAUSAL_PRIMARY_LABELS = {
    "GRU-MSE",
    "Transformer-MSE",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
}

ABLATION_SPECS = [
    ("full_physics", "pignn_v3", {"training_mode": "balanced_mode"}),
    ("no_thermal_residual", "pignn_v3", {"training_mode": "balanced_mode", "lambda_thermal": 0.0}),
    ("no_hydraulic_residual", "pignn_v3", {"training_mode": "balanced_mode", "lambda_hydraulic": 0.0}),
    ("no_boundary_loss", "pignn_v3", {"training_mode": "balanced_mode", "lambda_boundary": 0.0}),
    ("no_energy_residual", "pignn_v3", {"training_mode": "balanced_mode", "lambda_energy": 0.0}),
    ("no_sensor_consistency_loss", "pignn_v3", {"training_mode": "balanced_mode", "lambda_sensor": 0.0}),
    ("no_smoothness_loss", "pignn_v3", {"training_mode": "balanced_mode", "lambda_smooth": 0.0}),
    ("no_graph_topology", "pignn_v3_no_graph", {"training_mode": "balanced_mode"}),
    ("no_temporal_gru", "pignn_v3_no_temporal", {"training_mode": "balanced_mode"}),
    ("no_interpolation_residual_connection", "pignn_v3_no_interp", {"training_mode": "balanced_mode"}),
]

LAYOUTS_REQUIRED = [
    "S1_inlet_only",
    "S2_inlet_outlet",
    "S3_inlet_middle_outlet",
    "S4_five_sensors",
    "S5_noisy_inlet_outlet",
    "S6_dropout_five_sensors",
    "S7_xai4heat_substations",
    "S8_random_three_sensors",
    "S9_optimized_three_sensors",
    "S10_optimized_five_sensors",
    "S11_middle_only",
    "S12_inlet_two_middle_outlet",
    "S13_noisy_inlet_only",
    "S14_outlet_only",
    "S15_noisy_inlet_outlet_5pct",
    "S16_peak_dropout_five_sensors",
    "S17_optimized_two_sensors",
]


def preprocess_available_datasets(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    registry = DataRegistry()
    processed: dict[str, pd.DataFrame] = {}
    notes: dict[str, str] = {}
    statuses: dict[str, str] = {}
    real_cfg = config.get("real_data", {})
    optional_demand = real_cfg.get("optional_demand_dataset", "aalborg")
    primary = real_cfg.get("primary_dataset", "sonderborg")
    frozen_rel = real_cfg.get("canonical_sonderborg_processed_path")
    frozen_path = PROJECT_ROOT / str(frozen_rel) if frozen_rel else None
    for name in registry.names():
        if name == optional_demand and not real_cfg.get("use_optional_demand_dataset", False):
            notes[name] = "available but optional; not processed in core real-data study"
            statuses[name] = "available_optional_not_processed" if check_dataset_available(name) else "manual_required"
            continue
        try:
            if name == primary == "sonderborg" and real_cfg.get("freeze_canonical_processed", False) and frozen_path and frozen_path.exists():
                processed[name] = preprocess_dataset(pd.DataFrame(), name, config)
                notes[name] = "checksum-pinned canonical processed input"
                statuses[name] = "canonical_processed_available"
                continue
            local_processed = PROJECT_ROOT / "data" / "processed" / f"{name}_processed.csv"
            if local_processed.exists():
                df = pd.read_csv(local_processed)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
                processed[name] = df.sort_values("timestamp").reset_index(drop=True)
                notes[name] = "local processed input"
                statuses[name] = "processed_available_local"
                continue
            if not check_dataset_available(name):
                notes[name] = "manual download required"
                statuses[name] = "manual_required"
                continue
            raw = load_dataset_by_name(name)
            processed[name] = preprocess_dataset(raw, name, config)
            notes[name] = "processed from local raw input"
            statuses[name] = "available_local"
        except Exception as exc:
            notes[name] = f"load/preprocess failed: {exc}"
            statuses[name] = "preprocess_failed"
            print(f"{name}: {exc}")
    write_data_availability_report(notes, statuses)
    return processed


def load_primary_or_demo(config: dict[str, Any], allow_fallback: bool) -> tuple[pd.DataFrame | None, bool]:
    primary = config["real_data"]["primary_dataset"]
    real_cfg = config.get("real_data", {})
    frozen_rel = real_cfg.get("canonical_sonderborg_processed_path")
    frozen_path = PROJECT_ROOT / str(frozen_rel) if frozen_rel else None
    if primary == "sonderborg" and real_cfg.get("freeze_canonical_processed", False) and frozen_path and frozen_path.exists():
        return preprocess_dataset(pd.DataFrame(), primary, config), False
    if check_dataset_available(primary):
        raw = load_dataset_by_name(primary)
        return preprocess_dataset(raw, primary, config), False
    if allow_fallback:
        df = synthetic_realistic_operating_data(config, n_steps=max(220, config["model"]["window_steps"] * 16))
        return preprocess_dataset(df, "fallback_synthetic_realistic", config, save=False), True
    warning = PROJECT_ROOT / "results" / "DATA_WARNING_REAL_DATA_NOT_FOUND.txt"
    ensure_dir(warning.parent)
    warning.write_text(
        "Sonderborg primary real data were not found. Journal results were not generated. "
        "Place raw files in data/raw/sonderborg/ and rerun run_real_data_study.py.\n",
        encoding="utf-8",
    )
    (PROJECT_ROOT / "results" / "REAL_DATA_REQUIRED_FOR_JOURNAL_STUDY.txt").write_text(
        "Real Sonderborg data are required for the journal-study workflow. "
        "Fallback synthetic data are restricted to the quick software demo.\n",
        encoding="utf-8",
    )
    return None, False


def build_loaders(sim: dict[str, Any], sensors: dict[str, Any], config: dict[str, Any], stats: dict[str, Any] | None = None):
    arrays = build_state_arrays(sim, sensors, config)
    valid_starts = contiguous_window_starts(arrays["trajectory_start"], int(config["model"]["window_steps"]))
    train_idx, val_idx, test_idx = split_window_indices(
        n_steps=arrays["target"].shape[0],
        window_steps=config["model"]["window_steps"],
        train_fraction=config["dataset"]["train_fraction"],
        val_fraction=config["dataset"]["val_fraction"],
        embargo_steps=config["dataset"].get("embargo_steps", int(config["model"]["window_steps"]) - 1),
        valid_window_starts=valid_starts,
    )
    train_ds = StateWindowDataset(arrays, train_idx, config, fit_stats=True) if stats is None else StateWindowDataset(arrays, train_idx, config, stats=stats)
    val_ds = StateWindowDataset(arrays, val_idx, config, stats=train_ds.stats)
    test_ds = StateWindowDataset(arrays, test_idx, config, stats=train_ds.stats)
    batch_size = config["training"]["batch_size"]
    return {
        "arrays": arrays,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        "valid_window_starts": valid_starts,
    }


def _a_norm_for(arrays: dict[str, np.ndarray]) -> torch.Tensor:
    n_nodes = arrays["target"].shape[1]
    return torch.tensor(normalized_adjacency(build_line_graph_adjacency(n_nodes)), dtype=torch.float32)


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _write_secondary_result_tables(metrics_df: pd.DataFrame, output_dir: Path, training_rows: list[dict[str, Any]]) -> None:
    physics_cols = [
        "model",
        "loss_mode",
        "heat_loss_error_percent",
        "energy_balance_residual",
        "thermal_residual_mean",
        "hydraulic_residual_mean",
        "boundary_residual_mean",
        "heat_load_consistency_error_percent",
    ]
    present_physics = [c for c in physics_cols if c in metrics_df.columns]
    if present_physics:
        metrics_df[present_physics].to_csv(output_dir / "physics_consistency_comparison.csv", index=False)
        metrics_df[present_physics].to_csv(output_dir / "physics_consistency_comparison_improved.csv", index=False)
    heat_cols = [c for c in ["model", "RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "thermal_residual_mean"] if c in metrics_df.columns]
    if heat_cols:
        metrics_df[heat_cols].sort_values("heat_loss_error_percent").to_csv(output_dir / "heat_loss_comparison.csv", index=False)
    ranking_rows = []
    summary_rows = []
    proposed_names = [
        "Proposed PI-GNN-GRU-v3 balanced_mode",
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 physics_mode",
        "Proposed PI-GNN-GRU-v2",
        "Proposed PI-GNN-GRU improved",
        "Proposed PI-GNN-GRU",
    ]
    for metric in ["RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "thermal_residual_mean", "boundary_residual_mean"]:
        if metric in metrics_df.columns:
            ranking = metrics_df[["model", metric]].copy()
            ranking[metric] = pd.to_numeric(ranking[metric], errors="coerce")
            ranking = ranking.dropna().sort_values(metric).reset_index(drop=True)
            for rank, (_, row) in enumerate(ranking.iterrows(), start=1):
                ranking_rows.append({"metric": metric, "rank": rank, "model": row["model"], "value": row[metric]})
            proposed_row = ranking[ranking["model"].astype(str).isin(proposed_names)].head(1)
            v2_row = ranking[ranking["model"].astype(str).eq("Proposed PI-GNN-GRU-v2")].head(1)
            v3_rows = ranking[ranking["model"].astype(str).str.contains("PI-GNN-GRU-v3", regex=False)]
            v3_row = v3_rows.head(1)
            v3_balanced = ranking[ranking["model"].astype(str).eq("Proposed PI-GNN-GRU-v3 balanced_mode")].head(1)
            v3_improves_v2 = False
            if not v3_row.empty and not v2_row.empty:
                v3_improves_v2 = bool(float(v3_row.iloc[0][metric]) < float(v2_row.iloc[0][metric]))
            summary_rows.append(
                {
                    "metric": metric,
                    "best_model": ranking.iloc[0]["model"] if not ranking.empty else "",
                    "best_value": ranking.iloc[0][metric] if not ranking.empty else np.nan,
                    "proposed_model": proposed_row.iloc[0]["model"] if not proposed_row.empty else "not_available",
                    "proposed_rank": int(proposed_row.index[0]) + 1 if not proposed_row.empty else np.nan,
                    "proposed_value": proposed_row.iloc[0][metric] if not proposed_row.empty else np.nan,
                    "pignn_gru_v2_rank": int(v2_row.index[0]) + 1 if not v2_row.empty else np.nan,
                    "pignn_gru_v2_value": v2_row.iloc[0][metric] if not v2_row.empty else np.nan,
                    "pignn_gru_v3_best_model": v3_row.iloc[0]["model"] if not v3_row.empty else "not_available",
                    "pignn_gru_v3_best_rank": int(v3_row.index[0]) + 1 if not v3_row.empty else np.nan,
                    "pignn_gru_v3_best_value": v3_row.iloc[0][metric] if not v3_row.empty else np.nan,
                    "pignn_gru_v3_balanced_rank": int(v3_balanced.index[0]) + 1 if not v3_balanced.empty else np.nan,
                    "pignn_gru_v3_balanced_value": v3_balanced.iloc[0][metric] if not v3_balanced.empty else np.nan,
                    "v3_improves_over_v2": v3_improves_v2,
                    "v3_best_in_metric": bool((not v3_row.empty) and int(v3_row.index[0]) == 0),
                    "interpretation": "lower is better; report GRU/Transformer wins honestly when proposed_rank is not 1",
                }
            )
    pd.DataFrame(ranking_rows).to_csv(output_dir / "model_ranking_by_metric.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "model_ranking_summary.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(output_dir / "model_ranking_by_metric_final.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "model_ranking_summary_final.csv", index=False)
    cost = metrics_df[[c for c in ["model", "inference_time_ms"] if c in metrics_df.columns]].copy()
    if training_rows:
        cost = cost.merge(pd.DataFrame(training_rows), on="model", how="left")
    cost.to_csv(output_dir / "computational_cost.csv", index=False)
    metrics_df.to_csv(output_dir / "baseline_comparison_final.csv", index=False)
    if present_physics:
        metrics_df[present_physics].to_csv(output_dir / "physics_consistency_comparison_final.csv", index=False)
    if training_rows:
        training_df = pd.DataFrame(training_rows)
        training_df.to_csv(output_dir / "training_stability_summary_final.csv", index=False)
        v3_training = training_df[training_df["model"].astype(str).str.contains("PI-GNN-GRU-v3", regex=False)]
        if not v3_training.empty:
            v3_training.to_csv(output_dir / "proposed_v3_training_summary.csv", index=False)


def write_final_package_file_audit() -> Path:
    rows = []
    for path in sorted((PROJECT_ROOT / "results").glob("*")):
        if path.is_file():
            name = path.name
            status = "included"
            note = "final result artifact"
            if "QUICK_DEMO" in name or "FALLBACK" in name or name.startswith("DATA_WARNING_REAL_DATA_NOT_FOUND"):
                status = "demo_or_stale_warning"
                note = "should not be cited as journal evidence"
            if name.startswith("XAI4HEAT_NOT"):
                status = "limitation_marker"
                note = "documents unavailable XAI4HEAT local data"
            rows.append({"file_path": str(path.relative_to(PROJECT_ROOT)), "status": status, "note": note})
    out = PROJECT_ROOT / "results" / "final_package_file_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def train_and_evaluate_specs(
    specs: list[dict[str, Any]],
    loaders: dict[str, Any],
    config: dict[str, Any],
    quick: bool,
    output_prefix: str = "",
    epochs_override: int | None = None,
    write_secondary_tables: bool = True,
) -> tuple[pd.DataFrame, dict[str, torch.nn.Module], dict[str, dict[str, np.ndarray]]]:
    arrays = loaders["arrays"]
    n_nodes = arrays["target"].shape[1]
    a_norm = _a_norm_for(arrays)
    metrics = []
    training_rows: list[dict[str, Any]] = []
    trained: dict[str, torch.nn.Module] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for spec in specs:
        label = spec["label"]
        model_name = spec["model"]
        if model_name == "interpolation":
            model = InterpolationBaseline(n_nodes=n_nodes)
            result, pred = evaluate_model(model, loaders["test_loader"], config, loaders["train_ds"].stats, label, return_predictions=True)
            training_rows.append({"model": label, "training_time_s": 0.0, "parameter_count": 0})
        else:
            model = build_model(model_name, input_dim=arrays["features"].shape[-1], n_nodes=n_nodes, a_norm=a_norm, config=config)
            start_train = time.perf_counter()
            best_model, history = train_model(
                model,
                f"{output_prefix}{label}",
                loaders["train_loader"],
                loaders["val_loader"],
                config,
                loaders["train_ds"].stats,
                output_dir=PROJECT_ROOT / "results",
                quick=quick,
                loss_mode=spec.get("loss_mode", "physics"),
                loss_weights=spec.get("loss_weights"),
                epochs_override=epochs_override,
            )
            training_time_s = time.perf_counter() - start_train
            trained[label] = best_model
            result, pred = evaluate_model(best_model, loaders["test_loader"], config, loaders["train_ds"].stats, label, return_predictions=True)
            best_history = history.loc[history["selection_loss"].idxmin()] if not history.empty and "selection_loss" in history else pd.Series()
            training_rows.append(
                {
                    "model": label,
                    "training_time_s": training_time_s,
                    "parameter_count": _parameter_count(best_model),
                    "epochs_ran": int(len(history)),
                    "best_epoch": int(best_history.get("epoch", len(history))),
                    "best_selection_loss": float(best_history.get("selection_loss", np.nan)),
                    "selection_metric": str(best_history.get("selection_metric", "normalized_state_mse")),
                }
            )
        result["loss_mode"] = spec.get("loss_mode", "none")
        result["architecture"] = model_name
        result["training_time_s"] = float(training_rows[-1]["training_time_s"])
        result["parameter_count"] = int(training_rows[-1]["parameter_count"])
        metrics.append(result)
        predictions[label] = pred
    metrics_df = pd.DataFrame(metrics)
    if write_secondary_tables:
        _write_secondary_result_tables(metrics_df, PROJECT_ROOT / "results", training_rows)
    return metrics_df, trained, predictions


def greedy_optimized_sensors(sim: dict[str, Any], n_sensors: int) -> list[int]:
    n_nodes = len(sim["x_m"])
    candidate_nodes = list(range(1, n_nodes - 1))
    selected = [0, n_nodes - 1] if n_sensors > 1 else [0]
    state = np.stack([sim["Ts"], sim["Tr"]], axis=-1)
    while len(set(selected)) < max(1, n_sensors):
        best_node = candidate_nodes[0]
        best_score = float("inf")
        for node in candidate_nodes:
            if node in selected:
                continue
            nodes = sorted(set(selected + [node]))
            pred = np.zeros_like(state)
            for t in range(state.shape[0]):
                for var in range(2):
                    pred[t, :, var] = np.interp(np.arange(n_nodes), nodes, state[t, nodes, var])
            rmse_ts = float(np.sqrt(np.mean((pred[..., 0] - state[..., 0]) ** 2)))
            rmse_tr = float(np.sqrt(np.mean((pred[..., 1] - state[..., 1]) ** 2)))
            true_loss = np.mean(np.abs(state[..., 0] - state[..., 1]))
            pred_loss = np.mean(np.abs(pred[..., 0] - pred[..., 1]))
            heat_loss_error = 100.0 * abs(pred_loss - true_loss) / max(abs(true_loss), 1e-6)
            thermal_proxy = float(np.nanmean(np.abs(np.diff(pred[..., 0], axis=0) - np.diff(state[..., 0], axis=0)))) if pred.shape[0] > 1 else 0.0
            energy_proxy = float(abs(np.nanmean(pred[..., 0] - pred[..., 1]) - np.nanmean(state[..., 0] - state[..., 1])))
            score = rmse_ts + rmse_tr + 0.2 * heat_loss_error + 0.2 * thermal_proxy + 0.1 * energy_proxy
            if score < best_score:
                best_score = score
                best_node = node
        selected.append(best_node)
    return sorted(set(selected))[:n_sensors]


def greedy_optimized_three_sensors(sim: dict[str, Any]) -> list[int]:
    return greedy_optimized_sensors(sim, 3)


def _prediction_detail_metrics(pred_payload: dict[str, np.ndarray]) -> dict[str, float]:
    pred = pred_payload["pred"]
    true = pred_payload["true"]
    err_ts = np.abs(pred[..., 0] - true[..., 0])
    err_tr = np.abs(pred[..., 1] - true[..., 1])
    worst_node_temperature_error = float(np.nanmax(np.nanmean(err_ts + err_tr, axis=(0, 1)) / 2.0))
    outlet_node_error = float(np.nanmean((err_ts[:, :, -1] + err_tr[:, :, -1]) / 2.0))
    peak_period_error = float(np.nanpercentile(err_ts, 90))
    return {
        "worst_node_temperature_error_C": worst_node_temperature_error,
        "outlet_node_temperature_error_C": outlet_node_error,
        "peak_period_error_C": peak_period_error,
    }


def _sensor_layout_interpretation(layout: str, nodes: list[int], sim: dict[str, Any]) -> dict[str, Any]:
    n_nodes = len(sim["x_m"])
    x = np.asarray(sim["x_m"], dtype=float)
    if nodes:
        node_arr = np.asarray(sorted(set(nodes)), dtype=int)
        distances = np.min(np.abs(x[:, None] - x[node_arr][None, :]), axis=1)
        anchors = sorted(set([0, n_nodes - 1, *node_arr.tolist()]))
        max_gap = float(np.max(np.diff(x[anchors]))) if len(anchors) > 1 else float(x[-1] - x[0])
        middle = int(np.argmin(np.abs(x - x[-1] / 2.0)))
        includes_middle = bool(any(abs(int(node) - middle) <= 1 for node in node_arr))
    else:
        distances = x * np.nan
        max_gap = float(x[-1] - x[0])
        includes_middle = False
    mean_dist = float(np.nanmean(distances)) if nodes else float(x[-1] - x[0])
    info_gain = float(np.log1p(len(nodes)) / (1.0 + mean_dist / max(float(x[-1] - x[0]), 1.0)))
    return {
        "sensor_layout": layout,
        "n_sensors": len(nodes),
        "sensor_count": len(nodes),
        "information_gain_proxy": info_gain,
        "mean_distance_to_nearest_sensor_m": mean_dist,
        "nearest_sensor_distance_mean_km": mean_dist / 1000.0,
        "maximum_unobserved_segment_length_m": max_gap,
        "max_unobserved_distance_km": max_gap / 1000.0,
        "middle_pipe_sensor_included": includes_middle,
        "contains_middle_sensor": includes_middle,
        "contains_outlet_sensor": bool(any(int(node) == n_nodes - 1 for node in nodes)),
    }


def run_sensor_layouts(sim: dict[str, Any], config: dict[str, Any], stats: dict[str, Any] | None = None, quick: bool = False) -> pd.DataFrame:
    rows = []
    sim = dict(sim)
    sim["optimized_sensor_nodes"] = greedy_optimized_three_sensors(sim)
    sim["optimized_sensor_nodes_2"] = greedy_optimized_sensors(sim, 2)
    sim["optimized_sensor_nodes_5"] = greedy_optimized_sensors(sim, 5)
    interpretation_rows = []
    for layout in LAYOUTS_REQUIRED:
        layout_noise = 0.0
        if layout in {"S5_noisy_inlet_outlet", "S11_noisy_inlet_only"}:
            layout_noise = float(config["dataset"].get("noise_std_fraction", 0.01))
        sensors = apply_sensor_layout(sim, layout, config, noise_std_fraction=layout_noise)
        loaders = build_loaders(sim, sensors, config, stats=stats)
        spec = [{"label": "Proposed PI-GNN-GRU-v3 balanced_mode", "model": "pignn_v3", "loss_mode": "physics", "loss_weights": {"training_mode": "balanced_mode"}}]
        metrics_df, _, predictions = train_and_evaluate_specs(
            spec,
            loaders,
            config,
            quick=True,
            output_prefix=f"layout_{layout}_",
            epochs_override=int(config["training"].get("epochs_layout", 20) if not quick else min(2, int(config["training"]["epochs_quick"]))),
            write_secondary_tables=False,
        )
        row = metrics_df.iloc[0].to_dict()
        row["sensor_layout"] = layout
        row["sensor_nodes"] = ";".join(map(str, sensors["sensor_nodes"]))
        row["uses_real_xai4heat"] = False
        if "Proposed PI-GNN-GRU-v3 balanced_mode" in predictions:
            row.update(_prediction_detail_metrics(predictions["Proposed PI-GNN-GRU-v3 balanced_mode"]))
        rows.append(row)
        interpretation_rows.append(_sensor_layout_interpretation(layout, sensors["sensor_nodes"], sim))
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "sensor_layout_comparison.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "sensor_layout_comparison_detailed.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "sensor_layout_comparison_improved.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "sensor_layout_comparison_final.csv", index=False)
    pd.DataFrame(interpretation_rows).to_csv(PROJECT_ROOT / "results" / "sensor_layout_interpretation.csv", index=False)
    pd.DataFrame(interpretation_rows).to_csv(PROJECT_ROOT / "results" / "sensor_layout_interpretation_final.csv", index=False)
    (PROJECT_ROOT / "results" / "optimized_sensor_locations.json").write_text(
        json.dumps(
            {
                "optimized_three_layout": "S9_optimized_three_sensors",
                "optimized_three_sensor_nodes": sim["optimized_sensor_nodes"],
                "optimized_three_sensor_distances_m": [float(sim["x_m"][i]) for i in sim["optimized_sensor_nodes"]],
                "optimized_five_layout": "S10_optimized_five_sensors",
                "optimized_two_sensor_nodes": sim["optimized_sensor_nodes_2"],
                "optimized_two_sensor_distances_m": [float(sim["x_m"][i]) for i in sim["optimized_sensor_nodes_2"]],
                "optimized_five_sensor_nodes": sim["optimized_sensor_nodes_5"],
                "optimized_five_sensor_distances_m": [float(sim["x_m"][i]) for i in sim["optimized_sensor_nodes_5"]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (PROJECT_ROOT / "results" / "optimized_sensor_locations_final.json").write_text(
        (PROJECT_ROOT / "results" / "optimized_sensor_locations.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return df


def run_noise_dropout_robustness(
    sim: dict[str, Any],
    config: dict[str, Any],
    model: torch.nn.Module | dict[str, torch.nn.Module] | None,
    stats: dict[str, Any],
) -> pd.DataFrame:
    if model is None:
        df = pd.DataFrame([{"condition": "not_run", "note": "Proposed model unavailable"}])
        df.to_csv(PROJECT_ROOT / "results" / "noise_dropout_robustness.csv", index=False)
        return df
    if isinstance(model, dict):
        preferred = [
            "GRU-MSE",
            "Transformer-MSE",
            "PureGNN-MSE",
            "Proposed PI-GNN-GRU-v3 balanced_mode",
            "Proposed PI-GNN-GRU-v3 accuracy_mode",
            "Proposed PI-GNN-GRU-v2",
            "Proposed PI-GNN-GRU improved",
            "Proposed PI-GNN-GRU",
        ]
        model_items = [(name, model[name]) for name in preferred if name in model]
    else:
        model_items = [("Proposed PI-GNN-GRU", model)]
    conditions = [
        ("nominal_five_sensors", "S4_five_sensors", 0.0),
        ("noise_1_percent", "S4_five_sensors", 0.01),
        ("noise_3_percent", "S4_five_sensors", 0.03),
        ("noise_5_percent", "S4_five_sensors", 0.05),
        ("sensor_dropout", "S6_dropout_five_sensors", float(config["dataset"].get("noise_std_fraction", 0.0))),
    ]
    rows = []
    for condition, layout, noise in conditions:
        sensors = apply_sensor_layout(sim, layout, config, noise_std_fraction=noise)
        loaders = build_loaders(sim, sensors, config, stats=stats)
        for model_label, one_model in model_items:
            metrics = evaluate_model(one_model, loaders["test_loader"], config, stats, f"{model_label} {condition}")
            row = dict(metrics)
            row["base_model"] = model_label
            row["condition"] = condition
            row["sensor_layout"] = layout
            row["noise_std_fraction"] = noise
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "results" / "noise_dropout_robustness.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "noise_dropout_robustness_final.csv", index=False)
    return df


def run_ablation_study(loaders: dict[str, Any], config: dict[str, Any], quick: bool = False) -> pd.DataFrame:
    specs = [
        {
            "label": f"Proposed PI-GNN-GRU-v3 {name}",
            "model": model_name,
            "loss_mode": "physics",
            "loss_weights": weights,
        }
        for name, model_name, weights in ABLATION_SPECS
    ]
    metrics_df, _, _ = train_and_evaluate_specs(
        specs,
        loaders,
        config,
        quick=True,
        output_prefix="ablation_",
        epochs_override=int(config["training"].get("epochs_ablation", 30) if not quick else min(2, int(config["training"]["epochs_quick"]))),
        write_secondary_tables=False,
    )
    metrics_df["ablation"] = [name for name, _, _ in ABLATION_SPECS]
    metrics_df.to_csv(PROJECT_ROOT / "results" / "ablation_study.csv", index=False)
    metrics_df.to_csv(PROJECT_ROOT / "results" / "ablation_study_final.csv", index=False)
    hist_dir = ensure_dir(PROJECT_ROOT / "results" / "ablation_training_histories")
    for hist in (PROJECT_ROOT / "results").glob("ablation_*_training_history.csv"):
        try:
            target = hist_dir / hist.name
            target.write_text(hist.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    return metrics_df


def run_external_validation(
    flensburg_df: pd.DataFrame | None,
    config: dict[str, Any],
    calibrated_params: dict[str, float],
    model: torch.nn.Module | None,
    stats: dict[str, Any],
    primary_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    if flensburg_df is None or flensburg_df.empty or model is None:
        (PROJECT_ROOT / "results" / "EXTERNAL_VALIDATION_NOT_RUN.txt").write_text(
            "Flensburg external validation was not run because local Flensburg data or a trained proposed model were unavailable.\n",
            encoding="utf-8",
        )
        return None
    timestamps = pd.to_datetime(flensburg_df["timestamp"], utc=True, errors="raise").sort_values()
    observed_intervals_s = timestamps.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    native_dt_s = float(np.median(observed_intervals_s[observed_intervals_s > 0])) if len(observed_intervals_s) else float(config["system"]["dt_s"])
    if not np.isfinite(native_dt_s) or native_dt_s <= 0:
        raise ValueError("Flensburg native timestamp cadence could not be inferred.")
    # Flensburg is an hourly public dataset. Do not treat its native one-hour
    # samples as consecutive 15-min transitions merely because the source-model
    # configuration uses a 15-min cadence. The neural weights are unchanged;
    # this is intentionally part of the external temporal-resolution shift.
    ext_config = copy.deepcopy(config)
    ext_config["system"]["dt_s"] = int(round(native_dt_s))
    ext_boundary = build_boundary_conditions(flensburg_df, ext_config)
    ext_sim = simulate_thermo_hydraulics(ext_boundary, ext_config, params=calibrated_params)
    ext_sensors = apply_sensor_layout(ext_sim, "S2_inlet_outlet", ext_config, noise_std_fraction=ext_config["dataset"].get("noise_std_fraction", 0.0))
    loaders = build_loaders(ext_sim, ext_sensors, ext_config, stats=stats)
    metrics, pred = evaluate_model(model, loaders["test_loader"], ext_config, stats, "Proposed PI-GNN-GRU Flensburg transfer", return_predictions=True)
    df = pd.DataFrame([metrics])
    df["external_dataset"] = "flensburg"
    df["return_temperature_assumed"] = True
    df["validation_scope"] = "measured-node supply/feed error plus simulator-based hidden-state consistency"
    df.to_csv(PROJECT_ROOT / "results" / "external_validation_flensburg.csv", index=False)
    df.to_csv(PROJECT_ROOT / "results" / "external_validation_flensburg_final.csv", index=False)

    times = np.arange(pred["pred"].shape[1])
    ts_df = pd.DataFrame(
        {
            "window_step": times,
            "measured_or_boundary_supply_C": pred["sensor"][0, :, 0, 0],
            "predicted_supply_C": pred["pred"][0, :, 0, 0],
            "simulator_hidden_supply_C": pred["true"][0, :, 0, 0],
        }
    )
    ts_df["residual_C"] = ts_df["predicted_supply_C"] - ts_df["measured_or_boundary_supply_C"]
    ts_df["note"] = "Return temperature assumed as 50 C when unavailable in Flensburg."
    ts_df.to_csv(PROJECT_ROOT / "results" / "external_validation_flensburg_timeseries.csv", index=False)

    diagnostic = {
        "mean_heat_load_sonderborg_kw": float(primary_df["heat_load_kw"].mean()) if primary_df is not None and "heat_load_kw" in primary_df else np.nan,
        "mean_heat_load_flensburg_kw": float(flensburg_df["heat_load_kw"].mean()) if "heat_load_kw" in flensburg_df else np.nan,
        "supply_temp_min_sonderborg_C": float(primary_df["supply_temp_C"].min()) if primary_df is not None and "supply_temp_C" in primary_df else np.nan,
        "supply_temp_max_sonderborg_C": float(primary_df["supply_temp_C"].max()) if primary_df is not None and "supply_temp_C" in primary_df else np.nan,
        "supply_temp_min_flensburg_C": float(flensburg_df["supply_temp_C"].min()) if "supply_temp_C" in flensburg_df else np.nan,
        "supply_temp_max_flensburg_C": float(flensburg_df["supply_temp_C"].max()) if "supply_temp_C" in flensburg_df else np.nan,
        "return_temperature_assumed": bool(flensburg_df.get("return_temp_assumed", pd.Series([True])).fillna(True).astype(bool).any()),
        "ambient_available": bool("ambient_temp_C" in flensburg_df and not flensburg_df["ambient_temp_C"].isna().all()),
        "sampling_resolution_primary": "15 min",
        "sampling_resolution_external": f"native {native_dt_s / 3600.0:g} h",
        "external_native_dt_s": native_dt_s,
        "external_cadence_treatment": "simulator and sequence windows use native hourly cadence; no 15-min transition is assumed across hourly samples",
        "missing_variable_summary": "distributed pressure/head/flow unavailable; return temperature assumed if absent",
    }
    if np.isfinite(diagnostic["mean_heat_load_sonderborg_kw"]) and diagnostic["mean_heat_load_sonderborg_kw"] != 0:
        diagnostic["mean_heat_load_difference_percent"] = 100.0 * (
            diagnostic["mean_heat_load_flensburg_kw"] - diagnostic["mean_heat_load_sonderborg_kw"]
        ) / diagnostic["mean_heat_load_sonderborg_kw"]
    else:
        diagnostic["mean_heat_load_difference_percent"] = np.nan
    pd.DataFrame([diagnostic]).to_csv(PROJECT_ROOT / "results" / "flensburg_transfer_diagnostics.csv", index=False)

    mode_rows = []
    direct = dict(metrics)
    direct.update({"mode": "direct_transfer", "mode_status": "run", "note": "Sonderborg-trained model and Sonderborg-calibrated simulator parameters."})
    mode_rows.append(direct)
    class OutputBiasAdapter(torch.nn.Module):
        def __init__(self, base: torch.nn.Module, bias_norm: np.ndarray) -> None:
            super().__init__()
            self.base = base
            self.register_buffer("bias_norm", torch.tensor(bias_norm, dtype=torch.float32).view(1, 1, 1, 4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.bias_norm.to(x.device)

    def estimate_sensor_bias(payload: dict[str, np.ndarray], fraction: float = 0.25) -> np.ndarray:
        n = max(1, int(payload["pred"].shape[0] * fraction))
        pred_phys = payload["pred"][:n]
        sensor = payload["sensor"][:n]
        mask = payload["mask"][:n] > 0.5
        bias = np.zeros(4, dtype=np.float32)
        for var in range(4):
            if mask[..., var].any():
                bias[var] = float(np.nanmean(sensor[..., var][mask[..., var]] - pred_phys[..., var][mask[..., var]]))
        target_std = np.asarray(stats["target_std"], dtype=np.float32)
        return bias / np.maximum(target_std, 1e-6)

    try:
        offset_bias = estimate_sensor_bias(pred, fraction=0.20)
        offset_model = OutputBiasAdapter(model, offset_bias)
        offset_metrics = evaluate_model(offset_model, loaders["test_loader"], ext_config, stats, "Proposed PI-GNN-GRU Flensburg calibration-offset")
        offset_metrics.update(
            {
                "mode": "calibration_only_offset_adaptation",
                "mode_status": "run",
                "note": "No neural retraining; output temperature bias estimated from a short Flensburg calibration window.",
            }
        )
        mode_rows.append(offset_metrics)
    except Exception as exc:
        mode_rows.append({"mode": "calibration_only_offset_adaptation", "mode_status": "failed", "note": str(exc)})
    try:
        train_metrics, train_payload = evaluate_model(model, loaders["train_loader"], ext_config, stats, "Proposed PI-GNN-GRU Flensburg calibration slice", return_predictions=True)
        few_bias = estimate_sensor_bias(train_payload, fraction=1.0)
        few_model = OutputBiasAdapter(model, few_bias)
        few_metrics = evaluate_model(few_model, loaders["test_loader"], ext_config, stats, "Proposed PI-GNN-GRU Flensburg few-shot bias")
        few_metrics.update(
            {
                "mode": "few_shot_decoder_bias_adaptation",
                "mode_status": "run",
                "note": "Few-shot adaptation of final output bias using a small Flensburg calibration slice; not zero-shot transfer.",
            }
        )
        mode_rows.append(few_metrics)
    except Exception as exc:
        mode_rows.append({"mode": "few_shot_decoder_bias_adaptation", "mode_status": "failed", "note": str(exc)})
    try:
        norm_loaders = build_loaders(ext_sim, ext_sensors, ext_config, stats=None)
        norm_metrics = evaluate_model(model, norm_loaders["test_loader"], ext_config, norm_loaders["train_ds"].stats, "Proposed PI-GNN-GRU Flensburg boundary-normalized")
        norm_metrics.update({"mode": "normalized_transfer_flensburg_boundary_statistics", "mode_status": "diagnostic", "note": "Uses Flensburg normalization statistics to diagnose scale shift; model weights are not retrained."})
        mode_rows.append(norm_metrics)
    except Exception as exc:
        mode_rows.append({"mode": "normalized_transfer_flensburg_boundary_statistics", "mode_status": "failed", "note": str(exc)})
    pd.DataFrame(mode_rows).to_csv(PROJECT_ROOT / "results" / "external_validation_flensburg_modes.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(PROJECT_ROOT / "results" / "external_validation_flensburg_modes_final.csv", index=False)
    measured_only = pd.DataFrame(mode_rows).copy()
    measured_only = measured_only.loc[measured_only.get("mode_status", pd.Series(dtype=str)).eq("run") | measured_only.get("mode_status", pd.Series(dtype=str)).eq("diagnostic")].copy()
    measured_only = pd.DataFrame(
        {
            "mode": measured_only.get("mode", pd.Series(dtype=str)),
            "measured supply RMSE_C": pd.to_numeric(measured_only.get("RMSE_supply_measured_C"), errors="coerce"),
            "heat-load consistency_pct": pd.to_numeric(measured_only.get("heat_load_consistency_error_percent"), errors="coerce"),
            "return reference": "assumed 50 degC",
            "return metric status": "assumption-consistency; excluded from external validation",
            "native_cadence_s": native_dt_s,
            "cadence_treatment": "native hourly simulator and sequence cadence; no 15-min transition assumed",
        }
    )
    measured_only.to_csv(PROJECT_ROOT / "results" / "flensburg_measured_only_validation.csv", index=False)
    stale = PROJECT_ROOT / "results" / "EXTERNAL_VALIDATION_NOT_RUN.txt"
    if stale.exists():
        stale.unlink()
    fig10_external_validation(df)
    return df


def run_xai4heat_validation(
    xai4heat_df: pd.DataFrame | None,
    sim: dict[str, Any],
    config: dict[str, Any],
    model: torch.nn.Module | None,
    stats: dict[str, Any],
) -> pd.DataFrame | None:
    if xai4heat_df is None or xai4heat_df.empty or model is None:
        (PROJECT_ROOT / "results" / "XAI4HEAT_NOT_RUN.txt").write_text(
            "XAI4HEAT sparse-substation validation was not run because local XAI4HEAT data or a trained proposed model were unavailable.\n",
            encoding="utf-8",
        )
        (PROJECT_ROOT / "results" / "XAI4HEAT_NOT_AVAILABLE_FINAL.txt").write_text(
            "XAI4HEAT sparse-substation data were not available in the final local run. "
            "The manuscript therefore treats XAI4HEAT as a future measured-substation validation opportunity rather than completed evidence.\n",
            encoding="utf-8",
        )
        return None
    sparse = build_sparse_substation_measurements(xai4heat_df, len(sim["x_m"]))
    sensors = apply_sensor_layout(sim, "S7_xai4heat_substations", config, sparse_real_measurements=sparse)
    loaders = build_loaders(sim, sensors, config, stats=stats)
    metrics = evaluate_model(model, loaders["test_loader"], config, stats, "Proposed PI-GNN-GRU XAI4HEAT sparse nodes")
    rows = []
    node_map = sparse.get("node_map", {})
    for substation_id, node in node_map.items():
        row = dict(metrics)
        row["substation_id"] = substation_id
        row["mapped_node"] = node
        row["validation_scope"] = "sparse measured-node consistency where variables are available"
        rows.append(row)
    df = pd.DataFrame(rows or [metrics])
    df.to_csv(PROJECT_ROOT / "results" / "xai4heat_sparse_substation_validation.csv", index=False)
    fig12_xai4heat_sparse_substations(df)
    return df


def write_final_study_status(verdict: str | None = None) -> Path:
    results_dir = ensure_dir(PROJECT_ROOT / "results")
    qgate = PROJECT_ROOT / "results" / "paper_quality_gate_report.txt"
    if verdict is None and qgate.exists():
        first = qgate.read_text(encoding="utf-8").splitlines()[0]
        verdict = first.replace("Final verdict:", "").strip() if "Final verdict:" in first else "unknown"
    verdict = verdict or "unknown"
    metrics = PROJECT_ROOT / "results" / "metrics_summary.csv"
    fallback = False
    if metrics.exists():
        try:
            df = pd.read_csv(metrics)
            fallback = bool(df.get("used_fallback_synthetic", pd.Series([False])).fillna(False).astype(bool).any())
        except Exception:
            fallback = False
    lines = [
        f"Sonderborg available: {'yes' if check_dataset_available('sonderborg') else 'no'}",
        f"Flensburg available: {'yes' if check_dataset_available('flensburg') else 'no'}",
        f"XAI4HEAT available: {'yes' if check_dataset_available('xai4heat') else 'no'}",
        f"Main results use fallback synthetic data: {'yes' if fallback else 'no'}",
        f"Calibration completed: {'yes' if (PROJECT_ROOT / 'results' / 'calibration_metrics.csv').exists() else 'no'}",
        f"External validation completed: {'yes' if (PROJECT_ROOT / 'results' / 'external_validation_flensburg.csv').exists() else 'no'}",
        f"XAI4HEAT validation completed: {'yes' if (PROJECT_ROOT / 'results' / 'xai4heat_sparse_substation_validation.csv').exists() else 'no'}",
        f"Quality gate verdict: {verdict}",
    ]
    path = results_dir / "final_study_status.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_demo_workflow(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config["dataset"]["seed"])
    df, used_fallback = load_primary_or_demo(config, allow_fallback=True)
    assert df is not None
    df = df.head(220).copy()
    boundary = build_boundary_conditions(df, config)
    calibrated = calibrate_simulator(boundary, config, quick=True)
    config = apply_calibrated_params_to_config(config, calibrated["params"])
    sim = simulate_thermo_hydraulics(boundary, config, params=calibrated["params"])
    sim["used_fallback_synthetic"] = used_fallback
    save_model_verification(sim)
    run_discretization_study(boundary, config, calibrated["params"])
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"]["noise_std_fraction"])
    loaders = build_loaders(sim, sensors, config)
    quick_specs = [
        *BASELINE_SPECS,
    ]
    metrics_df, trained, predictions = train_and_evaluate_specs(quick_specs, loaders, config, quick=True, output_prefix="demo_")
    save_evaluation_outputs(metrics_df, PROJECT_ROOT / "results", fallback=used_fallback)
    if used_fallback:
        (PROJECT_ROOT / "results" / "QUICK_DEMO_USES_FALLBACK_SYNTHETIC.txt").write_text(
            "The quick demo used fallback synthetic-realistic data. This is not journal evidence.\n",
            encoding="utf-8",
        )
    (PROJECT_ROOT / "results" / "XAI4HEAT_NOT_RUN.txt").write_text(
        "XAI4HEAT sparse-substation validation was not run in the software quick demo. "
        "Run run_real_data_study.py with XAI4HEAT raw files available for measured-substation validation.\n",
        encoding="utf-8",
    )
    run_ablation_study(loaders, config, quick=True)
    run_sensor_layouts(sim, config, stats=loaders["train_ds"].stats, quick=True)
    run_noise_dropout_robustness(sim, config, trained, loaders["train_ds"].stats)
    if os.environ.get("CAUSAL_SKIP_ASSET_GENERATION", "0") != "1":
        generate_all_figures(sim, predictions, metrics_df, sensors, df, config)
        make_paper_assets(config)
    return {
        "used_fallback": used_fallback,
        "metrics": metrics_df,
        "trained": trained,
        "sim": sim,
        "stats": loaders["train_ds"].stats,
    }


def run_real_data_workflow(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config["dataset"]["seed"])
    processed = preprocess_available_datasets(config)
    primary = config["real_data"]["primary_dataset"]
    if primary not in processed:
        make_paper_assets(config)
        return {"used_fallback": False, "ran_main_results": False, "processed": processed}
    df = processed[primary].copy()
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    df = df.head(min(len(df), max_steps)).copy()
    boundary = build_boundary_conditions(df, config)
    calibrated = calibrate_simulator(boundary, config, quick=False)
    config = apply_calibrated_params_to_config(config, calibrated["params"])
    sim = simulate_thermo_hydraulics(boundary, config, params=calibrated["params"])
    save_model_verification(sim)
    run_discretization_study(boundary, config, calibrated["params"])
    sim["optimized_sensor_nodes"] = greedy_optimized_three_sensors(sim)
    sensors = apply_sensor_layout(sim, "S4_five_sensors", config, noise_std_fraction=config["dataset"]["noise_std_fraction"])
    loaders = build_loaders(sim, sensors, config)

    causal_core_only = os.environ.get("CAUSAL_RERUN_CORE_ONLY", "0") == "1"
    benchmark_specs = [spec for spec in BASELINE_SPECS if spec["label"] in CAUSAL_PRIMARY_LABELS] if causal_core_only else BASELINE_SPECS
    metrics_df, trained, predictions = train_and_evaluate_specs(benchmark_specs, loaders, config, quick=False, output_prefix="real_")
    save_evaluation_outputs(metrics_df, PROJECT_ROOT / "results", fallback=False)
    if not causal_core_only:
        run_ablation_study(loaders, config, quick=False)
        run_sensor_layouts(sim, config, stats=loaders["train_ds"].stats, quick=False)
        run_noise_dropout_robustness(sim, config, trained, loaders["train_ds"].stats)
    else:
        (PROJECT_ROOT / "results" / "CAUSAL_PROXY_DEPENDENT_STUDIES_PENDING.txt").write_text(
            "Primary benchmark, calibration, and external-evaluation artifacts were regenerated with causal proxies. "
            "Single-run layout, ablation, robustness, and scenario studies require the separate causal dependent-study rerun "
            "before they may be cited again. Pre-causal versions are archived under results/archive_precausal_proxy_20260801/.\n",
            encoding="utf-8",
        )

    proposed = (
        trained.get("Proposed PI-GNN-GRU-v3 balanced_mode")
        or trained.get("Proposed PI-GNN-GRU-v3 accuracy_mode")
        or trained.get("Proposed PI-GNN-GRU-v2")
        or trained.get("Proposed PI-GNN-GRU improved")
        or trained.get("Proposed PI-GNN-GRU")
    )
    run_external_validation(
        processed.get(config["real_data"]["external_validation_dataset"]),
        config,
        calibrated["params"],
        proposed,
        loaders["train_ds"].stats,
        primary_df=df,
    )
    run_xai4heat_validation(processed.get(config["real_data"]["sparse_dataset"]), sim, config, proposed, loaders["train_ds"].stats)
    if os.environ.get("CAUSAL_SKIP_ASSET_GENERATION", "0") != "1":
        generate_all_figures(sim, predictions, metrics_df, sensors, df, config)
        make_paper_assets(config)
    return {
        "used_fallback": False,
        "ran_main_results": True,
        "metrics": metrics_df,
        "trained": trained,
        "sim": sim,
        "stats": loaders["train_ds"].stats,
        "processed": processed,
    }
