"""Strengthen measured-node and external validation without changing benchmark labels.

The analyses in this module are deliberately separate from the distributed
calibrated-simulator benchmark:

* a training-only affine readout calibration for the 20 frozen principal
  checkpoints, evaluated against held-out measured return temperature;
* chronological leave-one-substation-out XAI4HEAT temperature estimation; and
* causal Flensburg measured-supply forecasting at 1, 6, and 24 h horizons.

No pressure/head, flow, distributed temperature, or heat-loss quantity is
treated as a measured target here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ate_figure_style import PALETTE, add_panel_label, save_ate_figure, set_ate_style, style_axes
from src.config import PROJECT_ROOT, load_config
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.effective_physics import apply_calibrated_params_to_config
from src.evaluate import evaluate_model
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import _a_norm_for, build_loaders
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import ensure_dir, load_json


RESULTS = PROJECT_ROOT / "results"
TABLES = PROJECT_ROOT / "paper" / "tables"
FIGURES = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures" / "final"
LOCKED_SONDERBORG = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv"
FLENSBURG = PROJECT_ROOT / "data" / "processed" / "flensburg_processed.csv"
XAI4HEAT = PROJECT_ROOT / "data" / "processed" / "xai4heat_processed.csv"
SEEDS = (11, 22, 33, 44, 55)
MODEL_SPECS = (
    ("GRU-MSE", "gru"),
    ("Transformer-MSE", "transformer"),
    ("Proposed PI-GNN-GRU-v3 accuracy_mode", "pignn_v3"),
    ("Proposed PI-GNN-GRU-v3 balanced_mode", "pignn_v3"),
)


def _escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_")):
        text = text.replace(old, new)
    return text


def _write_table(frame: pd.DataFrame, path: Path, caption: str, label: str, *, resize: bool = False) -> None:
    ensure_dir(path.parent)
    columns = list(frame.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        *([r"\resizebox{\textwidth}{!}{%"] if resize else []),
        f"\\begin{{tabular}}{{{'l' + 'r' * (len(columns) - 1)}}}",
        r"\toprule",
        " & ".join(_escape(column) for column in columns) + " \\\\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(_escape(row[column]) for column in columns) + " \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append("}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_unique(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(time_s, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    unique, inverse = np.unique(times, return_inverse=True)
    return unique, np.array([np.mean(values[inverse == index]) for index in range(len(unique))])


def _score(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        "RMSE_C": float(np.sqrt(np.mean(error**2))),
        "MAE_C": float(np.mean(np.abs(error))),
        "bias_C": float(np.mean(error)),
    }


def measured_return_readout_adaptation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a two-parameter readout on training M targets and score held-out M targets."""
    config = load_config()
    parameters = load_json(RESULTS / "calibrated_parameters.json")
    config = apply_calibrated_params_to_config(config, parameters)
    max_steps = int(config["dataset"]["n_scenarios_full"] * config["system"]["horizon_h"] * 3600 / config["system"]["dt_s"])
    frame = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config).head(max_steps).copy()
    boundary = build_boundary_conditions(frame, config)
    simulator = simulate_thermo_hydraulics(boundary, config, params=parameters)

    s4 = apply_sensor_layout(simulator, "S4_five_sensors", config, noise_std_fraction=config["dataset"].get("noise_std_fraction", 0.0))
    base_loaders = build_loaders(simulator, s4, config)
    statistics = base_loaders["train_ds"].stats

    shape = (len(simulator["time_s"]), len(simulator["x_m"]), 4)
    measurements = np.zeros(shape, dtype=np.float32)
    masks = np.zeros(shape, dtype=np.float32)
    measurements[:, 0, 0] = np.asarray(simulator["T_source"], dtype=np.float32)
    masks[:, 0, 0] = 1.0
    blind_loaders = build_loaders(
        simulator,
        {
            "layout_name": "measured_source_boundary_only",
            "sensor_nodes": [0],
            "measurements": measurements,
            "masks": masks,
            "variables": ["Ts"],
        },
        config,
        stats=statistics,
    )
    adjacency = _a_norm_for(blind_loaders["arrays"])
    measured_lookup = {
        float(time): float(value)
        for time, value in zip(simulator["time_s"], boundary["T_return_measured"])
    }

    rows: list[dict[str, Any]] = []
    test_times_reference: np.ndarray | None = None
    for seed in SEEDS:
        for label, architecture in MODEL_SPECS:
            checkpoint = RESULTS / f"seed_{seed}_{label}_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model = build_model(
                architecture,
                input_dim=blind_loaders["arrays"]["features"].shape[-1],
                n_nodes=blind_loaders["arrays"]["target"].shape[1],
                a_norm=adjacency,
                config=config,
            )
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            payload: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for split in ("train", "test"):
                _, result = evaluate_model(
                    model,
                    blind_loaders[f"{split}_loader"],
                    config,
                    statistics,
                    label,
                    return_predictions=True,
                )
                times, prediction = _aggregate_unique(result["time_s"], np.asarray(result["pred"])[..., 0, 1])
                measured = np.array([measured_lookup[float(time)] for time in times])
                payload[split] = (times, prediction, measured)

            train_times, train_prediction, train_measured = payload["train"]
            test_times, test_prediction, test_measured = payload["test"]
            overlap = int(np.intersect1d(train_times, test_times).size)
            if overlap:
                raise RuntimeError(f"Measured-node readout train/test overlap for {label}, seed {seed}: {overlap}")
            if test_times_reference is None:
                test_times_reference = test_times.copy()

            offset = float(np.mean(train_measured - train_prediction))
            design_train = np.column_stack([train_prediction, np.ones(len(train_prediction))])
            # Fixed weak slope regularization; the intercept remains unpenalized.
            coefficients = np.linalg.solve(
                design_train.T @ design_train + np.diag([1.0e-3, 0.0]),
                design_train.T @ train_measured,
            )
            variants = {
                "raw frozen checkpoint": test_prediction,
                "training-only offset": test_prediction + offset,
                "training-only affine readout": np.column_stack([test_prediction, np.ones(len(test_prediction))]) @ coefficients,
            }
            raw_rmse = _score(test_measured, test_prediction)["RMSE_C"]
            for mode, prediction in variants.items():
                metrics = _score(test_measured, prediction)
                rows.append(
                    {
                        "model": label,
                        "seed": seed,
                        "adaptation": mode,
                        **metrics,
                        "raw_RMSE_C": raw_rmse,
                        "RMSE_reduction_vs_raw_percent": 100.0 * (raw_rmse - metrics["RMSE_C"]) / raw_rmse,
                        "readout_slope": float(coefficients[0]) if mode == "training-only affine readout" else np.nan,
                        "readout_intercept_C": float(coefficients[1]) if mode == "training-only affine readout" else (offset if mode == "training-only offset" else np.nan),
                        "training_unique_timestamps": len(train_times),
                        "test_unique_timestamps": len(test_times),
                        "train_test_overlap": overlap,
                        "current_return_measurement_used_at_test": False,
                        "internal_simulator_sensor_values_used_at_test": False,
                        "state_type": "real_measured_node",
                        "safe_claim": "Training-only scalar readout adaptation of a mixed-target checkpoint; held-out measured return target; not distributed field validation.",
                    }
                )

    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "measured_return_checkpoint_adaptation.csv", index=False)
    summary = (
        detail.groupby(["model", "adaptation"], as_index=False)
        .agg(
            mean_RMSE_C=("RMSE_C", "mean"),
            std_RMSE_C=("RMSE_C", "std"),
            mean_MAE_C=("MAE_C", "mean"),
            mean_RMSE_reduction_percent=("RMSE_reduction_vs_raw_percent", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("mean_RMSE_C")
    )
    if test_times_reference is None:
        raise RuntimeError("No measured-return test timestamps were produced.")
    simulator_times = np.asarray(simulator["time_s"], dtype=float)
    measured_return = np.asarray(boundary["T_return_measured"], dtype=float)
    index = {float(time): position for position, time in enumerate(simulator_times)}
    target = np.array([measured_return[index[float(time)]] for time in test_times_reference])
    persistence = np.array([measured_return[max(0, index[float(time)] - 1)] for time in test_times_reference])
    persistence_metrics = _score(target, persistence)
    baseline = pd.DataFrame(
        [{
            "model": "Measured-return persistence",
            "adaptation": "none",
            "mean_RMSE_C": persistence_metrics["RMSE_C"],
            "std_RMSE_C": np.nan,
            "mean_MAE_C": persistence_metrics["MAE_C"],
            "mean_RMSE_reduction_percent": np.nan,
            "n_seeds": 0,
        }]
    )
    summary = pd.concat([baseline, summary], ignore_index=True)
    summary.to_csv(RESULTS / "measured_return_checkpoint_adaptation_summary.csv", index=False)

    compact = summary[
        summary["adaptation"].isin(["none", "raw frozen checkpoint", "training-only affine readout"])
    ].copy()
    compact["Estimator"] = compact["model"].replace(
        {
            "Measured-return persistence": "Persistence",
            "Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-v3 accuracy",
            "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-v3 balanced",
        }
    )
    compact["Protocol"] = compact["adaptation"].replace(
        {"none": "M persistence", "raw frozen checkpoint": "raw mixed-target checkpoint", "training-only affine readout": "M affine readout"}
    )
    compact["Return RMSE, mean (SD)"] = compact.apply(
        lambda row: f"{row['mean_RMSE_C']:.3f}" if not np.isfinite(row["std_RMSE_C"]) else f"{row['mean_RMSE_C']:.3f} ({row['std_RMSE_C']:.3f})",
        axis=1,
    )
    compact["Return MAE"] = compact["mean_MAE_C"].map(lambda value: f"{value:.3f}")
    _write_table(
        compact[["Estimator", "Protocol", "Return RMSE, mean (SD)", "Return MAE"]],
        TABLES / "table_measured_return_model_adaptation.tex",
        "Held-out measured-return validation before and after training-only scalar readout calibration. Current return and all internal C-class thermal and S-class hydraulic sensor inputs are hidden at test time. Persistence remains the strongest short-horizon reference.",
        "tab:measured_return_adaptation",
        resize=True,
    )
    return detail, summary


def _xai_design(data: pd.DataFrame, target: str, variable: str) -> tuple[pd.DataFrame, pd.Series]:
    substations = sorted(data["substation_id"].dropna().unique())
    pivot = data.pivot_table(
        index="timestamp",
        columns="substation_id",
        values=["supply_temp_C", "return_temp_C"],
        aggfunc="mean",
    ).sort_index()
    features = pd.concat(
        [
            pivot[(state, station)].rename(f"{state}_{station}")
            for station in substations
            if station != target
            for state in ("supply_temp_C", "return_temp_C")
        ],
        axis=1,
    )
    hours = features.index.hour.to_numpy()
    days = features.index.dayofyear.to_numpy()
    features["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    features["year_sin"] = np.sin(2.0 * np.pi * days / 365.25)
    features["year_cos"] = np.cos(2.0 * np.pi * days / 365.25)
    target_series = pivot[(variable, target)].rename("target")
    return features, target_series


def _spatial_interpolation_prediction(data: pd.DataFrame, target: str, variable: str, timestamps: pd.Index) -> np.ndarray:
    positions = {
        station: float("".join(character for character in station if character.isdigit()))
        for station in data["substation_id"].dropna().unique()
    }
    ordered = sorted(positions, key=positions.get)
    pivot = data.pivot_table(index="timestamp", columns="substation_id", values=variable, aggfunc="mean").reindex(columns=ordered)
    train = pivot.copy()
    train[target] = np.nan
    prediction = train.interpolate(axis=1, method="linear", limit_direction="both")[target]
    return prediction.reindex(timestamps).to_numpy(dtype=float)


def xai4heat_chronological_withholding() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(XAI4HEAT)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp", "substation_id"]).sort_values("timestamp")
    substations = sorted(data["substation_id"].unique())
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    rows: list[dict[str, Any]] = []
    for target in substations:
        for variable in ("supply_temp_C", "return_temp_C"):
            features, target_series = _xai_design(data, target, variable)
            joined = pd.concat([features, target_series], axis=1).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
            n = len(joined)
            train_end = int(0.60 * n)
            validation_end = int(0.80 * n)
            train = joined.iloc[:train_end]
            validation = joined.iloc[train_end:validation_end]
            test = joined.iloc[validation_end:]
            best = (float("inf"), None)
            for alpha in alphas:
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(train.drop(columns="target"), train["target"])
                rmse = float(np.sqrt(mean_squared_error(validation["target"], model.predict(validation.drop(columns="target")))))
                if rmse < best[0]:
                    best = (rmse, alpha)
            fit = joined.iloc[:validation_end]
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(best[1])))
            model.fit(fit.drop(columns="target"), fit["target"])
            ridge_prediction = model.predict(test.drop(columns="target"))
            interpolation_prediction = _spatial_interpolation_prediction(data, target, variable, test.index)
            for estimator, prediction in (
                ("spatial interpolation", interpolation_prediction),
                ("chronological multi-station ridge", ridge_prediction),
            ):
                mask = np.isfinite(prediction) & np.isfinite(test["target"].to_numpy(dtype=float))
                metrics = _score(test["target"].to_numpy(dtype=float)[mask], np.asarray(prediction)[mask])
                rows.append(
                    {
                        "substation_id": target,
                        "variable": variable,
                        "estimator": estimator,
                        **metrics,
                        "alpha_selected_on_validation": float(best[1]) if estimator.startswith("chronological") else np.nan,
                        "training_samples": train_end,
                        "validation_samples": validation_end - train_end,
                        "test_samples": int(mask.sum()),
                        "test_start": test.index.min().isoformat(),
                        "test_end": test.index.max().isoformat(),
                        "withheld_target_temperature_used_as_feature": False,
                        "target_conditioned_order_filter_used": False,
                        "state_type": "real_measured_node",
                        "safe_claim": "Chronological measured-substation temperature withholding; no pressure/head, flow, heat-loss, or internal-pipe validation.",
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "xai4heat_chronological_withholding.csv", index=False)
    summary = (
        detail.groupby(["variable", "estimator"], as_index=False)
        .agg(
            mean_fold_RMSE_C=("RMSE_C", "mean"),
            std_fold_RMSE_C=("RMSE_C", "std"),
            mean_fold_MAE_C=("MAE_C", "mean"),
            total_test_samples=("test_samples", "sum"),
            n_substations=("substation_id", "nunique"),
        )
        .sort_values(["variable", "mean_fold_RMSE_C"])
    )
    summary.to_csv(RESULTS / "xai4heat_chronological_withholding_summary.csv", index=False)
    compact = summary.copy()
    compact["Variable"] = compact["variable"].map({"supply_temp_C": "Primary supply", "return_temp_C": "Primary return"})
    compact["Estimator"] = compact["estimator"].replace(
        {"spatial interpolation": "Spatial interpolation", "chronological multi-station ridge": "Chronological ridge"}
    )
    compact["RMSE, mean (SD)"] = compact.apply(lambda row: f"{row['mean_fold_RMSE_C']:.3f} ({row['std_fold_RMSE_C']:.3f})", axis=1)
    compact["MAE"] = compact["mean_fold_MAE_C"].map(lambda value: f"{value:.3f}")
    compact["Samples"] = compact["total_test_samples"].map(lambda value: f"{int(value):,}")
    _write_table(
        compact[["Variable", "Estimator", "RMSE, mean (SD)", "MAE", "Samples"]],
        TABLES / "table_xai4heat_chronological_withholding.tex",
        r"Chronological XAI4HEAT leave-one-substation-out temperature estimation on the final 20\% of each fold. Ridge hyperparameters use only earlier training/validation data; withheld target temperatures never enter the features and no target-conditioned order filter is applied.",
        "tab:xai_chronological",
        resize=True,
    )
    return detail, summary


def _hourly_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data = data.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return data[["supply_temp_C", "heat_load_kw"]].resample("1h").mean()


def _forecast_design(data: pd.DataFrame, horizon_h: int) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    for lag in (0, 1, 2, 6, 24, 168):
        features[f"supply_lag_{lag}h"] = data["supply_temp_C"].shift(lag)
    for lag in (0, 1, 2, 6, 24):
        features[f"load_lag_{lag}h"] = data["heat_load_kw"].shift(lag)
    hours = features.index.hour.to_numpy()
    days = features.index.dayofyear.to_numpy()
    features["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    features["year_sin"] = np.sin(2.0 * np.pi * days / 365.25)
    features["year_cos"] = np.cos(2.0 * np.pi * days / 365.25)
    return pd.concat([features, data["supply_temp_C"].shift(-horizon_h).rename("target")], axis=1).dropna()


def _select_ridge(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[float, Any]:
    best = (float("inf"), None, None)
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train.drop(columns="target"), train["target"])
        rmse = float(np.sqrt(mean_squared_error(validation["target"], model.predict(validation.drop(columns="target")))))
        if rmse < best[0]:
            best = (rmse, alpha, model)
    return float(best[1]), best[2]


def flensburg_causal_forecast() -> tuple[pd.DataFrame, pd.DataFrame]:
    sonderborg = _hourly_data(LOCKED_SONDERBORG)
    flensburg = _hourly_data(FLENSBURG)
    rows: list[dict[str, Any]] = []
    prediction_sample: pd.DataFrame | None = None
    for horizon in (1, 6, 24):
        source = _forecast_design(sonderborg, horizon)
        source = source[source.index < pd.Timestamp("2018-01-01", tz="UTC")]
        source_split = max(1, int(0.80 * len(source)))
        source_train = source.iloc[:source_split]
        source_validation = source.iloc[source_split:]
        source_alpha, _ = _select_ridge(source_train, source_validation)
        source_model = make_pipeline(StandardScaler(), Ridge(alpha=source_alpha))
        source_model.fit(source.drop(columns="target"), source["target"])

        external = _forecast_design(flensburg, horizon)
        local_train = external[external.index < pd.Timestamp("2018-01-01", tz="UTC")]
        local_validation = external[
            (external.index >= pd.Timestamp("2018-01-01", tz="UTC"))
            & (external.index < pd.Timestamp("2018-04-01", tz="UTC"))
        ]
        test = external[external.index >= pd.Timestamp("2018-04-02", tz="UTC")]
        local_alpha, _ = _select_ridge(local_train, local_validation)
        local_fit = pd.concat([local_train, local_validation]).sort_index()
        local_model = make_pipeline(StandardScaler(), Ridge(alpha=local_alpha))
        local_model.fit(local_fit.drop(columns="target"), local_fit["target"])

        predictions = {
            "persistence": test["supply_lag_0h"].to_numpy(dtype=float),
            "daily persistence": test["supply_lag_24h"].to_numpy(dtype=float),
            "Sonderborg-trained ridge": source_model.predict(test.drop(columns="target")),
            "Flensburg local-history ridge": local_model.predict(test.drop(columns="target")),
        }
        persistence_rmse = _score(test["target"].to_numpy(dtype=float), predictions["persistence"])["RMSE_C"]
        for estimator, prediction in predictions.items():
            metrics = _score(test["target"].to_numpy(dtype=float), prediction)
            rows.append(
                {
                    "horizon_h": horizon,
                    "estimator": estimator,
                    **metrics,
                    "RMSE_change_vs_persistence_percent": 100.0 * (metrics["RMSE_C"] - persistence_rmse) / persistence_rmse,
                    "source_training_samples": len(source) if estimator == "Sonderborg-trained ridge" else 0,
                    "local_training_samples": len(local_fit) if estimator == "Flensburg local-history ridge" else 0,
                    "test_samples": len(test),
                    "test_start": test.index.min().isoformat(),
                    "test_end": test.index.max().isoformat(),
                    "future_supply_used_as_feature": False,
                    "return_temperature_used": False,
                    "state_type": "real_measured_node",
                    "safe_claim": "Causal measured-supply boundary forecasting; not distributed thermo-hydraulic state validation.",
                }
            )
        if horizon == 6:
            selection = slice(0, min(24 * 21, len(test)))
            prediction_sample = pd.DataFrame(
                {
                    "timestamp": test.index[selection],
                    "measured_supply_C": test["target"].to_numpy(dtype=float)[selection],
                    "persistence_C": predictions["persistence"][selection],
                    "source_ridge_C": predictions["Sonderborg-trained ridge"][selection],
                    "local_history_ridge_C": predictions["Flensburg local-history ridge"][selection],
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(RESULTS / "flensburg_causal_supply_forecast.csv", index=False)
    summary = detail[["horizon_h", "estimator", "RMSE_C", "MAE_C", "bias_C", "RMSE_change_vs_persistence_percent", "test_samples"]].copy()
    summary.to_csv(RESULTS / "flensburg_causal_supply_forecast_summary.csv", index=False)
    if prediction_sample is not None:
        prediction_sample.to_csv(RESULTS / "flensburg_causal_supply_forecast_timeseries_sample.csv", index=False)

    compact = summary[summary["estimator"].isin(["persistence", "Sonderborg-trained ridge", "Flensburg local-history ridge"])].copy()
    compact["Horizon"] = compact["horizon_h"].map(lambda value: f"{int(value)} h")
    compact["Estimator"] = compact["estimator"].replace(
        {"persistence": "Persistence", "Sonderborg-trained ridge": "Source-network ridge", "Flensburg local-history ridge": "Local-history ridge"}
    )
    compact["RMSE"] = compact["RMSE_C"].map(lambda value: f"{value:.3f}")
    compact["MAE"] = compact["MAE_C"].map(lambda value: f"{value:.3f}")
    compact["Change vs persistence"] = compact["RMSE_change_vs_persistence_percent"].map(lambda value: f"{value:+.1f}%")
    _write_table(
        compact[["Horizon", "Estimator", "RMSE", "MAE", "Change vs persistence"]],
        TABLES / "table_flensburg_causal_forecast.tex",
        "Causal Flensburg measured-supply forecasting. The S{\\o}nderborg model uses only pre-2018 source-network data; the local-history model uses Flensburg 2017 for fitting and January--March 2018 for validation. Testing begins after an embargo on 2 April 2018. Return temperature is unavailable and is not scored.",
        "tab:flensburg_causal_forecast",
        resize=True,
    )
    return detail, summary


def validation_upgrade_figure(
    adaptation: pd.DataFrame,
    xai_summary: pd.DataFrame,
    flensburg: pd.DataFrame,
) -> None:
    set_ate_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)

    selected = adaptation[
        adaptation["model"].isin(["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 balanced_mode"])
        & adaptation["adaptation"].isin(["raw frozen checkpoint", "training-only affine readout"])
    ].copy()
    pivot = selected.pivot(index="model", columns="adaptation", values="mean_RMSE_C")
    labels = ["GRU", "Transformer", "PI-GNN balanced"]
    index = ["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 balanced_mode"]
    x = np.arange(len(index))
    width = 0.36
    raw_group = selected[selected["adaptation"].eq("raw frozen checkpoint")].set_index("model").loc[index]
    adapted_group = selected[selected["adaptation"].eq("training-only affine readout")].set_index("model").loc[index]
    axes[0, 0].bar(x - width / 2, raw_group["mean_RMSE_C"], width, yerr=raw_group["std_RMSE_C"], capsize=3, color=PALETTE["baseline"], edgecolor=PALETTE["edge"], label="Raw checkpoint")
    axes[0, 0].bar(x + width / 2, adapted_group["mean_RMSE_C"], width, yerr=adapted_group["std_RMSE_C"], capsize=3, color=PALETTE["proposed"], edgecolor=PALETTE["edge"], label="Training-only M readout")
    axes[0, 0].axhline(0.133946, color=PALETTE["measured"], ls="--", lw=1.2, label="Persistence")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel(r"Measured-return RMSE ($^\circ$C)")
    axes[0, 0].set_title("Held-out measured-return adaptation", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7)
    style_axes(axes[0, 0])

    for variable, color, marker in (("supply_temp_C", PALETTE["proposed"], "o"), ("return_temp_C", PALETTE["safe"], "s")):
        subset = xai_summary[xai_summary["variable"].eq(variable)]
        for _, row in subset.iterrows():
            position = 0 if row["estimator"] == "spatial interpolation" else 1
            axes[0, 1].scatter(position, row["mean_fold_RMSE_C"], s=65, color=color, marker=marker, edgecolor=PALETTE["edge"], linewidth=0.7)
    axes[0, 1].set_xticks([0, 1], ["Spatial\ninterpolation", "Chronological\nridge"])
    axes[0, 1].set_ylabel(r"Mean fold RMSE ($^\circ$C)")
    axes[0, 1].set_title("XAI4HEAT measured-node withholding", fontweight="bold")
    axes[0, 1].plot([], [], color=PALETTE["proposed"], marker="o", ls="", label="Supply")
    axes[0, 1].plot([], [], color=PALETTE["safe"], marker="s", ls="", label="Return")
    axes[0, 1].legend(frameon=False, fontsize=7)
    style_axes(axes[0, 1])

    colors = {"persistence": PALETTE["measured"], "Sonderborg-trained ridge": PALETTE["baseline"], "Flensburg local-history ridge": PALETTE["proposed"]}
    for estimator, group in flensburg[flensburg["estimator"].isin(colors)].groupby("estimator"):
        axes[1, 0].plot(group["horizon_h"], group["RMSE_C"], marker="o", lw=1.7, color=colors[estimator], label=estimator.replace("Flensburg ", "").replace("Sonderborg-", "Sønderborg "))
    axes[1, 0].set_xticks([1, 6, 24])
    axes[1, 0].set_xlabel("Forecast horizon (h)")
    axes[1, 0].set_ylabel(r"Measured-supply RMSE ($^\circ$C)")
    axes[1, 0].set_title("Flensburg causal external forecast", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=7)
    style_axes(axes[1, 0])

    sample = pd.read_csv(RESULTS / "flensburg_causal_supply_forecast_timeseries_sample.csv")
    sample["timestamp"] = pd.to_datetime(sample["timestamp"], utc=True)
    axes[1, 1].plot(sample["timestamp"], sample["measured_supply_C"], color=PALETTE["measured"], lw=1.4, label="Measured")
    axes[1, 1].plot(sample["timestamp"], sample["source_ridge_C"], color=PALETTE["baseline"], lw=1.0, label="Sønderborg ridge")
    axes[1, 1].plot(sample["timestamp"], sample["local_history_ridge_C"], color=PALETTE["proposed"], lw=1.0, label="Local-history ridge")
    axes[1, 1].set_ylabel(r"Supply temperature ($^\circ$C)")
    axes[1, 1].set_title("Six-hour Flensburg forecast sample", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    axes[1, 1].tick_params(axis="x", rotation=20)
    style_axes(axes[1, 1])

    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.flat):
        add_panel_label(axis, label)
    save_ate_figure(fig, FIGURES, "fig_measured_external_validation_upgrade")
    ensure_dir(PAPER_FIGURES)
    for suffix in ("pdf", "png"):
        source = FIGURES / f"fig_measured_external_validation_upgrade.{suffix}"
        (PAPER_FIGURES / source.name).write_bytes(source.read_bytes())


def write_protocol_manifest() -> None:
    sources = [LOCKED_SONDERBORG, FLENSBURG, XAI4HEAT, RESULTS / "calibrated_parameters.json"]
    manifest = {
        "created_by": "src/final_measured_validation_upgrade.py",
        "source_files": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sources
        ],
        "protocols": {
            "measured_return_readout": "training-only offset/affine calibration; 102 untouched test timestamps; no current return or internal C-class thermal or S-class hydraulic sensor input at test",
            "xai4heat": "60/20/20 chronological split per fold; target temperature channels excluded from features; all-valid-range protocol",
            "flensburg": "causal 1/6/24 h measured-supply forecasts; pre-2018 source model and 2017+Q1-2018 local-history model; test from 2018-04-02",
        },
        "evidence_boundary": "Only measured temperature targets are validated. Distributed temperature, heat loss, pressure/head, and flow are not field validated by these analyses.",
    }
    (RESULTS / "final_measured_validation_upgrade_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RESULTS)
    ensure_dir(TABLES)
    detail, adaptation_summary = measured_return_readout_adaptation()
    _, xai_summary = xai4heat_chronological_withholding()
    flensburg_detail, _ = flensburg_causal_forecast()
    validation_upgrade_figure(adaptation_summary, xai_summary, flensburg_detail)
    write_protocol_manifest()
    print("Final measured-node and external-validation upgrade completed.")
    print(adaptation_summary.to_string(index=False))
    print(xai_summary.to_string(index=False))
    print(flensburg_detail.to_string(index=False))


if __name__ == "__main__":
    main()
