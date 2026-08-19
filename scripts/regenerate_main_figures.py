"""Regenerate the main ATE figures from numerical project artifacts only.

This script intentionally does not read existing PNG/PDF/SVG figures.  It uses
CSV/JSON/YAML-like text files, NPZ/NPY arrays, and PT files only when they
contain paired reference/prediction arrays.  For dense field figures, missing
paired arrays are reported and the figure is skipped rather than replaced by a
synthetic or raster-derived surface.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import textwrap
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.titlesize": 12,
        "axes.linewidth": 0.85,
    }
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
CONFIG = ROOT / "config"
OUT = ROOT / "figures" / "regenerated_main"
ZIP_PATH = ROOT / "figures" / "regenerated_main_figures.zip"

PALETTE = {
    "Reference": "#111111",
    "Measured": "#111111",
    "GRU-MSE": "#0072B2",
    "Transformer-MSE": "#E69F00",
    "PureGNN-MSE": "#009E73",
    "PI-GNN-GRU-v3": "#CC79A7",
    "PI-LSTM": "#D55E00",
    "Interpolation": "#555555",
    "Energy": "#56B4E9",
    "HeatLoss": "#F0E442",
    "Alert": "#D55E00",
    "Good": "#009E73",
    "Blue": "#0000E6",
    "Orange": "#FF6626",
    "Green": "#55D600",
    "Yellow": "#F2E600",
    "Magenta": "#E600E6",
    "Gray": "#555555",
}

MODEL_ALIASES = {
    "Proposed PI-GNN-GRU-v3 balanced_mode": "PI-GNN-GRU-v3",
    "Proposed PI-GNN-GRU-v3 accuracy_mode": "PI-GNN-GRU-v3",
    "Proposed PI-GNN-GRU-v3 physics_mode": "PI-GNN-GRU-v3",
    "Proposed PI-GNN-GRU-v3": "PI-GNN-GRU-v3",
    "Transformer-MSE": "Transformer-MSE",
    "GRU-MSE": "GRU-MSE",
    "PureGNN-MSE": "PureGNN-MSE",
    "PI-LSTM": "PI-LSTM",
    "Interpolation": "Interpolation",
}

EXPORT_MODEL_PRIORITY = [
    "Proposed PI-GNN-GRU-v3 balanced_mode",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Transformer-MSE",
    "GRU-MSE",
]


def safe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@dataclass
class ProvenanceRow:
    figure_id: str
    panel: str
    source_file: str
    columns_or_keys: str
    transformations: str
    assumptions: str
    evidence_class: str
    generation_status: str
    output_files: str
    notes: str = ""


@dataclass
class BuildState:
    provenance: list[ProvenanceRow] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    outputs: dict[str, list[Path]] = field(default_factory=dict)

    def add_prov(
        self,
        figure_id: str,
        panel: str,
        source_file: Path | str,
        columns_or_keys: str,
        transformations: str,
        assumptions: str,
        evidence_class: str,
        generation_status: str,
        output_files: Iterable[Path] | Path | str,
        notes: str = "",
    ) -> None:
        if isinstance(output_files, str):
            output_text = output_files
        elif isinstance(output_files, Path):
            output_text = str(output_files.relative_to(ROOT)) if output_files.is_absolute() else str(output_files)
        else:
            output_text = "; ".join(str(p.relative_to(ROOT)) for p in output_files)
        source_text = str(source_file)
        try:
            source_text = str(Path(source_file).relative_to(ROOT))
        except Exception:
            pass
        self.provenance.append(
            ProvenanceRow(
                figure_id=figure_id,
                panel=panel,
                source_file=source_text,
                columns_or_keys=columns_or_keys,
                transformations=transformations,
                assumptions=assumptions,
                evidence_class=evidence_class,
                generation_status=generation_status,
                output_files=output_text,
                notes=notes,
            )
        )

    def mark_missing(self, message: str) -> None:
        self.missing.append(message)
        self.log.append("MISSING: " + message)

    def note(self, message: str) -> None:
        self.log.append(message)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_csv(path: Path, required: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
    return df


def clean_model_name(name: Any) -> str:
    text = str(name)
    return MODEL_ALIASES.get(text, text)


def numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def save_figure(fig: plt.Figure, stem: str, state: BuildState) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("svg", "pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        outputs.append(path)
    for mirror in [ROOT / "figures" / "final", ROOT / "paper" / "figures" / "final"]:
        mirror.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            shutil.copy2(path, mirror / path.name)
    state.outputs[stem] = outputs
    plt.close(fig)
    state.note(f"Generated {stem}: {', '.join(rel(p) for p in outputs)}")
    return outputs


def panel_label(ax: plt.Axes, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(
            -0.08,
            1.04,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
        return
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )


def set_grid(ax: plt.Axes) -> None:
    ax.grid(True, color="#E5E5E5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def candidate_dense_files() -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.npz", "*.npy", "*.pt", "*.pth"):
        paths.extend(RESULTS.rglob(pattern))
        paths.extend((ROOT / "data").rglob(pattern))
    return sorted(set(paths))


def _flatten_payload_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 4:
        return arr.reshape(-1, *arr.shape[2:])
    if arr.ndim == 3:
        return arr
    return arr.reshape(-1)


def export_dense_reconstruction_payloads(state: BuildState) -> Path | None:
    """Export paired dense arrays using the repository's model/test pipeline."""
    out_path = RESULTS / "dense_reconstruction_payloads.npz"
    try:
        import torch  # noqa: F401
        from src.config import load_config
        from src.thermo_hydraulic_coupling_analysis import prepare_context
    except Exception as exc:
        state.mark_missing(f"Dense payload export unavailable because required runtime imports failed: {exc}")
        return out_path if out_path.exists() else None

    try:
        config = load_config()
        _, sim, sensors, loaders, params, trained, payloads = prepare_context(config)
    except Exception as exc:
        state.mark_missing(f"Dense payload export failed while preparing context/evaluating models: {exc}")
        return out_path if out_path.exists() else None

    if not payloads:
        state.mark_missing("Dense payload export produced no model payloads from saved checkpoints.")
        return out_path if out_path.exists() else None

    chosen_name = next((name for name in EXPORT_MODEL_PRIORITY if name in payloads), next(iter(payloads)))
    chosen = payloads[chosen_name]
    true = _flatten_payload_array(chosen["true"])
    pred = _flatten_payload_array(chosen["pred"])
    if true.shape != pred.shape or true.ndim != 3 or true.shape[-1] < 4:
        state.mark_missing(f"Dense payload export found incompatible true/pred shapes: {true.shape} vs {pred.shape}.")
        return out_path if out_path.exists() else None

    arrays: dict[str, np.ndarray] = {
        "Ts_reference": true[:, :, 0],
        "Ts_prediction": pred[:, :, 0],
        "Tr_reference": true[:, :, 1],
        "Tr_prediction": pred[:, :, 1],
        "H_reference": true[:, :, 2],
        "H_prediction": pred[:, :, 2],
        "q_reference": true[:, :, 3],
        "q_prediction": pred[:, :, 3],
        "time_index": np.arange(true.shape[0], dtype=np.float32),
        "distance_km": np.asarray(sim.get("x_m", np.linspace(0, 20000, true.shape[1])), dtype=np.float32) / 1000.0,
        "sensor_nodes": np.asarray(sensors.get("sensor_nodes", []), dtype=np.int64),
        "chosen_model": np.asarray([chosen_name]),
    }
    heat_load = np.asarray(chosen.get("heat_load_kw", []))
    if heat_load.size:
        arrays["heat_load_kw"] = heat_load.reshape(-1).astype(np.float32)

    for model_name, payload in payloads.items():
        model_pred = _flatten_payload_array(payload["pred"])
        if model_pred.shape == true.shape:
            key = safe_key(model_name)
            arrays[f"Ts_prediction_{key}"] = model_pred[:, :, 0]
            arrays[f"Tr_prediction_{key}"] = model_pred[:, :, 1]
            arrays[f"H_prediction_{key}"] = model_pred[:, :, 2]
            arrays[f"q_prediction_{key}"] = model_pred[:, :, 3]

    np.savez_compressed(out_path, **arrays)
    state.note(
        f"Exported dense paired reconstruction arrays to {rel(out_path)} "
        f"from model '{chosen_name}' with shape {true.shape}."
    )
    state.add_prov(
        "DenseArrayExport",
        "paired true/pred arrays",
        out_path,
        "Ts_reference/Ts_prediction, Tr_reference/Tr_prediction, H_reference/H_prediction, q_reference/q_prediction",
        "Arrays exported from prepare_context + evaluate_model(return_predictions=True); no raster or synthetic surface used.",
        "Reference arrays are calibrated-simulator hidden states; predictions are saved-checkpoint reconstructions on the test loader.",
        "calibrated_simulator + simulator_assisted_hidden_state",
        "generated",
        out_path,
        f"Chosen reconstruction model for paired figure surfaces: {chosen_name}",
    )
    return out_path


def _torch_load(path: Path) -> Any | None:
    try:
        import torch  # type: ignore

        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _to_numpy(obj: Any) -> np.ndarray | None:
    if obj is None:
        return None
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass
    try:
        arr = np.asarray(obj)
    except Exception:
        return None
    if arr.ndim == 0 or arr.dtype == object:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


def flatten_mapping(obj: Any, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            arr = _to_numpy(v)
            if arr is not None:
                out[name] = arr
            elif isinstance(v, dict):
                out.update(flatten_mapping(v, name))
    return out


def load_array_file(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    if path.suffix.lower() == ".npz":
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    arr = _to_numpy(data[key])
                    if arr is not None:
                        arrays[key] = arr
        except Exception:
            return {}
    elif path.suffix.lower() == ".npy":
        try:
            arr = np.load(path, allow_pickle=False)
            arr2 = _to_numpy(arr)
            if arr2 is not None:
                arrays[path.stem] = arr2
        except Exception:
            return {}
    elif path.suffix.lower() in {".pt", ".pth"}:
        loaded = _torch_load(path)
        arrays = flatten_mapping(loaded)
    return arrays


def find_dense_pair(quantity: str, state: BuildState) -> dict[str, Any] | None:
    """Find paired 2D reference/prediction arrays for a quantity."""
    key_sets = {
        "Ts": {
            "ref": [
                "ts_reference",
                "ts_ref",
                "ts_true",
                "supply_reference",
                "supply_ref",
                "supply_true",
                "y_reference_ts",
                "reference_ts",
            ],
            "pred": [
                "ts_prediction",
                "ts_pred",
                "supply_prediction",
                "supply_pred",
                "pred_ts",
                "prediction_ts",
                "y_pred_ts",
            ],
        },
        "Tr": {
            "ref": [
                "tr_reference",
                "tr_ref",
                "tr_true",
                "return_reference",
                "return_ref",
                "return_true",
                "y_reference_tr",
                "reference_tr",
            ],
            "pred": [
                "tr_prediction",
                "tr_pred",
                "return_prediction",
                "return_pred",
                "pred_tr",
                "prediction_tr",
                "y_pred_tr",
            ],
        },
        "H": {
            "ref": [
                "h_reference",
                "h_ref",
                "head_reference",
                "head_ref",
                "head_true",
                "pressure_head_reference",
            ],
            "pred": [
                "h_prediction",
                "h_pred",
                "head_prediction",
                "head_pred",
                "prediction_head",
                "pred_head",
            ],
        },
        "q": {
            "ref": [
                "q_reference",
                "q_ref",
                "flow_reference",
                "flow_ref",
                "flow_true",
                "mass_flow_reference",
            ],
            "pred": [
                "q_prediction",
                "q_pred",
                "flow_prediction",
                "flow_pred",
                "prediction_flow",
                "pred_flow",
            ],
        },
    }
    distance_keys = {"distance_km", "x_km", "node_distance_km", "distance_m", "x_m"}
    time_keys = {"time_h", "time_hours", "t_h", "time", "timestamp_index"}
    wanted = key_sets[quantity]

    candidates = candidate_dense_files()
    loadable_artifacts: list[Path] = []
    pt_candidates = [p for p in candidates if p.suffix.lower() in {".pt", ".pth"}]
    torch_available = False
    if pt_candidates:
        try:
            import torch  # noqa: F401

            torch_available = True
        except Exception:
            torch_available = False

    for path in candidates:
        arrays = load_array_file(path)
        if not arrays:
            continue
        loadable_artifacts.append(path)
        lower = {k.lower().replace(" ", "_"): k for k in arrays}

        ref_key = next((lower[k] for k in wanted["ref"] if k in lower), None)
        pred_key = next((lower[k] for k in wanted["pred"] if k in lower), None)
        if ref_key is None or pred_key is None:
            continue

        ref = np.asarray(arrays[ref_key], dtype=float)
        pred = np.asarray(arrays[pred_key], dtype=float)
        if ref.shape != pred.shape or ref.ndim != 2:
            continue

        d_key = next((lower[k] for k in distance_keys if k in lower), None)
        t_key = next((lower[k] for k in time_keys if k in lower), None)
        distance = None
        time = None
        if d_key is not None:
            distance = np.asarray(arrays[d_key], dtype=float).ravel()
            if distance.size == ref.shape[1] and np.nanmax(distance) > 100:
                distance = distance / 1000.0
        if time is None and t_key is not None:
            time = np.asarray(arrays[t_key]).ravel()
        if distance is None or distance.size != ref.shape[1]:
            distance = np.linspace(0, 20, ref.shape[1])
        if time is None or len(time) != ref.shape[0]:
            time = np.arange(ref.shape[0])
        return {
            "path": path,
            "ref_key": ref_key,
            "pred_key": pred_key,
            "arrays": arrays,
            "ref": ref,
            "pred": pred,
            "distance_km": distance,
            "time": time,
        }

    state.mark_missing(
        f"No paired 2D {quantity} reference/prediction arrays found. Expected shape: "
        "[time, node] with keys like "
        f"{wanted['ref'][0]}/{wanted['pred'][0]} in NPZ/NPY/PT artifacts. "
        f"Found {len(candidates)} numeric artifact candidates "
        f"({len(pt_candidates)} PT/PTH, {len(candidates) - len(pt_candidates)} NPY/NPZ). "
        f"Loaded {len(loadable_artifacts)} artifacts with numeric arrays. "
        f"Torch available for PT/PTH inspection: {torch_available}. "
        "No loadable artifact contained paired 2D reference/prediction keys with matching shapes."
    )
    return None


def choose_profile_indices(n_time: int) -> tuple[int, int]:
    processed = DATA / "processed" / "sonderborg_processed.csv"
    if processed.exists():
        try:
            df = pd.read_csv(processed)
            if "heat_load_kw" in df.columns and len(df) >= n_time:
                heat = pd.to_numeric(df["heat_load_kw"].iloc[:n_time], errors="coerce").to_numpy()
                normal = int(np.nanargmin(np.abs(heat - np.nanmedian(heat))))
                difficult = int(np.nanargmax(heat))
                return normal, difficult
        except Exception:
            pass
    return max(0, n_time // 3), max(0, (2 * n_time) // 3)


def model_prediction_series(pair: dict[str, Any], quantity: str) -> list[tuple[str, np.ndarray, str, str]]:
    arrays: dict[str, np.ndarray] = pair.get("arrays", {})
    prefix = f"{quantity}_prediction_"
    wanted = [
        ("GRU-MSE", "gru_mse", PALETTE["GRU-MSE"], "-."),
        ("Transformer-MSE", "transformer_mse", PALETTE["Transformer-MSE"], ":"),
        ("PureGNN-MSE", "puregnn_mse", PALETTE["PureGNN-MSE"], (0, (3, 1, 1, 1))),
        ("PI-GNN-GRU-v3", "proposed_pi_gnn_gru_v3_balanced_mode", PALETTE["PI-GNN-GRU-v3"], "--"),
        ("PI-GNN-GRU-v3", "proposed_pi_gnn_gru_v3_accuracy_mode", PALETTE["PI-GNN-GRU-v3"], "--"),
    ]
    out: list[tuple[str, np.ndarray, str, str]] = []
    used_labels: set[str] = set()
    lower = {k.lower(): k for k in arrays}
    for label, suffix, color, ls in wanted:
        key = lower.get((prefix + suffix).lower())
        if key is None or label in used_labels:
            continue
        arr = np.asarray(arrays[key], dtype=float)
        if arr.shape == pair["ref"].shape:
            out.append((label, arr, color, ls))
            used_labels.add(label)
    if not out:
        out.append(("Reconstructed", pair["pred"], PALETTE["PI-GNN-GRU-v3"], "--"))
    return out


def plot_temperature_profile(quantity: str, label: str, units: str, stem: str, fig_id: str, state: BuildState) -> None:
    pair = find_dense_pair(quantity, state)
    if pair is None:
        for panel in ("normal-load profile", "high-demand profile", "3D absolute-error surface"):
            state.add_prov(
                fig_id,
                panel,
                "not generated",
                "required paired arrays absent",
                "none",
                "strict no synthetic/raster replacement rule",
                "simulator_assisted_hidden_state",
                "skipped",
                "",
                "Dense profile/surface figure skipped because paired arrays are unavailable.",
            )
        return

    ref = pair["ref"]
    pred = pair["pred"]
    err = np.abs(ref - pred)
    x = pair["distance_km"]
    t = np.arange(ref.shape[0])
    normal_idx, difficult_idx = choose_profile_indices(ref.shape[0])

    fig = plt.figure(figsize=(10.5, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")

    for ax, idx, title, panel in [
        (ax1, normal_idx, "Normal-load profile", "(a)"),
        (ax2, difficult_idx, "High-demand profile", "(b)"),
    ]:
        ax.plot(x, ref[idx], color=PALETTE["Reference"], lw=2.0, label="Reference")
        for label_model, pred_arr, color, ls in model_prediction_series(pair, quantity):
            ax.plot(x, pred_arr[idx], color=color, lw=1.65 if label_model != "PI-GNN-GRU-v3" else 2.05, ls=ls, label=label_model)
        ax.set_title(title)
        ax.set_xlabel("Distance from source (km)")
        ax.set_ylabel(f"{label} ({units})")
        set_grid(ax)
        panel_label(ax, panel)

    T, X = np.meshgrid(t, x, indexing="ij")
    stride_t = max(1, ref.shape[0] // 80)
    stride_x = max(1, ref.shape[1] // 40)
    surf = ax3.plot_surface(
        X[::stride_t, ::stride_x],
        T[::stride_t, ::stride_x],
        err[::stride_t, ::stride_x],
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax3.set_title("Absolute-error surface")
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Time index")
    ax3.set_zlabel(f"|Error| ({units})")
    panel_label(ax3, "(c)")
    fig.colorbar(surf, ax=ax3, shrink=0.62, pad=0.08, label=f"|Error| ({units})")
    handles, labels = ax1.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=min(5, len(unique)), frameon=False, bbox_to_anchor=(0.38, 1.07))
    outputs = save_figure(fig, stem, state)
    for panel, keys in [
        ("normal-load profile", f"{pair['ref_key']}, {pair['pred_key']} at t={normal_idx}"),
        ("high-demand profile", f"{pair['ref_key']}, {pair['pred_key']} at t={difficult_idx}"),
        ("3D absolute-error surface", f"E=abs({pair['ref_key']}-{pair['pred_key']})"),
    ]:
        state.add_prov(
            fig_id,
            panel,
            pair["path"],
            keys,
            "Profiles use paired reference/reconstruction arrays; errors computed as absolute difference.",
            "Distance defaults to 0-20 km if no distance array with matching node count is stored.",
            "simulator_assisted_hidden_state",
            "generated",
            outputs,
        )


def plot_field_reconstruction(quantity: str, label: str, units: str, stem: str, fig_id: str, state: BuildState) -> None:
    pair = find_dense_pair(quantity, state)
    if pair is None:
        for panel in ("reference field", "reconstructed field", "absolute-error field"):
            state.add_prov(
                fig_id,
                panel,
                "not generated",
                "required paired arrays absent",
                "none",
                "strict no synthetic/raster replacement rule",
                "simulator_assisted_hidden_state",
                "skipped",
                "",
                "Dense field figure skipped because paired arrays are unavailable.",
            )
        return
    ref = pair["ref"]
    pred = pair["pred"]
    err = np.abs(ref - pred)
    x = pair["distance_km"]
    y = np.arange(ref.shape[0])
    vmin = float(np.nanmin([np.nanmin(ref), np.nanmin(pred)]))
    vmax = float(np.nanmax([np.nanmax(ref), np.nanmax(pred)]))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
    panels = [
        (axes[0], ref, "Reference field", "(a)", vmin, vmax, "viridis"),
        (axes[1], pred, "Reconstructed field", "(b)", vmin, vmax, "viridis"),
        (axes[2], err, "Absolute-error field", "(c)", 0, float(np.nanmax(err)), "magma"),
    ]
    for ax, arr, title, lab, lo, hi, cmap in panels:
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            extent=[float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))],
            vmin=lo,
            vmax=hi,
            cmap=cmap,
        )
        ax.set_title(title)
        ax.set_xlabel("Distance from source (km)")
        ax.set_ylabel("Time index")
        panel_label(ax, lab)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label(f"{label} ({units})" if title != "Absolute-error field" else f"|Error| ({units})")
    outputs = save_figure(fig, stem, state)
    for panel, keys in [
        ("reference field", pair["ref_key"]),
        ("reconstructed field", pair["pred_key"]),
        ("absolute-error field", f"E=abs({pair['ref_key']}-{pair['pred_key']})"),
    ]:
        state.add_prov(
            fig_id,
            panel,
            pair["path"],
            keys,
            "Field panels use paired 2D arrays; reference and reconstruction share identical color scales.",
            "Hydraulic quantities are simulator-assisted hidden states unless real dense hydraulic arrays are explicitly provided.",
            "simulator_assisted_hidden_state",
            "generated",
            outputs,
        )


def parse_sensor_nodes() -> dict[str, list[int]]:
    layouts: dict[str, list[int]] = {}
    p = RESULTS / "sensor_layout_definitions_table.csv"
    if p.exists():
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            layout = str(row.get("layout", ""))
            nodes_text = str(row.get("representative_nodes", ""))
            nodes = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", nodes_text)]
            if layout:
                layouts[layout] = nodes
    p2 = RESULTS / "sensor_layout_comparison_final.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        for _, row in df.iterrows():
            layout = str(row.get("sensor_layout", ""))
            nodes_text = str(row.get("sensor_nodes", ""))
            nodes = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", nodes_text)]
            if layout and nodes:
                layouts.setdefault(layout, nodes)
    return layouts


def plot_network_schematic(state: BuildState) -> None:
    fig_id = "Fig05"
    layouts = parse_sensor_nodes()
    s3 = layouts.get("S3_inlet_middle_outlet", [0, 10, 20])
    opt = layouts.get("S9_optimized_three_sensors") or layouts.get("S10_optimized_five_sensors") or []
    max_node = max([20] + s3 + opt)
    length_km = 20.0
    def node_x(n: int) -> float:
        return length_km * n / max(max_node, 1)

    fig, ax = plt.subplots(figsize=(10.4, 3.4), constrained_layout=True)
    ax.set_xlim(-1.2, 21.2)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    ax.set_title("Physical 20 km district-heating system and sparse sensors")
    ax.plot([0, 20], [0.45, 0.45], color=PALETTE["Orange"], lw=5, solid_capstyle="round")
    ax.plot([20, 0], [-0.45, -0.45], color=PALETTE["Blue"], lw=5, solid_capstyle="round")
    ax.annotate("", xy=(19.0, 0.45), xytext=(16.0, 0.45), arrowprops=dict(arrowstyle="->", lw=1.8, color=PALETTE["Orange"]))
    ax.annotate("", xy=(1.0, -0.45), xytext=(4.0, -0.45), arrowprops=dict(arrowstyle="->", lw=1.8, color=PALETTE["Blue"]))
    ax.text(10, 0.68, "Supply pipe: measured/simulated temperature field", ha="center", va="bottom", color=PALETTE["Orange"], fontweight="bold")
    ax.text(10, -0.68, "Return pipe: measured/simulated temperature field", ha="center", va="top", color=PALETTE["Blue"], fontweight="bold")
    ax.add_patch(plt.Rectangle((-0.82, -0.85), 1.18, 1.7, facecolor="#F4F4F4", edgecolor=PALETTE["Reference"], lw=1.4))
    ax.text(-0.23, 0.0, "Heat\nsource\n+pump", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.add_patch(plt.Rectangle((19.65, -0.85), 1.18, 1.7, facecolor="#F4F4F4", edgecolor=PALETTE["Reference"], lw=1.4))
    ax.text(20.24, 0.0, "Load /\nconsumer", ha="center", va="center", fontsize=9, fontweight="bold")
    for x in np.linspace(2, 18, 7):
        ax.annotate("", xy=(x, 0.12), xytext=(x, 0.38), arrowprops=dict(arrowstyle="->", lw=1.0, color=PALETTE["HeatLoss"]))
        ax.text(x, 0.06, "heat loss", ha="center", va="top", fontsize=6.7, color="#6B6400")
    for i, n in enumerate(s3):
        x = node_x(n)
        ax.plot([x], [0.45], marker="o", ms=10, color=PALETTE["Magenta"], mec="#111111", mew=0.8, zorder=5)
        ax.text(x, 0.98 if i != 1 else 1.12, f"S3 node {n}", ha="center", va="bottom", fontsize=8.3, color=PALETTE["Magenta"])
    for n in opt:
        x = node_x(n)
        ax.plot([x], [-0.45], marker="s", ms=8, color=PALETTE["Green"], mec="#111111", mew=0.8, zorder=5)
    if opt:
        ax.text(10, -1.05, "Optimized layout sensors shown as green squares", ha="center", va="center", fontsize=8.5, color="#236000")
    ax.plot(np.linspace(1, 19, 16), np.full(16, 0.0), marker=".", ls="", color="#AAAAAA", ms=5)
    ax.text(10, 0.05, "Virtual internal nodes", ha="center", va="bottom", fontsize=8.0, color="#666666")
    ax.text(
        10,
        -1.22,
        "Pressure/head and flow are simulator-assisted hidden hydraulic states.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=PALETTE["Reference"],
    )
    ax.annotate("20 km", xy=(0, -0.98), xytext=(20, -0.98), arrowprops=dict(arrowstyle="<->", lw=1.1, color=PALETTE["Reference"]), ha="center", va="center")
    outputs = save_figure(fig, "fig05_physical_20km_network_schematic", state)
    state.add_prov(
        fig_id,
        "network topology and sensors",
        RESULTS / "sensor_layout_definitions_table.csv",
        "layout, representative_nodes",
        "Node indices converted to distance using 20 km / max_node.",
        "Twenty-kilometre pipe length follows the study network definition; hidden hydraulic labels are simulator-assisted.",
        "calibrated_simulator + simulator_assisted_hidden_state",
        "generated",
        outputs,
    )


def plot_framework(state: BuildState) -> None:
    fig_id = "Fig06"
    data_report = RESULTS / "data_availability_report.csv"
    calib = RESULTS / "calibration_metrics.csv"
    fig, ax = plt.subplots(figsize=(11.2, 4.1), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Real-data-assisted digital-twin framework and evidence boundary")

    boxes = [
        (0.03, 0.58, 0.17, 0.24, "Real operating data\nSønderborg + Flensburg\nmeasured plant/substation nodes", PALETTE["Reference"]),
        (0.25, 0.58, 0.15, 0.24, "Preprocessing\nunit checks, gaps,\ntrain/validation/test", PALETTE["Gray"]),
        (0.45, 0.58, 0.16, 0.24, "Calibrated\nthermo-hydraulic\nsimulator", PALETTE["Orange"]),
        (0.66, 0.58, 0.14, 0.24, "Sparse-sensor\nobservations", PALETTE["Magenta"]),
        (0.84, 0.58, 0.13, 0.24, "Benchmark models\nGRU / Transformer /\nPI-GNN-GRU-v3", PALETTE["Blue"]),
        (0.18, 0.15, 0.20, 0.23, "Measured-node validation\nsupply/return temperature,\nheat/energy variables", PALETTE["Green"]),
        (0.43, 0.15, 0.21, 0.23, "Simulator-assisted\nhidden-state reconstruction\nT, H, q, heat loss", PALETTE["Orange"]),
        (0.70, 0.15, 0.20, 0.23, "Uncertainty, anomaly,\nand operational KPI layer", PALETTE["Magenta"]),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color + "18", edgecolor=color, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center", fontsize=8.7, color=PALETTE["Reference"])
    arrow_pairs = [(0.20, 0.70, 0.25, 0.70), (0.40, 0.70, 0.45, 0.70), (0.61, 0.70, 0.66, 0.70), (0.80, 0.70, 0.84, 0.70), (0.52, 0.58, 0.28, 0.38), (0.52, 0.58, 0.535, 0.38), (0.91, 0.58, 0.80, 0.38)]
    for x0, y0, x1, y1 in arrow_pairs:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.5, color=PALETTE["Reference"]))
    ax.text(0.50, 0.02, "Evidence boundary: real measured-node validation is separated from calibrated-simulator and simulator-assisted hidden-state evaluation.", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.7, fontweight="bold")
    outputs = save_figure(fig, "fig06_real_data_assisted_dt_framework", state)
    state.add_prov(
        fig_id,
        "workflow",
        f"{rel(data_report)}; {rel(calib)}",
        "dataset_name, available; calibration metrics",
        "Conceptual workflow drawn from repository data/status artifacts only.",
        "Boxes separate measured-node evidence, calibrated simulator outputs, and simulator-assisted hidden states.",
        "real_measured_node + calibrated_simulator + simulator_assisted_hidden_state",
        "generated",
        outputs,
    )


def plot_heat_energy(state: BuildState) -> None:
    fig_id = "Fig07"
    e_path = RESULTS / "energy_balance_time_series.csv"
    h_path = RESULTS / "heat_loss_profile_metrics.csv"
    impact_path = RESULTS / "operational_energy_impact_timeseries.csv"
    if not e_path.exists() or not h_path.exists():
        state.mark_missing("Figure 7 requires energy_balance_time_series.csv and heat_loss_profile_metrics.csv.")
        return
    energy = pd.read_csv(e_path)
    heat = pd.read_csv(h_path)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2), constrained_layout=True)
    step = np.arange(len(energy))
    ax = axes[0, 0]
    ax.plot(step, energy["measured_boundary_heat_load_kw"] / 1000, color=PALETTE["Reference"], lw=1.6, label="Measured boundary load")
    ax.plot(step, energy["simulator_delivered_heat_kw"] / 1000, color=PALETTE["Orange"], lw=1.3, label="Simulator delivered")
    ax.plot(step, energy["pignn_v3_delivered_heat_kw"] / 1000, color=PALETTE["Magenta"], lw=1.3, label="PI-GNN-GRU-v3 estimate")
    ax.set_ylabel("Heat rate (MW)")
    ax.set_xlabel("Time index")
    ax.set_title("Delivered heat tracking")
    set_grid(ax)
    panel_label(ax, "(a)")
    ax = axes[0, 1]
    ax.bar(heat["segment_midpoint_km"], heat["simulator_heat_loss_kW"], width=0.75, color=PALETTE["HeatLoss"], edgecolor=PALETTE["Reference"], label="Simulator")
    ax.plot(heat["segment_midpoint_km"], heat["pignn_v3_heat_loss_kW"], color=PALETTE["Magenta"], marker="o", ms=3, lw=1.4, label="PI-GNN-GRU-v3")
    ax.set_xlabel("Distance from source (km)")
    ax.set_ylabel("Segment heat loss (kW)")
    ax.set_title("Segment heat-loss profile")
    set_grid(ax)
    panel_label(ax, "(b)")
    ax = axes[1, 0]
    cum_sim = np.cumsum(heat["simulator_heat_loss_kW"])
    cum_pred = np.cumsum(heat["pignn_v3_heat_loss_kW"])
    ax.plot(heat["segment_midpoint_km"], cum_sim, color=PALETTE["Reference"], lw=1.8, label="Simulator cumulative")
    ax.plot(heat["segment_midpoint_km"], cum_pred, color=PALETTE["Magenta"], lw=1.8, ls="--", label="PI-GNN-GRU-v3 cumulative")
    ax.set_xlabel("Distance from source (km)")
    ax.set_ylabel("Cumulative heat loss (kW)")
    ax.set_title("Cumulative heat loss")
    set_grid(ax)
    panel_label(ax, "(c)")
    ax = axes[1, 1]
    if impact_path.exists():
        impact = pd.read_csv(impact_path)
        y = numeric_series(impact, "energy_balance_residual_percent")
        x = np.arange(len(y))
        label = "Energy residual (%)"
    else:
        y = energy["pignn_v3_energy_residual_kw"] / energy["measured_boundary_heat_load_kw"].replace(0, np.nan) * 100
        x = step
        label = "Energy residual (%)"
    ax.plot(x, y, color=PALETTE["Blue"], lw=1.3)
    ax.axhline(0, color=PALETTE["Reference"], lw=0.8)
    ax.set_xlabel("Time index")
    ax.set_ylabel(label)
    ax.set_title("Energy-balance residual")
    set_grid(ax)
    panel_label(ax, "(d)")
    handles, labels = [], []
    for a in axes.ravel()[:3]:
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    outputs = save_figure(fig, "fig07_heat_loss_delivered_heat_energy_balance", state)
    for panel, src, cols, evidence in [
        ("delivered heat tracking", e_path, "measured_boundary_heat_load_kw, simulator_delivered_heat_kw, pignn_v3_delivered_heat_kw", "real_measured_node + calibrated_simulator"),
        ("segment heat loss", h_path, "segment_midpoint_km, simulator_heat_loss_kW, pignn_v3_heat_loss_kW", "calibrated_simulator"),
        ("cumulative heat loss", h_path, "cumulative sum of simulator_heat_loss_kW and pignn_v3_heat_loss_kW", "calibrated_simulator"),
        ("energy residual", impact_path if impact_path.exists() else e_path, "energy_balance_residual_percent or pignn_v3_energy_residual_kw / measured load", "calibrated_simulator"),
    ]:
        state.add_prov(fig_id, panel, src, cols, "Direct plotting and cumulative sums only; no smoothing.", "Heat-loss labels are calibrated-simulator quantities.", evidence, "generated", outputs)


def normalized_physics_score(df: pd.DataFrame) -> pd.Series:
    metrics = ["heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean", "thermal_residual_mean"]
    vals = []
    for m in metrics:
        s = numeric_series(df, m).abs()
        denom = s.max() - s.min()
        vals.append((s - s.min()) / (denom if denom > 0 else 1.0))
    return pd.concat(vals, axis=1).mean(axis=1)


def plot_model_ranking(state: BuildState) -> None:
    fig_id = "Fig08"
    b_path = RESULTS / "baseline_comparison_final.csv"
    rank_path = RESULTS / "concept_model_value_rank_matrix.csv"
    if not b_path.exists() or not rank_path.exists():
        state.mark_missing("Figure 8 requires baseline_comparison_final.csv and concept_model_value_rank_matrix.csv.")
        return
    base = pd.read_csv(b_path)
    ranks = pd.read_csv(rank_path)
    ranks["model_clean"] = ranks["model"].map(clean_model_name)
    keep_models = ["GRU-MSE", "Transformer-MSE", "PureGNN-MSE", "PI-LSTM", "PI-GNN-GRU-v3", "Interpolation"]
    base["model_clean"] = base["model"].map(clean_model_name)

    # Fill missing concept-matrix cells from the final baseline table where a
    # one-to-one metric exists.  This is a script-derived rank from existing
    # CSV values, not a manual or fabricated result.
    fallback_map = {
        "Supply-temperature RMSE": "RMSE_Ts_full",
        "Return-temperature RMSE": "RMSE_Tr_full",
        "Heat-loss error": "heat_loss_error_percent",
        "Energy-balance residual": "energy_balance_residual",
        "Boundary residual": "boundary_residual_mean",
    }
    extra_rows = []
    for metric, col in fallback_map.items():
        if col not in base.columns:
            continue
        collapsed = base[base["model_clean"].isin(keep_models)].groupby("model_clean", as_index=False)[col].min()
        collapsed["rank"] = collapsed[col].rank(method="min", ascending=True)
        for _, row in collapsed.iterrows():
            already = (
                (ranks["metric"].astype(str) == metric)
                & (ranks["model_clean"].astype(str) == row["model_clean"])
            ).any()
            if not already:
                extra_rows.append(
                    {
                        "metric": metric,
                        "model": row["model_clean"],
                        "model_clean": row["model_clean"],
                        "value": row[col],
                        "rank": row["rank"],
                        "source": f"{b_path.name}; script-derived rank fallback",
                    }
                )
    if extra_rows:
        ranks = pd.concat([ranks, pd.DataFrame(extra_rows)], ignore_index=True)
    ranks = ranks[ranks["model_clean"].isin(keep_models)].copy()
    metrics_order = list(dict.fromkeys(ranks["metric"].astype(str)))
    pivot = ranks.pivot_table(index="model_clean", columns="metric", values="rank", aggfunc="min").reindex(keep_models)
    pivot = pivot[[m for m in metrics_order if m in pivot.columns]]

    scatter = base[base["model_clean"].isin(keep_models)].copy()
    scatter["thermal_rmse"] = numeric_series(scatter, "RMSE_Ts_full") + numeric_series(scatter, "RMSE_Tr_full")
    scatter["physics_score"] = normalized_physics_score(scatter)
    scatter = (
        scatter.sort_values(["model_clean", "physics_score", "thermal_rmse"])
        .drop_duplicates("model_clean", keep="first")
        .copy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    matrix = pivot.to_numpy(dtype=float)
    cmap = ListedColormap(["#2A9D8F", "#8BD17C", "#F4E285", "#F4A261", "#E76F51", "#B94E48", "#6D6875"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=1, vmax=max(6, np.nanmax(matrix)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_title("Metric-dependent model ranks (1 = best)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=7.5, color=PALETTE["Reference"])
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.7, color="#777777")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Rank")
    panel_label(ax, "(a)")

    ax = axes[1]
    for _, row in scatter.iterrows():
        name = row["model_clean"]
        color = PALETTE.get(name, PALETTE["Gray"])
        size = 88 if name == "PI-GNN-GRU-v3" else 58
        ax.scatter(row["thermal_rmse"], row["physics_score"], s=size, color=color, edgecolor=PALETTE["Reference"], zorder=3, label=name)
    ax.set_xlabel("Direct thermal RMSE sum, Ts + Tr (°C)")
    ax.set_ylabel("Normalized physical-consistency score (lower is better)")
    ax.set_title("Accuracy-physics trade-off")
    ymin = float(scatter["physics_score"].min())
    ymax = float(scatter["physics_score"].max())
    pad = max(0.06, 0.12 * (ymax - ymin))
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    xmin = float(scatter["thermal_rmse"].min())
    xmax = float(scatter["thermal_rmse"].max())
    xpad = max(0.08, 0.05 * (xmax - xmin))
    ax.set_xlim(xmin - xpad, xmax + xpad)
    set_grid(ax)
    panel_label(ax, "(b)")
    ax.legend(loc="lower right", frameon=True, fontsize=7.1, ncol=1)
    outputs = save_figure(fig, "fig08_model_ranking_heatmap_accuracy_physics_tradeoff", state)
    state.add_prov(fig_id, "rank heatmap", rank_path, "metric, model, rank", "Pivoted rank matrix; no rescaling of rank values.", "Lower rank is better; values come from project rank matrix.", "calibrated_simulator + real_measured_node", "generated", outputs)
    state.add_prov(fig_id, "accuracy-physics scatter", b_path, "RMSE_Ts_full, RMSE_Tr_full, heat_loss_error_percent, energy_balance_residual, boundary_residual_mean, thermal_residual_mean", "Thermal RMSE=sum(Ts,Tr); physics score=mean min-max-normalized absolute consistency metrics; duplicate v3 modes collapsed to the lowest physics-score row for readability.", "Score is for visualization only and does not replace raw table values.", "calibrated_simulator + simulator_assisted_hidden_state", "generated", outputs)


def plot_external_validation(state: BuildState) -> None:
    fig_id = "Fig09"
    f_ts_path = RESULTS / "external_validation_flensburg_timeseries.csv"
    f_modes_path = RESULTS / "external_validation_flensburg_modes_final.csv"
    shift_path = RESULTS / "flensburg_domain_shift_analysis.csv"
    xai_path = RESULTS / "xai4heat_sparse_substation_validation_final.csv"
    needed = [f_ts_path, f_modes_path, shift_path, xai_path]
    if not all(p.exists() for p in needed):
        state.mark_missing("Figure 9 requires Flensburg timeseries/modes/domain-shift CSVs and XAI4HEAT measured-node validation CSV.")
        return
    f_ts = pd.read_csv(f_ts_path)
    modes = pd.read_csv(f_modes_path)
    shift = pd.read_csv(shift_path)
    xai = pd.read_csv(xai_path)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6), constrained_layout=True)
    ax = axes[0, 0]
    x = numeric_series(f_ts, "window_step", np.arange(len(f_ts)))
    ax.plot(x, f_ts["measured_or_boundary_supply_C"], color=PALETTE["Reference"], lw=1.8, marker="o", ms=3, label="Measured/boundary supply")
    ax.plot(x, f_ts["predicted_supply_C"], color=PALETTE["Magenta"], lw=1.7, marker="s", ms=3, label="Predicted supply")
    ax.plot(x, f_ts["simulator_hidden_supply_C"], color=PALETTE["Orange"], lw=1.3, ls="--", label="Simulator hidden supply")
    ax.set_xlabel("Flensburg validation window")
    ax.set_ylabel("Supply temperature (°C)")
    ax.set_title("Flensburg measured-node supply transfer")
    set_grid(ax)
    panel_label(ax, "(a)")
    ax.legend(loc="upper right", frameon=True, fontsize=7.2)

    ax = axes[0, 1]
    plot_modes = modes.copy()
    mode_map = {
        "direct_transfer": "Direct\ntransfer",
        "calibration_only_offset_adaptation": "Calib.-offset\n(no ML retrain)",
        "few_shot_decoder_bias_adaptation": "Few-shot\ndecoder",
        "normalized_transfer_flensburg_boundary_statistics": "Normalized\nboundary stats",
    }
    plot_modes["mode_label"] = plot_modes["mode"].map(mode_map).fillna(plot_modes["mode"].astype(str).str.replace("_", "\n"))
    xpos = np.arange(len(plot_modes))
    width = 0.36
    ax.bar(xpos - width / 2, plot_modes["RMSE_supply_measured_C"], width, color=PALETTE["Blue"], edgecolor=PALETTE["Reference"], label="Supply RMSE")
    ax.bar(xpos + width / 2, plot_modes["RMSE_return_measured_C"], width, color=PALETTE["Orange"], edgecolor=PALETTE["Reference"], alpha=0.76, label="Return/proxy RMSE")
    ax.set_xticks(xpos)
    ax.set_xticklabels(plot_modes["mode_label"])
    ax.set_ylabel("Measured-node RMSE (°C)")
    ax.set_title("Transfer modes")
    set_grid(ax)
    panel_label(ax, "(b)")
    ax.legend(loc="upper left", frameon=True, fontsize=7.2)

    ax = axes[1, 0]
    key_metrics = shift[shift["metric"].astype(str).str.contains("mean_heat_load|mean_supply|sampling|return|difference", case=False, regex=True)].copy()
    key_metrics = key_metrics.head(7)
    labels = key_metrics["metric"].astype(str).str.replace("_", " ").str[:30]
    values = pd.to_numeric(key_metrics["value"], errors="coerce")
    numeric_mask = values.notna()
    ax.barh(labels[numeric_mask], values[numeric_mask], color=PALETTE["Green"], edgecolor=PALETTE["Reference"])
    ax.set_xlabel("Value")
    ax.set_title("Domain-shift indicators")
    set_grid(ax)
    panel_label(ax, "(c)")

    ax = axes[1, 1]
    xai_small = xai[xai["category"].astype(str).str.contains("temperature|energy", case=False, regex=True)].copy()
    if xai_small.empty:
        xai_small = xai.copy()
    metric_col = "mean_nRMSE_percent" if "mean_nRMSE_percent" in xai_small.columns else "mean_RMSE"
    xai_small = xai_small.sort_values(metric_col)
    ax.barh(xai_small["variable_label"].astype(str).str[:31], xai_small[metric_col], color=PALETTE["Magenta"], edgecolor=PALETTE["Reference"])
    ax.set_xlabel("Mean nRMSE (%)" if metric_col == "mean_nRMSE_percent" else "Mean RMSE (native units)")
    ax.set_title("XAI4HEAT measured-node validation")
    set_grid(ax)
    panel_label(ax, "(d)")
    outputs = save_figure(fig, "fig09_external_validation_domain_shift", state)
    state.add_prov(fig_id, "Flensburg supply transfer", f_ts_path, "measured_or_boundary_supply_C, predicted_supply_C, simulator_hidden_supply_C", "Direct window-wise plot; residuals are not smoothed.", "Return temperature may be assumed when unavailable per source notes.", "real_measured_node + calibrated_simulator", "generated", outputs)
    state.add_prov(fig_id, "Flensburg transfer modes", f_modes_path, "mode, RMSE_supply_measured_C, RMSE_return_measured_C", "Bar plot of reported transfer-mode metrics.", "Mode labels are shortened for display only.", "real_measured_node", "generated", outputs)
    state.add_prov(fig_id, "Flensburg domain-shift indicators", shift_path, "metric, value", "Numeric shift indicators plotted where parseable.", "Mixed-unit diagnostic panel; detailed units remain in CSV/table.", "real_measured_node", "generated", outputs)
    state.add_prov(fig_id, "XAI4HEAT measured-node validation", xai_path, f"variable_label, {metric_col}", "Measured-node validation metrics plotted directly; normalized RMSE is preferred when available to avoid mixing native units.", "XAI4HEAT pressure/head and flow are not measured; no hidden hydraulic validation is implied.", "real_measured_node", "generated", outputs)


def plot_robustness_uncertainty_anomaly(state: BuildState) -> None:
    fig_id = "Fig10"
    u_path = RESULTS / "uncertainty_quantification_metrics.csv"
    r_path = RESULTS / "thermo_hydraulic_robustness.csv"
    a_path = RESULTS / "anomaly_detection_metrics_improved.csv"
    ats_path = RESULTS / "anomaly_detection_timeseries_improved.csv"
    if not all(p.exists() for p in [u_path, r_path, a_path, ats_path]):
        state.mark_missing("Figure 10 requires uncertainty, thermo-hydraulic robustness, and anomaly detection CSVs.")
        return
    u = pd.read_csv(u_path)
    r = pd.read_csv(r_path)
    a = pd.read_csv(a_path)
    ats = pd.read_csv(ats_path)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    u_temp = u[u["quantity"].astype(str).str.contains("temperature|heat_loss|head|flow", case=False, regex=True)].copy()
    preferred = u_temp[u_temp["interval"].astype(str).str.contains("90", regex=False)].copy()
    if preferred.empty:
        preferred = u_temp.copy()
    preferred = preferred.sort_values("coverage").tail(8)
    labels = preferred["quantity"].astype(str).str.replace("_", " ").str[:18]
    ax.barh(labels, preferred["coverage"], color=PALETTE["Blue"], edgecolor=PALETTE["Reference"])
    ax.axvline(90, color=PALETTE["Orange"], ls="--", lw=1.2, label="90% target")
    ax.set_xlabel("Empirical coverage (%)")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Uncertainty coverage")
    set_grid(ax)
    panel_label(ax, "(a)")
    ax.legend(loc="lower right", fontsize=7.2, frameon=True)

    ax = axes[0, 1]
    rob = r[r["model"].astype(str).str.contains("GRU-MSE|Transformer-MSE|PI-GNN-GRU-v3", regex=True)].copy()
    rob["model_clean"] = rob["model"].map(clean_model_name)
    grouped = rob.groupby("model_clean", as_index=False).agg(
        supply_rmse=("RMSE_Ts_supply_C", "mean"),
        heat_loss=("heat_loss_error_percent", "mean"),
        energy=("energy_balance_residual", "mean"),
    )
    x = np.arange(len(grouped))
    width = 0.25
    ax.bar(x - width, grouped["supply_rmse"], width, color=PALETTE["Blue"], edgecolor=PALETTE["Reference"], label="Supply RMSE (°C)")
    ax.bar(x, grouped["heat_loss"], width, color=PALETTE["Yellow"], edgecolor=PALETTE["Reference"], label="Heat-loss error (%)")
    ax.bar(x + width, grouped["energy"], width, color=PALETTE["Magenta"], edgecolor=PALETTE["Reference"], label="Energy residual")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["model_clean"], rotation=15, ha="right")
    ax.set_title("Noise/dropout/parameter robustness")
    set_grid(ax)
    panel_label(ax, "(b)")
    ax.legend(loc="upper left", fontsize=7.0, frameon=True, ncol=1)

    ax = axes[1, 0]
    a_plot = a.copy()
    ax.barh(a_plot["case"].astype(str).str.replace("_", " ").str[:26], a_plot["detection_rate_percent"], color=PALETTE["Green"], edgecolor=PALETTE["Reference"], label="Detection")
    ax.scatter(a_plot["false_alarm_rate_percent"], np.arange(len(a_plot)), color=PALETTE["Orange"], edgecolor=PALETTE["Reference"], label="False alarms")
    ax.set_xlabel("Rate (%)")
    ax.set_title("Controlled-perturbation anomaly evidence")
    set_grid(ax)
    panel_label(ax, "(c)")
    ax.legend(loc="lower right", fontsize=7.2, frameon=True)

    ax = axes[1, 1]
    case = "normal_operation"
    if "case" in ats.columns and (ats["case"] != "normal_operation").any():
        case = str(ats.loc[ats["case"] != "normal_operation", "case"].iloc[0])
    ts = ats[ats["case"] == case].head(350).copy()
    x = np.arange(len(ts))
    score_col = "residual_score" if "residual_score" in ts.columns else "combined_score"
    ax.plot(x, ts[score_col], color=PALETTE["Magenta"], lw=1.2, label="Residual score")
    if "warning_threshold" in ts.columns:
        ax.axhline(pd.to_numeric(ts["warning_threshold"], errors="coerce").median(), color=PALETTE["Orange"], ls="--", lw=1.3, label="Warning threshold")
    if "alarm_threshold" in ts.columns:
        ax.axhline(pd.to_numeric(ts["alarm_threshold"], errors="coerce").median(), color=PALETTE["Reference"], ls=":", lw=1.3, label="Alarm threshold")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Residual score")
    ax.set_title(f"Anomaly residual trace: {case.replace('_', ' ')}")
    set_grid(ax)
    panel_label(ax, "(d)")
    ax.legend(loc="upper right", fontsize=7.0, frameon=True)
    outputs = save_figure(fig, "fig10_robustness_uncertainty_anomaly_evidence", state)
    state.add_prov(fig_id, "uncertainty coverage", u_path, "quantity, interval, mean_interval_width, coverage", "Scatter plot of saved interval coverage metrics.", "Intervals are confidence bands from saved uncertainty artifacts.", "real_measured_node + simulator_assisted_hidden_state", "generated", outputs)
    state.add_prov(fig_id, "robustness comparison", r_path, "model, RMSE_Ts_supply_C, heat_loss_error_percent, energy_balance_residual", "Mean over saved robustness conditions by model.", "Hydraulic effects are simulator-assisted where pressure/head/flow appear.", "real_measured_node + simulator_assisted_hidden_state", "generated", outputs)
    state.add_prov(fig_id, "anomaly summary", a_path, "case, detection_rate_percent, false_alarm_rate_percent", "Direct bar/scatter plot of controlled-perturbation metrics.", "Anomaly cases are controlled perturbations, not observed field fault labels.", "controlled_perturbation_of_real_profile", "generated", outputs)
    state.add_prov(fig_id, "anomaly residual trace", ats_path, "case, residual_score, warning_threshold, alarm_threshold", "First non-normal case residual trace plotted without smoothing.", "Thresholds are from saved anomaly artifact.", "controlled_perturbation_of_real_profile", "generated", outputs)


def write_reports(state: BuildState) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prov_path = OUT / "figure_provenance.csv"
    with prov_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProvenanceRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in state.provenance:
            writer.writerow(row.__dict__)

    missing_path = OUT / "missing_data_report.txt"
    with missing_path.open("w", encoding="utf-8") as f:
        if state.missing:
            f.write("Missing or skipped figure data\n")
            f.write("==============================\n\n")
            for msg in state.missing:
                f.write(f"- {msg}\n")
        else:
            f.write("No missing data detected for requested figures.\n")

    log_path = OUT / "generation_log.txt"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Repository root: {ROOT}\n")
        f.write(f"Output directory: {OUT}\n")
        f.write(f"Matplotlib backend: {matplotlib.get_backend()}\n\n")
        for line in state.log:
            f.write(line + "\n")

    readme_path = OUT / "README.md"
    generated_figs = sorted(state.outputs.keys())
    skipped = sorted(set(row.figure_id for row in state.provenance if row.generation_status == "skipped"))
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("# Main ATE Figures Regenerated From Numerical Artifacts\n\n")
        f.write("This package was generated by `regenerate_main_figures.py` using CSV/JSON/NPZ/NPY/PT numerical artifacts only. Existing raster/vector figures were not used as quantitative inputs.\n\n")
        f.write("## Generated Figures\n\n")
        for stem in generated_figs:
            f.write(f"- `{stem}.svg`, `{stem}.pdf`, `{stem}.png`\n")
        if skipped:
            f.write("\n## Skipped Figures\n\n")
            f.write("The strict paired-array rule caused these dense field figures/panels to be skipped:\n\n")
            for item in skipped:
                f.write(f"- `{item}`\n")
            f.write("\nTo enable skipped dense reconstruction figures, export paired 2D arrays with shape `[time, node]`, such as `Ts_reference/Ts_prediction`, `Tr_reference/Tr_prediction`, `head_reference/head_prediction`, and `flow_reference/flow_prediction`, plus optional `distance_km` and `time_h` arrays.\n")
        f.write("\n## ATE Figure Sufficiency Note\n\n")
        f.write("The generated set is strong for an ATE manuscript when paired dense reconstruction arrays are available. If Figures 1-4 are skipped, the paper should either export those arrays from the simulator/model pipeline or move dense field claims to supplementary text until array provenance is complete. Additional seasonal, stress-test, and parameter-sensitivity figures are useful as supplementary evidence rather than extra main figures unless the journal page budget allows them.\n")
        f.write("\n## Evidence Boundary\n\n")
        f.write("Measured-node panels use public operating data. Distributed temperature, pressure/head, flow, and heat-loss fields are simulator-assisted hidden states unless an input artifact explicitly provides real dense field measurements.\n")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(OUT)))

    # Also copy requested report names to repository root for quick discovery.
    # Keep the figure-package README inside the package directory to avoid
    # overwriting the repository's main README on future runs.
    for src in [prov_path, missing_path, log_path]:
        shutil.copy2(src, ROOT / src.name)
    shutil.copy2(readme_path, ROOT / "README_main_figures_real_data.md")


def main() -> int:
    state = BuildState()
    state.note("Starting strict main-figure regeneration from numerical artifacts only.")
    OUT.mkdir(parents=True, exist_ok=True)
    export_dense_reconstruction_payloads(state)

    # Dense figures with strict paired-array requirement.
    plot_temperature_profile("Ts", "Supply temperature", "°C", "fig01_supply_temperature_reconstruction", "Fig01", state)
    plot_temperature_profile("Tr", "Return temperature", "°C", "fig02_return_temperature_reconstruction", "Fig02", state)
    plot_field_reconstruction("H", "Head", "m", "fig03_pressure_head_reference_reconstruction_error", "Fig03", state)
    plot_field_reconstruction("q", "Flow", "m$^3$ s$^{-1}$", "fig04_flow_reference_reconstruction_error", "Fig04", state)

    # CSV-backed figures.
    plot_network_schematic(state)
    plot_framework(state)
    plot_heat_energy(state)
    plot_model_ranking(state)
    plot_external_validation(state)
    plot_robustness_uncertainty_anomaly(state)

    write_reports(state)
    state.note(f"Wrote package zip: {ZIP_PATH}")
    print(f"Generated {len(state.outputs)} successful figure groups in {OUT}")
    print(f"Skipped/missing items: {len(state.missing)}")
    print(f"Provenance: {OUT / 'figure_provenance.csv'}")
    print(f"Missing data report: {OUT / 'missing_data_report.txt'}")
    print(f"Zip: {ZIP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
