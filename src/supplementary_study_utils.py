from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

for _font_file in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    _font_path = Path("C:/Windows/Fonts") / _font_file
    if _font_path.exists():
        font_manager.fontManager.addfont(_font_path)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.sans-serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 1200,
    }
)

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.data_loaders import load_dataset_by_name
from src.data_preprocessing import preprocess_dataset
from src.evaluate import compute_thermo_hydraulic_metric_rows, evaluate_model
from src.effective_physics import apply_calibrated_params_to_config as _apply_calibrated_params_to_config
from src.graph_utils import build_line_graph_adjacency, normalized_adjacency
from src.models import build_model
from src.real_data_mapper import build_boundary_conditions
from src.sensor_layouts import apply_sensor_layout
from src.study_workflow import build_loaders
from src.thermo_hydraulic_simulator import simulate_thermo_hydraulics
from src.utils import ensure_dir


SUPPLEMENTARY_MODEL_SPECS = [
    ("GRU-MSE", "gru", "seed_11_GRU-MSE_best.pt"),
    ("Transformer-MSE", "transformer", "seed_11_Transformer-MSE_best.pt"),
    ("Proposed PI-GNN-GRU-v3 accuracy_mode", "pignn_v3", "seed_11_Proposed PI-GNN-GRU-v3 accuracy_mode_best.pt"),
    ("Proposed PI-GNN-GRU-v3 balanced_mode", "pignn_v3", "seed_11_Proposed PI-GNN-GRU-v3 balanced_mode_best.pt"),
]


def load_sonderborg_processed(config: dict[str, Any]) -> pd.DataFrame:
    real_cfg = config.get("real_data", {})
    frozen_rel = real_cfg.get("canonical_sonderborg_processed_path")
    frozen = PROJECT_ROOT / str(frozen_rel) if frozen_rel else None
    if real_cfg.get("freeze_canonical_processed", False) and frozen and frozen.exists():
        df = preprocess_dataset(pd.DataFrame(), "sonderborg", config)
    else:
        processed = PROJECT_ROOT / "data" / "processed" / "sonderborg_processed.csv"
        if processed.exists():
            df = pd.read_csv(processed)
        else:
            df = preprocess_dataset(load_dataset_by_name("sonderborg"), "sonderborg", config)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def load_calibrated_params() -> dict[str, Any]:
    path = PROJECT_ROOT / "results" / "calibrated_parameters.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def apply_calibrated_params_to_config(
    config: dict[str, Any], params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a copy whose effective physics parameters match the simulator."""
    return _apply_calibrated_params_to_config(config, params or load_calibrated_params())


def simulate_from_dataframe(df: pd.DataFrame, config: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    boundary = build_boundary_conditions(df, config)
    return simulate_thermo_hydraulics(boundary, config, params=params or load_calibrated_params())


def boundary_from_sim(sim: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_s": np.asarray(sim["time_s"], dtype=float),
        "T_source": np.asarray(sim["T_source"], dtype=float),
        "T_return_measured": np.asarray(sim["T_return_measured"], dtype=float),
        "Q_load_W": np.asarray(sim["Q_load"], dtype=float),
        "Ta": np.asarray(sim["Ta"], dtype=float),
        "alpha_estimated": np.asarray(sim["alpha"], dtype=float),
        "q_proxy": np.asarray(sim.get("q_proxy", sim["q"][:, 0]), dtype=float),
        "flow_proxy_mode": str(sim.get("flow_proxy_mode", "causal_lagged_return")),
        "proxy_causality_version": str(sim.get("proxy_causality_version", "causal_proxy_v1_training_prefix_trailing_mean")),
        "alpha_provenance": str(sim.get("alpha_provenance", "causal proxy copied from simulator")),
        "q_proxy_provenance": str(sim.get("q_proxy_provenance", "causal proxy copied from simulator")),
        "return_temperature_assumed": bool(sim.get("return_temperature_assumed", False)),
        "source_dataset": str(sim.get("source_dataset", "sonderborg")),
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    for ax in fig.axes:
        if hasattr(ax, "spines") and ax.get_visible():
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#111111")
                spine.set_linewidth(1.0)
    for out_dir in [
        ensure_dir(PROJECT_ROOT / "figures" / "final"),
        ensure_dir(PROJECT_ROOT / "paper" / "figures" / "final"),
    ]:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.svg", format="svg", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def load_models_for_sim(
    sim: dict[str, Any],
    sensors: dict[str, Any],
    config: dict[str, Any],
    stats: dict[str, Any] | None = None,
    model_specs: list[tuple[str, str, str]] | None = None,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    loaders = build_loaders(sim, sensors, config, stats=stats)
    arrays = loaders["arrays"]
    n_nodes = arrays["target"].shape[1]
    a_norm = torch.tensor(normalized_adjacency(build_line_graph_adjacency(n_nodes)), dtype=torch.float32)
    trained: dict[str, torch.nn.Module] = {}
    for label, model_name, state_name in model_specs or SUPPLEMENTARY_MODEL_SPECS:
        state_path = PROJECT_ROOT / "results" / state_name
        if not state_path.exists():
            continue
        model = build_model(model_name, arrays["features"].shape[-1], n_nodes, a_norm, config)
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        trained[label] = model
    return trained, loaders


def _metric_lookup(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        try:
            out[str(row["metric"])] = float(row["value"])
        except Exception:
            continue
    return out


def evaluate_models_on_sim(
    sim: dict[str, Any],
    config: dict[str, Any],
    trained: dict[str, torch.nn.Module],
    stats: dict[str, Any],
    layout: str = "S4_five_sensors",
    sensors_override: dict[str, Any] | None = None,
    noise: float = 0.0,
    case_label: str = "",
    regime: str = "",
    note: str = "",
    selected_models: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    sensors = sensors_override or apply_sensor_layout(sim, layout, config, noise_std_fraction=noise)
    loaders = build_loaders(sim, sensors, config, stats=stats)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, np.ndarray]] = {}
    for model_name, model in trained.items():
        if selected_models and model_name not in selected_models:
            continue
        metrics, payload = evaluate_model(model, loaders["test_loader"], config, stats, model_name, return_predictions=True)
        thermo_rows, _ = compute_thermo_hydraulic_metric_rows(payload, config, model_name, sensor_nodes=sensors.get("sensor_nodes", []))
        thermo = _metric_lookup(thermo_rows)
        pred = np.asarray(payload["pred"], dtype=float)
        true = np.asarray(payload["true"], dtype=float)
        max_temp_error = float(np.nanmax(np.abs(pred[..., :2] - true[..., :2])))
        max_head_error = float(np.nanmax(np.abs(pred[..., 2] - true[..., 2])))
        row = {
            "case": case_label,
            "regime": regime,
            "model": model_name,
            "sensor_layout": layout,
            "supply_RMSE_C": metrics.get("RMSE_Ts_full"),
            "return_RMSE_C": metrics.get("RMSE_Tr_full"),
            "head_RMSE_m": metrics.get("RMSE_H_full"),
            "flow_RMSE_m3_s": metrics.get("RMSE_q_full"),
            "pressure_drop_error_percent": thermo.get("pressure_drop_error_percent"),
            "heat_loss_error_percent": thermo.get("heat_loss_error_percent", metrics.get("heat_loss_error_percent")),
            "delivered_heat_error_percent": thermo.get("delivered_heat_error_percent", metrics.get("heat_load_consistency_error_percent")),
            "energy_balance_residual_percent": thermo.get("energy_balance_residual_percent", metrics.get("energy_balance_residual")),
            "thermal_delay_error_min": thermo.get("thermal_delay_error_min"),
            "boundary_residual_mean_C": metrics.get("boundary_residual_mean"),
            "max_temperature_error_C": max_temp_error,
            "max_head_error_m": max_head_error,
            "state_type": "calibrated_simulator + simulator_assisted_hidden_state",
            "safe_claim": (
                "Real operating data define boundary conditions, but these perturbation metrics compare "
                "calibrated-simulator and simulator-assisted hidden-state targets; they are not independent "
                "real measured-node validation."
            ),
            "note": note,
        }
        rows.append(row)
        payloads[model_name] = payload
    return pd.DataFrame(rows), payloads


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, resize: bool = True) -> None:
    ensure_dir(path.parent)
    if df.empty:
        df = pd.DataFrame([{"status": "not run"}])
    latex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    if resize or len(df.columns) > 5:
        latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
        latex = latex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    latex = latex.replace(r"\$\textasciicircum \textbackslash circ\$C", r"$^\circ$C")
    path.write_text(latex, encoding="utf-8")


def copy_final_figures_to_root_and_paper() -> None:
    final_dir = PROJECT_ROOT / "figures" / "final"
    paper_final = ensure_dir(PROJECT_ROOT / "paper" / "figures" / "final")
    root_figures = ensure_dir(PROJECT_ROOT / "figures")
    for fig in final_dir.glob("*.*"):
        if fig.suffix.lower() in {".pdf", ".png"}:
            (paper_final / fig.name).write_bytes(fig.read_bytes())
            (root_figures / fig.name).write_bytes(fig.read_bytes())
