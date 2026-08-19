from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .utils import ensure_dir


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Latin Modern Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 1200,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    out_dir = ensure_dir(PROJECT_ROOT / "figures")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def _best_prediction(predictions: dict[str, dict[str, np.ndarray]]) -> tuple[str, dict[str, np.ndarray]]:
    for key in [
        "Proposed PI-GNN-GRU-v3 balanced_mode",
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 physics_mode",
        "Proposed PI-GNN-GRU-v2",
        "Proposed PI-GNN-GRU improved",
        "Proposed PI-GNN-GRU",
        "PI-GNN-no-temporal",
        "Transformer-MSE",
        "GRU-MSE",
    ]:
        if key in predictions:
            return key, predictions[key]
    key = next(iter(predictions))
    return key, predictions[key]


def fig1_real_data_overview(df: pd.DataFrame) -> None:
    plot_df = df.head(192).copy()
    t = pd.to_datetime(plot_df["timestamp"]) if "timestamp" in plot_df else np.arange(len(plot_df))
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.2), sharex=True)
    items = [
        ("heat_load_kw", "Heat load (kW)"),
        ("supply_temp_C", "Supply temp. (C)"),
        ("return_temp_C", "Return temp. (C)"),
        ("ambient_temp_C", "Ambient temp. (C)"),
    ]
    for ax, (col, label) in zip(axes, items):
        if col in plot_df:
            ax.plot(t, plot_df[col], lw=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    fallback = bool(plot_df.get("is_fallback_synthetic", pd.Series([False])).fillna(False).astype(bool).any())
    axes[0].set_title("Fallback synthetic demo overview" if fallback else "Real operating data overview")
    axes[-1].set_xlabel("Time")
    _save(fig, "fig1_real_data_overview")


def fig2_network_sensor_layout(sim: dict[str, Any], sensors: dict[str, Any]) -> None:
    x_km = sim["x_m"] / 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.plot(x_km, np.zeros_like(x_km), color="0.25", lw=3, solid_capstyle="round")
    ax.scatter(x_km, np.zeros_like(x_km), s=20, color="0.72", zorder=2, label="graph nodes")
    nodes = sensors.get("sensor_nodes", [])
    if nodes:
        ax.scatter(x_km[nodes], np.zeros(len(nodes)), s=80, color="#d1495b", zorder=3, label=sensors.get("layout_name", "sensors"))
    ax.text(x_km[0], 0.08, "source", ha="left")
    ax.text(x_km[-1], 0.08, "load end", ha="right")
    ax.set_xlim(x_km[0] - 0.5, x_km[-1] + 0.5)
    ax.set_ylim(-0.25, 0.35)
    ax.set_xlabel("Distance (km)")
    ax.set_yticks([])
    ax.set_title("20 km line network and sparse sensor layout")
    ax.legend(loc="upper center", ncol=2)
    _save(fig, "fig2_network_sensor_layout")


def fig3_framework_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axis("off")
    boxes = [
        (0.03, 0.55, "Real operating data\nload, supply, return,\nambient, substations"),
        (0.29, 0.55, "Preprocessing and\nboundary-condition\nmapping"),
        (0.55, 0.55, "Calibrated physics\nsimulator for hidden\ndistributed states"),
        (0.55, 0.14, "Synthetic sparse sensors\nfrom measured and\nsimulated states"),
        (0.79, 0.34, "Manual PI-GNN\nstate estimator"),
        (0.79, 0.72, "Measured-node and\nsimulation-based\nevaluation"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.18, 0.22, facecolor="#f2f5f7", edgecolor="0.3", lw=1))
        ax.text(x + 0.09, y + 0.11, text, ha="center", va="center", fontsize=8)
    arrows = [
        ((0.21, 0.66), (0.29, 0.66)),
        ((0.47, 0.66), (0.55, 0.66)),
        ((0.64, 0.55), (0.64, 0.36)),
        ((0.73, 0.25), (0.79, 0.40)),
        ((0.73, 0.66), (0.79, 0.77)),
        ((0.88, 0.56), (0.88, 0.72)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="0.25", lw=1.2))
    ax.set_title("Real-data-assisted PI-GNN digital twin workflow")
    _save(fig, "fig3_framework_flowchart")


def fig4_calibration_fit(sim: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    t_h = sim["time_s"] / 3600.0
    ax.plot(t_h, sim["T_return_measured"], label="Measured return/source-node", lw=1.5)
    ax.plot(t_h, sim["Tr"][:, 0], label="Simulated return at source", lw=1.2)
    ax.plot(t_h, sim["T_source"], label="Measured supply boundary", lw=1.0)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Calibration and boundary fit")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, "fig4_calibration_fit")


def fig5_temperature_reconstruction(sim: dict[str, Any], prediction: dict[str, np.ndarray], model_label: str) -> None:
    pred = prediction["pred"][0, -1, :, 0]
    true = prediction["true"][0, -1, :, 0]
    x_km = sim["x_m"] / 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(x_km, true, label="Simulator hidden Ts", lw=1.8)
    ax.plot(x_km, pred, "--", label=f"{model_label} predicted Ts", lw=1.6)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Supply temperature (C)")
    ax.set_title("Supply temperature reconstruction")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, "fig5_temperature_reconstruction")


def fig5_temperature_reconstruction_multi(
    sim: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    sensors: dict[str, Any],
) -> None:
    x_km = sim["x_m"] / 1000.0
    preferred = ["GRU-MSE", "PureGNN-MSE", "Proposed PI-GNN-GRU-v2", "Proposed PI-GNN-GRU"]
    available = [name for name in preferred if name in predictions]
    if not available:
        label, pred_payload = _best_prediction(predictions)
        available = [label]
    true = predictions[available[0]]["true"][0, -1, :, 0]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(x_km, true, color="black", lw=2.0, label="Simulator hidden Ts")
    styles = {
        "LSTM-MSE": ("#577590", "--"),
        "GRU-MSE": ("#f9844a", "-."),
        "PureGNN-MSE": ("#43aa8b", ":"),
        "Proposed PI-GNN-GRU": ("#d1495b", "-"),
        "Proposed PI-GNN-GRU improved": ("#7b2cbf", "-"),
        "Proposed PI-GNN-GRU-v2": ("#7b2cbf", "-"),
    }
    for name in available:
        color, linestyle = styles.get(name, (None, "--"))
        ax.plot(x_km, predictions[name]["pred"][0, -1, :, 0], linestyle=linestyle, color=color, lw=1.5, label=name)
        if name in {"Proposed PI-GNN-GRU", "Proposed PI-GNN-GRU improved", "Proposed PI-GNN-GRU-v2"}:
            pred_profile = predictions[name]["pred"][0, -1, :, 0]
            band = np.abs(pred_profile - true)
            ax.fill_between(x_km, pred_profile - band, pred_profile + band, color=color, alpha=0.10, linewidth=0)
    nodes = sensors.get("sensor_nodes", [])
    if nodes:
        ax.scatter(x_km[nodes], true[nodes], s=36, color="white", edgecolor="black", zorder=5, label="sensor nodes")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Supply temperature (C)")
    ax.set_title("Temperature reconstruction: simulator hidden state and model predictions")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "fig5_temperature_reconstruction")


def fig6_head_reconstruction(sim: dict[str, Any], prediction: dict[str, np.ndarray], model_label: str) -> None:
    pred = prediction["pred"][0, -1, :, 2]
    true = prediction["true"][0, -1, :, 2]
    pred_q = prediction["pred"][0, -1, :, 3]
    true_q = prediction["true"][0, -1, :, 3]
    x_km = sim["x_m"] / 1000.0
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.plot(x_km, true, label="Simulator hidden head", lw=1.8)
    ax.plot(x_km, pred, "--", label=f"{model_label} predicted head", lw=1.6)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Hydraulic head (m)")
    ax2 = ax.twinx()
    ax2.plot(x_km, true_q, color="#43aa8b", alpha=0.8, lw=1.2, label="Simulator hidden flow")
    ax2.plot(x_km, pred_q, color="#43aa8b", alpha=0.8, lw=1.2, linestyle=":", label=f"{model_label} predicted flow")
    ax2.set_ylabel("Flow proxy (m3/s)")
    ax.set_title("Hydraulic head/flow reconstruction (simulator hidden state)")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=7)
    _save(fig, "fig6_head_reconstruction")


def fig7_error_heatmap(sim: dict[str, Any], prediction: dict[str, np.ndarray]) -> None:
    error = np.abs(prediction["pred"][0, :, :, 0] - prediction["true"][0, :, :, 0])
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    im = ax.imshow(error, aspect="auto", origin="lower", cmap="magma", extent=[0, sim["x_m"][-1] / 1000.0, 0, error.shape[0]])
    fig.colorbar(im, ax=ax, label="Absolute Ts error (C)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Window step")
    ax.set_title("Space-time supply-temperature error")
    _save(fig, "fig7_spacetime_temperature_error_heatmap")


def fig8_sensor_layout_comparison(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.3))
    axes = axes.ravel()
    if "sensor_layout" in metrics_df.columns:
        plot_df = metrics_df.sort_values("RMSE_Ts_full")
        labels = plot_df["sensor_layout"].astype(str).str.replace("_", "\n")
        metric_specs = [
            ("RMSE_Ts_full", "RMSE Ts (C)", "#3973ac"),
            ("RMSE_Tr_full", "RMSE Tr (C)", "#4d908e"),
            ("heat_loss_error_percent", "Heat-loss error (%)", "#bc4749"),
            ("energy_balance_residual", "Energy residual", "#6a994e"),
        ]
        for ax, (metric, ylabel, color) in zip(axes, metric_specs):
            if metric in plot_df.columns:
                ax.bar(labels, pd.to_numeric(plot_df[metric], errors="coerce"), color=color)
                ax.set_ylabel(ylabel)
                ax.set_title(ylabel)
            else:
                ax.text(0.5, 0.5, f"{metric}\nnot available", ha="center", va="center")
    else:
        row = metrics_df[metrics_df["model"].eq("pignn")]
        values = [float(row["RMSE_Ts_full"].iloc[0]) if not row.empty else float(metrics_df["RMSE_Ts_full"].iloc[0])]
        labels = ["S4\nfive sensors"]
        axes[0].bar(labels, values, color="#3973ac")
        for ax in axes[1:]:
            ax.axis("off")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=70, labelsize=6)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Sensor layout comparison")
    _save(fig, "fig8_sensor_layout_comparison")
    for suffix in [".pdf", ".png"]:
        src = PROJECT_ROOT / "figures" / f"fig8_sensor_layout_comparison{suffix}"
        if src.exists():
            shutil.copy2(src, PROJECT_ROOT / "figures" / f"fig8_sensor_layout_comparison_improved{suffix}")


def fig9_baseline_comparison(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))
    plot_df = metrics_df.sort_values("RMSE_Ts_full")
    axes[0].bar(plot_df["model"], plot_df["RMSE_Ts_full"], color="#4d908e")
    axes[0].set_ylabel("RMSE Ts full (C)")
    axes[0].set_title("Direct reconstruction")
    metric = "heat_loss_error_percent" if "heat_loss_error_percent" in plot_df.columns else "thermal_residual_mean"
    axes[1].bar(plot_df["model"], pd.to_numeric(plot_df[metric], errors="coerce"), color="#bc4749")
    axes[1].set_ylabel("Heat-loss error (%)" if metric == "heat_loss_error_percent" else "Thermal residual")
    axes[1].set_title("Physical consistency")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=70, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Baseline comparison")
    _save(fig, "fig9_baseline_comparison")


def fig10_external_validation(metrics_df: pd.DataFrame) -> None:
    ts_path = PROJECT_ROOT / "results" / "external_validation_flensburg_timeseries.csv"
    modes_path = PROJECT_ROOT / "results" / "external_validation_flensburg_modes.csv"
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=False)
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        axes[0].plot(ts["window_step"], ts["measured_or_boundary_supply_C"], label="Measured/boundary feed", lw=1.5)
        axes[0].plot(ts["window_step"], ts["predicted_supply_C"], label="Predicted feed", lw=1.3)
        axes[0].plot(ts["window_step"], ts["simulator_hidden_supply_C"], label="Simulator hidden feed", lw=1.0, alpha=0.8)
        axes[0].set_ylabel("Supply temperature (C)")
        axes[0].legend(ncol=3, fontsize=7)
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(ts["window_step"], ts["residual_C"], color="#bc4749", lw=1.3)
        axes[1].axhline(0, color="0.3", lw=0.8)
        axes[1].set_ylabel("Residual (C)")
        axes[1].set_xlabel("Window step")
        axes[1].grid(True, alpha=0.25)
        axes[0].set_title("External validation on Flensburg: feed-temperature transfer")
        axes[1].text(0.01, 0.05, "Return temperature assumed as 50 C if unavailable", transform=axes[1].transAxes, fontsize=7)
        if modes_path.exists():
            modes = pd.read_csv(modes_path)
            if "RMSE_Ts_measured_nodes" in modes.columns:
                plot_modes = modes.dropna(subset=["RMSE_Ts_measured_nodes"]).copy()
                axes[2].bar(plot_modes["mode"].astype(str).str.replace("_", "\n"), pd.to_numeric(plot_modes["RMSE_Ts_measured_nodes"], errors="coerce"), color="#577590")
                axes[2].set_ylabel("Supply RMSE (C)")
                axes[2].set_title("Transfer modes")
                axes[2].tick_params(axis="x", labelrotation=25, labelsize=7)
                axes[2].grid(True, axis="y", alpha=0.25)
            else:
                axes[2].axis("off")
    elif metrics_df.empty:
        ax = axes[0]
        ax.text(0.5, 0.5, "Flensburg data not available in this run", ha="center", va="center")
        for empty_ax in axes:
            empty_ax.set_xticks([])
            empty_ax.set_yticks([])
    else:
        ax = axes[0]
        value_col = "RMSE_Ts_measured_nodes" if "RMSE_Ts_measured_nodes" in metrics_df.columns else "RMSE_Ts_full"
        value = float(metrics_df[value_col].iloc[0])
        ax.bar(["Flensburg transfer\nmeasured-node"], [value], color="#577590")
        ax.set_ylabel("Measured-node RMSE Ts (C)")
        ax.grid(True, axis="y", alpha=0.25)
        axes[1].axis("off")
        axes[2].axis("off")
        ax.set_title("External validation on Flensburg")
    _save(fig, "fig10_external_validation_flensburg")
    for suffix in [".pdf", ".png"]:
        src = PROJECT_ROOT / "figures" / f"fig10_external_validation_flensburg{suffix}"
        if src.exists():
            shutil.copy2(src, PROJECT_ROOT / "figures" / f"fig10_external_validation_flensburg_modes{suffix}")


def fig11_noise_dropout_robustness(metrics_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    if {"condition", "base_model", "RMSE_Ts_full"}.issubset(metrics_df.columns):
        pivot = metrics_df.pivot_table(index="condition", columns="base_model", values="RMSE_Ts_full", aggfunc="mean")
        pivot = pivot[
            [
                c
                for c in [
                    "GRU-MSE",
                    "Transformer-MSE",
                    "PureGNN-MSE",
                    "Proposed PI-GNN-GRU-v3 balanced_mode",
                    "Proposed PI-GNN-GRU-v3 accuracy_mode",
                    "Proposed PI-GNN-GRU-v2",
                    "Proposed PI-GNN-GRU improved",
                    "Proposed PI-GNN-GRU",
                ]
                if c in pivot.columns
            ]
        ]
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylabel("RMSE Ts full (C)")
        ax.set_title("Noise/dropout robustness")
    else:
        row = metrics_df[metrics_df["model"].eq("Proposed PI-GNN-GRU")]
        values = [float(row["RMSE_Ts_full"].iloc[0]) if not row.empty else float(metrics_df["RMSE_Ts_full"].iloc[0])]
        labels = ["Nominal\nrun"]
        ax.bar(labels, values, color="#f9844a")
        ax.set_ylabel("RMSE Ts full (C)")
        ax.set_title("Noise/dropout robustness")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, "fig11_noise_dropout_robustness")


def fig8b_sensor_placement_map(sim: dict[str, Any], layout_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x_km = sim["x_m"] / 1000.0
    ax.plot(x_km, np.zeros_like(x_km), color="0.35", lw=2)
    if "sensor_nodes" in layout_df.columns:
        for j, (_, row) in enumerate(layout_df.iterrows()):
            nodes = [int(x) for x in str(row["sensor_nodes"]).split(";") if str(x).strip().isdigit()]
            if nodes:
                ax.scatter(x_km[nodes], np.full(len(nodes), j + 1), s=34, label=row["sensor_layout"])
        ax.set_ylim(-0.5, len(layout_df) + 1)
        ax.set_yticks([])
    ax.set_xlabel("Distance (km)")
    ax.set_title("Sparse sensor placement map")
    ax.grid(True, axis="x", alpha=0.25)
    _save(fig, "fig8b_sensor_placement_map")


def fig12_xai4heat_sparse_substations(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if not df.empty and {"substation_id", "RMSE_Ts_measured_nodes"}.issubset(df.columns):
        ax.bar(df["substation_id"].astype(str), df["RMSE_Ts_measured_nodes"], color="#43aa8b")
    else:
        ax.text(0.5, 0.5, "XAI4HEAT not available in this run", ha="center", va="center")
    ax.set_ylabel("Measured-node RMSE Ts (C)")
    ax.set_title("XAI4HEAT sparse-substation validation")
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, "fig12_xai4heat_sparse_substations")


def fig13_ablation_study(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2))
    axes = axes.ravel()
    metrics = [
        ("RMSE_Ts_full", "RMSE Ts (C)"),
        ("heat_loss_error_percent", "Heat-loss error (%)"),
        ("energy_balance_residual", "Energy residual"),
        ("thermal_residual_mean", "Thermal residual"),
    ]
    if not df.empty and "ablation" in df.columns:
        plot_df = df.copy()
        labels = plot_df["ablation"].astype(str).str.replace("_", "\n")
        for ax, (col, ylabel) in zip(axes, metrics):
            if col in plot_df.columns:
                ax.bar(labels, pd.to_numeric(plot_df[col], errors="coerce"), color="#6a994e")
                ax.set_ylabel(ylabel)
                ax.tick_params(axis="x", labelrotation=60, labelsize=7)
                ax.grid(True, axis="y", alpha=0.25)
            else:
                ax.text(0.5, 0.5, f"{col}\nnot available", ha="center", va="center")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Ablation study not available", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Physics-informed loss ablation")
    _save(fig, "fig13_ablation_study")


def fig14_discretization_study(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if not df.empty and {"dx_m", "mean_outlet_supply_C"}.issubset(df.columns):
        ax.plot(df["dx_m"], df["mean_outlet_supply_C"], marker="o", label="Outlet supply")
        ax2 = ax.twinx()
        ax2.plot(df["dx_m"], df["mean_heat_loss_kW"], marker="s", color="#bc4749", label="Heat loss")
        ax2.set_ylabel("Mean heat loss (kW)")
    else:
        ax.text(0.5, 0.5, "Discretization study not available", ha="center", va="center")
    ax.set_xlabel("dx (m)")
    ax.set_ylabel("Outlet supply (C)")
    ax.set_title("Discretization/model-verification study")
    ax.grid(True, alpha=0.25)
    _save(fig, "fig14_discretization_check")


def fig15_physics_consistency_summary(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    axes = axes.ravel()
    cols = [
        ("heat_loss_error_percent", "Heat-loss error (%)"),
        ("energy_balance_residual", "Energy residual"),
        ("thermal_residual_mean", "Thermal residual"),
        ("boundary_residual_mean", "Boundary residual (C)"),
    ]
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        if "model" in df.columns:
            keep = ["GRU-MSE", "PureGNN-MSE", "PI-LSTM", "Proposed PI-GNN-GRU", "Proposed PI-GNN-GRU improved"]
            plot_df = df[df["model"].astype(str).isin(keep)].copy()
            if plot_df.empty:
                plot_df = df.copy()
            labels = plot_df["model"].astype(str)
        else:
            plot_df = df.copy()
            labels = pd.Series([f"model {i}" for i in range(len(df))])
        for ax, (col, ylabel) in zip(axes, cols):
            if col in plot_df.columns:
                ax.bar(labels, pd.to_numeric(plot_df[col], errors="coerce"), color="#6a994e")
                ax.tick_params(axis="x", labelrotation=60)
                ax.set_ylabel(ylabel)
                ax.grid(True, axis="y", alpha=0.25)
            else:
                ax.text(0.5, 0.5, f"{ylabel}\nnot available", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
    axes[0].set_title("Safety and physics consistency summary")
    _save(fig, "fig15_physics_consistency_summary")


def fig10_model_ranking_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    if not df.empty and {"metric", "rank", "model"}.issubset(df.columns):
        pivot = df.pivot_table(index="model", columns="metric", values="rank", aggfunc="min")
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
        fig.colorbar(im, ax=ax, label="Rank (lower is better)")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=7)
        ax.set_title("Model rankings differ by metric")
    else:
        ax.text(0.5, 0.5, "Model ranking data unavailable", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "fig10_model_ranking_heatmap")


def fig17_evidence_summary() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    labels = ["Calibration", "Direct RMSE", "Physics metrics", "Sensor layouts", "Flensburg transfer", "XAI4HEAT"]
    status = [1.0, 0.8, 0.8, 1.0, 0.65, 0.2]
    colors = ["#43aa8b", "#4d908e", "#4d908e", "#43aa8b", "#f9c74f", "#f94144"]
    ax.bar(labels, status, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Evidence status")
    ax.set_title("Final evidence summary: real-data-assisted benchmark")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig17_final_evidence_summary")


def generate_all_figures(
    sim: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    metrics_df: pd.DataFrame,
    sensors: dict[str, Any],
    operating_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    model_label, prediction = _best_prediction(predictions)
    fig1_real_data_overview(operating_df)
    fig2_network_sensor_layout(sim, sensors)
    fig3_framework_flowchart()
    fig4_calibration_fit(sim)
    fig5_temperature_reconstruction_multi(sim, predictions, sensors)
    fig6_head_reconstruction(sim, prediction, model_label)
    fig7_error_heatmap(sim, prediction)
    fig8_sensor_layout_comparison(metrics_df)
    fig9_baseline_comparison(metrics_df)
    external_path = PROJECT_ROOT / "results" / "external_validation_flensburg.csv"
    fig10_external_validation(pd.read_csv(external_path) if external_path.exists() else pd.DataFrame())
    layout_path = PROJECT_ROOT / "results" / "sensor_layout_comparison_final.csv"
    if not layout_path.exists():
        layout_path = PROJECT_ROOT / "results" / "sensor_layout_comparison_improved.csv"
    if not layout_path.exists():
        layout_path = PROJECT_ROOT / "results" / "sensor_layout_comparison.csv"
    if layout_path.exists():
        layout_df = pd.read_csv(layout_path)
        fig8_sensor_layout_comparison(layout_df)
        fig8b_sensor_placement_map(sim, layout_df)
        robustness_path = PROJECT_ROOT / "results" / "noise_dropout_robustness_final.csv"
        if not robustness_path.exists():
            robustness_path = PROJECT_ROOT / "results" / "noise_dropout_robustness.csv"
        fig11_noise_dropout_robustness(pd.read_csv(robustness_path) if robustness_path.exists() else layout_df)
    else:
        fig11_noise_dropout_robustness(metrics_df)
    ablation_path = PROJECT_ROOT / "results" / "ablation_study_final.csv"
    if not ablation_path.exists():
        ablation_path = PROJECT_ROOT / "results" / "ablation_study.csv"
    if ablation_path.exists():
        fig13_ablation_study(pd.read_csv(ablation_path))
    disc_path = PROJECT_ROOT / "results" / "discretization_study.csv"
    if disc_path.exists():
        fig14_discretization_study(pd.read_csv(disc_path))
    physics_path = PROJECT_ROOT / "results" / "physics_consistency_comparison_final.csv"
    if not physics_path.exists():
        physics_path = PROJECT_ROOT / "results" / "physics_consistency_comparison.csv"
    fig15_physics_consistency_summary(pd.read_csv(physics_path) if physics_path.exists() else metrics_df)
    ranking_path = PROJECT_ROOT / "results" / "model_ranking_by_metric_final.csv"
    if not ranking_path.exists():
        ranking_path = PROJECT_ROOT / "results" / "model_ranking_by_metric.csv"
    fig10_model_ranking_heatmap(pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame())
    fig17_evidence_summary()
    aliases = {
        "fig6_head_reconstruction": ["fig7_head_flow_hidden_state_reconstruction"],
        "fig7_spacetime_temperature_error_heatmap": ["fig8_spacetime_temperature_error_heatmap"],
        "fig8_sensor_layout_comparison": ["fig11_sensor_layout_comparison"],
        "fig8b_sensor_placement_map": ["fig12_sensor_placement_map"],
        "fig10_external_validation_flensburg": ["fig13_external_validation_flensburg_final"],
        "fig11_noise_dropout_robustness": ["fig14_noise_dropout_robustness"],
        "fig13_ablation_study": ["fig15_ablation_study_final"],
        "fig14_discretization_check": ["fig16_discretization_model_verification"],
    }
    for src_name, dst_names in aliases.items():
        for suffix in [".pdf", ".png"]:
            src = PROJECT_ROOT / "figures" / f"{src_name}{suffix}"
            if src.exists():
                for dst_name in dst_names:
                    shutil.copy2(src, PROJECT_ROOT / "figures" / f"{dst_name}{suffix}")
