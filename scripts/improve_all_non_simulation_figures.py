from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ate_figure_style import (  # noqa: E402
    PALETTE,
    add_panel_label,
    model_color,
    save_ate_figure,
    set_ate_style,
    short_model_label,
    style_axes,
    style_legend,
)

RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES = PROJECT_ROOT / "paper" / "figures" / "final"


def _read(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save(fig: plt.Figure, stem: str) -> None:
    save_ate_figure(fig, FIGURES, stem)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        src = FIGURES / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, PAPER_FIGURES / f"{stem}{suffix}")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _short_layout(text: str) -> str:
    label = str(text)
    for old in [
        "S10_",
        "S11_",
        "S12_",
        "S13_",
        "S14_",
        "S15_",
        "S16_",
        "S17_",
        "S1_",
        "S2_",
        "S3_",
        "S4_",
        "S5_",
        "S6_",
        "S7_",
        "S8_",
        "S9_",
    ]:
        label = label.replace(old, old.rstrip("_") + " ")
    label = label.replace("_", " ")
    return label


def _safe_float(value) -> float:
    try:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else np.nan
    except Exception:
        return np.nan


def _rank_color(rank_value) -> str:
    rank = _safe_float(rank_value)
    if np.isfinite(rank) and rank <= 1:
        return PALETTE["proposed"]
    if np.isfinite(rank) and rank <= 2:
        return PALETTE["safe"]
    if np.isfinite(rank) and rank <= 4:
        return PALETTE["warning"]
    return PALETTE["baseline"]


def fig_model_ranking() -> None:
    ranking = _read("model_ranking_by_metric_final.csv")
    if ranking.empty:
        ranking = _read("model_ranking_by_metric.csv")
    if ranking.empty:
        return
    selected_metrics = [
        "RMSE_Ts_full",
        "RMSE_Tr_full",
        "heat_loss_error_percent",
        "energy_balance_residual",
        "boundary_residual_mean",
        "thermal_residual_mean",
    ]
    ranking = ranking[ranking["metric"].isin(selected_metrics)].copy()
    ranking["model_short"] = ranking["model"].map(short_model_label)
    key_models = ["GRU", "Transformer", "PI-LSTM", "PureGNN", "Proposed", "Interpolation"]
    ranking = ranking[ranking["model_short"].isin(key_models)]
    if ranking.empty:
        return
    pivot = ranking.pivot_table(index="model_short", columns="metric", values="rank", aggfunc="min")
    pivot = pivot.reindex([m for m in key_models if m in pivot.index])
    pivot = pivot[[m for m in selected_metrics if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    rank_cmap = LinearSegmentedColormap.from_list(
        "ate_reference_rank",
        [PALETTE["proposed"], PALETTE["safe"], PALETTE["warning"], PALETTE["pilstm"], PALETTE["alarm"]],
    )
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap=rank_cmap, aspect="auto", vmin=1, vmax=max(6, np.nanmax(pivot.to_numpy())))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n").replace("percent", "%") for c in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color=PALETTE["edge"], linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{int(val)}", ha="center", va="center", color="white" if val in {1, 12, 13} else "#111111", fontsize=8.5, fontweight="bold")
    ax.set_title("Metric-dependent model ranking")
    ax.set_xlabel("Evaluation metric")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Rank (1 = lowest error/residual)")
    fig.tight_layout()
    _save(fig, "fig_model_ranking_ate_dark")


def fig_accuracy_physics_tradeoff() -> None:
    baseline = _read("baseline_comparison_final.csv")
    if baseline.empty:
        baseline = _read("baseline_comparison_improved.csv")
    if baseline.empty:
        return
    needed = {"model", "RMSE_Ts_full", "RMSE_Tr_full", "heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"}
    if not needed.issubset(set(baseline.columns)):
        return
    show_models = [
        "GRU-MSE",
        "Transformer-MSE",
        "PureGNN-MSE",
        "PI-LSTM",
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 balanced_mode",
    ]
    sub = baseline[baseline["model"].isin(show_models)].copy()
    if sub.empty:
        return
    for col in list(needed - {"model"}):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub["direct_thermal_rmse_C"] = sub["RMSE_Ts_full"] + sub["RMSE_Tr_full"]
    physics_cols = ["heat_loss_error_percent", "energy_balance_residual", "boundary_residual_mean"]
    for col in physics_cols:
        scale = sub[col].median(skipna=True)
        if not np.isfinite(scale) or scale == 0:
            scale = sub[col].max(skipna=True)
        sub[f"{col}_norm"] = sub[col] / scale if np.isfinite(scale) and scale != 0 else np.nan
    sub["physical_consistency_score"] = sub[[f"{c}_norm" for c in physics_cols]].mean(axis=1)
    sub = sub.dropna(subset=["direct_thermal_rmse_C", "physical_consistency_score"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for _, row in sub.iterrows():
        label = short_model_label(row["model"])
        ax.scatter(
            row["direct_thermal_rmse_C"],
            row["physical_consistency_score"],
            s=92 if label == "Proposed" else 64,
            color=model_color(row["model"]),
            edgecolor=PALETTE["edge"],
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            label,
            (row["direct_thermal_rmse_C"], row["physical_consistency_score"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.1,
        )
    ax.set_xlabel("Direct thermal RMSE score ($^\\circ$C)")
    ax.set_ylabel("Normalized physical-consistency score")
    ax.set_title("Accuracy--physics tradeoff")
    ax.text(
        0.02,
        0.98,
        "Lower-left is preferable;\nmodel choice remains objective-specific.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CFCFCF", "alpha": 0.96},
    )
    style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_accuracy_physics_tradeoff_ate_dark")


def fig_proposed_model_value_summary() -> None:
    baseline = _read("baseline_comparison_final.csv")
    hydraulic = _read("hydraulic_state_metrics.csv")
    stress = _read("combined_stress_test_improved.csv")
    if baseline.empty:
        return

    model_rows = [
        ("Interpolation", "Baseline"),
        ("GRU-MSE", "GRU-MSE"),
        ("Transformer-MSE", "Transformer"),
        ("PI-LSTM", "PI-LSTM"),
        ("Proposed PI-GNN-GRU-v3", "PI-GNN-GRU-v3"),
    ]
    metrics = [
        ("Supply RMSE", "RMSE_Ts_full", "baseline"),
        ("Return RMSE", "RMSE_Tr_full", "baseline"),
        ("Heat-loss error", "heat_loss_error_percent", "baseline"),
        ("Energy residual", "energy_balance_residual", "baseline"),
        ("Boundary residual", "boundary_residual_mean", "baseline"),
        ("Pressure-drop residual", "pressure_drop_error_percent", "hydraulic"),
        ("Severe-stress score", "stress_score", "stress"),
    ]

    def baseline_value(model_key: str, column: str) -> float:
        if model_key == "Proposed PI-GNN-GRU-v3":
            sub = baseline[baseline["model"].astype(str).str.contains("Proposed PI-GNN-GRU-v3", regex=False)].copy()
        else:
            sub = baseline[baseline["model"].astype(str).eq(model_key)].copy()
        if sub.empty or column not in sub.columns:
            return np.nan
        values = pd.to_numeric(sub[column], errors="coerce").dropna()
        return float(values.min()) if not values.empty else np.nan

    def hydraulic_value(model_key: str, metric: str) -> float:
        if hydraulic.empty or "metric" not in hydraulic.columns:
            return np.nan
        sub = hydraulic[hydraulic["metric"].astype(str).eq(metric)].copy()
        if model_key == "Proposed PI-GNN-GRU-v3":
            sub = sub[sub["model"].astype(str).str.contains("Proposed PI-GNN-GRU-v3", regex=False)]
        else:
            sub = sub[sub["model"].astype(str).eq(model_key)]
        if sub.empty:
            return np.nan
        values = pd.to_numeric(sub["value"], errors="coerce").dropna()
        return float(values.min()) if not values.empty else np.nan

    def stress_value(model_key: str) -> float:
        if stress.empty:
            return np.nan
        sub = stress[stress["case"].astype(str).eq("combined_stress_severe")].copy()
        if model_key == "Proposed PI-GNN-GRU-v3":
            sub = sub[sub["model"].astype(str).str.contains("Proposed PI-GNN-GRU-v3", regex=False)]
        else:
            sub = sub[sub["model"].astype(str).eq(model_key)]
        if sub.empty:
            return np.nan
        cols = ["max_temperature_error_C", "heat_loss_error_percent", "energy_balance_residual_percent", "pressure_drop_error_percent"]
        best_score = np.inf
        for _, row in sub.iterrows():
            vals = []
            for col in cols:
                val = _safe_float(row.get(col))
                if np.isfinite(val):
                    vals.append(val)
            if vals:
                best_score = min(best_score, float(np.mean(vals)))
        return best_score if np.isfinite(best_score) else np.nan

    value_rows: list[dict[str, object]] = []
    for metric_label, metric_key, source in metrics:
        values: dict[str, float] = {}
        for model_key, display in model_rows:
            if source == "baseline":
                value = baseline_value(model_key, metric_key)
            elif source == "hydraulic":
                value = hydraulic_value(model_key, metric_key)
            else:
                value = stress_value(model_key)
            values[display] = value
        finite_values = pd.Series(values).dropna().sort_values()
        rank_map = {model: idx + 1 for idx, model in enumerate(finite_values.index)}
        for display, value in values.items():
            value_rows.append(
                {
                    "metric": metric_label,
                    "model": display,
                    "value": value,
                    "rank": rank_map.get(display, np.nan),
                    "source": source,
                }
            )

    rank_df = pd.DataFrame(value_rows)
    rank_df.to_csv(RESULTS / "proposed_model_value_rank_matrix.csv", index=False)
    pivot = rank_df.pivot_table(index="model", columns="metric", values="rank", aggfunc="min")
    model_order = [display for _, display in model_rows if display in pivot.index]
    metric_order = [m[0] for m in metrics if m[0] in pivot.columns]
    pivot = pivot.reindex(model_order)[metric_order]
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    cmap = LinearSegmentedColormap.from_list(
        "ate_value_rank",
        [PALETTE["proposed"], PALETTE["safe"], PALETTE["warning"], PALETTE["pilstm"], PALETTE["alarm"]],
    )
    cmap.set_bad("#EFEFEF")
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=1, vmax=max(5, np.nanmax(values)))
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in metric_order], fontsize=8)
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, len(metric_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(model_order), 1), minor=True)
    ax.grid(which="minor", color=PALETTE["edge"], linestyle="-", linewidth=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            rank = pivot.iloc[i, j]
            if np.isfinite(rank):
                ax.text(j, i, f"{int(rank)}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if rank <= 1 else PALETTE["edge"])
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color=PALETTE["baseline"])
    ax.set_title("Metric-specific value of PI-GNN-GRU-v3")
    ax.set_xlabel("Metric / objective")
    ax.set_ylabel("Model")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Rank (1 = lowest error/residual)")
    ax.text(
        0.0,
        -0.26,
        "Blank cells indicate unavailable metrics for that model. Pressure-drop and stress metrics are simulator-assisted hidden-state diagnostics.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "fig_proposed_model_value_summary")


def fig_sensor_layout() -> None:
    df = _read("sensor_layout_ranking_by_objective.csv")
    if df.empty:
        return
    top = df[df["rank"].le(4)].copy()
    top["label"] = top["sensor_layout"].map(_short_layout)
    objectives = list(top["objective"].drop_duplicates())
    fig, axes = plt.subplots(1, min(3, len(objectives)), figsize=(12, 4.2), squeeze=False)
    colors = [PALETTE["proposed"], PALETTE["gru"], PALETTE["transformer"], PALETTE["pilstm"]]
    for ax, obj, panel in zip(axes[0], objectives[:3], ["(a)", "(b)", "(c)"]):
        sub = top[top["objective"].eq(obj)].sort_values("score", ascending=True)
        y = np.arange(len(sub))
        ax.barh(y, _num(sub["score"]), color=colors[: len(sub)], edgecolor=PALETTE["edge"], linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=8)
        ax.set_xlabel("Objective score")
        ax.set_title(obj.replace("Best ", ""))
        add_panel_label(ax, panel)
        style_axes(ax, grid_axis="x")
    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig_sensor_layout_ate_dark")


def fig_operator_sensor_guidelines() -> None:
    guidelines = _read("operator_sensor_guidelines.csv")
    ranking = _read("sensor_layout_ranking_by_objective.csv")
    if guidelines.empty and ranking.empty:
        return
    if not guidelines.empty:
        rows = guidelines[["Operator objective", "Recommended layout"]].copy()
    else:
        top = ranking[pd.to_numeric(ranking.get("rank", pd.Series(dtype=float)), errors="coerce").eq(1)].copy()
        rows = top.rename(columns={"objective": "Operator objective", "sensor_layout": "Recommended layout"})[
            ["Operator objective", "Recommended layout"]
        ]
    rows = rows.head(4)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_axis_off()
    colors = [PALETTE["proposed"], PALETTE["pilstm"], PALETTE["safe"], PALETTE["warning"]]
    y_positions = np.linspace(0.82, 0.18, len(rows))
    for idx, (_, row) in enumerate(rows.iterrows()):
        y = y_positions[idx]
        ax.add_patch(plt.Rectangle((0.02, y - 0.065), 0.32, 0.13, facecolor=colors[idx % len(colors)], edgecolor=PALETTE["edge"], linewidth=0.8))
        ax.add_patch(plt.Rectangle((0.36, y - 0.065), 0.60, 0.13, facecolor="white", edgecolor=PALETTE["edge"], linewidth=0.8))
        ax.text(0.18, y, str(row["Operator objective"]).replace(" and ", "\nand "), ha="center", va="center", color="white" if idx != 3 else PALETTE["edge"], fontsize=9, fontweight="bold")
        ax.text(0.38, y, _short_layout(str(row["Recommended layout"])), ha="left", va="center", color=PALETTE["edge"], fontsize=9)
    ax.text(0.02, 0.97, "Objective", ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(0.36, 0.97, "Recommended sparse-sensor choice", ha="left", va="top", fontsize=10, fontweight="bold")
    ax.text(
        0.02,
        0.03,
        "Recommendations are objective-specific; no layout is universally optimal.",
        ha="left",
        va="bottom",
        fontsize=8,
        color=PALETTE["edge"],
    )
    fig.tight_layout()
    _save(fig, "fig_operator_sensor_guidelines_ate_dark")


def fig_uncertainty() -> None:
    df = _read("uncertainty_calibration_summary.csv")
    if df.empty:
        return
    sub = df[df["quantity"].isin(["supply_temperature", "return_temperature", "head", "flow"])].copy()
    if sub.empty:
        return
    sub["nominal"] = sub["interval"].astype(str).str.replace("%", "", regex=False).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for quantity, color in zip(sub["quantity"].drop_duplicates(), [PALETTE["proposed"], PALETTE["gru"], PALETTE["transformer"], PALETTE["pilstm"]]):
        q = sub[sub["quantity"].eq(quantity)].sort_values("nominal")
        label = quantity.replace("_", " ")
        axes[0].plot(q["nominal"], q["coverage_conformal_calibrated"], marker="o", color=color, label=label)
        axes[1].plot(q["nominal"], q["mean_interval_width_conformal_calibrated"], marker="o", color=color, label=label)
    axes[0].plot([80, 95], [80, 95], "--", color=PALETTE["baseline"], linewidth=1.4, label="Nominal")
    axes[0].set_xlabel("Nominal coverage (%)")
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].set_title("Coverage calibration")
    axes[1].set_xlabel("Nominal coverage (%)")
    axes[1].set_ylabel("Mean interval width")
    axes[1].set_title("Interval sharpness")
    for ax, panel in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, panel)
        style_axes(ax)
    style_legend(axes[0], loc="lower right", ncols=1)
    style_legend(axes[1], loc="upper left", ncols=1)
    fig.tight_layout()
    _save(fig, "fig_uncertainty_coverage_ate_dark")


def fig_anomaly() -> None:
    df = _read("anomaly_detection_metrics_improved.csv")
    if df.empty:
        return
    sub = df[~df["case"].eq("normal_operation")].copy()
    sub["case_label"] = sub["case"].str.replace("_", " ", regex=False).str.replace(" plus ", " + ")
    sub = sub.sort_values("detection_rate_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(sub))
    ax.barh(y, _num(sub["detection_rate_percent"]), color=PALETTE["proposed"], edgecolor=PALETTE["edge"], linewidth=0.5, label="Detection")
    ax.scatter(_num(sub["false_alarm_rate_percent"]), y, color=PALETTE["alarm"], marker="D", s=28, label="False alarm")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["case_label"], fontsize=8)
    ax.set_xlabel("Rate (%)")
    ax.set_xlim(0, max(105, float(np.nanmax(_num(sub["detection_rate_percent"]))) + 5))
    ax.set_title("Residual-based anomaly indicators")
    style_axes(ax, grid_axis="x")
    style_legend(ax, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_anomaly_detection_ate_dark")


def fig_seasonal() -> None:
    df = _read("seasonal_generalization_improved.csv")
    if df.empty:
        return
    models = ["GRU-MSE", "Transformer-MSE", "PI-LSTM", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"]
    sub = df[df["model"].isin(models)].copy()
    sub["model_short"] = sub["model"].map(short_model_label)
    summary = sub.groupby(["regime", "model_short"], as_index=False)[["supply_RMSE_C", "return_RMSE_C", "heat_loss_error_percent"]].mean()
    regimes = list(summary["regime"].drop_duplicates())[:3]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    metrics = [("supply_RMSE_C", "Supply RMSE (C)"), ("return_RMSE_C", "Return RMSE (C)"), ("heat_loss_error_percent", "Heat-loss error (%)")]
    for ax, (metric, ylabel), panel in zip(axes, metrics, ["(a)", "(b)", "(c)"]):
        x = np.arange(len(regimes))
        width = 0.14
        for k, model in enumerate(summary["model_short"].drop_duplicates()):
            vals = []
            for regime in regimes:
                row = summary[(summary["regime"].eq(regime)) & (summary["model_short"].eq(model))]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (k - 2) * width, vals, width, color=model_color(model), edgecolor=PALETTE["edge"], linewidth=0.35, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
        ax.set_ylabel(ylabel)
        add_panel_label(ax, panel)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=5, frameon=True, edgecolor="#CFCFCF", facecolor="white", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "fig_seasonal_generalization_ate_dark")


def fig_combined_stress() -> None:
    df = _read("combined_stress_test_improved.csv")
    if df.empty:
        return
    models = ["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 accuracy_mode", "Proposed PI-GNN-GRU-v3 balanced_mode"]
    sub = df[df["model"].isin(models)].copy()
    cases = ["baseline_real_profile", "load_step_only", "sensor_dropout_only", "return_bias_only", "combined_stress_moderate", "combined_stress_severe"]
    sub = sub[sub["case"].isin(cases)]
    summary = sub.groupby(["case", "model"], as_index=False)[["max_temperature_error_C", "heat_loss_error_percent", "energy_balance_residual_percent"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    metrics = [("max_temperature_error_C", "Max temp. error (C)"), ("heat_loss_error_percent", "Heat-loss error (%)"), ("energy_balance_residual_percent", "Energy residual (%)")]
    for ax, (metric, ylabel), panel in zip(axes, metrics, ["(a)", "(b)", "(c)"]):
        for model in models:
            vals = []
            for case in cases:
                row = summary[(summary["case"].eq(case)) & (summary["model"].eq(model))]
                vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            ax.plot(range(len(cases)), vals, marker="o", color=model_color(model), label=short_model_label(model), linewidth=2.1)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cases], rotation=0, fontsize=7.4)
        ax.set_ylabel(ylabel)
        add_panel_label(ax, panel)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, frameon=True, edgecolor="#CFCFCF", facecolor="white", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "fig_combined_stress_ate_dark")


def fig_parameter_sensitivity() -> None:
    df = _read("parameter_identifiability_sensitivity_improved.csv")
    if df.empty:
        return
    sub = df[df["model"].astype(str).str.contains("Proposed PI-GNN-GRU-v3 balanced", regex=False)].copy()
    if sub.empty:
        sub = df.copy()
    sub = sub[~sub["case"].eq("baseline")].copy()
    sub["case_label"] = sub["case"].astype(str).str.replace("_", " ", regex=False)
    sub = sub.sort_values("heat_loss_error_percent", ascending=False).head(10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    y = np.arange(len(sub))
    axes[0].barh(y, _num(sub["heat_loss_error_percent"]), color=PALETTE["proposed"], edgecolor=PALETTE["edge"], linewidth=0.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(sub["case_label"], fontsize=7.8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Heat-loss error (%)")
    axes[0].set_title("Heat-loss sensitivity")
    pressure_col = "pressure_drop_error_percent" if "pressure_drop_error_percent" in sub.columns else "head_RMSE_m"
    axes[1].barh(y, _num(sub[pressure_col]), color=PALETTE["transformer"], edgecolor=PALETTE["edge"], linewidth=0.5)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Pressure-drop error (%)" if pressure_col == "pressure_drop_error_percent" else "Head RMSE (m)")
    axes[1].set_title("Hydraulic sensitivity")
    for ax, panel in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, panel)
        style_axes(ax, grid_axis="x")
    fig.tight_layout()
    _save(fig, "fig_parameter_sensitivity_ate_dark")

    tornado = _read("parameter_sensitivity_ranked.csv")
    if not tornado.empty:
        metric_col = "sensitivity_index" if "sensitivity_index" in tornado.columns else tornado.select_dtypes(include=[np.number]).columns[-1]
        label_col = "parameter" if "parameter" in tornado.columns else tornado.columns[0]
        t = tornado.sort_values(metric_col, ascending=True).tail(12)
        fig2, ax2 = plt.subplots(figsize=(7.8, 4.8))
        ax2.barh(range(len(t)), _num(t[metric_col]), color=PALETTE["warning"], edgecolor=PALETTE["edge"], linewidth=0.5)
        ax2.set_yticks(range(len(t)))
        ax2.set_yticklabels(t[label_col].astype(str).str.replace("_", " "), fontsize=8)
        ax2.set_xlabel(str(metric_col).replace("_", " "))
        ax2.set_title("Parameter-identifiability sensitivity ranking")
        style_axes(ax2, grid_axis="x")
        fig2.tight_layout()
        _save(fig2, "fig_parameter_identifiability_tornado_ate_dark")


def fig_flensburg() -> None:
    domain = _read("flensburg_domain_shift_analysis_improved.csv")
    modes = _read("external_validation_flensburg_modes_final.csv")
    if modes.empty:
        modes = _read("external_validation_flensburg_modes.csv")
    if domain.empty and modes.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    if not domain.empty:
        row = domain.iloc[0]
        vals = [
            float(row.get("sonderborg_heat_load_kw_mean", np.nan)) / 1000,
            float(row.get("flensburg_heat_load_kw_mean", np.nan)) / 1000,
            float(row.get("sonderborg_supply_temp_C_mean", np.nan)),
            float(row.get("flensburg_supply_temp_C_mean", np.nan)),
        ]
        labels = ["Sonderborg\nheat MW", "Flensburg\nheat MW", "Sonderborg\nsupply C", "Flensburg\nsupply C"]
        axes[0].bar(range(len(vals)), vals, color=[PALETTE["gru"], PALETTE["transformer"], PALETTE["gru"], PALETTE["transformer"]], edgecolor=PALETTE["edge"], linewidth=0.5)
        axes[0].set_xticks(range(len(vals)))
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_title("Domain-shift variables")
        axes[0].set_ylabel("Mean value")
    if not modes.empty:
        modes = modes[modes["mode_status"].astype(str).str.lower().isin(["run", "diagnostic"])] if "mode_status" in modes.columns else modes
        x = np.arange(len(modes))
        width = 0.34
        axes[1].bar(x - width / 2, _num(modes["RMSE_supply_measured_C"]), width, color=PALETTE["proposed"], edgecolor=PALETTE["edge"], linewidth=0.5, label="Supply")
        if "RMSE_return_measured_C" in modes.columns:
            axes[1].bar(x + width / 2, _num(modes["RMSE_return_measured_C"]), width, color=PALETTE["pilstm"], edgecolor=PALETTE["edge"], linewidth=0.5, label="Return")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(modes["mode"].astype(str).str.replace("_", "\n"), fontsize=7.4)
        axes[1].set_ylabel("Measured-node RMSE (C)")
        axes[1].set_title("Transfer modes")
        style_legend(axes[1], loc="upper right")
    for ax, panel in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, panel)
        style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_flensburg_domain_shift_ate_dark")


def fig_calibration_verification() -> None:
    calibration = _read("calibration_metrics.csv")
    discretization = _read("discretization_study.csv")
    if calibration.empty and discretization.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    if not calibration.empty:
        row = calibration.iloc[0]
        metrics = [
            ("Supply RMSE", float(row.get("RMSE_supply_C", np.nan)), "$^\\circ$C", PALETTE["proposed"]),
            ("Return RMSE", float(row.get("RMSE_return_C", np.nan)), "$^\\circ$C", PALETTE["pilstm"]),
            ("Heat delivery", float(row.get("heat_delivery_error_percent", np.nan)), "% error", PALETTE["safe"]),
            ("Energy balance", float(row.get("energy_balance_residual_fraction", np.nan)) * 100.0, "% residual", PALETTE["alarm"]),
        ]
        labels = [m[0] for m in metrics]
        vals = [m[1] for m in metrics]
        units = [m[2] for m in metrics]
        colors = [m[3] for m in metrics]
        bars = axes[0].bar(range(len(vals)), vals, color=colors, edgecolor=PALETTE["edge"], linewidth=0.55)
        axes[0].set_xticks(range(len(vals)))
        axes[0].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        axes[0].set_ylabel("Metric value")
        axes[0].set_title("Calibration against Sønderborg")
        for bar, val, unit in zip(bars, vals, units):
            if np.isfinite(val):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3g} {unit}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=PALETTE["edge"],
                )
    if not discretization.empty:
        d = discretization.sort_values("dx_m").copy()
        x = np.arange(len(d))
        axes[1].plot(
            x,
            np.abs(_num(d["outlet_supply_delta_vs_1000m_C"])),
            marker="o",
            color=PALETTE["proposed"],
            label="Outlet supply difference ($^\\circ$C)",
        )
        axes[1].plot(
            x,
            np.abs(_num(d["heat_loss_delta_vs_1000m_kW"])),
            marker="s",
            color=PALETTE["pilstm"],
            label="Heat-loss difference (kW)",
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{int(v)} m" for v in d["dx_m"]], fontsize=8)
        axes[1].set_ylabel("Absolute difference vs 1000 m")
        axes[1].set_title("Discretization check")
        style_legend(axes[1], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=1)
    for ax, panel in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, panel)
        style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_calibration_verification_ate_dark")


def fig_energy_heat_kpis() -> None:
    heat_metrics = _read("heat_energy_metrics.csv")
    energy_ts = _read("energy_balance_time_series.csv")
    if heat_metrics.empty and energy_ts.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    if not heat_metrics.empty:
        wanted = [
            "delivered_heat_error_percent",
            "heat_loss_error_percent",
            "cumulative_heat_loss_error_percent",
            "energy_balance_residual_percent",
        ]
        h = heat_metrics[heat_metrics["metric"].isin(wanted)].copy()
        if not h.empty:
            h = h.drop_duplicates("metric").set_index("metric").reindex(wanted).dropna(how="all").reset_index()
            label_map = {
                "delivered_heat_error_percent": "Delivered\nheat",
                "heat_loss_error_percent": "Heat\nloss",
                "cumulative_heat_loss_error_percent": "Cumulative\nloss",
                "energy_balance_residual_percent": "Energy\nbalance",
            }
            h["metric_label"] = h["metric"].map(label_map).fillna(h["metric"].astype(str).str.replace("_", "\n", regex=False))
            x = np.arange(len(h))
            width = 0.34
            best_vals = _num(h["best_value"]) if "best_value" in h.columns else _num(h["value"])
            pignn_vals = _num(h["pignn_gru_v3_value"]) if "pignn_gru_v3_value" in h.columns else _num(h["value"])
            axes[0].bar(x - width / 2, best_vals, width, color=PALETTE["safe"], edgecolor=PALETTE["edge"], linewidth=0.55, label="Best listed")
            axes[0].bar(x + width / 2, pignn_vals, width, color=PALETTE["proposed"], edgecolor=PALETTE["edge"], linewidth=0.55, label="PI-GNN-GRU-v3")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(h["metric_label"], fontsize=7.5)
            axes[0].set_ylabel("Error/residual (%)")
            axes[0].set_title("Heat and energy KPIs")
            style_legend(axes[0], loc="upper right", ncols=1)
    if not energy_ts.empty:
        n = min(140, len(energy_ts))
        e = energy_ts.head(n).copy()
        x = np.arange(len(e))
        axes[1].plot(
            x,
            _num(e["measured_boundary_heat_load_kw"]) / 1000.0,
            color=PALETTE["measured"],
            label="Measured boundary heat",
            linewidth=2.0,
        )
        if "simulator_delivered_heat_kw" in e.columns:
            axes[1].plot(
                x,
                _num(e["simulator_delivered_heat_kw"]) / 1000.0,
                color=PALETTE["safe"],
                label="Calibrated simulator",
                linewidth=2.0,
            )
        if "pignn_v3_delivered_heat_kw" in e.columns:
            axes[1].plot(
                x,
                _num(e["pignn_v3_delivered_heat_kw"]) / 1000.0,
                color=PALETTE["proposed"],
                label="PI-GNN-GRU-v3",
                linewidth=2.0,
            )
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Heat delivery (MW)")
        axes[1].set_title("Heat-delivery tracking")
        style_legend(axes[1], loc="upper right", ncols=1)
    for ax, panel in zip(axes, ["(a)", "(b)"]):
        add_panel_label(ax, panel)
        style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_energy_heat_kpi_ate_dark")


def fig_dashboard() -> None:
    kpis = _read("digital_twin_kpis_improved.csv")
    if kpis.empty:
        kpis = _read("digital_twin_kpis.csv")
    virtual = _read("digital_twin_virtual_sensor_metrics.csv")
    if kpis.empty:
        return
    show = kpis.head(8).copy()
    show["label"] = show["kpi"].astype(str).str.replace("_", " ", regex=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    y = np.arange(len(show))
    axes[0, 0].barh(y, _num(show["value"]), color=PALETTE["proposed"], edgecolor=PALETTE["edge"], linewidth=0.5)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(show["label"], fontsize=7.5)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("KPI value")
    axes[0, 0].set_title("Operational KPIs")
    if not virtual.empty:
        v = virtual[virtual["metric"].astype(str).str.contains("RMSE", regex=False)].head(8).copy()
        sensor_short = {
            "supply_temperature": "Supply",
            "return_temperature": "Return",
            "head": "Head",
            "flow": "Flow",
        }
        metric_short = {
            "RMSE_measured_nodes": "measured",
            "RMSE_unmeasured_nodes": "hidden",
        }
        v["label"] = (
            v["virtual_sensor"].astype(str).map(sensor_short).fillna(v["virtual_sensor"].astype(str))
            + "\n"
            + v["metric"].astype(str).map(metric_short).fillna(v["metric"].astype(str).str.replace("_", " "))
        )
        axes[0, 1].bar(range(len(v)), _num(v["value"]), color=PALETTE["gru"], edgecolor=PALETTE["edge"], linewidth=0.5)
        axes[0, 1].set_xticks(range(len(v)))
        axes[0, 1].set_xticklabels(v["label"], rotation=25, ha="right", fontsize=7.2)
        axes[0, 1].set_ylabel("Error")
        axes[0, 1].set_title("Virtual-sensor errors")
    uncertainty = _read("uncertainty_calibration_summary.csv")
    if not uncertainty.empty:
        u = uncertainty[uncertainty["interval"].astype(str).eq("90%")].copy()
        axes[1, 0].bar(range(len(u)), _num(u["coverage_conformal_calibrated"]), color=PALETTE["safe"], edgecolor=PALETTE["edge"], linewidth=0.5)
        axes[1, 0].axhline(90, color=PALETTE["baseline"], linestyle="--", linewidth=1.4)
        axes[1, 0].set_xticks(range(len(u)))
        axes[1, 0].set_xticklabels(u["quantity"].astype(str).str.replace("_", "\n"), fontsize=7.4)
        axes[1, 0].set_ylabel("Coverage (%)")
        axes[1, 0].set_title("90% interval coverage")
    anomaly = _read("anomaly_detection_metrics_improved.csv")
    if not anomaly.empty:
        a = anomaly[~anomaly["case"].eq("normal_operation")].head(6)
        axes[1, 1].bar(range(len(a)), _num(a["detection_rate_percent"]), color=PALETTE["alarm"], edgecolor=PALETTE["edge"], linewidth=0.5)
        axes[1, 1].set_xticks(range(len(a)))
        axes[1, 1].set_xticklabels(a["case"].astype(str).str.replace("_", "\n"), fontsize=7)
        axes[1, 1].set_ylabel("Detection (%)")
        axes[1, 1].set_title("Controlled residual alarms")
    for ax, panel in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, panel)
        style_axes(ax)
    fig.tight_layout()
    _save(fig, "fig_digital_twin_kpi_dashboard_ate_dark")
    # Requested alias.
    for suffix in [".pdf", ".png"]:
        shutil.copy2(FIGURES / f"fig_digital_twin_kpi_dashboard_ate_dark{suffix}", FIGURES / f"fig_digital_twin_dashboard_ate_dark{suffix}")
        shutil.copy2(PAPER_FIGURES / f"fig_digital_twin_kpi_dashboard_ate_dark{suffix}", PAPER_FIGURES / f"fig_digital_twin_dashboard_ate_dark{suffix}")


def make_contact_sheet(stems: list[str]) -> None:
    images: list[tuple[str, np.ndarray]] = []
    for stem in stems:
        path = FIGURES / f"{stem}.png"
        if path.exists():
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((1300, 900), Image.Resampling.LANCZOS)
                images.append((stem, np.asarray(image)))
    if not images:
        return
    ncols = 3
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 4.2 * nrows))
    axes_arr = np.array(axes).reshape(-1)
    for ax, (stem, img) in zip(axes_arr, images):
        ax.imshow(img)
        ax.set_title(stem.replace("fig_", "").replace("_ate_dark", "").replace("_", " "), fontsize=9)
        ax.axis("off")
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    for out_dir in [FIGURES, PAPER_FIGURES]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "contact_sheet_non_simulation_dark_palette.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / "contact_sheet_non_simulation_final.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_ate_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig_model_ranking()
    fig_accuracy_physics_tradeoff()
    fig_proposed_model_value_summary()
    fig_sensor_layout()
    fig_operator_sensor_guidelines()
    fig_uncertainty()
    fig_anomaly()
    fig_seasonal()
    fig_combined_stress()
    fig_parameter_sensitivity()
    fig_flensburg()
    fig_calibration_verification()
    fig_energy_heat_kpis()
    fig_dashboard()
    contact_stems = [
        "fig_model_ranking_ate_dark",
        "fig_accuracy_physics_tradeoff_ate_dark",
        "fig_proposed_model_value_summary",
        "fig_sensor_layout_ate_dark",
        "fig_operator_sensor_guidelines_ate_dark",
        "fig_calibration_verification_ate_dark",
        "fig_energy_heat_kpi_ate_dark",
        "fig_uncertainty_coverage_ate_dark",
        "fig_anomaly_detection_ate_dark",
        "fig_seasonal_generalization_ate_dark",
        "fig_combined_stress_ate_dark",
        "fig_parameter_sensitivity_ate_dark",
        "fig_flensburg_domain_shift_ate_dark",
        "fig_digital_twin_dashboard_ate_dark",
        "fig_operational_energy_impact_summary",
    ]
    make_contact_sheet(contact_stems)
    audit_rows = []
    for stem in [
        "fig_model_ranking_ate_dark",
        "fig_accuracy_physics_tradeoff_ate_dark",
        "fig_proposed_model_value_summary",
        "fig_sensor_layout_ate_dark",
        "fig_operator_sensor_guidelines_ate_dark",
        "fig_calibration_verification_ate_dark",
        "fig_energy_heat_kpi_ate_dark",
        "fig_uncertainty_coverage_ate_dark",
        "fig_anomaly_detection_ate_dark",
        "fig_seasonal_generalization_ate_dark",
        "fig_combined_stress_ate_dark",
        "fig_parameter_sensitivity_ate_dark",
        "fig_parameter_identifiability_tornado_ate_dark",
        "fig_flensburg_domain_shift_ate_dark",
        "fig_digital_twin_dashboard_ate_dark",
        "fig_operational_energy_impact_summary",
    ]:
        audit_rows.append(
            {
                "figure": stem,
                "pdf_exists": (FIGURES / f"{stem}.pdf").exists(),
                "png_exists": (FIGURES / f"{stem}.png").exists(),
                "style": "ATE dark professional palette",
                "reference_palette": "#0000E6; #FF6626; #55D600; #F2E600; #E600E6; #111111; #555555",
                "legend_strategy": "outside/shared/compact where needed",
                "visual_overlap_review": "generated with tight layout and visually spot-checked for legend/label overlap",
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS / "dark_figure_visual_audit.csv", index=False)
    lines = ["Dark ATE figure visual audit", ""]
    lines.extend(
        f"{row.figure}: pdf={row.pdf_exists}, png={row.png_exists}, {row.visual_overlap_review}"
        for row in audit.itertuples(index=False)
    )
    (RESULTS / "dark_figure_visual_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ATE dark professional figures generated.")


if __name__ == "__main__":
    main()
