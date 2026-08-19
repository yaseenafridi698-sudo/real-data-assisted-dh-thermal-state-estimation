from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

from ate_concept_figure_style import COLORS, FIGURES, contact_sheet, read_csv, save_figure, set_style


COMBINED_STEMS = [
    "fig_thermo_hydraulic_reconstruction_summary",
    "fig_heat_energy_balance_summary",
    "fig_operational_energy_pressure_summary",
    "fig_uncertainty_anomaly_summary",
    "fig_seasonal_stress_sensitivity_summary",
]

MODEL_COLORS = {
    "GRU-MSE": COLORS["blue"],
    "Transformer-MSE": COLORS["orange"],
    "PureGNN-MSE": COLORS["gray"],
    "PI-LSTM": COLORS["green"],
    "Proposed PI-GNN-GRU-v3 balanced_mode": COLORS["magenta"],
    "Proposed PI-GNN-GRU-v3 accuracy_mode": COLORS["magenta"],
    "Proposed PI-GNN-GRU-v3 physics_mode": COLORS["magenta"],
}

PREFERRED_MODELS = [
    "GRU-MSE",
    "Transformer-MSE",
    "PI-LSTM",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
]


def _short_model(name: object) -> str:
    text = str(name)
    return (
        text.replace("Proposed PI-GNN-GRU-v3 balanced_mode", "PI-GNN-v3\nbalanced")
        .replace("Proposed PI-GNN-GRU-v3 accuracy_mode", "PI-GNN-v3\naccuracy")
        .replace("Proposed PI-GNN-GRU-v3 physics_mode", "PI-GNN-v3\nphysics")
        .replace("Transformer-MSE", "Transformer")
        .replace("PI-LSTM", "PI-LSTM")
        .replace("GRU-MSE", "GRU")
    )


def _color_for(label: object) -> str:
    return MODEL_COLORS.get(str(label), COLORS["blue"])


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["black"],
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
    )


def _finish_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    if grid:
        ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _numeric(values: pd.Series | list | np.ndarray) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _thin(df: pd.DataFrame, max_points: int = 260) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, int(math.ceil(len(df) / max_points)))
    return df.iloc[::step].copy()


def _safe_series(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    if df.empty or col not in df.columns:
        return np.full(len(df), default)
    return _numeric(df[col])


def _value(summary: pd.DataFrame, source: str, kpi: str) -> float:
    if summary.empty:
        return np.nan
    sub = summary[
        summary.get("source", pd.Series(dtype=str)).astype(str).eq(source)
        & summary.get("kpi", pd.Series(dtype=str)).astype(str).eq(kpi)
    ]
    if sub.empty:
        return np.nan
    return pd.to_numeric(sub.iloc[0].get("value"), errors="coerce")


def _metric_frame(name: str, metric: str, models: list[str] | None = None) -> pd.DataFrame:
    df = read_csv(name)
    if df.empty or "metric" not in df.columns:
        return pd.DataFrame()
    sub = df[df["metric"].astype(str).eq(metric)].copy()
    if sub.empty:
        return sub
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"])
    if models is not None and "model" in sub.columns:
        sub = sub[sub["model"].astype(str).isin(models)]
        sub["order"] = sub["model"].astype(str).map({m: i for i, m in enumerate(models)})
        sub = sub.sort_values("order")
    return sub


def _bar_panel(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    colors: list[str] | None = None,
    rotate: int = 0,
) -> None:
    x = np.arange(len(labels))
    vals = np.array(values, dtype=float)
    plot_vals = np.nan_to_num(vals, nan=0.0)
    colors = colors or [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["magenta"], COLORS["gray"]]
    bars = ax.bar(x, plot_vals, color=colors[: len(labels)], edgecolor=COLORS["black"], linewidth=0.9, zorder=3)
    ax.set_title(title, fontweight="bold", fontsize=9.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmin(finite) >= 0:
        ymax = max(np.nanmax(finite) * 1.18, 1.0)
        ax.set_ylim(0, ymax)
    for bar, value in zip(bars, vals):
        if not np.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.04,
                "N/A",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                color=COLORS["gray"],
            )
        else:
            label = f"{value:.2f}" if abs(value) < 100 else f"{value:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=COLORS["black"],
            )
    _finish_axis(ax)


def _imshow_panel(ax: plt.Axes, stem: str, title: str) -> None:
    path = FIGURES / f"{stem}.png"
    if not path.exists():
        ax.text(0.5, 0.5, "Panel not available", ha="center", va="center", color=COLORS["gray"])
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        return
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is not None:
        left, upper, right, lower = bbox
        pad_x = max(8, int(0.015 * img.width))
        pad_y = max(8, int(0.015 * img.height))
        img = img.crop((max(0, left - pad_x), max(0, upper - pad_y), min(img.width, right + pad_x), min(img.height, lower + pad_y)))
    ax.imshow(np.asarray(img), aspect="auto")
    ax.set_title(title, fontweight="bold", fontsize=9.2)
    ax.axis("off")


def make_thermo_hydraulic_reconstruction_summary() -> None:
    set_style()
    panels = [
        ("fig5_supply_temperature_reconstruction", "Supply temperature"),
        ("fig6_return_temperature_reconstruction", "Return temperature"),
        ("fig_temperature_error_spacetime", "Supply-temperature error"),
        ("fig_return_temperature_error_spacetime", "Return-temperature error"),
        ("fig7_pressure_head_reconstruction", "Pressure/head field"),
        ("fig8_flow_reconstruction", "Flow reconstruction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.6))
    for ax, (stem, title), label in zip(axes.ravel(), panels, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        _imshow_panel(ax, stem, title)
        _panel_label(ax, label)
    fig.tight_layout(pad=0.35)
    save_figure(fig, "fig_thermo_hydraulic_reconstruction_summary")


def make_heat_energy_balance_summary() -> None:
    set_style()
    ts_full = read_csv("operational_energy_impact_timeseries.csv")
    heat_metric = _metric_frame("heat_energy_metrics.csv", "heat_loss_error_percent", PREFERRED_MODELS)
    source_order = ["measured_boundary", "calibrated_simulator", "pignn_v3_balanced"]
    source_labels = {
        "measured_boundary": "Measured boundary",
        "calibrated_simulator": "Calibrated simulator",
        "pignn_v3_balanced": "PI-GNN-v3 balanced",
    }
    source_colors = {
        "measured_boundary": COLORS["black"],
        "calibrated_simulator": COLORS["blue"],
        "pignn_v3_balanced": COLORS["magenta"],
    }

    def plot_sources(ax: plt.Axes, y_col: str, title: str, ylabel: str, *, scale: float = 1.0, cumulative: bool = False, ratio: bool = False) -> None:
        for source in source_order:
            if source == "measured_boundary" and y_col in {"heat_loss_kw", "energy_balance_residual_percent"}:
                continue
            sub = ts_full[ts_full.get("source", pd.Series(dtype=str)).astype(str).eq(source)].copy()
            if sub.empty:
                continue
            sub = _thin(sub, 220)
            x = _safe_series(sub, "time_h")
            if not np.isfinite(x).any():
                x = np.arange(len(sub))
            if ratio:
                y = 100.0 * _safe_series(sub, "heat_loss_kw") / np.maximum(_safe_series(sub, "delivered_heat_kw"), 1e-9)
            elif cumulative:
                y = np.nancumsum(np.nan_to_num(_safe_series(sub, y_col), nan=0.0)) * 0.25 / 1000.0
            else:
                y = _safe_series(sub, y_col) * scale
            if not np.isfinite(y).any():
                continue
            ax.plot(x, y, color=source_colors[source], linewidth=1.75, label=source_labels[source])
        ax.set_title(title, fontweight="bold", fontsize=9.6)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(ylabel)
        _finish_axis(ax)

    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.5))
    axes = axes.ravel()
    plot_sources(axes[0], "delivered_heat_kw", "Delivered heat", "Heat (MW)", scale=1 / 1000.0)
    plot_sources(axes[1], "heat_loss_kw", "Heat loss", "Heat loss (kW)")
    plot_sources(axes[2], "heat_loss_kw", "Cumulative heat loss", "Heat loss (MWh)", cumulative=True)
    plot_sources(axes[3], "energy_balance_residual_percent", "Energy-balance residual", "Residual (%)")
    for ax, label in zip(axes[:4], ["(a)", "(b)", "(c)", "(d)"]):
        _panel_label(ax, label)

    if not heat_metric.empty:
        labels = [_short_model(v) for v in heat_metric["model"].astype(str)]
        values = heat_metric["value"].astype(float).to_list()
        colors = [_color_for(v) for v in heat_metric["model"].astype(str)]
    else:
        labels, values, colors = ["N/A"], [np.nan], [COLORS["gray"]]
    _bar_panel(
        axes[4],
        labels,
        values,
        title="Heat-loss error",
        ylabel="Error (%)",
        colors=colors,
        rotate=25,
    )
    _panel_label(axes[4], "(e)")

    plot_sources(axes[5], "heat_loss_kw", "Operational heat-loss ratio", "Heat-loss ratio (%)", ratio=True)
    _panel_label(axes[5], "(f)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.9)
    save_figure(fig, "fig_heat_energy_balance_summary")


def make_operational_energy_pressure_summary() -> None:
    set_style()
    summary = read_csv("operational_energy_impact_summary.csv")
    sources = ["measured_boundary", "calibrated_simulator", "pignn_v3_balanced"]
    labels = ["Measured\nboundary", "Calibrated\nsimulator", "PI-GNN-v3\nbalanced"]
    colors = [COLORS["black"], COLORS["blue"], COLORS["magenta"]]
    panels = [
        ("delivered_heat_MWh_per_day", "Delivered heat", "MWh/day"),
        ("heat_loss_MWh_per_day", "Heat loss", "MWh/day"),
        ("pump_energy_proxy_kWh_per_day", "Pump-energy proxy", "kWh/day"),
        ("pressure_drop_residual_percent", "Pressure-drop residual", "%"),
        ("mean_energy_balance_residual_percent", "Energy-balance residual", "%"),
        ("cost_proxy_EUR_per_day", "Cost / CO2 proxy", "EUR/day"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.4))
    for ax, (kpi, title, unit), label in zip(axes.ravel(), panels, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        if kpi == "pressure_drop_residual_percent":
            _bar_panel(
                ax,
                ["Benchmark"],
                [_value(summary, "benchmark_metrics", "pressure_drop_residual_percent")],
                title=title,
                ylabel=unit,
                colors=[COLORS["orange"]],
            )
            _panel_label(ax, label)
            continue
        values = [_value(summary, src, kpi) for src in sources]
        if kpi in {"heat_loss_MWh_per_day", "pump_energy_proxy_kWh_per_day", "mean_energy_balance_residual_percent"}:
            values[0] = np.nan
        if kpi == "cost_proxy_EUR_per_day":
            _bar_panel(ax, labels, values, title=title, ylabel=unit, colors=colors)
            ax2 = ax.twinx()
            co2 = [_value(summary, src, "CO2_proxy_kg_per_day") for src in sources]
            ax2.plot(np.arange(len(labels)), co2, marker="D", color=COLORS["orange"], linewidth=1.6, markersize=4.6)
            ax2.set_ylabel("CO2 proxy (kg/day)")
            ax2.spines["top"].set_visible(False)
        else:
            _bar_panel(ax, labels, values, title=title, ylabel=unit, colors=colors)
        _panel_label(ax, label)
    fig.tight_layout(pad=0.9)
    save_figure(fig, "fig_operational_energy_pressure_summary")


def make_uncertainty_anomaly_summary() -> None:
    set_style()
    ci = read_csv("virtual_sensor_confidence_intervals_calibrated.csv")
    coverage = read_csv("uncertainty_calibration_summary.csv")
    anomaly_metrics = read_csv("anomaly_detection_metrics_improved.csv")
    anomaly_ts = _thin(read_csv("anomaly_detection_timeseries_improved.csv"), 280)
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.5))
    axes = axes.ravel()

    def band_panel(ax: plt.Axes, quantity: str, title: str, ylabel: str, color: str) -> None:
        sub = ci[ci.get("quantity", pd.Series(dtype=str)).astype(str).eq(quantity)].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", color=COLORS["gray"], fontweight="bold")
        else:
            node = sub["node"].iloc[0] if "node" in sub.columns else None
            if node is not None:
                sub = sub[sub["node"].eq(node)]
            sub = _thin(sub, 220)
            x_local = _safe_series(sub, "window_index")
            mean = _safe_series(sub, "mean")
            lo = _safe_series(sub, "lower_90")
            hi = _safe_series(sub, "upper_90")
            ax.fill_between(x_local, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(x_local, mean, color=color, linewidth=1.8)
        ax.set_title(title, fontweight="bold", fontsize=9.6)
        ax.set_xlabel("Window")
        ax.set_ylabel(ylabel)
        _finish_axis(ax)

    band_panel(axes[0], "supply_temperature", "Temperature uncertainty band", r"Temperature ($^\circ$C)", COLORS["blue"])
    _panel_label(axes[0], "(a)")
    band_panel(axes[1], "total_heat_loss", "Heat-loss uncertainty band", "Total heat loss (MWh)", COLORS["orange"])
    _panel_label(axes[1], "(b)")

    cov_sub = coverage[coverage.get("quantity", pd.Series(dtype=str)).astype(str).isin(["supply_temperature", "return_temperature", "heat_loss", "flow"])]
    cov_sub = cov_sub[cov_sub.get("interval", pd.Series(dtype=str)).astype(str).eq("90%")]
    if cov_sub.empty:
        _bar_panel(axes[2], ["N/A"], [np.nan], title="Uncertainty coverage", ylabel="Coverage (%)", colors=[COLORS["gray"]])
    else:
        labels = [str(q).replace("_", "\n") for q in cov_sub["quantity"]]
        vals = _numeric(cov_sub["coverage_conformal_calibrated"]).tolist()
        _bar_panel(
            axes[2],
            labels,
            vals,
            title="Uncertainty coverage",
            ylabel="Coverage (%)",
            colors=[COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["magenta"]],
        )
        axes[2].axhline(90, color=COLORS["black"], linestyle="--", linewidth=1.0)
    _panel_label(axes[2], "(c)")

    if anomaly_ts.empty:
        axes[3].text(0.5, 0.5, "N/A", ha="center", va="center", color=COLORS["gray"], fontweight="bold")
    else:
        case = "combined_stress_severe" if "combined_stress_severe" in set(anomaly_ts["case"].astype(str)) else anomaly_ts["case"].astype(str).iloc[0]
        sub = anomaly_ts[anomaly_ts["case"].astype(str).eq(case)]
        x_local = _safe_series(sub, "window_index")
        axes[3].plot(x_local, _safe_series(sub, "residual_score"), color=COLORS["magenta"], linewidth=1.7, label="Residual score")
        axes[3].plot(x_local, _safe_series(sub, "warning_threshold"), color=COLORS["orange"], linestyle="--", linewidth=1.2, label="Warning")
        axes[3].plot(x_local, _safe_series(sub, "alarm_threshold"), color=COLORS["black"], linestyle=":", linewidth=1.2, label="Alarm")
        axes[3].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=True, fontsize=7.0)
    axes[3].set_title("Anomaly residual score", fontweight="bold", fontsize=9.6)
    axes[3].set_xlabel("Window")
    axes[3].set_ylabel("Score")
    _panel_label(axes[3], "(d)")
    _finish_axis(axes[3])

    if anomaly_metrics.empty:
        labels, det, fa = ["N/A"], [np.nan], [np.nan]
    else:
        rows = anomaly_metrics[anomaly_metrics["case"].astype(str).ne("normal_operation")].head(4)
        labels = [str(c).replace("_", "\n") for c in rows["case"]]
        det = _numeric(rows["detection_rate_percent"]).tolist()
        fa = _numeric(rows["false_alarm_rate_percent"]).tolist()
    xloc = np.arange(len(labels))
    width = 0.36
    axes[4].bar(xloc - width / 2, np.nan_to_num(det, nan=0), width, color=COLORS["green"], edgecolor=COLORS["black"], label="Detection")
    axes[4].bar(xloc + width / 2, np.nan_to_num(fa, nan=0), width, color=COLORS["yellow"], edgecolor=COLORS["black"], label="False alarm")
    axes[4].set_title("Warning/alarm threshold", fontweight="bold", fontsize=9.6)
    axes[4].set_ylabel("Rate (%)")
    axes[4].set_xticks(xloc)
    axes[4].set_xticklabels(labels, rotation=25, ha="right")
    axes[4].legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=True, fontsize=7.0)
    _panel_label(axes[4], "(e)")
    _finish_axis(axes[4])

    if anomaly_ts.empty:
        _bar_panel(axes[5], ["N/A"], [np.nan], title="Sensor-health status", ylabel="Score", colors=[COLORS["gray"]])
    else:
        health = anomaly_ts.groupby("case", as_index=False)["sensor_health_score"].mean().head(5)
        labels = [str(c).replace("_", "\n") for c in health["case"]]
        _bar_panel(
            axes[5],
            labels,
            _numeric(health["sensor_health_score"]).tolist(),
            title="Sensor-health status",
            ylabel="Mean health score",
            colors=[COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["magenta"], COLORS["gray"]],
            rotate=25,
        )
    _panel_label(axes[5], "(f)")
    fig.tight_layout(pad=0.9)
    save_figure(fig, "fig_uncertainty_anomaly_summary")


def make_seasonal_stress_sensitivity_summary() -> None:
    set_style()
    seasonal = read_csv("seasonal_generalization.csv")
    stress = read_csv("combined_stress_test.csv")
    sens = read_csv("parameter_identifiability_sensitivity.csv")
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.55))
    axes = axes.ravel()

    if not seasonal.empty:
        rows = seasonal[seasonal["model"].astype(str).isin(PREFERRED_MODELS)].copy()
        rows = rows.groupby("regime", as_index=False)["supply_RMSE_C"].mean()
        _bar_panel(
            axes[0],
            [str(v).replace("_", "\n") for v in rows["regime"]],
            _numeric(rows["supply_RMSE_C"]).tolist(),
            title="Seasonal split performance",
            ylabel=r"Supply RMSE ($^\circ$C)",
            colors=[COLORS["blue"], COLORS["orange"], COLORS["green"]],
            rotate=20,
        )
    _panel_label(axes[0], "(a)")

    if not stress.empty:
        rows = stress[stress["model"].astype(str).isin(["GRU-MSE", "Transformer-MSE", "Proposed PI-GNN-GRU-v3 balanced_mode"])].copy()
        rows = rows.groupby("case", as_index=False)["energy_balance_residual_percent"].mean().tail(5)
        _bar_panel(
            axes[1],
            [str(v).replace("_", "\n") for v in rows["case"]],
            _numeric(rows["energy_balance_residual_percent"]).tolist(),
            title="Combined stress performance",
            ylabel="Energy residual (%)",
            colors=[COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["magenta"], COLORS["gray"]],
            rotate=24,
        )
    _panel_label(axes[1], "(b)")

    if not sens.empty:
        base = sens[sens["case"].astype(str).eq("baseline")]
        baseline = pd.to_numeric(base["heat_loss_error_percent"], errors="coerce").mean() if not base.empty else 0.0
        rows = sens.copy()
        rows["delta"] = pd.to_numeric(rows["heat_loss_error_percent"], errors="coerce") - baseline
        top = rows[rows["parameter"].astype(str).ne("baseline")].groupby("parameter", as_index=False)["delta"].apply(lambda s: s.abs().max())
        top = top.rename(columns={"delta": "max_abs_delta"}).sort_values("max_abs_delta", ascending=True).tail(6)
        y = np.arange(len(top))
        axes[2].barh(y, top["max_abs_delta"], color=COLORS["orange"], edgecolor=COLORS["black"], zorder=3)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels([str(v).replace("_", " ") for v in top["parameter"]])
        axes[2].set_title("Parameter sensitivity tornado", fontweight="bold", fontsize=9.6)
        axes[2].set_xlabel("Max heat-loss error change (%)")
        _finish_axis(axes[2])
    _panel_label(axes[2], "(c)")

    def sensitivity_panel(ax: plt.Axes, key: str, title: str, ylabel: str, label: str) -> None:
        if sens.empty:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", color=COLORS["gray"], fontweight="bold")
        else:
            rows = sens[sens["parameter"].astype(str).str.contains(key, case=False, na=False)].copy()
            if rows.empty:
                rows = sens[sens["case"].astype(str).str.contains(key, case=False, na=False)].copy()
            rows = rows.groupby("perturbation", as_index=False)[ylabel].mean() if ylabel in rows.columns and not rows.empty else pd.DataFrame()
            if rows.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", color=COLORS["gray"], fontweight="bold")
            else:
                vals = _numeric(rows[ylabel])
                x = np.arange(len(rows))
                ax.plot(x, vals, marker="o", color=COLORS["blue"], linewidth=2.0)
                ax.set_xticks(x)
                ax.set_xticklabels([str(v).replace("_", "\n") for v in rows["perturbation"]], rotation=18, ha="right")
        ax.set_title(title, fontweight="bold", fontsize=9.6)
        ax.set_ylabel(label)
        _finish_axis(ax)

    sensitivity_panel(axes[3], "heat_loss", "Heat-loss coefficient sensitivity", "heat_loss_error_percent", "Heat-loss error (%)")
    _panel_label(axes[3], "(d)")
    sensitivity_panel(axes[4], "friction", "Friction-factor sensitivity", "pressure_drop_error_percent", "Pressure-drop error (%)")
    _panel_label(axes[4], "(e)")
    sensitivity_panel(axes[5], "return", "Return-temperature bias sensitivity", "return_RMSE_C", r"Return RMSE ($^\circ$C)")
    _panel_label(axes[5], "(f)")
    fig.tight_layout(pad=0.9)
    save_figure(fig, "fig_seasonal_stress_sensitivity_summary")


def main() -> None:
    make_thermo_hydraulic_reconstruction_summary()
    make_heat_energy_balance_summary()
    make_operational_energy_pressure_summary()
    make_uncertainty_anomaly_summary()
    make_seasonal_stress_sensitivity_summary()
    contact_sheet(COMBINED_STEMS, "contact_sheet_combined_simulation_figures.png", title="Combined simulation figure package")


if __name__ == "__main__":
    main()
