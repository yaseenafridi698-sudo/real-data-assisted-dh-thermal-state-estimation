from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ate_figure_style import PALETTE, set_ate_style

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures" / "final"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures" / "final"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_matplotlib() -> None:
    set_ate_style()
    plt.rcParams.update({"savefig.dpi": 1200, "axes.titlesize": 10.5})


def save_figure(fig: plt.Figure, stem: str) -> None:
    for out_dir in [ensure_dir(FIGURES_DIR), ensure_dir(PAPER_FIGURES_DIR)]:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


def copy_to_paper_figures(stem: str) -> None:
    ensure_dir(PAPER_FIGURES_DIR)
    for suffix in [".pdf", ".png"]:
        src = FIGURES_DIR / f"{stem}{suffix}"
        dst = PAPER_FIGURES_DIR / f"{stem}{suffix}"
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def load_virtual_field(quantity: str, *, pressure: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = RESULTS_DIR / "virtual_sensor_confidence_intervals_calibrated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required virtual-sensor file: {path}")
    df = pd.read_csv(path)
    sub = df[df["quantity"].eq(quantity)].copy()
    if sub.empty:
        raise ValueError(f"Quantity '{quantity}' not found in {path.name}")
    sub["window_index"] = pd.to_numeric(sub["window_index"], errors="coerce")
    sub["distance_km"] = pd.to_numeric(sub["distance_km"], errors="coerce")
    sub["mean"] = pd.to_numeric(sub["mean"], errors="coerce")
    sub = sub.dropna(subset=["window_index", "distance_km", "mean"])
    pivot = sub.pivot_table(index="window_index", columns="distance_km", values="mean", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    distances = pivot.columns.to_numpy(dtype=float)
    windows = pivot.index.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    if pressure:
        values = values * 9.81

    # Interpolate to a visually smooth distance axis without inventing new evidence;
    # the interpolated surface is a display of the saved simulator-assisted field.
    dense_distances = np.linspace(distances.min(), distances.max(), 81)
    dense_values = np.vstack([np.interp(dense_distances, distances, row) for row in values])

    # The saved windows are not always contiguous. Use ordered windows as elapsed time.
    time_h = np.arange(len(windows), dtype=float) * 0.25
    X, Y = np.meshgrid(dense_distances, time_h)
    return X, Y, dense_values


def infer_sensor_distances(prefer: str = "optimized_three") -> list[float]:
    candidates = [
        RESULTS_DIR / "optimized_sensor_locations_final.json",
        RESULTS_DIR / "optimized_sensor_locations.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        key = "optimized_three_sensor_distances_m" if prefer == "optimized_three" else "optimized_five_sensor_distances_m"
        vals = data.get(key)
        if vals:
            return [float(v) / 1000.0 for v in vals]
    return [0.0, 10.0, 20.0]


def plot_surface(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    stem: str,
    z_label: str,
    colorbar_label: str,
    title: str,
    cmap: str,
    sensor_distances_km: list[float] | None = None,
    z_floor_pad: float = 0.08,
    z_percentiles: tuple[float, float] = (1.0, 99.0),
) -> None:
    configure_matplotlib()
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError(f"No finite values available for {stem}")
    z_min = float(np.nanpercentile(finite, z_percentiles[0]))
    z_max = float(np.nanpercentile(finite, z_percentiles[1]))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        z_min = float(np.nanmin(finite))
        z_max = float(np.nanmax(finite))
    z_range = max(z_max - z_min, 1e-6)
    z_floor = z_min - z_floor_pad * z_range

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        rcount=80,
        ccount=80,
    )
    ax.set_xlabel("Distance from source (km)", labelpad=7)
    ax.set_ylabel("Elapsed test time (h)", labelpad=8)
    ax.set_zlabel(z_label, labelpad=8)
    ax.set_title(title, pad=12)
    ax.view_init(elev=27, azim=-132)
    ax.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
    ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
    ax.set_zlim(z_floor, z_max + 0.04 * z_range)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
    ax.grid(True, alpha=0.25)

    sensors = sensor_distances_km or infer_sensor_distances("optimized_three")
    y_floor = np.full(len(sensors), float(np.nanmin(Y)))
    z_mark = np.full(len(sensors), z_floor)
    ax.scatter(
        sensors,
        y_floor,
        z_mark,
        marker="^",
        s=32,
        c=PALETTE["edge"],
        edgecolors="white",
        linewidths=0.45,
        depthshade=False,
        label="Sparse sensor locations",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.96,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, aspect=20)
    cbar.set_label(colorbar_label)
    save_figure(fig, stem)


def load_heat_loss_density_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    profile_path = RESULTS_DIR / "heat_loss_profile_metrics.csv"
    interval_path = RESULTS_DIR / "virtual_sensor_confidence_intervals_calibrated.csv"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing heat-loss profile file: {profile_path}")
    if not interval_path.exists():
        raise FileNotFoundError(f"Missing virtual-sensor interval file: {interval_path}")
    profile = pd.read_csv(profile_path)
    if "segment_midpoint_km" not in profile.columns:
        raise ValueError("heat_loss_profile_metrics.csv must contain segment_midpoint_km")
    value_col = "pignn_v3_heat_loss_kW" if "pignn_v3_heat_loss_kW" in profile.columns else "simulator_heat_loss_kW"
    x = pd.to_numeric(profile["segment_midpoint_km"], errors="coerce").to_numpy(dtype=float)
    base_segment_kw = pd.to_numeric(profile[value_col], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(x)
    x = x[order]
    base_segment_kw = base_segment_kw[order]
    finite = np.isfinite(x) & np.isfinite(base_segment_kw)
    x = x[finite]
    base_segment_kw = base_segment_kw[finite]
    if x.size < 2:
        raise ValueError("Need at least two heat-loss segments for a 3D heat-loss surface")
    segment_length_km = float(np.median(np.diff(x)))
    if not np.isfinite(segment_length_km) or segment_length_km <= 0:
        segment_length_km = 1.0

    intervals = pd.read_csv(interval_path)
    total = intervals[intervals["quantity"].eq("total_heat_loss")].copy()
    total["window_index"] = pd.to_numeric(total["window_index"], errors="coerce")
    total["mean"] = pd.to_numeric(total["mean"], errors="coerce")
    total = total.dropna(subset=["window_index", "mean"]).sort_values("window_index")
    if total.empty:
        # Fall back to a constant field if only the segment profile exists.
        total_values = np.full(80, np.nansum(base_segment_kw))
    else:
        total_values = total["mean"].to_numpy(dtype=float)

    base_sum = float(np.nansum(base_segment_kw))
    if not np.isfinite(base_sum) or base_sum <= 0:
        base_sum = 1.0
    scale = total_values / base_sum
    segment_kw = np.outer(scale, base_segment_kw)
    density_kw_per_km = segment_kw / segment_length_km
    time_h = np.arange(segment_kw.shape[0], dtype=float) * 0.25

    dense_x = np.linspace(float(x.min()), float(x.max()), 81)
    dense_z = np.vstack([np.interp(dense_x, x, row) for row in density_kw_per_km])
    X, Y = np.meshgrid(dense_x, time_h)
    report = {
        "source_profile_file": str(profile_path.relative_to(PROJECT_ROOT)),
        "source_total_heat_loss_file": str(interval_path.relative_to(PROJECT_ROOT)),
        "heat_loss_value_column": value_col,
        "segment_length_km": segment_length_km,
        "input_unit": "kW per segment",
        "plotted_unit": "kW/km",
        "conversion": "segment_heat_loss_kW / segment_length_km",
    }
    return X, Y, dense_z, report


def write_heat_loss_unit_report(report: dict[str, float | str]) -> None:
    ensure_dir(RESULTS_DIR)
    pd.DataFrame([report]).to_csv(RESULTS_DIR / "main_3d_heat_loss_unit_report.csv", index=False)
    lines = ["Main 3D heat-loss unit report", ""]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    (RESULTS_DIR / "main_3d_heat_loss_unit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_contact_sheet(stems: list[str]) -> None:
    configure_matplotlib()
    images: list[tuple[str, np.ndarray]] = []
    for stem in stems:
        path = FIGURES_DIR / f"{stem}.png"
        if path.exists():
            images.append((stem, mpimg.imread(path)))
    if not images:
        return
    ncols = 2
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.9 * nrows))
    axes_arr = np.array(axes).reshape(-1)
    for ax, (stem, img) in zip(axes_arr, images):
        ax.imshow(img)
        ax.set_title(stem.replace("_", " "), fontsize=9)
        ax.axis("off")
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.tight_layout()
    ensure_dir(FIGURES_DIR)
    ensure_dir(PAPER_FIGURES_DIR)
    fig.savefig(FIGURES_DIR / "contact_sheet_main_3d_figures.png", dpi=450, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / "contact_sheet_main_3d_figures.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippets() -> None:
    latex_dir = ensure_dir(PROJECT_ROOT / "paper" / "latex")
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/final/main_3d_supply_temperature_surface.pdf}
\caption{Three-dimensional supply-temperature field reconstructed over distance and time. Distributed temperature labels are generated by the calibrated thermo-hydraulic simulator, while real operating data provide boundary conditions, calibration, and measured-node thermal validation. Sparse-sensor positions are marked on the floor edge.}
\label{fig:main_3d_supply_temperature}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/final/main_3d_heat_loss_surface.pdf}
\caption{Segment-wise heat-loss density over distance and time. The heat-loss field is derived from calibrated-simulator heat-loss profiles and saved total heat-loss intervals; it is reported in kW/km after segment-length conversion and is not a direct pipe heat-loss measurement.}
\label{fig:main_3d_heat_loss}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/final/main_3d_pressure_surface.pdf}
\caption{Three-dimensional pressure field converted from hydraulic head using water density and gravity. Pressure/head fields are simulator-assisted hidden hydraulic states because dense distributed pressure measurements are not available in the public datasets.}
\label{fig:main_3d_pressure}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/final/main_3d_return_temperature_surface.pdf}
\caption{Supplementary three-dimensional return-temperature field. Distributed return-temperature labels are generated by the calibrated thermo-hydraulic simulator; real measured return temperature supports plant-level calibration and measured-node validation where available.}
\label{fig:supp_3d_return_temperature}
\end{figure}
"""
    (latex_dir / "main_3d_figure_snippets.tex").write_text(text, encoding="utf-8")
