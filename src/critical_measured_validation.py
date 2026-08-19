from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Numerical validation remains runnable in minimal environments.
    plt = None


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sonderborg"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "final"
PAPER_FIGURES = ROOT / "paper" / "figures" / "final"
TABLES = ROOT / "paper" / "tables"

BLUE = "#0000E6"
ORANGE = "#FF6626"
GREEN = "#55A800"
MAGENTA = "#C000C0"
BLACK = "#111111"
GRAY = "#555555"


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    err = pred - y
    return {
        "n_samples": int(mask.sum()),
        "RMSE_C": float(np.sqrt(np.mean(err**2))),
        "MAE_C": float(np.mean(np.abs(err))),
        "Bias_C": float(np.mean(err)),
        "MaxAE_C": float(np.max(np.abs(err))),
    }


def _feature_matrix(df: pd.DataFrame, include_load: str = "current", include_target: bool = False) -> tuple[np.ndarray, list[str]]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    hour = ts.dt.hour.to_numpy(float) + ts.dt.minute.to_numpy(float) / 60.0
    doy = ts.dt.dayofyear.to_numpy(float)
    cols = [
        np.ones(len(df)),
        df["supply_temp_C"].to_numpy(float),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
    ]
    names = ["intercept", "supply_temp_C", "hour_sin", "hour_cos", "year_sin", "year_cos"]
    if "lag_return_C" in df:
        cols.append(df["lag_return_C"].to_numpy(float))
        names.append("return_temp_C[k-1]")
    if include_load == "lagged":
        cols.append(np.log1p(np.maximum(df["lag_heat_load_kw"].to_numpy(float), 0.0)))
        names.append("log1p_heat_load_kw[k-1]")
    elif include_load == "current":
        cols.append(np.log1p(np.maximum(df["heat_load_kw"].to_numpy(float), 0.0)))
        names.append("log1p_heat_load_kw[k]")
    if "ambient_temp_C" in df:
        cols.append(df["ambient_temp_C"].to_numpy(float))
        names.append("ambient_temp_C[k]")
    if include_target:
        cols.append(df["return_temp_C"].to_numpy(float))
        names.append("return_temp_C[k] (oracle leakage)")
    return np.column_stack(cols), names


def _ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    mu = np.nanmean(train_x[:, 1:], axis=0)
    sd = np.nanstd(train_x[:, 1:], axis=0)
    sd[sd < 1e-9] = 1.0
    xtr = train_x.copy()
    xte = test_x.copy()
    xtr[:, 1:] = (xtr[:, 1:] - mu) / sd
    xte[:, 1:] = (xte[:, 1:] - mu) / sd
    reg = np.eye(xtr.shape[1]) * ridge
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtr.T @ xtr + reg, xtr.T @ train_y)
    return xte @ beta


def load_plant_level() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(RAW.glob("sfjv_dhs_data_*.csv")):
        raw = pd.read_csv(path, sep=";")
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True)
        for plant in range(1, 8):
            part = pd.DataFrame(
                {
                    "timestamp": raw["date"],
                    "plant_id": f"plant_{plant}",
                    "heat_load_kw": pd.to_numeric(raw[f"plant{plant}_heat_load"], errors="coerce") * 1000.0,
                    "supply_temp_C": pd.to_numeric(raw[f"plant{plant}_temp_feed_flow"], errors="coerce"),
                    "return_temp_C": pd.to_numeric(raw[f"plant{plant}_temp_back_flow"], errors="coerce"),
                }
            )
            frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["timestamp", "heat_load_kw", "supply_temp_C", "return_temp_C"])
    out = out[
        out["heat_load_kw"].ge(50.0)
        & out["supply_temp_C"].between(30.0, 120.0)
        & out["return_temp_C"].between(10.0, 95.0)
    ].copy()
    out["year"] = out["timestamp"].dt.year
    out["source_dataset"] = "sonderborg_seven_plant_raw"
    out["state_type"] = "real_measured_node"
    out = out.sort_values(["timestamp", "plant_id"]).reset_index(drop=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "sonderborg_plant_level_processed.csv", index=False)
    return out


def plant_withholding(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plant in sorted(data["plant_id"].unique()):
        # Strongest test: a plant and a future year are both absent from model fitting.
        train = data[(data["plant_id"] != plant) & (data["year"] <= 2018)].copy()
        test = data[(data["plant_id"] == plant) & (data["year"] == 2019)].copy()
        if len(test) < 100:
            continue
        xtr, names = _feature_matrix(train, include_load="current")
        xte, _ = _feature_matrix(test, include_load="current")
        pred = _ridge_fit_predict(xtr, train["return_temp_C"].to_numpy(float), xte)
        result = _metrics(test["return_temp_C"].to_numpy(float), pred)
        rows.append(
            {
                "protocol": "combined_plant_and_time_withholding",
                "held_out_plant": plant,
                "train_years": "2016-2018",
                "test_year": 2019,
                "estimator": "pooled ridge regression",
                "available_target_plant_inputs": "current measured supply and heat load; calendar terms; no target return",
                "feature_names": "; ".join(names),
                **result,
                "state_type": "real_measured_node",
                "safe_interpretation": "blind measured return-temperature transfer to an unseen plant and future year; plants are separate production sites, not pipe nodes",
            }
        )

        persistence = test["return_temp_C"].shift(1)
        pm = _metrics(test["return_temp_C"].to_numpy(float)[1:], persistence.to_numpy(float)[1:])
        rows.append(
            {
                "protocol": "within_test_online_persistence_reference",
                "held_out_plant": plant,
                "train_years": "none",
                "test_year": 2019,
                "estimator": "last observation persistence",
                "available_target_plant_inputs": "previous measured return",
                "feature_names": "return_temp_C[k-1]",
                **pm,
                "state_type": "real_measured_node",
                "safe_interpretation": "strong online temporal reference; not a fully plant-blind estimator because past target measurements are available",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sonderborg_blind_plant_validation.csv", index=False)
    summary = (
        out.groupby(["protocol", "estimator"], as_index=False)
        .agg(
            plants=("held_out_plant", "nunique"),
            total_samples=("n_samples", "sum"),
            mean_RMSE_C=("RMSE_C", "mean"),
            std_RMSE_C=("RMSE_C", "std"),
            median_RMSE_C=("RMSE_C", "median"),
            mean_MAE_C=("MAE_C", "mean"),
        )
    )
    summary["state_type"] = "real_measured_node"
    summary["safe_claim"] = np.where(
        summary["protocol"].eq("combined_plant_and_time_withholding"),
        "measured multi-plant/future-year transfer; not spatial validation along one pipe",
        "online persistence reference with past target measurements",
    )
    summary.to_csv(RESULTS / "sonderborg_blind_plant_validation_summary.csv", index=False)
    return out


def leave_one_year_out(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in sorted(data["year"].unique()):
        train = data[data["year"] != year].copy()
        test = data[data["year"] == year].copy()
        xtr, _ = _feature_matrix(train, include_load="current")
        xte, _ = _feature_matrix(test, include_load="current")
        pred = _ridge_fit_predict(xtr, train["return_temp_C"].to_numpy(float), xte)
        rows.append(
            {
                "held_out_year": int(year),
                "estimator": "pooled ridge regression",
                **_metrics(test["return_temp_C"].to_numpy(float), pred),
                "state_type": "real_measured_node",
                "safe_interpretation": "leave-one-year-out measured return-temperature prediction across seven plants",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sonderborg_leave_one_year_out.csv", index=False)
    return out


def causal_load_ablation() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "sonderborg_processed.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").copy()
    interval_minutes = df["timestamp"].diff().dt.total_seconds().div(60.0)
    df["trajectory_start"] = interval_minutes.isna() | interval_minutes.gt(22.5)
    df["trajectory_id"] = df["trajectory_start"].cumsum().astype(int)
    # Never form a causal feature from the last sample before an observed gap.
    df["lag_return_C"] = df.groupby("trajectory_id", sort=False)["return_temp_C"].shift(1)
    df["lag_heat_load_kw"] = df.groupby("trajectory_id", sort=False)["heat_load_kw"].shift(1)
    required = ["supply_temp_C", "return_temp_C", "heat_load_kw", "lag_heat_load_kw", "lag_return_C", "ambient_temp_C"]
    df = df.dropna(subset=required).reset_index(drop=True)
    n = len(df)
    train_end = int(0.70 * n)
    val_start = train_end + 11
    val_end = val_start + int(0.15 * n)
    test_start = val_end + 11
    train = df.iloc[:train_end]
    test = df.iloc[test_start:]
    variants = [
        ("past_return_no_load", "none", False, True, "primary causal: current supply/ambient and measured return history"),
        ("past_return_lagged_load", "lagged", False, True, "primary causal with heat load available only through k-1"),
        ("past_return_current_load", "current", False, True, "diagnostic using current heat-load signal; possible target-derived information"),
        ("oracle_current_return", "current", True, False, "invalid oracle containing the prediction target; reported only to expose leakage"),
    ]
    rows: list[dict[str, object]] = []
    for name, load_mode, include_target, causal, note in variants:
        xtr, features = _feature_matrix(train, include_load=load_mode, include_target=include_target)
        xte, _ = _feature_matrix(test, include_load=load_mode, include_target=include_target)
        pred = _ridge_fit_predict(xtr, train["return_temp_C"].to_numpy(float), xte)
        rows.append(
            {
                "variant": name,
                "causal_operational_candidate": causal,
                "current_heat_load_used": load_mode == "current",
                "current_return_target_used": include_target,
                "features": "; ".join(features),
                **_metrics(test["return_temp_C"].to_numpy(float), pred),
                "state_type": "real_measured_node",
                "interpretation": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "causal_heat_load_input_ablation.csv", index=False)
    dependency = pd.DataFrame(
        [
            ["return temperature at k", "supply[k], ambient[k], return[k-1]", "yes", "primary causal no-load test"],
            ["return temperature at k", "supply[k], ambient[k], return[k-1], load[k-1]", "yes", "primary causal lagged-load test"],
            ["return temperature at k", "supply[k], ambient[k], return[k-1], load[k]", "conditional", "current heat load may be meter-derived from same-timestamp flow and delta-T"],
            ["return temperature at k", "supply[k], ambient[k], return[k-1], load[k], return[k]", "no", "oracle leakage control; never used as evidence"],
        ],
        columns=["target", "timestamp_level_inputs", "causal_for_target", "evidence_role"],
    )
    dependency.to_csv(RESULTS / "causal_timestamp_dependency_table.csv", index=False)
    return out


def _latex_escape(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def make_tables(plant: pd.DataFrame, causal: pd.DataFrame, year: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    p = plant[plant["protocol"].eq("combined_plant_and_time_withholding")].copy()
    lines = [
        "\\begin{table}[t]", "\\centering", "\\caption{Blind S{\\o}nderborg plant-and-time withholding using real measured plant data. Current target-plant supply and load are available, while all target-plant return temperatures and the 2019 test period are excluded from fitting. Plants are separate production sites, not nodes on one pipe.}",
        "\\label{tab:plant_withholding}", "\\small", "\\begin{tabular}{lrrrr}", "\\toprule", "Plant & Samples & RMSE ($^\\circ$C) & MAE ($^\\circ$C) & Bias ($^\\circ$C) \\\\", "\\midrule",
    ]
    for _, r in p.iterrows():
        lines.append(f"{_latex_escape(r['held_out_plant'])} & {int(r['n_samples'])} & {r['RMSE_C']:.3f} & {r['MAE_C']:.3f} & {r['Bias_C']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "table_sonderborg_blind_plant_validation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{table}[t]", "\\centering", "\\caption{Timestamp-level causal input ablation for blind return-temperature prediction. The current-load row is diagnostic because heat-meter load may contain same-timestamp flow and temperature-difference information; the oracle row is intentionally invalid.}",
        "\\label{tab:causal_load_ablation}", "\\small", "\\begin{tabular}{lrrrl}", "\\toprule", "Input configuration & RMSE ($^\\circ$C) & MAE ($^\\circ$C) & Causal & Evidence role \\\\", "\\midrule",
    ]
    for _, r in causal.iterrows():
        role = "primary" if bool(r["causal_operational_candidate"]) else "oracle only"
        lines.append(f"{_latex_escape(r['variant'])} & {r['RMSE_C']:.3f} & {r['MAE_C']:.3f} & {'yes' if r['causal_operational_candidate'] else 'no'} & {role} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "table_causal_heat_load_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "\\begin{table}[t]", "\\centering", "\\caption{Leave-one-year-out return-temperature prediction across all seven S{\\o}nderborg plants. These are measured-node temporal-transfer results.}",
        "\\label{tab:leave_one_year_out}", "\\small", "\\begin{tabular}{rrrr}", "\\toprule", "Held-out year & Samples & RMSE ($^\\circ$C) & MAE ($^\\circ$C) \\\\", "\\midrule",
    ]
    for _, r in year.iterrows():
        lines.append(f"{int(r['held_out_year'])} & {int(r['n_samples'])} & {r['RMSE_C']:.3f} & {r['MAE_C']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLES / "table_sonderborg_leave_one_year_out.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(plant: pd.DataFrame, causal: pd.DataFrame, year: pd.DataFrame) -> None:
    if plt is None:
        (RESULTS / "critical_measured_validation_figure_status.txt").write_text(
            "Figure not regenerated: Matplotlib is unavailable in this runtime. "
            "The measured-data CSVs and LaTeX tables were generated successfully.\n",
            encoding="utf-8",
        )
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9.5, "axes.labelsize": 9.5,
        "axes.titlesize": 10.5, "legend.fontsize": 8.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.45), constrained_layout=True)
    p = plant[plant["protocol"].eq("combined_plant_and_time_withholding")]
    axes[0].bar(p["held_out_plant"].str.replace("plant_", "P"), p["RMSE_C"], color=BLUE, edgecolor=BLACK, linewidth=0.8)
    axes[0].set_ylabel("Return-temperature RMSE ($^\\circ$C)")
    axes[0].set_title("Plant + future-year withholding")
    axes[0].tick_params(axis="x", rotation=0)

    c = causal[~causal["variant"].eq("oracle_current_return")]
    labels = ["No load", "Lagged load", "Current load"]
    axes[1].bar(labels, c["RMSE_C"], color=[GRAY, GREEN, ORANGE], edgecolor=BLACK, linewidth=0.8)
    axes[1].set_ylabel("Return-temperature RMSE ($^\\circ$C)")
    axes[1].set_title("Timestamp-level causal ablation")
    axes[1].tick_params(axis="x", rotation=18)

    axes[2].bar(year["held_out_year"].astype(str), year["RMSE_C"], color=MAGENTA, edgecolor=BLACK, linewidth=0.8)
    axes[2].set_ylabel("Return-temperature RMSE ($^\\circ$C)")
    axes[2].set_title("Leave-one-year-out transfer")
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.set_axisbelow(True)
    for ext, dpi in [("pdf", 300), ("png", 600), ("svg", 300)]:
        path = FIGURES / f"fig_blind_measured_validation_extended.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        (PAPER_FIGURES / path.name).write_bytes(path.read_bytes())
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_plant_level()
    plant = plant_withholding(data)
    year = leave_one_year_out(data)
    causal = causal_load_ablation()
    make_tables(plant, causal, year)
    make_figure(plant, causal, year)
    print(plant[plant["protocol"].eq("combined_plant_and_time_withholding")][["held_out_plant", "n_samples", "RMSE_C", "MAE_C"]].to_string(index=False))
    print("\nCausal heat-load ablation")
    print(causal[["variant", "RMSE_C", "MAE_C", "causal_operational_candidate"]].to_string(index=False))


if __name__ == "__main__":
    main()
