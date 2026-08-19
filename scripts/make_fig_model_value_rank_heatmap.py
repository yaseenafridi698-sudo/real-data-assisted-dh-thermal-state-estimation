from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ate_concept_figure_style import COLORS, model_short, rank_cmap, read_csv, save_figure, set_style


MODELS = [
    "Interpolation",
    "GRU-MSE",
    "Transformer-MSE",
    "PI-LSTM",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
]


def _metric_from_baseline(df: pd.DataFrame, column: str, label: str) -> list[dict]:
    rows = []
    if df.empty or column not in df.columns:
        return rows
    sub = df[df["model"].astype(str).isin(MODELS)][["model", column]].copy()
    sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return rows
    sub["rank"] = sub[column].rank(method="min", ascending=True)
    for _, row in sub.iterrows():
        rows.append({"metric": label, "model": row["model"], "value": row[column], "rank": row["rank"], "source": "baseline_comparison_final.csv"})
    return rows


def _metric_from_stress(df: pd.DataFrame, column: str, label: str, case: str = "baseline_real_profile") -> list[dict]:
    rows = []
    if df.empty or column not in df.columns:
        return rows
    sub = df[df["model"].astype(str).isin(MODELS)].copy()
    if "case" in sub.columns:
        case_sub = sub[sub["case"].astype(str).eq(case)].copy()
        if not case_sub.empty:
            sub = case_sub
    sub = sub[["model", column]].copy()
    sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return rows
    sub["rank"] = sub[column].rank(method="min", ascending=True)
    for _, row in sub.iterrows():
        rows.append({"metric": label, "model": row["model"], "value": row[column], "rank": row["rank"], "source": "combined_stress_test.csv"})
    return rows


def build_rank_matrix() -> pd.DataFrame:
    baseline = read_csv("baseline_comparison_final.csv")
    stress = read_csv("combined_stress_test.csv")
    rows: list[dict] = []
    rows += _metric_from_baseline(baseline, "RMSE_Ts_full", "Supply-temperature RMSE")
    rows += _metric_from_baseline(baseline, "RMSE_Tr_full", "Return-temperature RMSE")
    rows += _metric_from_baseline(baseline, "heat_loss_error_percent", "Heat-loss error")
    rows += _metric_from_baseline(baseline, "energy_balance_residual", "Energy-balance residual")
    rows += _metric_from_baseline(baseline, "boundary_residual_mean", "Boundary residual")
    rows += _metric_from_stress(stress, "pressure_drop_error_percent", "Pressure-drop residual")
    severe = stress[stress.get("case", pd.Series(dtype=str)).astype(str).eq("combined_stress_severe")] if not stress.empty else pd.DataFrame()
    rows += _metric_from_stress(severe if not severe.empty else stress, "max_temperature_error_C", "Severe-stress max error", case="combined_stress_severe")
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(read_csv.__globals__["RESULTS"] / "concept_model_value_rank_matrix.csv", index=False)
    return out


def main() -> None:
    set_style()
    ranks = build_rank_matrix()
    if ranks.empty:
        ranks = read_csv("proposed_model_value_rank_matrix.csv")
    if ranks.empty:
        return

    metrics = [
        "Supply-temperature RMSE",
        "Return-temperature RMSE",
        "Heat-loss error",
        "Energy-balance residual",
        "Boundary residual",
        "Pressure-drop residual",
        "Severe-stress max error",
    ]
    models = [m for m in MODELS if m in set(ranks["model"].astype(str))]
    matrix = np.full((len(models), len(metrics)), np.nan)
    values = {}
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            sub = ranks[(ranks["model"].astype(str).eq(model)) & (ranks["metric"].astype(str).eq(metric))]
            if not sub.empty:
                matrix[i, j] = float(sub["rank"].iloc[0])
                values[(i, j)] = float(sub["value"].iloc[0])

    max_rank = int(np.nanmax(matrix)) if np.isfinite(matrix).any() else 6
    cmap, norm = rank_cmap(max(6, max_rank))
    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_facecolor("#F5F5F5")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m.replace(" ", "\n", 1).replace(" residual", "\nresidual").replace(" error", "\nerror") for m in metrics], fontsize=8.2)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([model_short(m) for m in models], fontsize=8.4)

    for i in range(len(models)):
        for j in range(len(metrics)):
            if np.isfinite(matrix[i, j]):
                color = "white" if matrix[i, j] <= 1.5 else COLORS["black"]
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=9, fontweight="bold", color=color)
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color=COLORS["gray"])

    for i, model in enumerate(models):
        if "PI-GNN-GRU-v3" in model:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(metrics), 1, fill=False, edgecolor=COLORS["blue"], linewidth=2.0))

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, ticks=np.arange(1, max_rank + 1))
    cbar.set_label("Rank (lower is better)", fontsize=9)
    ax.set_title("Metric-specific model ranking: where PI-GNN-GRU-v3 adds value", fontweight="bold", pad=10, fontsize=11.3)
    ax.set_xlabel("Evaluation objective")
    ax.set_ylabel("Model")
    ax.text(
        0,
        -0.19,
        "GRU-MSE can win raw supply-temperature RMSE; PI-GNN-GRU-v3 adds value in selected thermo-hydraulic consistency metrics.\n"
        "Pressure/head and flow diagnostics are simulator-assisted hidden hydraulic states. No best-overall-model claim is made.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color=COLORS["black"],
    )
    fig.tight_layout()
    save_figure(fig, "fig_model_value_rank_heatmap")


if __name__ == "__main__":
    main()
