from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional for table/report generation.
    plt = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT


COLORS = {
    "blue": "#0000E6",
    "orange": "#FF6626",
    "green": "#55D600",
    "yellow": "#F2E600",
    "magenta": "#E600E6",
    "black": "#111111",
    "gray": "#555555",
}


def _read_csv(name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "results" / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _write_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df = pd.DataFrame([{"Status": "not available", "Interpretation": "Required source CSV was missing."}])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(df.columns) + r"}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row[c]) for c in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_identifiability_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Quantity": "pressure/head profile",
                "Directly measured in public data?": "no",
                "Current evidence": "pump-curve, friction, and calibrated simulator consistency",
                "Main confounding": "friction factor, flow proxy, pump boundary, topology simplification",
                "Recommended upgrade": "install pressure sensors at source/load and at least one interior hydraulic node",
                "Safe claim": "simulator-assisted hidden hydraulic state",
            },
            {
                "Quantity": "distributed flow",
                "Directly measured in public data?": "no",
                "Current evidence": "heat-load-derived flow proxy and flow-balance residuals",
                "Main confounding": "return-temperature assumption, cp/rho assumptions, unmodeled branch flow",
                "Recommended upgrade": "add ultrasonic or calibrated flow meters at source and representative branches",
                "Safe claim": "heat-load-proxy-informed simulator-assisted flow state",
            },
            {
                "Quantity": "friction factor",
                "Directly measured in public data?": "no",
                "Current evidence": "effective parameter sensitivity around literature/default value",
                "Main confounding": "pipe roughness, local losses, pump control, unknown valve states",
                "Recommended upgrade": "joint head-flow calibration with measured differential pressure and flow",
                "Safe claim": "weakly identifiable effective hydraulic parameter",
            },
            {
                "Quantity": "heat-loss coefficient",
                "Directly measured in public data?": "no direct pipe heat-loss measurement",
                "Current evidence": "thermal calibration, heat-delivery consistency, heat-loss sensitivity",
                "Main confounding": "ground/ambient model, insulation age, soil condition, flow/delay proxy",
                "Recommended upgrade": "distributed temperature sensing or pipe-section heat-loss audit",
                "Safe claim": "calibrated effective thermal parameter",
            },
            {
                "Quantity": "return-temperature bias",
                "Directly measured in public data?": "yes for Sønderborg boundary, assumed/limited for Flensburg",
                "Current evidence": "return-temperature calibration and bias sensitivity",
                "Main confounding": "consumer aggregation, sensor bias, missing return data in transfer set",
                "Recommended upgrade": "calibrate return-temperature sensors and include substation-level returns",
                "Safe claim": "measured-node thermal validation where available",
            },
        ]
    )


def _build_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "pressure_drop_error_percent",
        "flow_RMSE_m3_s",
        "heat_loss_error_percent",
        "energy_balance_residual_percent",
        "boundary_residual_mean_C",
    ]
    rows = []
    baseline = df[df["case"].astype(str).eq("baseline")]
    for parameter, group in df.groupby("parameter", dropna=False):
        if str(parameter) == "baseline":
            continue
        for metric in metrics:
            if metric not in group.columns:
                continue
            base_value = pd.to_numeric(baseline[metric], errors="coerce").mean()
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.dropna().empty or not np.isfinite(base_value):
                continue
            deltas = values - base_value
            idx = deltas.abs().idxmax()
            max_delta = float(deltas.loc[idx])
            rows.append(
                {
                    "Parameter": str(parameter),
                    "Metric": metric,
                    "Baseline mean": round(float(base_value), 4),
                    "Maximum absolute change": round(abs(max_delta), 4),
                    "Most sensitive case": str(df.loc[idx, "case"]),
                    "Interpretation": "Large changes indicate weak hydraulic/thermal identifiability under plant-level public data.",
                    "State type": "simulator_assisted_hidden_state",
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Maximum absolute change", ascending=False).reset_index(drop=True)
    return out


def _plot_sensitivity(summary: pd.DataFrame) -> None:
    final_dir = PROJECT_ROOT / "figures" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        (PROJECT_ROOT / "results" / "hydraulic_identifiability_figure_status.txt").write_text(
            "Hydraulic identifiability CSV and LaTeX tables were generated, but the optional figure was skipped because matplotlib is unavailable in this runtime.\n",
            encoding="utf-8",
        )
        return
    if summary.empty:
        return
    plot = summary.head(14).copy()
    labels = (plot["Parameter"] + "\n" + plot["Metric"].str.replace("_", " ")).tolist()
    values = plot["Maximum absolute change"].astype(float).to_numpy()
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["magenta"], COLORS["gray"]] * 4
    plt.rcParams.update(
        {
            "font.family": "serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": COLORS["black"],
            "axes.labelcolor": COLORS["black"],
            "text.color": COLORS["black"],
        }
    )
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    y = np.arange(len(plot))
    ax.barh(y, values, color=colors[: len(plot)], edgecolor=COLORS["black"], linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Maximum absolute metric change from baseline")
    ax.set_title("Hydraulic identifiability sensitivity of effective parameters", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    for ext, kwargs in {"pdf": {}, "png": {"dpi": 1200}}.items():
        fig.savefig(final_dir / f"fig_hydraulic_identifiability_sensitivity.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    tables_dir = PROJECT_ROOT / "paper" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = _read_csv("parameter_identifiability_sensitivity.csv")
    matrix = _build_identifiability_matrix()
    summary = _build_sensitivity_summary(sensitivity)

    matrix.to_csv(results_dir / "hydraulic_identifiability_matrix.csv", index=False)
    summary.to_csv(results_dir / "hydraulic_identifiability_sensitivity_summary.csv", index=False)

    _write_table(
        matrix,
        tables_dir / "table_hydraulic_identifiability_matrix.tex",
        "Hydraulic identifiability matrix. Pressure/head and flow are not directly measured in the public datasets and are therefore treated as simulator-assisted hidden hydraulic states.",
        "tab:hydraulic_identifiability_matrix",
    )
    _write_table(
        summary.head(12),
        tables_dir / "table_hydraulic_identifiability_sensitivity.tex",
        "Hydraulic-identifiability sensitivity summary from effective-parameter perturbations. Large changes indicate uncertainty in simulator-assisted hydraulic hidden states, not measured pressure/flow validation.",
        "tab:hydraulic_identifiability_sensitivity",
    )
    _plot_sensitivity(summary)
    print(results_dir / "hydraulic_identifiability_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
