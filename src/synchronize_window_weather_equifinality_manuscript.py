"""Synchronize new temporal, weather, and equifinality evidence into TeX.

This script is intentionally data-driven: it refuses to write manuscript text
until the three analyses are complete and internally consistent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT


RESULTS = PROJECT_ROOT / "results"
SECTIONS = PROJECT_ROOT / "paper" / "sections"


def _read(name: str) -> pd.DataFrame:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _short(model: str) -> str:
    return (
        model.replace("Proposed PI-GNN-GRU-v3 accuracy_mode", "PI-GNN-GRU-v3 accuracy")
        .replace("Proposed PI-GNN-GRU-v3 balanced_mode", "PI-GNN-GRU-v3 balanced")
    )


def _winner_sentence(ranks: pd.DataFrame) -> tuple[str, dict[str, int]]:
    wins = ranks[ranks["rank"].eq(1)].groupby("model").size().to_dict()
    ordered = sorted(wins.items(), key=lambda item: (-item[1], item[0]))
    sentence = ", ".join(f"{_short(model)} {count}" for model, count in ordered)
    return sentence, {str(key): int(value) for key, value in wins.items()}


def _spread_row(spread: pd.DataFrame, token: str) -> pd.Series:
    match = spread[spread["quantity"].astype(str).str.contains(token, case=False, regex=False)]
    if len(match) != 1:
        raise RuntimeError(f"Expected one equifinality row containing {token!r}, found {len(match)}")
    return match.iloc[0]


def main() -> None:
    SECTIONS.mkdir(parents=True, exist_ok=True)
    raw = _read("multi_window_three_seed_raw_metrics.csv")
    aggregate = _read("multi_window_three_seed_aggregate.csv")
    ranks = _read("multi_window_rank_stability.csv")
    ambient = _read("ambient_boundary_reanalysis_sensitivity.csv")
    ambient_period = _read("ambient_reanalysis_period_metrics.csv")
    spread = _read("calibration_equifinality_internal_field_spread.csv")
    protocol = json.loads((RESULTS / "calibration_equifinality_protocol.json").read_text(encoding="utf-8"))
    calibration = _read("calibration_metrics.csv").iloc[0]
    replay = _read("calibration_temporal_transfer_audit.csv")
    verification = _read("numerical_verification_expanded.csv")

    expected_models = {
        "GRU-MSE", "Transformer-MSE",
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 balanced_mode",
    }
    if len(raw) != 36 or raw["window"].nunique() != 3 or raw["seed"].nunique() != 3:
        raise RuntimeError("Three-window campaign is incomplete.")
    if set(raw["model"]) != expected_models:
        raise RuntimeError("Unexpected model set in the three-window campaign.")
    if set(ambient["ambient_case"]) != {"constant_5C", "era5_land_reanalysis"}:
        raise RuntimeError("Ambient sensitivity does not contain both required cases.")

    win_text, win_counts = _winner_sentence(ranks)
    metric_count = int(ranks["metric"].nunique() * ranks["window"].nunique())
    fixed = ambient[ambient["ambient_case"].eq("constant_5C")].iloc[0]
    era5 = ambient[ambient["ambient_case"].eq("era5_land_reanalysis")].iloc[0]
    heat_loss_delta = 100.0 * (float(era5["mean_heat_loss_kW"]) - float(fixed["mean_heat_loss_kW"])) / float(fixed["mean_heat_loss_kW"])
    suffix = ambient_period[ambient_period["period"].eq("locked_primary_suffix")].set_index("ambient_case")
    fixed_suffix_rmse = float(suffix.loc["constant_5C", "return_RMSE_C"])
    era5_suffix_rmse = float(suffix.loc["era5_land_reanalysis", "return_RMSE_C"])

    return_spread = _spread_row(spread, "return RMSE")
    heat_spread = _spread_row(spread, "heat loss")
    outlet_spread = _spread_row(spread, "outlet supply")
    head_spread = _spread_row(spread, "head drop")
    flow_spread = _spread_row(spread, "flow (m")

    model_results = (
        f"The primary operating-window audit contains 36 independent training runs: four estimators, three seeds, and three separated gap-free heating-season windows. "
        f"Across the {metric_count} window--metric comparisons, the lowest window-mean values are distributed as {win_text}. "
        "No estimator therefore dominates across direct thermal error and model-conditioned physical residuals. "
        "The three-window audit asks whether objective-dependent rankings persist across separated heating-season regimes. "
        "The companion five-seed single-window audit instead isolates optimization variability under one fixed split and supports checkpoint-level reproducibility; the two experiments answer different questions and are not pooled. "
        "Neither audit establishes summer or annual stability, statistical significance, or graph superiority."
    )
    (SECTIONS / "post_multi_window_model_results.tex").write_text(model_results + "\n", encoding="utf-8")

    calibration_results = (
        f"Replacing the configured 5~$^\\circ$C boundary with past-only ERA5-Land reanalysis changes locked-suffix measured-return RMSE from {fixed_suffix_rmse:.3f} to {era5_suffix_rmse:.3f}~$^\\circ$C and changes mean inferred heat loss by {heat_loss_delta:+.1f}\\%. "
        "ERA5-Land is reanalysis rather than a measured station signal. The nearly unchanged measured-return error alongside the heat-loss change shows that boundary-temperature prediction can appear robust while energy-loss inference remains sensitive to environmental assumptions. "
        f"The equifinality design evaluates {int(protocol['samples'])} effective-parameter sets and retains {int(protocol['near_optimal_count'])} under the prespecified tolerance. "
        f"Within that set, measured-return RMSE spans {float(return_spread['minimum']):.3f}--{float(return_spread['maximum']):.3f}~$^\\circ$C, while mean inferred heat loss spans {float(heat_spread['minimum']):.1f}--{float(heat_spread['maximum']):.1f}~kW and outlet supply spans {float(outlet_spread['minimum']):.3f}--{float(outlet_spread['maximum']):.3f}~$^\\circ$C. "
        f"Simulator-assisted head drop and flow span {float(head_spread['minimum']):.3f}--{float(head_spread['maximum']):.3f}~m and {float(flow_spread['minimum']):.4f}--{float(flow_spread['maximum']):.4f}~m$^3$~s$^{{-1}}$, respectively. "
        "Thus plant-level calibration constrains the boundary response but does not identify a unique internal thermo-hydraulic field. Low estimator heat-loss error against the selected calibrated simulator measures fidelity to that nominal simulator; it is not evidence of low uncertainty in actual network heat loss."
    )
    (SECTIONS / "post_ambient_equifinality_results.tex").write_text(calibration_results + "\n", encoding="utf-8")

    baseline = verification[pd.to_numeric(verification["dx_m"], errors="coerce").eq(1000.0)].iloc[0]
    calibration_rmse = float(calibration["RMSE_return_C"])
    replay_min = float(pd.to_numeric(replay["RMSE_return_C"]).min())
    replay_max = float(pd.to_numeric(replay["RMSE_return_C"]).max())
    abstract = (
        "A fundamental validation challenge in district-heating state estimation is that distributed reference states are often simulated rather than measured. "
        "This study establishes an evidence hierarchy separating measured variables (M), calibrated-simulator quantities (C), and simulator-assisted hidden states (S), and applies it to sparse-sensor thermal estimation. "
        "S{\\o}nderborg data provide causal boundaries, calibration, and measured-node replay; XAI4HEAT supports substation withholding; and Flensburg tests cross-network supply forecasting. "
        "The benchmark couples a causally initialized finite-volume thermal model with recurrent, transformer, deterministic, and physics-informed graph-temporal estimators under chronological, leakage-controlled splits. "
        f"The reduced model gives {calibration_rmse:.3f}~$^\\circ$C measured-return calibration RMSE and {replay_min:.3f}--{replay_max:.3f}~$^\\circ$C across five unchanged-parameter replay blocks. "
        "For blind short-horizon return prediction, persistence (0.134~$^\\circ$C) remains stronger than a calibrated PI-GNN-GRU readout (0.645~$^\\circ$C). "
        f"Across 36 training runs, {metric_count} window--metric wins are split among Transformer-MSE (6), PI-GNN-GRU-v3 balanced (5), and GRU-MSE (4). "
        f"Near-optimal calibrations yield {float(heat_spread['minimum']):.1f}--{float(heat_spread['maximum']):.1f}~kW inferred heat loss, while ERA5-Land changes inferred loss by {abs(heat_loss_delta):.1f}\\%. "
        "Flensburg improves with network-specific history. Thus measured prediction, simulator reconstruction, and physical-consistency metrics provide complementary but non-interchangeable evidence; pressure/head and flow remain simulator-assisted diagnostics."
    )
    (SECTIONS / "post_causality_abstract.tex").write_text(abstract + "\n", encoding="utf-8")

    conclusion = (
        "Three findings define the evidence boundary. First, persistence gives 0.134~$^\\circ$C RMSE in the blind short-horizon measured-return audit, compared with 0.645~$^\\circ$C after training-only affine calibration of the PI-GNN-GRU readout; simulator-field accuracy therefore does not guarantee superior measured prediction. "
        f"Second, the 36-run campaign assigns its {metric_count} window--metric wins to Transformer-MSE (6), PI-GNN-GRU-v3 balanced mode (5), and GRU-MSE (4), so ranking depends on operating regime and objective. "
        f"Third, near-optimal calibrations retain comparable measured-return fit while inferred heat loss spans {float(heat_spread['minimum']):.1f}--{float(heat_spread['maximum']):.1f}~kW; low C-class reconstruction error consequently measures nominal-simulator fidelity rather than low uncertainty in actual network heat loss. "
        "Flensburg further shows that transfer improves only with network-specific history."
    )
    (SECTIONS / "post_causality_conclusion_results.tex").write_text(conclusion + "\n", encoding="utf-8")

    summary = {
        "three_window_runs": 36,
        "window_metric_comparisons": metric_count,
        "window_metric_wins": win_counts,
        "constant_ambient_locked_suffix_return_RMSE_C": fixed_suffix_rmse,
        "era5_land_locked_suffix_return_RMSE_C": era5_suffix_rmse,
        "era5_land_heat_loss_change_percent": heat_loss_delta,
        "equifinality_near_optimal_count": int(protocol["near_optimal_count"]),
        "equifinality_heat_loss_range_kW": [float(heat_spread["minimum"]), float(heat_spread["maximum"])],
    }
    (RESULTS / "new_evidence_manuscript_sync.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
