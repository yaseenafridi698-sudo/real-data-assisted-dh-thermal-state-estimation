"""Final integrity audit for the post-causality ATE evidence package."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.freeze_final_submission_results import LOCK_ROOT


CANONICAL = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv"
CANONICAL_ROWS = 18703
CANONICAL_SHA256 = "35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e"
SEEDS = {11, 22, 33, 44, 55}
MODELS = {
    "GRU-MSE",
    "Transformer-MSE",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
}
REQUIRED_STATE_AUDITS = {
    "supply_temperature_state_uses_no_future_values",
    "return_temperature_state_uses_no_future_values",
    "head_state_uses_no_future_values",
    "flow_state_uses_no_future_values",
    "heat_loss_state_uses_no_future_values",
    "delivered_heat_state_uses_no_future_values",
}
REQUIRED_DEPENDENT_STAGES = {
    "aggregate_primary_five_seed_results",
    "build_primary_comparison_tables",
    "export_dense_reconstruction_payload",
    "cross_estimator_disagreement",
    "external_transfer_and_noise_dropout",
    "sensor_layouts",
    "physics_loss_ablations",
    "thermo_hydraulic_coupling_and_robustness",
    "uncertainty_and_conformal",
    "seasonal_generalization",
    "combined_stress",
    "parameter_identifiability",
    "critical_measured_validation",
    "xai4heat_measured_node_validation",
    "online_replay_and_gaussian_observer",
    "replay_robustness_summary",
}
REQUIRED_VERIFICATION = {
    "Analytical advection",
    "Analytical advection-loss",
    "Hydraulic analytical solution",
    "Energy/storage closure",
    "Manufactured solution",
    "Grid convergence",
    "Independent solver comparison",
    "Cross-estimator audit",
    "Locked later replay",
    "Cross-window/layout audit",
    "Causality/leakage audit",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(relative: str) -> pd.DataFrame:
    path = PROJECT_ROOT / relative
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _active_text_paths() -> list[Path]:
    paths = [
        PROJECT_ROOT / "paper" / "main_ate_submission_candidate.tex",
        PROJECT_ROOT / "paper" / "supplementary_material.tex",
    ]
    for source in list(paths):
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        for item in re.findall(r"\\input\{([^}]+)\}", text):
            candidate = source.parent / (item if item.endswith(".tex") else f"{item}.tex")
            if candidate.is_file():
                paths.append(candidate)
    return sorted(set(paths))


def run_integrity_audit() -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, evidence: str, note: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "evidence": evidence, "note": note})

    canonical_ok = CANONICAL.is_file() and sum(1 for _ in CANONICAL.open("rb")) - 1 == CANONICAL_ROWS and sha256(CANONICAL) == CANONICAL_SHA256
    add("canonical_dataset_frozen", canonical_ok, str(CANONICAL), f"required rows={CANONICAL_ROWS}; required SHA-256={CANONICAL_SHA256}")

    causality = read_csv("results/proxy_causality_audit.csv")
    passed_audits = set(causality.loc[causality.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("pass"), "audit"].astype(str)) if not causality.empty else set()
    state_rows = causality[causality.get("audit", pd.Series(dtype=str)).astype(str).isin(REQUIRED_STATE_AUDITS)] if not causality.empty else pd.DataFrame()
    causality_values = pd.to_numeric(state_rows["value"], errors="coerce") if "value" in state_rows else pd.Series(dtype=float)
    zero_deltas = bool(len(state_rows) == len(REQUIRED_STATE_AUDITS) and causality_values.fillna(np.inf).eq(0).all())
    add("complete_state_causality", REQUIRED_STATE_AUDITS.issubset(passed_audits) and zero_deltas, "results/proxy_causality_audit.csv", "all past full-state and proxy deltas must be exactly zero")

    calibration = read_csv("results/calibration_metrics.csv")
    later = read_csv("results/locked_later_replay_metrics.csv")
    calibrated = PROJECT_ROOT / "results" / "calibrated_parameters.json"
    calibration_finite = not calibration.empty and np.isfinite(calibration.select_dtypes(include=[np.number]).to_numpy()).all()
    later_finite = not later.empty and np.isfinite(later.select_dtypes(include=[np.number]).to_numpy()).all()
    add("corrected_recalibration_and_locked_replay", calibrated.is_file() and calibration_finite and later_finite, "results/calibrated_parameters.json; results/calibration_metrics.csv; results/locked_later_replay_metrics.csv", "later replay must be finite and explicitly use no retuning")
    temporal_replay = read_csv("results/calibration_temporal_transfer_audit.csv")
    temporal_ok = (
        len(temporal_replay) == 5
        and temporal_replay.get("retuned", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
        and np.isfinite(temporal_replay[["RMSE_return_C", "MAE_return_C", "dynamic_energy_residual_percent"]].to_numpy(float)).all()
    )
    add("unchanged_parameter_multiblock_replay", temporal_ok, "results/calibration_temporal_transfer_audit.csv", "five disjoint later blocks are scored against measured return without retuning")

    states = PROJECT_ROOT / "results" / "corrected_simulator_states.npz"
    provenance_path = PROJECT_ROOT / "results" / "corrected_simulator_states_provenance.json"
    required_keys = {"Ts", "Tr", "H", "q", "Q_loss", "delivered_heat_W", "energy_balance_residual_W"}
    state_keys = set(np.load(states).files) if states.is_file() else set()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    provenance_hash = provenance.get("canonical_dataset_sha256", provenance.get("source_sha256"))
    add("corrected_simulator_states_regenerated", required_keys.issubset(state_keys) and provenance_hash == CANONICAL_SHA256, "results/corrected_simulator_states.npz", "all active C-class thermal and S-class hydraulic state families must share the canonical hash")
    superseded = PROJECT_ROOT / "superseded_pre_full_state_causality_20260807"
    superseded_manifest = superseded / "superseded_archive_manifest.csv"
    add(
        "pre_correction_states_archived",
        superseded.is_dir() and superseded_manifest.is_file() and not pd.read_csv(superseded_manifest).empty,
        str(superseded),
        "pre-correction outputs are retained only in the explicitly superseded archive",
    )

    raw = read_csv("results/repeated_seed_raw_metrics.csv")
    pairs = set(zip(raw.get("model", pd.Series(dtype=str)).astype(str), pd.to_numeric(raw.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))) if not raw.empty else set()
    expected_pairs = {(model, seed) for model in MODELS for seed in SEEDS}
    checkpoints = [PROJECT_ROOT / "results" / f"seed_{seed}_{model}_best.pt" for model, seed in expected_pairs]
    state_mtime = states.stat().st_mtime if states.is_file() else np.inf
    checkpoints_are_post_correction = all(path.is_file() and path.stat().st_mtime > state_mtime for path in checkpoints)
    add("twenty_principal_training_runs", pairs == expected_pairs and checkpoints_are_post_correction, "results/repeated_seed_raw_metrics.csv and seed_*_best.pt", "four models x five seeds with one post-correction checkpoint per run")

    multi = read_csv("results/multi_window_three_seed_raw_metrics.csv")
    expected_multi = {
        (window, seed, model)
        for window in {"winter_2016", "shoulder_2016", "late_winter_2018"}
        for seed in {11, 22, 33}
        for model in MODELS
    }
    observed_multi = {
        (str(row["window"]), int(row["seed"]), str(row["model"]))
        for _, row in multi.iterrows()
    } if {"window", "seed", "model"}.issubset(multi.columns) else set()
    add(
        "three_window_three_seed_campaign",
        len(multi) == 36 and observed_multi == expected_multi,
        "results/multi_window_three_seed_raw_metrics.csv",
        "four models x three seeds x three separated gap-free heating-season windows",
    )

    ambient = read_csv("results/ambient_boundary_reanalysis_sensitivity.csv")
    ambient_provenance_path = PROJECT_ROOT / "results" / "ambient_reanalysis_provenance.json"
    ambient_provenance = json.loads(ambient_provenance_path.read_text(encoding="utf-8")) if ambient_provenance_path.is_file() else {}
    weather_path = PROJECT_ROOT / "data" / "external_weather" / "sonderborg_era5_land_2016_2019_hourly.csv"
    weather_hash_ok = weather_path.is_file() and ambient_provenance.get("source_sha256") == sha256(weather_path)
    add(
        "era5_land_boundary_sensitivity",
        set(ambient.get("ambient_case", pd.Series(dtype=str)).astype(str)) == {"constant_5C", "era5_land_reanalysis"}
        and ambient_provenance.get("reanalysis_model") == "ERA5-Land"
        and weather_hash_ok,
        "results/ambient_boundary_reanalysis_sensitivity.csv; results/ambient_reanalysis_provenance.json",
        "historical reanalysis is past-only, hash-verified, and kept distinct from local measurements",
    )

    equifinality = read_csv("results/calibration_equifinality_ensemble.csv")
    eq_protocol_path = PROJECT_ROOT / "results" / "calibration_equifinality_protocol.json"
    eq_protocol = json.loads(eq_protocol_path.read_text(encoding="utf-8")) if eq_protocol_path.is_file() else {}
    add(
        "calibration_equifinality_quantified",
        len(equifinality) >= 700
        and int(eq_protocol.get("near_optimal_count", 0)) >= 8
        and "5%/0.05" in str(eq_protocol.get("tolerance_rule", "")),
        "results/calibration_equifinality_ensemble.csv; results/calibration_equifinality_protocol.json",
        "internal-field non-uniqueness uses a prespecified near-optimal threshold",
    )

    second = read_csv("results/second_chronological_window_metrics.csv")
    second_pairs = set(zip(second.get("model", pd.Series(dtype=str)).astype(str), pd.to_numeric(second.get("seed", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))) if not second.empty else set()
    add("second_chronological_window", second_pairs == expected_pairs, "results/second_chronological_window_metrics.csv", "fixed-checkpoint transfer on the disjoint later window")

    observer = read_csv("results/gaussian_observer_baseline_metrics.csv")
    observer_ok = (
        len(observer) == 1
        and int(observer.iloc[0].get("train_test_timestamp_overlap_count", -1)) == 0
        and int(observer.iloc[0].get("held_out_unique_timestamp_count", 0)) == 102
        and str(observer.iloc[0].get("observer_type", "")) == "static_training_only_covariance_gaussian_conditioning"
    )
    add("training_only_gaussian_observer", observer_ok, "results/gaussian_observer_baseline_metrics.csv", "observer covariance uses training timestamps only and held-out scoring is unique-timestamp and disjoint")
    blind_return = read_csv("results/principal_models_blind_measured_return_summary.csv")
    blind_models = set(blind_return.get("model", pd.Series(dtype=str)).astype(str))
    add(
        "principal_checkpoint_blind_measured_return_audit",
        len(blind_return) == 5 and {"Measured-return persistence", *MODELS}.issubset(blind_models),
        "results/principal_models_blind_measured_return_summary.csv",
        "current return and internal C-class thermal and S-class hydraulic sensor inputs are withheld while a real return target is scored",
    )

    dependent = read_csv("results/full_dependent_regeneration_manifest.csv")
    completed_stages = set(dependent.loc[dependent.get("status", pd.Series(dtype=str)).astype(str).eq("complete"), "stage"].astype(str)) if not dependent.empty else set()
    add("all_dependent_analyses_regenerated", REQUIRED_DEPENDENT_STAGES.issubset(completed_stages), "results/full_dependent_regeneration_manifest.csv", "no legacy metric may be selectively retained")

    verification = read_csv("results/verification_campaign_status.csv")
    verification_names = set(verification.get("verification", pd.Series(dtype=str)).astype(str))
    no_fail = not verification.empty and not verification.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("FAIL").any()
    add("eleven_test_verification_campaign", verification_names == REQUIRED_VERIFICATION and no_fail, "results/verification_campaign_status.csv", "PASS or explicitly justified LIMITATION; no FAIL")

    active_text_paths = _active_text_paths()
    active_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in active_text_paths)
    former_acronym = "E" + "nKF"
    former_expansion = "ensemble" + " Kalman"
    old_name_absent = re.search(rf"\b{former_acronym}\b|{former_expansion}", active_text, flags=re.IGNORECASE) is None
    add("observer_renamed_in_active_publication", old_name_absent and "covariance-conditioned Gaussian observer" in active_text, "; ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in active_text_paths), "former method name must not remain in active prose or imported tables")
    evidence_boundary = all(phrase in active_text.lower() for phrase in ["measured-node validation", "simulator-assisted hidden", "dense distributed field validation remains future work"])
    add("evidence_hierarchy_explicit", evidence_boundary, "paper/main_ate_submission_candidate.tex", "M, C, and S evidence boundaries must remain explicit")

    measured_adaptation = read_csv("results/measured_return_checkpoint_adaptation.csv")
    adapted = measured_adaptation[
        measured_adaptation.get("adaptation", pd.Series(dtype=str)).astype(str).eq("training-only affine readout")
    ] if not measured_adaptation.empty else pd.DataFrame()
    adaptation_ok = (
        len(adapted) == 20
        and pd.to_numeric(adapted.get("train_test_overlap"), errors="coerce").eq(0).all()
        and pd.to_numeric(adapted.get("RMSE_C"), errors="coerce").lt(pd.to_numeric(adapted.get("raw_RMSE_C"), errors="coerce")).all()
        and adapted.get("current_return_measurement_used_at_test", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
    )
    add("measured_return_training_only_adaptation", adaptation_ok, "results/measured_return_checkpoint_adaptation.csv", "all 20 frozen checkpoints use disjoint training-only readouts and untouched measured-return targets")

    xai_chronological = read_csv("results/xai4heat_chronological_withholding.csv")
    xai_ok = (
        len(xai_chronological) == 20
        and xai_chronological.get("withheld_target_temperature_used_as_feature", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
        and xai_chronological.get("target_conditioned_order_filter_used", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
    )
    add("xai4heat_chronological_measured_withholding", xai_ok, "results/xai4heat_chronological_withholding.csv", "five-station chronological folds use no withheld target channel or target-conditioned filter")

    flensburg_causal = read_csv("results/flensburg_causal_supply_forecast.csv")
    flensburg_ok = (
        len(flensburg_causal) == 12
        and set(pd.to_numeric(flensburg_causal.get("horizon_h"), errors="coerce").dropna().astype(int)) == {1, 6, 24}
        and flensburg_causal.get("future_supply_used_as_feature", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
        and flensburg_causal.get("return_temperature_used", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
    )
    add("flensburg_causal_measured_supply_audit", flensburg_ok, "results/flensburg_causal_supply_forecast.csv", "three causal horizons use measured supply targets and exclude unavailable return temperature")
    figure_provenance = read_csv("results/active_figure_provenance_post_causality.csv")
    add(
        "active_core_figure_provenance",
        len(figure_provenance) == 4
        and figure_provenance.get("generated_after_all_sources", pd.Series(dtype=str)).astype(str).str.lower().eq("true").all(),
        "results/active_figure_provenance_post_causality.csv",
        "stale workflow/calibration/reconstruction/heat figures are replaced by source-traced post-correction assets",
    )
    static_latex = read_csv("results/latex_static_preflight.csv")
    add(
        "static_latex_preflight",
        not static_latex.empty and static_latex.get("passed", pd.Series(dtype=str)).astype(str).str.lower().eq("true").all(),
        "results/latex_static_preflight.csv",
        "all active inputs, figures, labels, citations, environments, and table rows pass static checks; this is not a TeX-engine compile",
    )

    if LOCK_ROOT.is_dir() and (LOCK_ROOT / "manifest_sha256.txt").is_file():
        manifest_rows = []
        for line in (LOCK_ROOT / "manifest_sha256.txt").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            target = LOCK_ROOT / relative
            manifest_rows.append(target.is_file() and sha256(target) == expected)
        locked_paths = [path.relative_to(LOCK_ROOT).as_posix().lower() for path in LOCK_ROOT.rglob("*") if path.is_file()]
        forbidden = any(any(token in relative for token in ["legacy_audits", "superseded", "enkf_baseline", "table_enkf"]) for relative in locked_paths)
        add("locked_archive_hashes_and_scope", bool(manifest_rows) and all(manifest_rows) and not forbidden, str(LOCK_ROOT / "manifest_sha256.txt"), "every locked file hash matches and no superseded evidence is included")
    else:
        add("locked_archive_hashes_and_scope", False, str(LOCK_ROOT), "run freeze_final_submission_results.py after the pre-lock audit")

    report = pd.DataFrame(checks)
    results = PROJECT_ROOT / "results"
    results.mkdir(exist_ok=True)
    report.to_csv(results / "final_integrity_audit.csv", index=False)
    lines = ["Final post-causality integrity audit", ""]
    for row in checks:
        lines.append(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['check']}: {row['note']} ({row['evidence']})")
    verdict = "PASS" if report["passed"].all() else "FAIL"
    lines.extend(["", f"verdict: {verdict}"])
    (results / "final_integrity_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    active_files = []
    for path in [CANONICAL, weather_path, ambient_provenance_path, eq_protocol_path, *active_text_paths, states, provenance_path, calibrated, *checkpoints]:
        if path.is_file():
            active_files.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(active_files).drop_duplicates(subset=["path"]).sort_values("path")
    manifest.to_csv(results / "final_integrity_active_file_manifest.csv", index=False)
    (results / "final_integrity_active_file_manifest.json").write_text(json.dumps(active_files, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    frame = run_integrity_audit()
    print(frame.to_string(index=False))
    if not frame["passed"].all():
        raise SystemExit(1)
