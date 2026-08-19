"""Focused, evidence-aware quality gate for the submission manuscript.

This gate deliberately avoids treating archived figures or legacy dashboards as
submission requirements.  It checks the locked causal workflow, evidence
boundaries, compact paper assets, and compile freshness instead.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT
from src.data_registry import check_dataset_available, processed_file_exists


FORBIDDEN = (
    "outperforms all baselines",
    "fully validated",
    "deployment-ready",
    "guaranteed",
    "best overall model",
    "universal generalization",
    "real distributed head measurements",
    "real distributed flow measurements",
)

CORE_MODELS = {
    "GRU-MSE",
    "Transformer-MSE",
    "Proposed PI-GNN-GRU-v3 accuracy_mode",
    "Proposed PI-GNN-GRU-v3 balanced_mode",
}


def _path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def _csv(relative: str) -> pd.DataFrame:
    path = _path(relative)
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _exists(relative: str) -> bool:
    return _path(relative).exists()


def _finite(df: pd.DataFrame, columns: list[str]) -> bool:
    if df.empty or any(column not in df for column in columns):
        return False
    values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return bool(np.isfinite(values).all())


def _current_pdf(tex_rel: str, pdf_rel: str) -> bool:
    tex, pdf = _path(tex_rel), _path(pdf_rel)
    if not tex.exists() or not pdf.exists():
        return False
    dependencies = [tex]
    expanded = _expanded_tex(tex)
    for graphic in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", expanded):
        candidates = [tex.parent / graphic, PROJECT_ROOT / graphic]
        dependencies.extend(path for path in candidates if path.exists())
    return pdf.stat().st_mtime >= max(path.stat().st_mtime for path in dependencies)


def _keyword_count(tex: str) -> int:
    match = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, flags=re.DOTALL)
    if not match:
        return 0
    content = match.group(1).strip()
    return 0 if not content else content.count("\\sep") + 1


def _abstract_word_count(tex: str) -> int:
    """Count readable abstract words without treating LaTeX markup as prose."""
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if not match:
        return 0
    plain = match.group(1)
    plain = re.sub(r"\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", plain)
    plain = re.sub(r"\$[^$]*\$", " ", plain)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?", plain))


def _expanded_tex(path: Path) -> str:
    """Read active TeX with local input files expanded for claim/value checks."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    for _ in range(3):
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            item = match.group(1)
            candidate = path.parent / (item if item.endswith(".tex") else f"{item}.tex")
            if not candidate.exists():
                return match.group(0)
            changed = True
            return candidate.read_text(encoding="utf-8", errors="ignore")

        text = re.sub(r"\\input\{([^}]+)\}", replace, text)
        if not changed:
            break
    return text


def _review_archive_integrity() -> tuple[bool, bool]:
    """Return source-hash and source-only checks for the reviewer archive."""
    archive = _path("paper/real_data_assisted_dh_review_archive.zip")
    main = _path("paper/main_ate_submission_candidate.tex")
    if not archive.exists() or not main.exists():
        return False, False
    try:
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            main_name = "submission_review_bundle/paper/main_ate_submission_candidate.tex"
            if main_name not in names:
                return False, False
            archive_hash = hashlib.sha256(zipped.read(main_name)).hexdigest()
            active_hash = hashlib.sha256(main.read_bytes()).hexdigest()
            forbidden_suffixes = (".aux", ".bbl", ".blg", ".log", ".out", ".spl")
            source_only = not any(name.lower().endswith(forbidden_suffixes) for name in names)
            stale_pdf_names = {
                "submission_review_bundle/paper/main_ate_submission_candidate.pdf",
                "submission_review_bundle/paper/supplementary_material.pdf",
            }
            source_only = source_only and not any(name.lower() in stale_pdf_names for name in names)
            return archive_hash == active_hash, source_only
    except (OSError, zipfile.BadZipFile):
        return False, False


def _write_claim_safety(main_text: str, supplement_text: str) -> bool:
    text = (main_text + "\n" + supplement_text).lower()
    checks: list[dict[str, object]] = []

    def add(check: str, passed: bool, note: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "note": note})

    for phrase in [
        "real-data-assisted",
        "measured-node validation",
        "simulator-assisted hidden",
        "domain shift",
        "dense distributed field validation remains future work",
        "causal load-derived pump-speed proxy",
        "not a universal",
    ]:
        add(f"required_phrase::{phrase}", phrase in text, "required evidence-boundary language")
    add(
        "required_phrase::aggregate_measured_supply_boundary_at_k",
        "aggregate measured supply boundary at k" in text or "aggregate measured supply boundary at $k$" in text,
        "causal supply-boundary timing is explicit",
    )
    add(
        "required_phrase::return_boundary_at_k_minus_1",
        "return boundary at k-1" in text or "return boundary at $k-1$" in text,
        "causal return-boundary timing is explicit",
    )
    add(
        "states_flow_and_head_are_simulator_assisted",
        "pressure/head and flow are simulator-assisted" in text,
        "hydraulic evidence boundary is explicit",
    )
    add(
        "states_return_assumption_is_not_measured_validation",
        "assumption-consistency diagnostics" in text and "excluded from measured external-validation claims" in text,
        "Flensburg assumed-return boundary is explicit",
    )
    add(
        "states_energy_metric_is_objective_aligned",
        "objective alignment" in text and "rather than independent validation" in text,
        "dynamic-energy metric limitation is explicit",
    )
    for phrase in FORBIDDEN:
        add(f"forbidden_phrase_absent::{phrase}", phrase not in text, "forbidden claim absent")

    report = pd.DataFrame(checks)
    report.to_csv(_path("results/manuscript_claim_safety_report_final.csv"), index=False)
    lines = ["Manuscript claim-safety report", ""]
    for row in checks:
        lines.append(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['check']}: {row['note']}")
    _path("results/manuscript_claim_safety_report_final.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bool(report["passed"].all())


def run_paper_quality_gate() -> str:
    _path("results").mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, note: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "note": note})

    # Real-data and calibrated-simulator basis.
    add("sonderborg_raw_data_available", check_dataset_available("sonderborg"), "S\u00f8nderborg raw operating data available")
    add("sonderborg_processed_data_available", processed_file_exists("sonderborg"), "processed S\u00f8nderborg data available")
    calibration = _csv("results/calibration_metrics.csv")
    add("calibration_artifacts_present", not calibration.empty and _exists("results/calibrated_parameters.json"), "calibration metrics and parameters present")
    add(
        "calibration_metrics_finite",
        _finite(calibration, ["RMSE_return_C", "heat_delivery_error_percent", "energy_balance_residual_fraction"]),
        "return fit, boundary closure, and calibration dynamic-residual ratio are finite",
    )
    add("numerical_verification_present", _finite(_csv("results/numerical_verification_expanded.csv"), ["outlet_Ts_L2_C", "cumulative_heat_loss_error_pct"]), "coordinated grid/time verification present")
    canonical_path = _path("data/locked/sonderborg_processed_18703.csv")
    canonical_hash = hashlib.sha256(canonical_path.read_bytes()).hexdigest() if canonical_path.exists() else ""
    canonical_rows = sum(1 for _ in canonical_path.open("rb")) - 1 if canonical_path.exists() else -1
    add(
        "canonical_18703_row_dataset_is_frozen",
        canonical_rows == 18703 and canonical_hash == "35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e",
        "canonical processed input has the required row count and SHA-256",
    )
    later = _csv("results/locked_later_replay_metrics.csv")
    later_ok = not later.empty and not bool(later.iloc[0].get("retuned_on_later_period", True)) and _finite(later, ["RMSE_return_C", "MAE_return_C"])
    add("corrected_calibration_locked_later_replay", later_ok, "corrected parameters applied to chronological suffix without retuning")
    temporal_replay = _csv("results/calibration_temporal_transfer_audit.csv")
    add(
        "unchanged_parameter_multiblock_replay_completed",
        len(temporal_replay) == 5
        and temporal_replay.get("retuned", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
        and _finite(temporal_replay, ["RMSE_return_C", "MAE_return_C", "dynamic_energy_residual_percent"]),
        "five later contiguous blocks are scored against measured return without retuning",
    )

    # Causal evidence provenance.
    audit = _csv("results/proxy_causality_audit.csv")
    audit_pass = not audit.empty and "status" in audit and audit["status"].astype(str).str.lower().eq("pass").all()
    add("future_perturbation_proxy_causality_audit_passes", audit_pass, "future values leave all earlier causal proxies invariant")
    required_full_states = {
        "supply_temperature_state_uses_no_future_values",
        "return_temperature_state_uses_no_future_values",
        "head_state_uses_no_future_values",
        "flow_state_uses_no_future_values",
        "heat_loss_state_uses_no_future_values",
        "delivered_heat_state_uses_no_future_values",
    }
    full_state_rows = audit[audit.get("audit", pd.Series(dtype=str)).astype(str).isin(required_full_states)] if not audit.empty else pd.DataFrame()
    full_state_values = pd.to_numeric(full_state_rows.get("value", pd.Series(dtype=float)), errors="coerce")
    add(
        "future_perturbation_complete_state_causality_passes",
        len(full_state_rows) == len(required_full_states) and full_state_rows["status"].astype(str).str.lower().eq("pass").all() and full_state_values.eq(0).all(),
        "future supply/return/load/ambient perturbations leave every earlier state exactly invariant",
    )
    manifest = _csv("results/causal_regeneration_manifest.csv")
    scopes = set(manifest.get("scope", pd.Series(dtype=str)).astype(str))
    add("causal_postbenchmark_regenerated", "postbenchmark" in scopes, "Flensburg and robustness outputs regenerated from causal checkpoints")
    add("causal_sensor_layouts_regenerated", "sensor_layouts" in scopes, "layout study retrained with causal proxies")
    add("causal_ablations_regenerated", "ablations" in scopes, "ablation study retrained with causal proxies")
    add("no_causal_rerun_pending_marker", not _exists("results/CAUSAL_PROXY_DEPENDENT_STUDIES_PENDING.txt"), "no unresolved causal dependent-study marker")

    # Primary fair model comparison.
    baseline = _csv("results/baseline_comparison_final.csv")
    baseline_models = set(baseline.get("model", pd.Series(dtype=str)).astype(str))
    add("causal_primary_benchmark_has_four_core_models", CORE_MODELS.issubset(baseline_models), "four causal core-model rows present")
    seed = _csv("results/repeated_seed_statistics.csv")
    seed_models = set(seed.get("model", pd.Series(dtype=str)).astype(str))
    n_seed_ok = not seed.empty and "n_seeds" in seed and (pd.to_numeric(seed["n_seeds"], errors="coerce") >= 5).all()
    add("five_seed_common_protocol_completed", CORE_MODELS.issubset(seed_models) and n_seed_ok, "five seed statistics available for each core model")
    multi_window = _csv("results/multi_window_three_seed_raw_metrics.csv")
    multi_window_pairs = {
        (str(row["window"]), int(row["seed"]), str(row["model"]))
        for _, row in multi_window.iterrows()
    } if {"window", "seed", "model"}.issubset(multi_window.columns) else set()
    expected_multi_window = {
        (window, seed_id, model)
        for window in {"winter_2016", "shoulder_2016", "late_winter_2018"}
        for seed_id in {11, 22, 33}
        for model in CORE_MODELS
    }
    add(
        "three_window_three_seed_neural_campaign_complete",
        multi_window_pairs == expected_multi_window and len(multi_window) == 36,
        "four core models are independently trained with three seeds in each of three separated gap-free heating-season windows",
    )
    add("common_selection_protocol_recorded", _exists("results/repeated_seed_protocol.json"), "common normalized state-MSE selection protocol recorded")
    observer = _csv("results/gaussian_observer_baseline_metrics.csv")
    add(
        "gaussian_observer_is_training_only_and_disjoint",
        len(observer) == 1
        and int(observer.iloc[0].get("train_test_timestamp_overlap_count", -1)) == 0
        and int(observer.iloc[0].get("training_timestamp_count", 0)) > 0
        and int(observer.iloc[0].get("held_out_unique_timestamp_count", 0)) == 102
        and str(observer.iloc[0].get("observer_type", "")) == "static_training_only_covariance_gaussian_conditioning",
        "deterministic observer is fitted on training timestamps and scored on disjoint unique held-out timestamps",
    )
    add(
        "observer_unique_timestamp_comparison_present",
        len(_csv("results/observer_heldout_unique_timestamp_comparison.csv")) == 5,
        "observer and four seed-11 checkpoints share a unique-held-out-timestamp diagnostic",
    )
    seed_protocol = _path("results/repeated_seed_protocol.json")
    protocol = json.loads(seed_protocol.read_text(encoding="utf-8")) if seed_protocol.exists() else {}
    raw_seed = _csv("results/repeated_seed_raw_metrics.csv")
    expected_seed_pairs = {(model, int(seed)) for model in CORE_MODELS for seed in protocol.get("seeds", [])}
    observed_seed_pairs = {(str(row["model"]), int(row["seed"])) for _, row in raw_seed.iterrows()} if {"model", "seed"}.issubset(raw_seed.columns) else set()
    add("all_twenty_seed_artifacts_present", expected_seed_pairs.issubset(observed_seed_pairs) and len(expected_seed_pairs) == 20, "all four models and five specified seeds have raw metric artifacts")
    second = _csv("results/second_chronological_window_metrics.csv")
    second_pairs = {(str(row["model"]), int(row["seed"])) for _, row in second.iterrows()} if {"model", "seed"}.issubset(second.columns) else set()
    add("second_chronological_window_completed", second_pairs == expected_seed_pairs, "all 20 fixed checkpoints evaluated on a disjoint later window")
    dependent = _csv("results/full_dependent_regeneration_manifest.csv")
    add(
        "post_correction_dependent_regeneration_complete",
        not dependent.empty and dependent.get("status", pd.Series(dtype=str)).astype(str).eq("complete").all() and len(dependent) >= 16,
        "all dependent analyses regenerated from corrected states/checkpoints",
    )
    verification = _csv("results/verification_campaign_status.csv")
    add(
        "eleven_test_verification_campaign_complete",
        len(verification) == 11 and not verification.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("FAIL").any(),
        "eleven prescribed checks report PASS or an explicit justified limitation",
    )

    # Dependent studies and external measured-node evidence.
    layouts = _csv("results/sensor_layout_comparison_final.csv")
    required_layouts = {"S1_inlet_only", "S2_inlet_outlet", "S3_inlet_middle_outlet", "S4_five_sensors", "S5_noisy_inlet_outlet", "S6_dropout_five_sensors"}
    add("core_sensor_layouts_present", required_layouts.issubset(set(layouts.get("sensor_layout", pd.Series(dtype=str)).astype(str))), "S1--S6 layouts present after causal rerun")
    add("ablation_present", not _csv("results/ablation_study_final.csv").empty, "single-run causal ablation is present and scoped as diagnostic")
    add("robustness_present", not _csv("results/noise_dropout_robustness_final.csv").empty, "noise/dropout robustness regenerated")
    add("seasonal_study_present", not _csv("results/seasonal_generalization.csv").empty, "seasonal C-class thermal and S-class hydraulic sensitivity present")
    ambient = _csv("results/ambient_boundary_reanalysis_sensitivity.csv")
    ambient_cases = set(ambient.get("ambient_case", pd.Series(dtype=str)).astype(str))
    try:
        ambient_provenance = json.loads(_path("results/ambient_reanalysis_provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ambient_provenance = {}
    add(
        "era5_land_ambient_boundary_sensitivity_complete",
        {"constant_5C", "era5_land_reanalysis"}.issubset(ambient_cases)
        and ambient_provenance.get("reanalysis_model") == "ERA5-Land"
        and bool(ambient_provenance.get("source_sha256")),
        "configured ambient boundary is compared with a provenance-hashed, past-only ERA5-Land boundary",
    )
    equifinality = _csv("results/calibration_equifinality_ensemble.csv")
    try:
        equifinality_protocol = json.loads(_path("results/calibration_equifinality_protocol.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        equifinality_protocol = {}
    add(
        "calibration_equifinality_is_quantified",
        len(equifinality) >= 700
        and int(equifinality_protocol.get("near_optimal_count", 0)) >= 8
        and "5%/0.05" in str(equifinality_protocol.get("tolerance_rule", "")),
        "near-optimal effective calibrations quantify internal-field non-uniqueness under a prespecified tolerance",
    )
    add("combined_stress_present", not _csv("results/combined_stress_test.csv").empty, "controlled stress study present")
    parameter = _csv("results/parameter_identifiability_sensitivity.csv")
    return_bias_changes = not parameter.empty and "case" in parameter and "return_temperature_bias" in " ".join(parameter["case"].astype(str))
    add("parameter_sensitivity_and_return_bias_propagation_present", return_bias_changes, "return-bias cases are regenerated through the causal proxy")
    add("flensburg_measured_supply_validation_present", not _csv("results/flensburg_measured_only_validation.csv").empty, "assumed return excluded from measured external table")
    xai = _csv("results/xai4heat_sparse_substation_validation_final.csv")
    add("xai4heat_all_valid_measured_node_validation_present", not xai.empty and set(xai.get("state_type", pd.Series(dtype=str))) == {"real_measured_node"}, "all-valid-range XAI4HEAT leave-one-substation-out result present")
    add("xai4heat_station_quality_audit_present", not _csv("results/xai4heat_station_quality_audit.csv").empty, "deduplication and station-level reversal audit present")
    blind_checkpoint = _csv("results/principal_models_blind_measured_return_summary.csv")
    add(
        "principal_checkpoints_have_blind_measured_return_audit",
        len(blind_checkpoint) == 5
        and {"Measured-return persistence", *CORE_MODELS}.issubset(set(blind_checkpoint.get("model", pd.Series(dtype=str)).astype(str)))
        and _finite(blind_checkpoint, ["mean_RMSE_return_measured_C", "mean_MAE_return_measured_C"]),
        "current return and internal C-class thermal and S-class hydraulic sensor inputs are withheld while a real measured return target is scored",
    )

    # Retained timestamp discontinuities must be treated as trajectory starts,
    # never as nominal 15-minute transitions.
    gap_path = _path("results/gap_handling_audit.json")
    dense_path = _path("results/dense_reconstruction_payload_provenance.json")
    online_gap = _csv("results/online_replay_gap_handling_audit.csv")
    try:
        gap = json.loads(gap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        gap = {}
    add(
        "retained_timestamp_gap_is_scientifically_handled",
        gap.get("gap_duration_hours") == 17.25
        and gap.get("gap_start_index") == 62
        and gap.get("excluded_cross_gap_window_count") == 11
        and gap.get("eligible_contiguous_window_count") == 746,
        "17.25-h discontinuity is a trajectory restart; 11 cross-gap windows are excluded",
    )
    try:
        dense = json.loads(dense_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        dense = {}
    add(
        "dense_payload_uses_corrected_contiguous_windows",
        dense.get("test_window_count") == 91 and dense.get("trajectory_start_indices") == [0, 62],
        "paired C-class thermal and S-class hydraulic reconstruction payload is regenerated from valid test windows only",
    )
    add(
        "online_replay_restarts_at_retained_gaps",
        not online_gap.empty and int(online_gap.iloc[0].get("retained_gap_restarts", 0)) > 0 and int(online_gap.iloc[0].get("excluded_test_restart_samples", 0)) > 0,
        "online replay resets state and excludes restart samples from contiguous scoring",
    )

    # Submission assets and manuscript claims.
    main_tex = _path("paper/main_ate_submission_candidate.tex")
    supp_tex = _path("paper/supplementary_material.tex")
    main_text = _expanded_tex(main_tex)
    supp_text = _expanded_tex(supp_tex)
    claim_ok = _write_claim_safety(main_text, supp_text)
    add("manuscript_claim_safety_passed", claim_ok, "required evidence language is present and unsafe claims are absent")
    former_observer_name = "E" + "nKF"
    add(
        "observer_name_is_correct_everywhere_active",
        former_observer_name.lower() not in (main_text + supp_text).lower()
        and "covariance-conditioned Gaussian observer" in (main_text + supp_text),
        "active manuscript and imported tables use the implemented static-observer name",
    )
    add("compact_main_manuscript_exists", main_tex.exists(), "active submission manuscript present")
    add("supplementary_manuscript_exists", supp_tex.exists(), "supplementary manuscript present")
    abstract_words = _abstract_word_count(main_text)
    add("abstract_within_190_word_limit", 1 <= abstract_words <= 190, f"main abstract contains {abstract_words} readable words")
    add("submission_consistency_audit_script_exists", _exists("scripts/audit_submission_consistency.py"), "data-driven source/result audit is available")
    add("reproducible_submission_build_script_exists", _exists("scripts/build_submission.sh"), "clean build, audit, compile, and font-preflight script is available")
    add("elsevier_keyword_limit_respected", _keyword_count(main_text) <= 6, f"main manuscript contains {_keyword_count(main_text)} keywords")
    final_title = "Real-Data-Assisted Thermal State Estimation in District Heating Networks: An Evidence-Separated Benchmark with Simulator-Assisted Hydraulics"
    add(
        "final_title_is_synchronized",
        final_title in main_text and final_title in supp_text and final_title in (PROJECT_ROOT / "paper" / "cover_letter_ate_draft.tex").read_text(encoding="utf-8", errors="ignore"),
        "main manuscript, supplement, and cover letter use the final title",
    )
    try:
        calibrated_params = json.loads(_path("results/calibrated_parameters.json").read_text(encoding="utf-8"))
        current_params = all(
            f"{float(calibrated_params[key]):.4f}" in main_text
            for key in ("heat_loss_U_W_m2K", "effective_velocity_factor", "flow_proxy_blend")
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        current_params = False
    add(
        "calibrated_parameter_prose_matches_json",
        current_params and "1.5401" not in main_text and "0.9801" not in main_text,
        "U, effective velocity, and flow-proxy blend are synchronized to calibrated_parameters.json",
    )
    uncertainty = _csv("results/uncertainty_conformal_evaluation_locked.csv")
    required_uncertainty = {"supply_temperature", "return_temperature", "head", "flow", "heat_loss"}
    uncertainty_current = not uncertainty.empty and set(uncertainty.get("quantity", pd.Series(dtype=str)).astype(str)) == required_uncertainty
    for _, uncertainty_row in uncertainty.iterrows():
        coverage = float(uncertainty_row["coverage"])
        width = float(uncertainty_row["mean_interval_width"])
        uncertainty_current = uncertainty_current and f"{coverage:.1f}\\%" in main_text and f"{width:.4g}" in main_text
    add(
        "conformal_uncertainty_prose_matches_locked_csv",
        uncertainty_current,
        "coverage and interval widths are current and reported separately by quantity",
    )
    bootstrap = _csv("results/moving_block_bootstrap_ci.csv")
    try:
        bootstrap_protocol = json.loads(_path("results/moving_block_bootstrap_protocol.json").read_text(encoding="utf-8"))
        bootstrap_current = (
            not bootstrap.empty
            and pd.to_numeric(bootstrap["source_window_rows"], errors="coerce").eq(1092).all()
            and pd.to_numeric(bootstrap["unique_timestamp_count"], errors="coerce").eq(102).all()
            and pd.to_numeric(bootstrap["block_length_steps"], errors="coerce").eq(12).all()
            and bootstrap_protocol.get("unique_timestamp_count") == 102
            and bootstrap_protocol.get("block_length_steps") == 12
            and "unique physical timestamp" in str(bootstrap_protocol.get("method", ""))
            and "102 unique chronological timestamps" in main_text
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        bootstrap_current = False
    add(
        "moving_block_bootstrap_uses_unique_chronological_timestamps",
        bool(bootstrap_current),
        "overlapping test-window predictions are collapsed before within-segment block resampling",
    )
    compact_main = re.sub(r"\s+", "", main_text)
    add(
        "effective_velocity_factor_is_in_displayed_thermal_equation",
        "C_k=\\frac{u^T_k\\Deltat}{\\Deltax}" in compact_main
        and "\\frac{\\eta_vq_{v,k}\\Deltat}{A\\Deltax}" in compact_main,
        "eta_v is explicitly connected to the thermal Courant number",
    )
    add(
        "calibration_stages_are_separated_and_weighted",
        "J_T(\\boldsymbol\\theta_T)" in compact_main
        and "+8.0" in compact_main
        and "f^*=\\underset" in compact_main,
        "thermal/effective-flow and friction-proxy stages are separately defined",
    )
    add(
        "decoded_source_boundary_residual_is_defined",
        "r_b=\\frac{1}{n}" in compact_main and "No output boundary replacement is applied during estimator evaluation" in main_text,
        "model decoder boundary metric is distinguished from the imposed simulator boundary",
    )
    add(
        "calibration_energy_ratio_matches_executed_code",
        "total absolute residual divided by total absolute imposed load" in main_text,
        "calibration report does not misstate an instantaneous-load denominator",
    )
    add(
        "review_archive_and_manifest_exist",
        _exists("paper/real_data_assisted_dh_review_archive.zip") and _exists("results/submission_package_audit.csv"),
        "current source-only reviewer archive and audit are present",
    )
    archive_hash_matches, archive_source_only = _review_archive_integrity()
    add("review_archive_matches_active_main_tex", archive_hash_matches, "archived main TeX hash matches active source")
    add("review_archive_excludes_stale_build_products", archive_source_only, "archive has no previous PDFs or LaTeX auxiliary products")
    add(
        "data_code_availability_declares_reviewer_archive",
        "reviewer archive" in main_text.lower()
        and "real\\_data\\_assisted\\_dh\\_review\\_archive.zip" in main_text
        and "sha-256" in main_text.lower(),
        "availability statement identifies the submission archive without inventing a DOI",
    )
    add(
        "multi_window_scope_is_explicitly_disclosed",
        "three separated" in main_text.lower()
        and "heating-season" in main_text.lower()
        and "does not establish full seasonal stability" in (main_text + supp_text).lower(),
        "multi-window repeated training is distinguished from annual or summer seasonal stability",
    )
    add(
        "architecture_screen_is_scoped_to_locked_archive",
        "A broader single-run architecture screen is retained in the locked reviewer archive for hypothesis generation" in main_text
        and "A broader single-run architecture screen is retained in the locked reviewer archive for hypothesis generation" in supp_text,
        "single-run hypothesis screen is not represented as primary manuscript evidence",
    )
    xai_table = _path("paper/tables/table_xai4heat_withholding_diagnostics.tex")
    xai_table_text = xai_table.read_text(encoding="utf-8", errors="ignore") if xai_table.exists() else ""
    xai_primary = _csv("results/xai4heat_sparse_substation_validation_final.csv")
    xai_tokens = []
    for _, row in xai_primary.iterrows():
        xai_tokens.extend([str(int(row["n_total_samples"])), f"{float(row['mean_RMSE']):.3f}"])
    add(
        "xai4heat_publication_table_uses_all_valid_protocol",
        bool(xai_tokens) and all(token in xai_table_text for token in xai_tokens) and "all_valid_range" in " ".join(xai_primary.get("primary_temperature_protocol", pd.Series(dtype=str)).astype(str)),
        "headline XAI4HEAT table matches the all-valid-range measured-node CSV",
    )
    residual_table = _path("paper/tables/table_residual_submission_risk_mitigation.tex")
    residual_text = residual_table.read_text(encoding="utf-8", errors="ignore") if residual_table.exists() else ""
    flensburg = _csv("results/flensburg_measured_only_validation.csv")
    try:
        direct = float(flensburg.loc[flensburg["mode"].eq("direct_transfer"), "measured supply RMSE_C"].iloc[0])
        offset = float(flensburg.loc[flensburg["mode"].eq("calibration_only_offset_adaptation"), "measured supply RMSE_C"].iloc[0])
        few = float(flensburg.loc[flensburg["mode"].eq("few_shot_decoder_bias_adaptation"), "measured supply RMSE_C"].iloc[0])
        reduction = 100.0 * (direct - few) / direct
        flensburg_table_current = all(f"{value:.3f}" in residual_text for value in (direct, offset, few)) and f"{reduction:.1f}" in residual_text
    except (KeyError, IndexError, TypeError, ValueError):
        flensburg_table_current = False
    add("flensburg_risk_table_is_current", flensburg_table_current, "risk table matches current measured-only Flensburg modes")
    cover = _path("paper/cover_letter_ate_draft.tex")
    cover_text = _expanded_tex(cover)
    add(
        "cover_letter_matches_primary_ranking",
        "rather than a single-model superiority claim" in cover_text
        and "Thirty-six independent training runs" in cover_text
        and "Transformer-MSE records six lowest window means" in cover_text
        and "PI-GNN-GRU-v3 balanced mode five" in cover_text
        and "GRU-MSE four" in cover_text
        and "Distributed hydraulic fields remain simulator-assisted diagnostics" in cover_text,
        "cover letter matches the objective- and window-dependent repeated-training evidence",
    )
    add("focused_main_tables_present", all(_exists(f"paper/tables/{name}") for name in [
        "table_main_1_data_evidence.tex", "table_main_2_method_definition.tex", "table_main_3_calibration_verification.tex",
        "table_main_4_benchmark.tex", "table_main_5_measured_node.tex", "table_main_6_sensor_layout.tex",
        "table_flensburg_causal_forecast.tex", "table_main_8_reproducibility.tex", "table_causal_proxy_audit.tex",
    ]), "compact evidence-separated table set present")
    measured_adaptation = _csv("results/measured_return_checkpoint_adaptation.csv")
    balanced_affine = measured_adaptation[
        measured_adaptation.get("model", pd.Series(dtype=str)).astype(str).eq("Proposed PI-GNN-GRU-v3 balanced_mode")
        & measured_adaptation.get("adaptation", pd.Series(dtype=str)).astype(str).eq("training-only affine readout")
    ] if not measured_adaptation.empty else pd.DataFrame()
    add(
        "heldout_measured_return_readout_adaptation_complete",
        len(balanced_affine) == 5
        and pd.to_numeric(balanced_affine.get("train_test_overlap"), errors="coerce").eq(0).all()
        and pd.to_numeric(balanced_affine.get("RMSE_C"), errors="coerce").lt(pd.to_numeric(balanced_affine.get("raw_RMSE_C"), errors="coerce")).all(),
        "five balanced-mode seeds use training-only readout calibration and disjoint measured-return test targets",
    )
    xai_chronological = _csv("results/xai4heat_chronological_withholding.csv")
    add(
        "xai4heat_chronological_target_withholding_complete",
        len(xai_chronological) == 20
        and xai_chronological.get("withheld_target_temperature_used_as_feature", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
        and xai_chronological.get("target_conditioned_order_filter_used", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all(),
        "five substations x two variables x two estimators use chronological target-independent withholding",
    )
    flensburg_causal = _csv("results/flensburg_causal_supply_forecast.csv")
    add(
        "flensburg_causal_measured_supply_forecast_complete",
        len(flensburg_causal) == 12
        and set(pd.to_numeric(flensburg_causal.get("horizon_h"), errors="coerce").dropna().astype(int)) == {1, 6, 24}
        and flensburg_causal.get("future_supply_used_as_feature", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all()
        and flensburg_causal.get("return_temperature_used", pd.Series(dtype=str)).astype(str).str.lower().eq("false").all(),
        "causal measured-supply forecasting covers three horizons without assumed return-temperature scoring",
    )
    add(
        "measured_and_external_validation_figures_present",
        _exists("figures/final/fig_measured_node_validation.pdf")
        and _exists("figures/final/fig_flensburg_causal_forecasting.pdf")
        and "figures/final/fig_measured_node_validation.pdf" in main_text
        and "figures/final/fig_flensburg_causal_forecasting.pdf" in main_text,
        "main paper separates measured-node withholding from causal Flensburg forecasting",
    )
    adaptation_summary = _csv("results/measured_return_checkpoint_adaptation_summary.csv")
    xai_summary = _csv("results/xai4heat_chronological_withholding_summary.csv")
    try:
        adapted_row = adaptation_summary[
            adaptation_summary["model"].astype(str).eq("Proposed PI-GNN-GRU-v3 balanced_mode")
            & adaptation_summary["adaptation"].astype(str).eq("training-only affine readout")
        ].iloc[0]
        xai_supply = xai_summary[
            xai_summary["variable"].astype(str).eq("supply_temp_C")
            & xai_summary["estimator"].astype(str).eq("chronological multi-station ridge")
        ].iloc[0]
        xai_return = xai_summary[
            xai_summary["variable"].astype(str).eq("return_temp_C")
            & xai_summary["estimator"].astype(str).eq("chronological multi-station ridge")
        ].iloc[0]
        local_flensburg = flensburg_causal[flensburg_causal["estimator"].astype(str).eq("Flensburg local-history ridge")].sort_values("horizon_h")
        expected_numbers = [
            f"{float(adapted_row['mean_RMSE_C']):.3f}",
            f"{float(adapted_row['std_RMSE_C']):.3f}",
            f"{float(xai_supply['mean_fold_RMSE_C']):.3f}",
            f"{float(xai_return['mean_fold_RMSE_C']):.3f}",
            *[f"{float(value):.3f}" for value in local_flensburg["RMSE_C"]],
        ]
        new_numbers_current = len(local_flensburg) == 3 and all(value in main_text for value in expected_numbers)
    except (KeyError, IndexError, TypeError, ValueError):
        new_numbers_current = False
    add(
        "new_measured_validation_numbers_match_csv_sources",
        new_numbers_current,
        "PI readout, XAI4HEAT, and Flensburg headline values are formatted directly from current CSV evidence",
    )
    figure_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_text)
    figure_paths = [path for path in figure_paths if path.startswith("figures/final/")]
    figures_exist = bool(figure_paths) and all(_exists(path) or _exists(f"paper/{path}") for path in figure_paths)
    add("main_figure_paths_resolve", figures_exist, "all figures referenced by the main manuscript resolve")
    required_figure_sequence = [
        "figures/final/fig_method_evidence_hierarchy.pdf",
        "figures/final/fig_thermo_hydraulic_governing_model.pdf",
        "figures/final/fig_evaluation_protocol_flowchart.pdf",
        "figures/final/fig_pignn_gru_v3_architecture.pdf",
        "figures/final/fig1_real_sonderborg_data_overview.pdf",
        "figures/final/fig4_calibration_and_discretization.pdf",
        "figures/final/fig_calibration_equifinality.pdf",
        "figures/final/fig_measured_node_validation.pdf",
        "figures/final/fig_flensburg_causal_forecasting.pdf",
        "figures/final/fig_multi_window_three_seed.pdf",
        "figures/final/fig_engineering_reconstruction_summary.pdf",
    ]
    add(
        "main_figure_count_and_sequence_are_locked",
        figure_paths == required_figure_sequence,
        f"main manuscript references {len(figure_paths)} final figures in the locked scientific sequence",
    )
    figure_provenance = _csv("results/active_figure_provenance_post_causality.csv")
    add(
        "active_core_figures_are_post_causality_and_source_traced",
        len(figure_provenance) == 4
        and figure_provenance.get("generated_after_all_sources", pd.Series(dtype=str)).astype(str).str.lower().eq("true").all()
        and figure_provenance.get("quantitative_source", pd.Series(dtype=str)).astype(str).str.contains("no raster digitization", case=False).all(),
        "workflow, calibration, reconstruction, and heat/energy figures are source-traced after corrected evidence",
    )
    add(
        "focused_workflow_figure_is_active",
        "figures/final/fig_method_evidence_hierarchy.pdf" in figure_paths
        and "figures/final/fig_digital_twin_workflow_concept.pdf" not in figure_paths,
        "main workflow shows evidence hierarchy rather than anomaly/KPI scope",
    )
    font_preflight = _csv("results/figure_font_preflight.csv")
    font_preflight_ok = (
        not font_preflight.empty
        and {"status", "type3_present", "unembedded_font_present", "mirror_hash_match"}.issubset(font_preflight.columns)
        and font_preflight["status"].astype(str).eq("PASS").all()
        and font_preflight["type3_present"].astype(str).str.lower().eq("false").all()
        and font_preflight["unembedded_font_present"].astype(str).str.lower().eq("false").all()
        and font_preflight["mirror_hash_match"].astype(str).str.lower().eq("true").all()
    )
    add("active_vector_figure_fonts_preflight_passes", font_preflight_ok, "active figure PDFs contain no Type 3 or used unembedded fonts")
    svg_audit = _csv("results/publication_svg_figure_audit.csv")
    svg_audit_ok = (
        not svg_audit.empty
        and {
            "status",
            "no_embedded_raster",
            "editable_text_present",
            "times_new_roman_declared",
            "vector_geometry_present",
        }.issubset(svg_audit.columns)
        and svg_audit["status"].astype(str).eq("PASS").all()
        and svg_audit["no_embedded_raster"].astype(str).str.lower().eq("true").all()
        and svg_audit["editable_text_present"].astype(str).str.lower().eq("true").all()
        and svg_audit["times_new_roman_declared"].astype(str).str.lower().eq("true").all()
        and svg_audit["vector_geometry_present"].astype(str).str.lower().eq("true").all()
    )
    add(
        "publication_active_svg_masters_are_true_vector_times",
        svg_audit_ok,
        "all active SVG masters use editable Times New Roman text and vector geometry with no embedded raster image",
    )
    figure_sync = _csv("results/publication_figure_sync_audit.csv")
    figure_sync_ok = (
        not figure_sync.empty
        and {"source_exists", "mirror_exists", "hash_match"}.issubset(figure_sync.columns)
        and figure_sync["source_exists"].astype(str).str.lower().eq("true").all()
        and figure_sync["mirror_exists"].astype(str).str.lower().eq("true").all()
        and figure_sync["hash_match"].astype(str).str.lower().eq("true").all()
    )
    add(
        "latex_figure_mirrors_match_audited_masters",
        figure_sync_ok,
        "all active SVG/PDF/PNG files in the LaTeX tree match the authoritative figure masters by SHA-256",
    )
    add(
        "supplementary_float_controls_present",
        "\\raggedbottom" in supp_text and "\\sloppy" not in supp_text and supp_text.count("\\FloatBarrier") >= 8,
        "supplementary source uses local float barriers and avoids global sloppy spacing",
    )
    supp_source_text = supp_tex.read_text(encoding="utf-8", errors="ignore") if supp_tex.exists() else ""
    supplementary_table_inputs = re.findall(r"\\input\{tables/([^}]+)\}", supp_source_text)
    supplementary_table_line_counts = []
    for name in supplementary_table_inputs:
        table_path = _path(f"paper/tables/{name if name.endswith('.tex') else name + '.tex'}")
        if table_path.exists():
            supplementary_table_line_counts.append(len(table_path.read_text(encoding="utf-8", errors="ignore").splitlines()))
        else:
            supplementary_table_line_counts.append(10**9)
    add(
        "supplementary_active_tables_are_compact",
        bool(supplementary_table_line_counts) and max(supplementary_table_line_counts) <= 35,
        "active supplementary tables are compact renders; full matrices are retained in the reviewer archive",
    )

    broken_units = any("textasciicircum" in path.read_text(encoding="utf-8", errors="ignore") or "textbackslash circ" in path.read_text(encoding="utf-8", errors="ignore") for path in list((PROJECT_ROOT / "paper").glob("*.tex")) + list((PROJECT_ROOT / "paper" / "tables").glob("*.tex")))
    add("no_broken_degree_symbols", not broken_units, "no broken degree-Celsius LaTex escaping")
    verification_table = _path("paper/tables/table_main_3_calibration_verification.tex")
    table_text = verification_table.read_text(encoding="utf-8", errors="ignore") if verification_table.exists() else ""
    numerical = _csv("results/numerical_verification_expanded.csv")
    try:
        baseline_row = numerical.loc[pd.to_numeric(numerical["dx_m"], errors="coerce").eq(1000.0)].iloc[0]
        expected_numeric = (
            f"{float(baseline_row['outlet_Ts_L2_C']):.4f}",
            f"{float(baseline_row['outlet_Ts_Linf_C']):.4f}",
            f"{float(baseline_row['source_Tr_L2_C']):.4f}",
            f"{float(baseline_row['cumulative_heat_loss_error_pct']):.4f}",
        )
        numerical_table_current = all(value in table_text for value in expected_numeric)
    except (KeyError, IndexError, TypeError, ValueError):
        numerical_table_current = False
    add("numerical_verification_table_matches_current_csv", numerical_table_current, "main verification table contains current 1000 m/900 s versus fine-grid values")

    main_pdf_current = _current_pdf("paper/main_ate_submission_candidate.tex", "paper/main_ate_submission_candidate.pdf")
    supp_pdf_current = _current_pdf("paper/supplementary_material.tex", "paper/supplementary_material.pdf")
    add("main_pdf_is_current", main_pdf_current, "main PDF is current" if main_pdf_current else "main TeX changed; compile required")
    add("supplementary_pdf_is_current", supp_pdf_current, "supplementary PDF is current" if supp_pdf_current else "supplementary TeX changed; compile required")

    metadata_placeholders = (
        "AUTHOR INPUT REQUIRED",
        "AUTHOR NAME REQUIRED",
        "AFFILIATION, CITY, AND COUNTRY REQUIRED",
    )
    metadata_complete = (
        "Authors omitted for review" not in main_text
        and "Department of Mechanical and Energy Systems Engineering" not in main_text
        and not any(marker in main_text for marker in metadata_placeholders)
        and "Declaration of Competing Interest" in main_text
        and "CRediT authorship contribution statement" in main_text
        and "Funding" in main_text
    )
    add(
        "submission_author_metadata_and_declarations_complete",
        metadata_complete,
        "requires factual author names, affiliations, funding, competing interests, and CRediT roles from the authors",
    )
    terminology_files = [
        _path("paper/main_ate_submission_candidate.tex"),
        _path("paper/supplementary_material.tex"),
        *sorted((_path("paper/sections")).glob("*.tex")),
        *sorted((_path("paper/tables")).glob("*.tex")),
    ]
    obsolete_combined_class = any(
        re.search(r"\bC/S\b|C/S-", path.read_text(encoding="utf-8", errors="ignore"))
        for path in terminology_files
        if path.exists()
    )
    add(
        "evidence_terminology_has_no_combined_class",
        not obsolete_combined_class and "mixed C+S dependency" in main_text,
        "M, C, and S remain distinct; joint metrics use the phrase mixed C+S dependency",
    )

    integrity = _csv("results/final_integrity_audit.csv")
    add(
        "post_causality_integrity_audit_passes",
        not integrity.empty and integrity.get("passed", pd.Series(dtype=str)).astype(str).str.lower().eq("true").all(),
        "canonical hash, corrected states, 20 runs, dependent analyses, verification, and archive hashes all pass",
    )

    df = pd.DataFrame(rows)
    df.to_csv(_path("results/paper_quality_gate_report_final.csv"), index=False)
    df.to_csv(_path("results/final_submission_quality_gate.csv"), index=False)

    external_completion_checks = {
        "main_pdf_is_current",
        "supplementary_pdf_is_current",
        "submission_author_metadata_and_declarations_complete",
    }
    hard_failure = not df.loc[~df["check"].isin(external_completion_checks), "passed"].all()
    metadata_complete = bool(
        df.loc[df["check"].eq("submission_author_metadata_and_declarations_complete"), "passed"].iloc[0]
    )
    if hard_failure:
        verdict = "ATE_EVIDENCE_PACKAGE_NEEDS_CORRECTION"
    elif not metadata_complete and main_pdf_current and supp_pdf_current:
        verdict = "ATE_SCIENTIFIC_EVIDENCE_READY_PENDING_AUTHOR_METADATA"
    elif not metadata_complete:
        verdict = "ATE_SCIENTIFIC_EVIDENCE_READY_PENDING_AUTHOR_METADATA_AND_COMPILE"
    elif main_pdf_current and supp_pdf_current:
        verdict = "ATE_SUBMISSION_READY_POST_CAUSALITY_LOCKED_EVIDENCE"
    else:
        verdict = "ATE_SUBMISSION_ARCHIVE_SYNCHRONIZED_PENDING_CLEAN_COMPILE"
    lines = [f"Final verdict: {verdict}", "", "Focused causal-evidence quality gate", ""]
    for row in rows:
        lines.append(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['check']}: {row['note']}")
    text = "\n".join(lines) + "\n"
    for name in [
        "paper_quality_gate_report_final.txt",
        "final_submission_quality_gate.txt",
        "paper_quality_gate_report.txt",
    ]:
        _path(f"results/{name}").write_text(text, encoding="utf-8")
    return verdict


if __name__ == "__main__":
    print(run_paper_quality_gate())
