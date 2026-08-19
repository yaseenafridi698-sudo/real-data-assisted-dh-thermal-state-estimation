"""Create a self-contained, source-only reviewer archive for the ATE submission.

The archive intentionally excludes old PDFs and LaTeX auxiliary files.  It is a
curated snapshot of the current manuscript sources, final vector figures, table
assets, compilation script, and numerical evidence needed to audit the reported
claims.  It does not recalculate or alter any result.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"
RESULTS = PROJECT_ROOT / "results"
BUNDLE = PROJECT_ROOT / "submission_review_bundle"
ARCHIVE = PAPER / "real_data_assisted_dh_review_archive.zip"


EVIDENCE_FILES = (
    "calibrated_parameters.json",
    "canonical_dataset_manifest.csv",
    "canonical_dataset_manifest.json",
    "canonical_dataset_manifest.md",
    "corrected_simulator_states.npz",
    "corrected_simulator_states_provenance.json",
    "locked_later_replay_metrics.csv",
    "verification_campaign_status.csv",
    "verification_campaign_status.json",
    "second_chronological_window_metrics.csv",
    "second_chronological_window_summary.csv",
    "second_chronological_window_protocol.json",
    "full_dependent_regeneration_manifest.csv",
    "full_dependent_regeneration_protocol.json",
    "cross_estimator_disagreement.csv",
    "gaussian_observer_baseline_metrics.csv",
    "observer_heldout_unique_timestamp_comparison.csv",
    "baseline_comparison_with_gaussian_observer.csv",
    "principal_models_blind_measured_return.csv",
    "principal_models_blind_measured_return_summary.csv",
    "calibration_temporal_transfer_audit.csv",
    "calibration_temporal_transfer_summary.json",
    "measured_return_checkpoint_adaptation.csv",
    "measured_return_checkpoint_adaptation_summary.csv",
    "xai4heat_chronological_withholding.csv",
    "xai4heat_chronological_withholding_summary.csv",
    "flensburg_causal_supply_forecast.csv",
    "flensburg_causal_supply_forecast_summary.csv",
    "flensburg_causal_supply_forecast_timeseries_sample.csv",
    "final_measured_validation_upgrade_protocol.json",
    "active_figure_provenance_post_causality.csv",
    "calibration_metrics.csv",
    "calibration_scope_audit.csv",
    "gap_handling_audit.csv",
    "gap_handling_audit.json",
    "numerical_verification_expanded.csv",
    "proxy_causality_audit.csv",
    "causal_regeneration_manifest.csv",
    "baseline_comparison_final.csv",
    "physics_consistency_comparison_final.csv",
    "repeated_seed_statistics.csv",
    "repeated_seed_raw_metrics.csv",
    "repeated_seed_completeness_audit.csv",
    "repeated_seed_checkpoint_audit.csv",
    "repeated_seed_stability_status.csv",
    "training_stability_summary_final.csv",
    "repeated_seed_protocol.json",
    "multi_window_three_seed_raw_metrics.csv",
    "multi_window_three_seed_summary.csv",
    "multi_window_three_seed_aggregate.csv",
    "multi_window_rank_stability.csv",
    "multi_window_three_seed_protocol.json",
    "ambient_boundary_reanalysis_sensitivity.csv",
    "ambient_reanalysis_period_metrics.csv",
    "ambient_reanalysis_provenance.json",
    "calibration_equifinality_ensemble.csv",
    "calibration_equifinality_internal_field_spread.csv",
    "calibration_equifinality_protocol.json",
    "new_evidence_manuscript_sync.json",
    "topology_scope_decision.md",
    "moving_block_bootstrap_ci.csv",
    "moving_block_bootstrap_protocol.json",
    "online_replay_metrics.csv",
    "online_replay_gap_handling_audit.csv",
    "dense_reconstruction_payloads.npz",
    "dense_reconstruction_payload_provenance.json",
    "gaussian_observer_dense_predictions.npz",
    "measured_node_baseline_comparison.csv",
    "causal_heat_load_input_ablation.csv",
    "sonderborg_blind_plant_validation_summary.csv",
    "xai4heat_sparse_substation_validation_final.csv",
    "xai4heat_sparse_substation_validation_by_substation.csv",
    "xai4heat_sparse_substation_validation_physically_ordered_sensitivity.csv",
    "xai4heat_station_quality_audit.csv",
    "xai4heat_raw_file_inventory.csv",
    "xai4heat_publication_protocol_resolution.md",
    "flensburg_measured_only_validation.csv",
    "flensburg_domain_shift_analysis.csv",
    "sensor_layout_ranking_by_objective.csv",
    "sensor_layout_comparison_final.csv",
    "sensor_layout_interpretation_final.csv",
    "sensor_layout_scenario_classification.csv",
    "sensor_layout_geometry_ranking.csv",
    "sensor_layout_robustness_scenarios.csv",
    "noise_dropout_robustness_final.csv",
    "uncertainty_conformal_evaluation_locked.csv",
    "ablation_study_final.csv",
    "seasonal_generalization.csv",
    "combined_stress_test.csv",
    "parameter_identifiability_sensitivity.csv",
    "parameter_sensitivity_summary.csv",
    "parameter_sensitivity_propagation_audit.csv",
    "computational_cost.csv",
    "strict_target_dependency_audit.csv",
    "corridor_length_sensitivity.csv",
    "hydraulic_identifiability_summary.csv",
    "final_submission_lock_report.txt",
    "audit_submission_consistency.txt",
    "audit_targeted_remaining_issues.txt",
    "latex_compile_report.txt",
    "font_preflight_report.txt",
    "manuscript_font_preflight.csv",
    "figure_font_preflight.csv",
    "figure_font_preflight.txt",
    "ate_submission_style_alignment.md",
    "final_integrity_audit.csv",
    "final_integrity_audit.txt",
    "final_integrity_active_file_manifest.csv",
    "final_integrity_active_file_manifest.json",
    "superseded_archive_status.txt",
)

# Deliberately curate executable sources.  The repository contains historical
# manuscript-polishing and exploratory scripts whose old wording is not part of
# the locked publication workflow; shipping all of ``src`` would undermine the
# point of an auditable final archive.
SOURCE_FILES = (
    "src/config.py",
    "src/utils.py",
    "src/data_registry.py",
    "src/data_loaders.py",
    "src/data_preprocessing.py",
    "src/freeze_canonical_dataset.py",
    "src/real_data_mapper.py",
    "src/effective_physics.py",
    "src/thermo_hydraulic_simulator.py",
    "src/calibration.py",
    "src/models.py",
    "src/losses.py",
    "src/train.py",
    "src/evaluate.py",
    "src/study_workflow.py",
    "src/sensor_layouts.py",
    "src/external_validation.py",
    "src/xai4heat_measured_node_validation.py",
    "src/critical_measured_validation.py",
    "src/reviewer_critical_fixes.py",
    "src/online_replay_validation.py",
    "src/submission_evidence_repairs.py",
    "src/final_measured_validation_upgrade.py",
    "src/export_dense_reconstruction_payload.py",
    "src/rerun_external_validation_from_checkpoint.py",
    "src/parameter_identifiability_sensitivity.py",
    "src/repeated_seed_statistics.py",
    "src/multi_window_three_seed_benchmark.py",
    "src/ambient_reanalysis_sensitivity.py",
    "src/calibration_equifinality_analysis.py",
    "src/synchronize_window_weather_equifinality_manuscript.py",
    "src/second_chronological_window.py",
    "src/recalibrate_corrected_simulator.py",
    "src/refresh_post_causality_metadata.py",
    "src/synchronize_post_causality_manuscript.py",
    "src/rebuild_dependent_evidence.py",
    "src/verification_campaign.py",
    "src/final_integrity_audit.py",
    "src/freeze_final_submission_results.py",
    "src/hash_superseded_archive.py",
    "src/reconcile_sensor_layout_evidence.py",
    "src/plots.py",
    "src/thermo_hydraulic_coupling_analysis.py",
    "src/supplementary_study_utils.py",
    "src/figure_font_preflight.py",
    "src/manuscript_font_preflight.py",
    "src/refresh_targeted_protocol_metadata.py",
    "src/gap_handling_audit.py",
    "src/make_paper_assets.py",
    "src/make_focused_main_tables.py",
    "src/synchronize_submission_evidence.py",
    "src/final_result_audit_for_ate.py",
    "src/reviewer_submission_additions.py",
    "src/create_submission_review_bundle.py",
    "src/create_final_audit_package.py",
    "src/verify_package_manifests.py",
    "src/write_final_audit_change_log.py",
    "src/paper_quality_gate.py",
    "scripts/audit_submission_consistency.py",
    "scripts/audit_targeted_remaining_issues.py",
    "scripts/static_latex_preflight.py",
    "scripts/make_focused_submission_figures.py",
    "scripts/make_method_figures_ate.py",
    "scripts/build_submission.sh",
    "scripts/rebuild_post_causality_evidence.ps1",
    "run_real_data_study.py",
    "run_download_data.py",
    "requirements.txt",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path, copied: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(destination)


def _copy_tex_dependencies(tex_names: tuple[str, ...], copied: list[Path]) -> None:
    """Copy exactly the table and figure assets imported by the active sources."""
    for tex_name in tex_names:
        source_tex = PAPER / tex_name
        text = source_tex.read_text(encoding="utf-8")
        for item in re.findall(r"\\input\{([^}]+)\}", text):
            source = PAPER / (item if item.endswith(".tex") else f"{item}.tex")
            if not source.exists():
                raise FileNotFoundError(f"Active TeX input is missing: {source}")
            _copy(source, BUNDLE / source.relative_to(PROJECT_ROOT), copied)
        for item in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            candidates = [PAPER / item, PROJECT_ROOT / item]
            if not Path(item).suffix:
                candidates.extend([candidate.with_suffix(".pdf") for candidate in list(candidates)])
            source = next((candidate for candidate in candidates if candidate.exists()), None)
            if source is None:
                raise FileNotFoundError(f"Active TeX figure is missing: {item}")
            _copy(source, BUNDLE / source.relative_to(PROJECT_ROOT), copied)


def create_submission_review_bundle() -> pd.DataFrame:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for name in (
        "main_ate_submission_candidate.tex",
        "supplementary_material.tex",
        "references.bib",
        "compile_submission.ps1",
        "highlights_ate.txt",
        "cover_letter_ate_draft.tex",
        "SUBMISSION_METADATA_REQUIRED.md",
    ):
        source = PAPER / name
        if source.exists():
            _copy(source, BUNDLE / "paper" / name, copied)

    _copy_tex_dependencies(("main_ate_submission_candidate.tex", "supplementary_material.tex", "cover_letter_ate_draft.tex"), copied)
    for name in EVIDENCE_FILES:
        source = RESULTS / name
        if source.exists():
            _copy(source, BUNDLE / "results" / name, copied)

    # Include the executable publication workflow, not all historical helpers.
    for relative in SOURCE_FILES:
        source = PROJECT_ROOT / relative
        if source.exists():
            _copy(source, BUNDLE / relative, copied)
    config = PROJECT_ROOT / "config"
    if config.exists():
        for source in config.rglob("*"):
            if source.is_file():
                _copy(source, BUNDLE / source.relative_to(PROJECT_ROOT), copied)

    # Exact repeated-seed checkpoints and epoch histories provide the complete
    # 4 x 5 artifact set documented by the protocol.  Processed input is
    # included for review reproducibility; raw public downloads are not
    # redistributed by this package.
    for source in sorted(list(RESULTS.glob("seed_*")) + list(RESULTS.glob("mw_*"))):
        if source.is_file():
            _copy(source, BUNDLE / source.relative_to(PROJECT_ROOT), copied)
    processed = PROJECT_ROOT / "data" / "locked" / "sonderborg_processed_18703.csv"
    if processed.exists():
        _copy(processed, BUNDLE / processed.relative_to(PROJECT_ROOT), copied)
    for relative in (
        "data/locked/sonderborg_processed_18703_era5_land.csv",
        "data/external_weather/sonderborg_era5_land_2016_2019_hourly.csv",
    ):
        source = PROJECT_ROOT / relative
        if source.exists():
            _copy(source, BUNDLE / relative, copied)

    compile_report = RESULTS / "latex_compile_report.txt"
    if compile_report.exists():
        _copy(compile_report, BUNDLE / compile_report.relative_to(PROJECT_ROOT), copied)

    rows = []
    for path in sorted(set(copied)):
        rows.append(
            {
                "relative_path": path.relative_to(BUNDLE).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "status": "included",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(BUNDLE / "submission_package_audit.csv", index=False)
    manifest = "\n".join(f"{row['sha256']}  {row['relative_path']}" for row in rows) + "\n"
    (BUNDLE / "manifest_sha256.txt").write_text(manifest, encoding="utf-8")
    readme = [
        "# Anonymized ATE Review Archive",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This is a source-only archive. It contains no previously compiled PDFs or LaTeX auxiliary files.",
        "Compile from `paper/` with `compile_submission.ps1`, or run the equivalent pdflatex/bibtex sequence.",
        "",
        "Evidence boundary:",
        "- M: direct measured-node quantities.",
        "- C: calibrated-simulator quantities.",
        "- S: simulator-assisted hidden hydraulic states.",
        "",
        "The 50 C Flensburg return assumption is excluded from measured external-validation claims.",
        "The primary stability comparison uses three seeds in each of three separated gap-free heating-season windows; it is not full seasonal stability evidence. A five-seed single-window audit is retained as a controlled companion.",
        "The configured 5 C ambient boundary is audited against a provenance-hashed ERA5-Land reanalysis series. ERA5-Land is not a local weather-station measurement.",
        "The graph/no-graph corridor ablation does not establish graph superiority; no recognized branched benchmark is claimed.",
        "XAI4HEAT headline evidence uses the all-valid-range primary-temperature protocol (170,523 supply and 170,521 return observations after content deduplication). Station-level heterogeneity is retained explicitly.",
        "Raw public downloads are not redistributed; download/preprocessing scripts and the processed S\u00f8nderborg input used for the locked benchmark are included.",
        "Python dependencies are listed in `requirements.txt`. This locked review package supports evidence inspection and source audit; it is not represented as a fully standalone clean-environment reproduction bundle. Public raw-data downloads, runtime dependencies, and optional utilities must be resolved separately.",
        "The numerical evidence files are copied artifacts; this bundling process does not alter metrics.",
    ]
    (BUNDLE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for source in sorted(BUNDLE.rglob("*")):
            if source.is_file():
                zipped.write(source, source.relative_to(BUNDLE.parent).as_posix())

    manifest_ok = all(_sha256(BUNDLE / row["relative_path"]) == row["sha256"] for row in rows)

    audit.to_csv(RESULTS / "submission_package_audit.csv", index=False)
    lines = [
        "Submission reviewer archive report",
        "",
        f"archive: {ARCHIVE}",
        f"archive_exists: {ARCHIVE.exists()}",
        f"source_file_count: {len(rows)}",
        f"manifest_validation: {'PASS' if manifest_ok else 'FAIL'}",
        "status: source-only reviewer archive created",
        "note: attach this archive to the manuscript submission; do not submit an older compiled PDF.",
    ]
    (RESULTS / "submission_package_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    audit = create_submission_review_bundle()
    print(f"Created {ARCHIVE.name} with {len(audit)} source artifacts.")
