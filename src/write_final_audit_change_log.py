"""Write a file-level inventory for the final gap-aware publication audit."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"


EDITED_SOURCES = (
    "src/real_data_mapper.py",
    "src/thermo_hydraulic_simulator.py",
    "src/dataset.py",
    "src/study_workflow.py",
    "src/calibration.py",
    "src/online_replay_validation.py",
    "src/reviewer_critical_fixes.py",
    "src/critical_measured_validation.py",
    "src/supplementary_study_utils.py",
    "src/thermo_hydraulic_coupling_analysis.py",
    "src/plots.py",
    "src/figure_font_preflight.py",
    "src/parameter_identifiability_sensitivity.py",
    "src/rerun_external_validation_from_checkpoint.py",
    "src/export_dense_reconstruction_payload.py",
    "src/gap_handling_audit.py",
    "src/refresh_targeted_protocol_metadata.py",
    "src/make_focused_main_tables.py",
    "src/synchronize_submission_evidence.py",
    "src/final_result_audit_for_ate.py",
    "src/reviewer_submission_additions.py",
    "src/make_paper_assets.py",
    "src/paper_quality_gate.py",
    "src/create_submission_review_bundle.py",
    "src/create_final_audit_package.py",
    "src/verify_package_manifests.py",
    "src/write_final_audit_change_log.py",
    "scripts/build_submission.sh",
    "scripts/audit_submission_consistency.py",
    "scripts/audit_targeted_remaining_issues.py",
    "scripts/make_focused_submission_figures.py",
    "scripts/make_combined_simulation_figures.py",
    "scripts/make_method_figures_ate.py",
)

RESULTS_REGENERATED = (
    "calibration_metrics.csv",
    "numerical_verification_expanded.csv",
    "repeated_seed_protocol.json",
    "gap_handling_audit.csv",
    "gap_handling_audit.json",
    "dense_reconstruction_payloads.npz",
    "dense_reconstruction_payload_provenance.json",
    "online_replay_metrics.csv",
    "online_replay_gap_handling_audit.csv",
    "measured_node_baseline_comparison.csv",
    "causal_heat_load_input_ablation.csv",
    "parameter_identifiability_sensitivity.csv",
    "parameter_sensitivity_summary.csv",
    "parameter_sensitivity_propagation_audit.csv",
    "flensburg_measured_only_validation.csv",
    "flensburg_transfer_diagnostics.csv",
    "sensor_layout_geometry_ranking.csv",
    "audit_submission_consistency.txt",
    "audit_targeted_remaining_issues.txt",
    "latex_compile_report.txt",
    "font_preflight_report.txt",
    "figure_font_preflight.csv",
    "figure_font_preflight.txt",
    "ate_submission_style_alignment.md",
    "active_audit_index.md",
    "package_manifest_verification.txt",
)


def _dependencies(tex: Path) -> tuple[list[Path], list[Path]]:
    text = tex.read_text(encoding="utf-8", errors="ignore")
    tables = []
    figures = []
    for item in re.findall(r"\\input\{([^}]+)\}", text):
        candidate = tex.parent / (item if item.endswith(".tex") else f"{item}.tex")
        tables.append(candidate)
    for item in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        raw = Path(item)
        options = [tex.parent / raw, ROOT / raw]
        if not raw.suffix:
            options.extend(path.with_suffix(".pdf") for path in list(options))
        figures.append(next((path for path in options if path.exists()), options[0]))
    return tables, figures


def main() -> None:
    rows: list[dict[str, str]] = []
    for relative in EDITED_SOURCES:
        rows.append({"file_path": relative, "final_pass_action": "edited or added", "authoritative_source": "gap-handling and audit implementation"})
    for relative in (
        "paper/main_ate_submission_candidate.tex",
        "paper/supplementary_material.tex",
        "paper/cover_letter_ate_draft.tex",
        "paper/highlights_ate.txt",
        "results/final_ate_positioning_summary.md",
        "FINAL_AUDIT_REPORT.md",
    ):
        rows.append({"file_path": relative, "final_pass_action": "edited", "authoritative_source": "current locked results and audit reports"})
    for tex_name in ("main_ate_submission_candidate.tex", "supplementary_material.tex"):
        tables, figures = _dependencies(PAPER / tex_name)
        for path in tables:
            rows.append({"file_path": path.relative_to(ROOT).as_posix(), "final_pass_action": "regenerated or verified active table", "authoritative_source": "current CSV/JSON results"})
        for path in figures:
            rows.append({"file_path": path.relative_to(ROOT).as_posix(), "final_pass_action": "regenerated or verified active figure", "authoritative_source": "current arrays/CSV and plotting scripts"})
    for name in RESULTS_REGENERATED:
        rows.append({"file_path": f"results/{name}", "final_pass_action": "regenerated or audited", "authoritative_source": "gap-aware workflow/checkpoints"})
    rows.append({"file_path": "paper/real_data_assisted_dh_review_archive.zip", "final_pass_action": "rebuilt", "authoritative_source": "submission_review_bundle manifest"})
    rows.append({"file_path": "final_ate_submission_package.zip", "final_pass_action": "rebuilt as source/evidence package", "authoritative_source": "delivery ZIP SHA-256 manifest"})
    frame = pd.DataFrame(rows).drop_duplicates(subset=["file_path"]).sort_values("file_path")
    frame.to_csv(RESULTS / "final_audit_changed_files.csv", index=False)
    print(RESULTS / "final_audit_changed_files.csv")


if __name__ == "__main__":
    main()
