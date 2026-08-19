from __future__ import annotations

import re
import runpy
import shutil
import sys
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    mpimg = None
    plt = None

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT

try:
    from src.utils import ensure_dir
except Exception:
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


TITLE = (
    "Real-Data-Assisted Thermo-Hydraulic Digital Twin Benchmark "
    "for Sparse-Sensor State Estimation in District Heating Networks"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def fix_latex_unit_symbols() -> None:
    """Fix escaped degree-Celsius math produced by pandas table escaping."""
    replacements = {
        r"\$\textasciicircum \textbackslash circ\$C": r"$^\circ$C",
        r"\$\textasciicircum{}\textbackslash{}circ\$C": r"$^\circ$C",
        r"\$\^\\circ\$C": r"$^\circ$C",
    }
    for folder in [PROJECT_ROOT / "paper", PROJECT_ROOT / "paper" / "tables"]:
        for path in folder.glob("*.tex"):
            text = _read(path)
            original = text
            for old, new in replacements.items():
                text = text.replace(old, new)
            if text != original:
                _write(path, text)


def _source_file(stem: str, suffix: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "figures" / "final" / f"{stem}{suffix}",
        PROJECT_ROOT / "figures" / f"{stem}{suffix}",
        PROJECT_ROOT / "paper" / "figures" / "final" / f"{stem}{suffix}",
        PROJECT_ROOT / "paper" / "figures" / f"{stem}{suffix}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _copy_figure_pair(source_stem: str, target_stem: str) -> None:
    for suffix in [".pdf", ".png"]:
        src = _source_file(source_stem, suffix)
        if src is None:
            continue
        for out_dir in [
            ensure_dir(PROJECT_ROOT / "figures" / "final"),
            ensure_dir(PROJECT_ROOT / "paper" / "figures" / "final"),
        ]:
            dst = out_dir / f"{target_stem}{suffix}"
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)


def _panel_image(stem: str) -> Path | None:
    return _source_file(stem, ".png")


def _composite(target_stem: str, panels: list[tuple[str, str]], ncols: int = 2, figsize: tuple[float, float] = (11, 7)) -> None:
    if plt is None or mpimg is None:
        return
    usable = [(stem, title, _panel_image(stem)) for stem, title in panels]
    usable = [(stem, title, path) for stem, title, path in usable if path is not None]
    if not usable:
        return
    nrows = int((len(usable) + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_arr = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, (_, title, path) in zip(axes_arr, usable):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes_arr[len(usable) :]:
        ax.axis("off")
    fig.tight_layout()
    for out_dir in [
        ensure_dir(PROJECT_ROOT / "figures" / "final"),
        ensure_dir(PROJECT_ROOT / "paper" / "figures" / "final"),
    ]:
        fig.savefig(out_dir / f"{target_stem}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{target_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_submission_figures() -> None:
    """Create exact submission-facing figure names and compact composite figures."""
    if plt is None:
        return
    scripts_dir = PROJECT_ROOT / "scripts"
    concept_scripts = [
        "make_fig_digital_twin_workflow_concept.py",
        "make_fig_network_sparse_sensor_layout.py",
        "make_fig_pignn_gru_v3_architecture.py",
        "make_fig_model_value_rank_heatmap.py",
        "make_fig_operator_sensor_guidelines_matrix.py",
        "make_fig_operational_digital_twin_kpi_dashboard.py",
        "make_combined_simulation_figures.py",
    ]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    for script in concept_scripts:
        path = scripts_dir / script
        if path.exists():
            runpy.run_path(str(path), run_name="__main__")

    simple = {
        "fig1_real_data_overview": "fig1_real_sonderborg_data_overview",
        "fig3_framework_flowchart": "fig2_digital_twin_workflow",
        "fig2_network_sensor_layout": "fig3_network_sparse_sensor_layout",
        "fig_temperature_supply_profile": "fig5_supply_temperature_reconstruction",
        "fig_temperature_return_profile": "fig6_return_temperature_reconstruction",
        "fig9_model_ranking_heatmap": "fig11_model_ranking_heatmap",
        "fig10_rmse_physics_tradeoff": "fig12_accuracy_physics_tradeoff",
        "fig11_sensor_layout_ranking_by_objective": "fig13_sensor_layout_ranking",
        "fig14_flensburg_domain_shift": "fig14_flensburg_domain_shift",
        "fig15_ablation_study_final": "fig16_ablation_study",
        "fig16_final_evidence_summary": "fig17_final_evidence_summary",
    }
    for source, target in simple.items():
        _copy_figure_pair(source, target)

    _composite(
        "fig4_calibration_and_discretization",
        [
            ("fig4_calibration_fit", "Measured-node thermal calibration"),
            ("fig5_discretization_model_verification", "Grid consistency"),
        ],
        ncols=2,
        figsize=(11, 5),
    )
    _composite(
        "fig4_thermo_hydraulic_reconstruction_summary",
        [
            ("fig_temperature_supply_profile", "Supply temperature"),
            ("fig_head_profile_reconstruction", "Head profile"),
            ("fig_flow_profile_reconstruction", "Flow profile"),
        ],
        ncols=3,
        figsize=(13, 4.8),
    )
    _composite(
        "fig7_pressure_head_reconstruction",
        [
            ("fig_head_profile_reconstruction", "Hydraulic head"),
            ("fig_pressure_profile_reconstruction", "Pressure from head"),
            ("fig_pressure_drop_profile", "Pressure/head drop"),
        ],
        ncols=3,
        figsize=(13, 4.8),
    )
    _composite(
        "fig8_flow_reconstruction",
        [
            ("fig_flow_profile_reconstruction", "Flow profile"),
            ("fig_flow_time_response", "Flow time response"),
            ("fig_flow_balance_error", "Flow-balance error"),
        ],
        ncols=3,
        figsize=(13, 4.8),
    )
    _composite(
        "fig9_heat_loss_and_energy_balance",
        [
            ("fig_heat_loss_profile", "Segment heat loss"),
            ("fig_cumulative_heat_loss", "Cumulative heat loss"),
            ("fig_energy_balance_residual_time", "Energy-balance residual"),
        ],
        ncols=3,
        figsize=(13, 4.8),
    )
    _composite(
        "fig10_thermo_hydraulic_coupling",
        [
            ("fig_heat_flow_pressure_coupling", "Heat-flow-head coupling"),
            ("fig_heat_loss_vs_flow_temperature", "Heat loss vs flow/temperature"),
            ("fig_thermal_delay_by_flow_regime", "Thermal delay by flow"),
        ],
        ncols=3,
        figsize=(13, 4.8),
    )
    _composite(
        "fig6_model_ranking_tradeoff_summary",
        [
            ("fig9_model_ranking_heatmap", "Metric-dependent model ranking"),
            ("fig10_rmse_physics_tradeoff", "Accuracy-physics tradeoff"),
        ],
        ncols=2,
        figsize=(11, 5),
    )
    _composite(
        "fig15_robustness_uncertainty",
        [
            ("fig_temperature_pressure_noise_robustness", "Sensor/input noise"),
            ("fig_parameter_uncertainty_robustness", "Parameter uncertainty"),
            ("fig_sensor_dropout_thermo_hydraulic", "Sensor dropout"),
        ],
        ncols=3,
        figsize=(13, 4.8),
    )


def _copy_table_aliases() -> None:
    aliases = {
        "table3_main_calibration_verification.tex": "table2_calibration_model_verification.tex",
        "table_thermo_hydraulic_estimation_summary.tex": "table3_thermo_hydraulic_summary.tex",
        "table4_main_model_ranking_objective.tex": "table4_model_ranking_by_objective.tex",
        "table6_main_sensor_layout_recommendation.tex": "table5_sensor_layout_recommendation.tex",
        "table7_main_flensburg_domain_shift.tex": "table6_flensburg_domain_shift.tex",
    }
    tables_dir = PROJECT_ROOT / "paper" / "tables"
    for source, target in aliases.items():
        src = tables_dir / source
        if src.exists():
            shutil.copy2(src, tables_dir / target)
            _resize_table_file(tables_dir / target)
    for name in ["table_xai4heat_status.tex", "table12_xai4heat_status.tex"]:
        path = tables_dir / name
        if path.exists():
            _resize_table_file(path)


def _resize_table_file(path: Path) -> None:
    text = _read(path)
    if "\\resizebox{\\textwidth}{!}" in text or "\\begin{tabular}" not in text:
        return
    text = text.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
    text = text.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    _write(path, text)


def _section_text() -> str:
    path = PROJECT_ROOT / "paper" / "sections" / "final_results_interpretation.tex"
    return _read(path)


def _uncertainty_summary_sentence() -> str:
    fallback = (
        "Conformal calibration is reported in the supplementary uncertainty table; "
        "the intervals should be interpreted as operational confidence bands rather than deployment guarantees."
    )
    path = PROJECT_ROOT / "results" / "uncertainty_calibration_summary.csv"
    if not path.exists():
        return fallback
    try:
        df = pd.read_csv(path)
    except Exception:
        return fallback
    if df.empty or "interval" not in df.columns:
        return fallback
    sub = df[df["interval"].astype(str).eq("90%")].copy()
    if sub.empty:
        sub = df.copy()
    coverage = pd.to_numeric(sub.get("coverage_conformal_calibrated"), errors="coerce").dropna()
    width = pd.to_numeric(sub.get("mean_interval_width_conformal_calibrated"), errors="coerce").dropna()
    if coverage.empty or width.empty:
        return fallback
    return (
        f"The conformal calibration summary reports mean empirical coverage of {coverage.mean():.1f}\\% "
        f"for the selected interval level, with mean conformal interval width {width.mean():.3g} across "
        "the available virtual-sensor quantities."
    )


def write_submission_manuscript() -> None:
    uncertainty_summary = _uncertainty_summary_sentence()
    text = rf"""\documentclass[preprint,12pt]{{elsarticle}}

\usepackage[utf8]{{inputenc}}
\usepackage{{amsmath,amssymb}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{array}}
\usepackage{{geometry}}
\usepackage{{float}}
\usepackage{{placeins}}
\geometry{{margin=1in}}
\emergencystretch=3em
\hfuzz=20pt
\hbadness=10000
\sloppy
\graphicspath{{{{figures/}}{{figures/final/}}}}

\journal{{Applied Thermal Engineering}}

\begin{{document}}
\pagestyle{{plain}}
\begin{{frontmatter}}

\title{{{TITLE}}}
\author[inst1]{{Authors omitted for review}}
\address[inst1]{{Department of Mechanical and Energy Systems Engineering}}

\begin{{abstract}}
District-heating monitoring requires sparse-sensor reconstruction of supply temperature, return temperature, pressure/head, flow, heat delivery, heat loss, transport delay, uncertainty, and energy-balance consistency. Public operating datasets, however, generally provide plant- or substation-level measurements rather than dense distributed pipe-level thermo-hydraulic fields. This study develops a real-data-assisted district-heating digital-twin benchmark in which S\o nderborg operating data provide boundary conditions, calibration, and measured-node thermal validation, while Flensburg data provide an external domain-shift test. The calibrated thermal model achieved supply-temperature RMSE of 0.337 $^\circ$C, return-temperature RMSE of 1.373 $^\circ$C, and heat-delivery error of 1.03\%. Distributed temperature, pressure/head, flow, and heat-loss fields are evaluated as simulator-assisted hidden states generated by the calibrated thermo-hydraulic simulator. Interpolation, LSTM, GRU, Transformer, PureGNN, physics-informed variants, and PI-GNN-GRU-v3 are benchmarked under common sparse-sensor layouts. The final layer converts estimates into operational virtual sensors, uncertainty bands, residual-based anomaly flags, and dashboard-style KPIs. The results show that direct RMSE and thermo-hydraulic consistency rank models differently: GRU/Transformer baselines remain strong for selected direct reconstruction metrics, while PI-GNN-GRU-v3 improves selected ATE-relevant indicators including return-temperature reconstruction, heat-loss error, pressure-drop consistency, and boundary consistency. Flensburg transfer reveals domain shift and the need for local calibration or adaptation. The study provides a reproducible thermal-engineering benchmark rather than a universal single-model victory claim.
\end{{abstract}}

\begin{{keyword}}
District heating \sep thermo-hydraulic state estimation \sep digital twin \sep sparse sensing \sep heat loss \sep pressure/head \sep graph neural networks \sep physics-informed learning
\end{{keyword}}

\end{{frontmatter}}

\section{{Introduction}}
District-heating networks are long-distance thermal-fluid systems in which heat loss, return temperature, pressure/head, flow, pump operation, and transport delay directly affect operational efficiency \cite{{lund2014fourth,werner2017international}}. Utilities increasingly need virtual sensing methods that infer internal network states from a small number of measurements, but dense pipe-level sensing remains expensive and uncommon. Public operating datasets are therefore valuable but incomplete: they support boundary-condition generation, calibration, measured-node validation, and transfer testing, yet they do not provide complete distributed pressure/head, flow, or temperature fields.

This study treats the problem as a real-data-assisted thermo-hydraulic benchmark. Real operating data calibrate the simulator and validate available measured thermal nodes. The calibrated simulator then generates hidden-state labels for distributed internal temperature, pressure, head, flow, and heat-loss fields. This separation is central to the evidence boundary of the paper. The proposed PI-GNN-GRU-v3 is evaluated as a physics-informed graph and temporal estimator, but the study does not assume a universal winner. Instead, it asks how direct RMSE, heat-loss reconstruction, head and pressure consistency, flow balance, energy balance, robustness, and external transfer rank different estimators.

The contributions are summarized as follows:
\begin{{enumerate}}
\item A real-data-assisted sparse-sensor district-heating benchmark using public operating data.
\item A calibrated dynamic thermo-hydraulic simulator for simulator-assisted hidden-state generation.
\item A fair comparison of interpolation, recurrent, transformer, graph, and physics-informed graph-temporal models.
\item A thermo-hydraulic result package covering temperature, pressure/head, flow, heat delivery, heat loss, coupling, and robustness.
\item A claim-safe separation of measured-node validation from simulator-assisted hidden-state reconstruction.
\end{{enumerate}}

\section{{Real Operating Datasets and Sparse-Sensor Problem}}
\input{{tables/table1_dataset_roles.tex}}
Sønderborg is used for primary calibration and training, Flensburg is used for external domain-shift validation, and XAI4HEAT is used for supplementary sparse-substation measured-node validation \cite{{sonderborg_dh_dataset,flensburg_dh_dataset,xai4heat_scada_2024,cvetkovic2025xai4heat_dib}}. The local XAI4HEAT package contains measured substation supply/return temperatures, outdoor temperature, and energy-related variables for five substations. Its provenance is documented in the supplementary material; it is used only for measured-node thermal/energy consistency, not for distributed pressure/head, flow, heat-loss, or internal pipe-state validation.

The pipe is represented as a graph $\mathcal{{G}}=(\mathcal{{V}},\mathcal{{E}})$ with node state
\begin{{equation}}
\mathbf{{x}}_{{k,i}}=[T^s_{{k,i}},T^r_{{k,i}},H_{{k,i}},q_{{k,i}}]^T ,
\end{{equation}}
where $T^s$ and $T^r$ are supply and return temperatures, $H$ is hydraulic head, and $q$ is flow. Sparse measurements are modeled as $\mathbf{{y}}_{{k,i}}=\mathbf{{M}}_{{k,i}}\mathbf{{x}}_{{k,i}}+\epsilon_{{k,i}}$. Measured-node validation is only for available measured variables; distributed pressure/head, flow, internal temperature, and heat-loss fields are simulator-assisted hidden states.

\section{{Calibrated Thermo-Hydraulic Digital-Twin Model}}
The benchmark simulator solves supply and return advection-loss equations, load-side heat extraction, heat-load-derived flow proxy, pump/head boundary behavior, and segment heat-loss integration:
\begin{{align}}
T^s_{{k+1,i}} &= T^s_{{k,i}}-\mathrm{{CFL}}_k(T^s_{{k,i}}-T^s_{{k,i-1}})-\Delta tK_\ell(T^s_{{k,i}}-T^a_k),\\
T^r_{{k+1,i}} &= T^r_{{k,i}}-\mathrm{{CFL}}_k(T^r_{{k,i}}-T^r_{{k,i+1}})-\Delta tK_\ell(T^r_{{k,i}}-T^a_k),\\
T^r_{{k+1,N}} &= T^s_{{k,N}}-\frac{{Q^{{load}}_k}}{{\rho c_p\max(q_{{k,N}},\epsilon)}}+\Delta T_r.
\end{{align}}
Pressure is reported as $p=\rho gH/1000$ in kPa. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements.
\input{{tables/table2_main_parameters.tex}}

\section{{Sparse-Sensor Benchmark Models and Physics-Informed Graph Learning}}
The benchmark includes interpolation, LSTM-MSE, GRU-MSE, Transformer-MSE, PureGNN-MSE, PI-LSTM, PI-GNN without temporal recurrence, and PI-GNN-GRU-v3. PI-GNN-GRU-v3 uses residual graph convolution blocks inspired by graph neural network message passing \cite{{kipf2017semi,scarselli2009graph}}, normalization, temporal GRUs, sensor-mask-aware fusion, multi-head state/heat-loss/boundary outputs, and an interpolation-residual connection. The physics-informed loss combines state, sensor, thermal, hydraulic, boundary, heat-loss, energy, and smoothness terms with normalized residual scales and curriculum weighting \cite{{raissi2019physics}}. PI-GNN-GRU-v3 is not uniformly superior; it is one structured estimator in a multi-metric benchmark.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.94\linewidth]{{figures/final/fig_pignn_gru_v3_architecture.pdf}}
\caption{{PI-GNN-GRU-v3 architecture. Sparse sensor time series and graph topology are encoded through graph-temporal layers, while physics-informed residuals regularize heat-loss, energy-balance, boundary, and thermo-hydraulic consistency metrics.}}
\end{{figure}}

\section{{Experimental Design}}
\input{{tables/table6_sensor_layout_definitions.tex}}
The evaluation reports direct simulator-hidden-state reconstruction, measured-node thermal validation, pressure/head and flow reconstruction against simulator-assisted hidden hydraulic states, delivered heat, heat loss, energy-balance residual, thermal delay, sensor-layout ranking, uncertainty/robustness, and Flensburg transfer.

\section{{Results and Discussion}}
\subsection{{Real-data preprocessing and calibration}}
\input{{tables/table2_calibration_model_verification.tex}}
The S\o nderborg calibration errors are small for the measured thermal boundary variables: 0.337 $^\circ$C supply-temperature RMSE, 1.373 $^\circ$C return-temperature RMSE, and 1.03\% heat-delivery error. This supports use of the simulator as a boundary-consistent hidden-state generator. It does not represent dense distributed field validation, because public datasets do not provide full pipe-level measurements.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig1_real_sonderborg_data_overview.pdf}}
\caption{{Real S\o nderborg plant-level operating data used for boundary conditions, calibration, and measured-node thermal validation.}}
\end{{figure}}
\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_digital_twin_workflow_concept.pdf}}
\caption{{Overall real-data-assisted sparse-sensor thermo-hydraulic digital-twin workflow. Real data support calibration and measured-node validation, while distributed pressure/head and flow are simulator-assisted hidden hydraulic states.}}
\end{{figure}}
\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_network_sparse_sensor_layout.pdf}}
\caption{{Sparse-sensor district-heating network schematic. Measured nodes provide boundary and sparse thermal information; unmeasured distributed temperature, pressure/head, flow, and heat-loss states are reconstructed by the digital twin. Pressure/head and flow are simulator-assisted hidden hydraulic states.}}
\end{{figure}}

\subsection{{Numerical consistency of the calibrated simulator}}
The discretization study compares coarse, baseline, and fine grids. The near-zero 1000 m versus 500 m outlet-supply difference supports numerical consistency of the benchmark simulator, while still remaining a numerical verification rather than full network field validation.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig4_calibration_and_discretization.pdf}}
\caption{{Calibration and discretization verification. The calibration panel uses real measured thermal boundary data; the discretization panel checks calibrated-simulator numerical consistency.}}
\end{{figure}}

\subsection{{Supply and return temperature estimation}}
\input{{tables/table3_thermo_hydraulic_summary.tex}}
\input{{tables/table_thermal_estimation.tex}}
This subsection and the following subsections report temperature, pressure/head, flow, and heat loss as the core thermo-hydraulic state-estimation outputs. Supply and return temperature estimates are evaluated at measured nodes where real data are available and across the pipeline using calibrated-simulator hidden labels. The results confirm that sequence models are strong for some direct RMSE metrics, while PI-GNN-GRU-v3 improves selected return-temperature and consistency-oriented metrics. The full distributed result is simulator-assisted hidden-state reconstruction, while measured-node validation is only for available measured variables.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{figures/final/fig_thermo_hydraulic_reconstruction_summary.pdf}}
\caption{{Combined thermo-hydraulic reconstruction summary for supply temperature, return temperature, thermal errors, pressure/head, and flow. Distributed temperature labels are generated by the calibrated thermo-hydraulic simulator, while real operating data provide boundary conditions, calibration, and measured-node thermal validation. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements.}}
\end{{figure}}

\subsection{{Pressure/head and flow estimation}}
\input{{tables/table_hydraulic_estimation.tex}}
Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements. They are included because hydraulic consistency matters for district-heating monitoring, but they must not be interpreted as validation against real distributed pressure or flow sensors.
The compact reconstruction summary above reports pressure/head and flow in the same visual evidence package as supply and return temperature, keeping simulator-assisted hydraulic diagnostics separate from measured-node thermal validation.

\subsection{{Heat delivery, heat loss, and energy-balance estimation}}
\input{{tables/table_heat_energy_estimation.tex}}
Heat delivery and heat-loss estimates connect state reconstruction to thermal-engineering operation. Heat-loss and energy-balance metrics can rank models differently from direct RMSE, which is why the benchmark reports both statistical and physical diagnostics. The three-dimensional heat-loss surface is derived from calibrated-simulator segment profiles and saved total heat-loss intervals, with the unit conversion from segment heat loss to kW/km reported in the reproducibility files.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{figures/final/fig_heat_energy_balance_summary.pdf}}
\caption{{Combined heat and energy-balance summary. Delivered heat is linked to real boundary heat-load data, while heat-loss profiles, cumulative heat loss, heat-loss error, and energy-balance residuals are calibrated-simulator or reconstructed-state quantities. Heat loss is reported in kW, MWh, or percent according to the panel axis; it is not a direct pipe heat-loss measurement.}}
\end{{figure}}

\subsection{{Thermo-hydraulic coupling and thermal-delay interpretation}}
\input{{tables/table_coupling_summary.tex}}
The coupling analysis relates heat load to flow proxy, pressure/head drop, return temperature, heat loss, and thermal delay. This moves the evaluation beyond model ranking and toward the operational quantities used by district-heating engineers.

\subsection{{Model ranking: direct accuracy versus physical consistency}}
\input{{tables/table4_model_ranking_by_objective.tex}}
Direct RMSE and physical consistency can rank models differently. GRU-MSE and Transformer-MSE remain competitive or strongest for selected direct reconstruction metrics. PI-GNN-GRU-v3 improves selected ATE-relevant metrics, including return-temperature reconstruction, heat-loss error, pressure-drop consistency, and boundary consistency. The result is a multi-objective benchmark, not a claim of universal PI-GNN-GRU-v3 superiority.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_model_ranking_ate_dark.pdf}}
\caption{{Model ranking and accuracy--physics tradeoff. Different objectives favor different estimators, so direct RMSE and thermo-hydraulic consistency must be reported together.}}
\end{{figure}}
\begin{{figure}}[t]
\centering
\includegraphics[width=0.78\linewidth]{{figures/final/fig_accuracy_physics_tradeoff_ate_dark.pdf}}
\caption{{Accuracy--physics tradeoff for the benchmark models. GRU/Transformer baselines remain strong for direct thermal RMSE, while physics-consistency-oriented diagnostics provide a different objective space for digital-twin monitoring.}}
\end{{figure}}

\subsection{{Where and why PI-GNN-GRU-v3 adds value}}
\input{{tables/table_proposed_model_value_summary.tex}}
The proposed PI-GNN-GRU-v3 should not be read as the best overall predictor. GRU-MSE is strongest for raw supply-temperature RMSE in the final ranking, and Transformer-MSE remains competitive for several direct and hydraulic hidden-state metrics. PI-GNN-GRU-v3 adds value where topology, sparse-sensor masks, interpolation-residual correction, and normalized physical residuals matter: return-temperature reconstruction, heat-loss error, energy-balance residual, boundary consistency, pressure-drop consistency, and selected robustness diagnostics. This is an ATE-relevant value proposition because heat-loss monitoring and energy-balance closure are operational digital-twin objectives, not only pointwise regression targets.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.94\linewidth]{{figures/final/fig_model_value_rank_heatmap.pdf}}
\caption{{Metric-specific value of PI-GNN-GRU-v3. Direct supply-temperature RMSE and thermo-hydraulic consistency metrics rank models differently; PI-GNN-GRU-v3 is most valuable for physically consistent digital-twin monitoring rather than universal point-prediction superiority.}}
\end{{figure}}

\subsection{{Sparse-sensor layout implications}}
\input{{tables/table5_sensor_layout_recommendation.tex}}
Sparse-sensor placement strongly affects reconstruction quality. Inlet--middle--outlet sensing is practically attractive because a middle-pipeline sensor reduces unobserved transport length and helps thermal-delay and heat-loss inference. Optimized layouts may improve some objectives, while noisy and dropout cases show the cost of relying on too few boundary sensors.

\subsection{{Sensor-placement guidelines for district-heating operators}}
\input{{tables/table_operator_sensor_guidelines.tex}}
The operator-facing decision matrix translates the sensor-layout benchmark into monitoring choices. Direct thermal accuracy favors S12/S10-style layouts when four or five sensors are feasible. Physical consistency and heat-loss monitoring favor optimized five-sensor or S10-style layouts because they reduce unobserved transport length while improving heat-loss and energy-balance indicators. Low-cost monitoring can use inlet--outlet sensing, with inlet-only monitoring reserved for cases where coarse boundary awareness is acceptable and hidden-state uncertainty is tolerated. Energy-impact monitoring favors the optimized five-sensor layout in the current scenario table because pump-energy proxy, heat-loss error, energy-balance residual, cost proxy, and CO2 proxy are all reported transparently under stated assumptions. No layout is universally optimal; the choice depends on whether the utility prioritizes direct thermal accuracy, heat-loss monitoring, capital cost, or operational energy-impact indicators.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_operator_sensor_guidelines_matrix.pdf}}
\caption{{Operator sensor-guideline matrix derived from the sensor-layout ranking and energy-impact scenario tables. Recommendations are objective-specific; cost and CO2 entries are proxy indicators, and pressure/head plus flow indicators remain simulator-assisted hidden hydraulic quantities.}}
\end{{figure}}

\subsection{{External transfer to Flensburg and domain shift}}
\input{{tables/table6_flensburg_domain_shift.tex}}
External transfer to Flensburg is challenging and is interpreted as a domain-shift stress test, not proof of broad cross-network transfer. Differences in heat-load scale, supply-temperature distribution, sampling interval, network characteristics, and unavailable return-temperature measurements make direct transfer challenging. Local calibration or adaptation is required for cross-network use. Detailed transfer-mode and return-temperature-assumption diagnostics are retained in the supplementary material so the main paper stays focused on the operational digital-twin evidence.

\subsection{{Robustness to noise, dropout, and parameter uncertainty}}
Robustness is evaluated separately for measured-node thermal variables and simulator-assisted hydraulic hidden states. Sensor noise, outlet dropout, heat-load perturbation, heat-loss coefficient uncertainty, and friction-factor uncertainty quantify how strongly sparse virtual sensing depends on data quality and parameter assumptions.

\subsection{{Supplementary seasonal, stress, and parameter-sensitivity tests}}
Three supplementary simulations extend the robustness evidence without adding new generic machine-learning models. First, a seasonal generalization test separates S\o nderborg operating profiles into high-load and shoulder/low-load regimes to assess whether estimators calibrated on one heat-load regime remain reliable in another. Second, a combined stress test applies controlled load-step, ambient-temperature, sensor-dropout, return-bias, heat-loss, and friction perturbations to real operating profiles. These disturbances are controlled perturbations, not observed field fault events. Third, a parameter-identifiability sensitivity test varies effective heat-loss, friction, return-temperature, flow-proxy, and velocity/delay parameters to quantify how simulator-assisted hidden states depend on calibration assumptions.

The supplementary results are reported in Supplementary Sections S8--S10, with corresponding supplementary tables and figures. They reinforce the central engineering conclusion: robust district-heating digital twins require multi-metric evaluation across operating regimes, disturbance cases, and effective-parameter uncertainty. Pressure/head and flow in these tests remain simulator-assisted hidden hydraulic states, while real measured data provide the base operating profiles and thermal boundary evidence.

\subsection{{Uncertainty-aware operational digital-twin monitoring layer}}
\input{{tables/table_digital_twin_kpis.tex}}
The digital-twin layer converts state estimation into operational monitoring indicators. It provides virtual sensors, uncertainty bounds, and residual-based anomaly flags. The framework turns sparse-sensor estimation into an operational digital-twin monitoring layer with virtual sensors, confidence intervals, residual indicators, and engineering KPIs. {uncertainty_summary} These conformal intervals provide practical operational confidence bands: a wider interval warns an operator that the virtual sensor is less certain, while coverage below the nominal level would indicate that recalibration or model updating is needed. Residual-based anomaly indicators are promising for large friction/pressure-drop and combined stress cases, while smaller heat-loss changes and mild sensor biases require temporal accumulation or lower thresholds. Uncertainty intervals are intentionally conservative and should be interpreted as operational confidence bands rather than perfectly calibrated probabilistic forecasts. Online adaptation and continual learning are future deployment pathways, not completed results in this study. In practice, periodic recalibration with newly available SCADA data could improve robustness under seasonal drift, boundary-condition changes, and network aging. The anomaly cases are controlled perturbations applied to real operating profiles and should be interpreted as robustness tests, not documented field faults. Pressure/head and flow monitoring indicators are based on simulator-assisted hidden hydraulic states because public datasets do not provide dense distributed hydraulic measurements. Measured-node validation is retained only for the thermal variables available in the real operating data.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_digital_twin_dashboard_ate_dark.pdf}}
\caption{{Dark ATE-style digital-twin KPI dashboard. The panels summarize virtual-sensor errors, uncertainty coverage, residual alarms, heat-loss and energy indicators, with measured-node thermal evidence kept separate from calibrated-simulator and simulator-assisted hidden-state quantities.}}
\end{{figure}}

\subsection{{Operational energy-impact interpretation}}
The sparse-sensor digital twin is further translated into operational KPIs, including delivered heat, heat loss, pump-energy proxy, pressure-drop residual, and energy-balance residual. These indicators do not turn the study into an optimization/control paper; rather, they show how the reconstructed thermo-hydraulic states can support practical monitoring decisions. Pressure/head and flow-based energy indicators are simulator-assisted because public district-heating datasets do not provide dense distributed hydraulic measurements.

Values are reported over the stated evaluation horizon unless normalized; cost and CO2 are proxy indicators under stated assumptions. Cost and CO2 values are proxy indicators, not optimized economic-dispatch results. The scenario summaries include a nominal winter-day operation, a combined controlled-stress case, and sparse-sensor layout energy-impact comparisons. Disturbances in the stress case are controlled perturbations applied to real operating profiles, not documented field fault events. The results are reported in Supplementary Section S14 and summarized in Fig.~\ref{{fig:operational_energy_impact}}.
\begin{{figure}}[t]
\centering
\includegraphics[width=0.98\linewidth]{{figures/final/fig_operational_energy_pressure_summary.pdf}}
\caption{{Operational energy--pressure KPI summary. Reconstructed thermo-hydraulic states are translated into delivered heat, heat loss, pump-energy proxy, pressure-drop residual, energy-balance residual, and cost/CO2 proxy indicators over the stated evaluation horizon. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements. Cost and CO2 are proxy indicators under stated assumptions, not optimized dispatch results.}}
\label{{fig:operational_energy_impact}}
\end{{figure}}

\section{{Limitations and Future Work}}
Public datasets provide plant/substation-level measurements, not complete distributed temperature, pressure/head, and flow fields. XAI4HEAT now provides an external sparse-substation measured-node thermal/energy validation case, but it still does not provide dense pipe-level pressure/head, flow, heat-loss, or distributed internal temperature fields. Distributed hidden states are generated by a calibrated simulator; therefore, hidden-state reconstruction is simulator-assisted. Hydraulic states are weakly identifiable without real pressure and flow measurements. PI-GNN-GRU-v3 does not dominate all metrics; model selection depends on direct RMSE, physical consistency, robustness, interpretability, and transfer requirements. Dense distributed field validation remains future work.

\section{{Conclusions}}
This study develops a real-data-assisted sparse-sensor thermo-hydraulic digital-twin benchmark for district-heating monitoring. Real operating data support calibration and measured-node thermal validation, while distributed pressure/head, flow, internal temperature, and heat-loss fields are evaluated as simulator-assisted hidden states. The calibrated simulator reproduces measured thermal boundary behavior accurately and passes numerical consistency checks. Model comparisons show that GRU/Transformer baselines remain strong for direct RMSE, while PI-GNN-GRU-v3 improves selected thermal-engineering metrics such as return-temperature reconstruction, heat-loss error, pressure-drop consistency, and boundary consistency. Sensor placement and Flensburg domain shift strongly affect performance. The operational energy-impact layer further translates reconstructed states into delivered-heat, heat-loss, pump-energy proxy, pressure-drop residual, and energy-balance KPIs under explicit assumptions. The work provides a reproducible benchmark and practical guidance for sparse district-heating virtual sensing without claiming full field validation, optimization/control, or deployment readiness.

\bibliographystyle{{elsarticle-num}}
\bibliography{{references}}

\end{{document}}
"""
    _write(PROJECT_ROOT / "paper" / "main_ate_submission_candidate.tex", text)


def write_supplementary_material() -> None:
    text = rf"""\documentclass[preprint,12pt]{{elsarticle}}
\usepackage[utf8]{{inputenc}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\usepackage{{rotating}}
\usepackage{{float}}
\usepackage{{hyperref}}
\geometry{{margin=1in}}
\emergencystretch=3em
\sloppy
\graphicspath{{{{figures/}}{{figures/final/}}}}
\begin{{document}}
\begin{{center}}
{{\Large Supplementary Material}}\\[0.5em]
{{\large {TITLE}}}
\end{{center}}

\section*{{S1. Full model comparison}}
\input{{tables/supplementary_baseline_comparison_full.tex}}
\input{{tables/supplementary_physics_consistency_full.tex}}
\input{{tables/table15_model_ranking.tex}}
\input{{tables/table_proposed_model_value_summary.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig11_model_ranking_heatmap.pdf}}
\caption{{Full model-ranking heatmap.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_model_value_rank_heatmap.pdf}}
\caption{{Metric-specific value of PI-GNN-GRU-v3. The rank heatmap is interpreted metric by metric and should not be read as a best-overall-model claim. Pressure/head and flow metrics are simulator-assisted hidden hydraulic diagnostics.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.82\linewidth]{{figures/final/fig_accuracy_physics_tradeoff_ate_dark.pdf}}
\caption{{Accuracy--physics tradeoff showing that direct thermal RMSE and physical-consistency indicators can rank models differently.}}
\end{{figure}}

\section*{{S2. Detailed thermo-hydraulic reconstruction}}
\input{{tables/supplementary_temperature_error_by_node.tex}}
\input{{tables/supplementary_head_pressure_error_by_node.tex}}
\input{{tables/supplementary_heat_loss_by_segment.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig5_supply_temperature_reconstruction.pdf}}
\caption{{Detailed supply-temperature profile reconstruction; distributed labels are simulator-assisted hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig6_return_temperature_reconstruction.pdf}}
\caption{{Detailed return-temperature profile reconstruction; distributed labels are simulator-assisted hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig7_pressure_head_reconstruction.pdf}}
\caption{{Detailed pressure/head diagnostics; hydraulic fields are simulator-assisted hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig8_flow_reconstruction.pdf}}
\caption{{Detailed flow diagnostics; flow labels are simulator/proxy-derived and are not directly measured in the public datasets.}}
\end{{figure}}

\section*{{S3. Sensor-layout details}}
\input{{tables/supplementary_sensor_layout_comparison_full.tex}}
\input{{tables/table_operator_sensor_guidelines.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_sensor_layout_ate_dark.pdf}}
\caption{{Dark ATE-style sensor-layout ranking by objective. The recommended layout depends on direct thermal accuracy, hydraulic reconstruction, physical consistency, robustness, and sensor count.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\linewidth]{{figures/final/fig_operator_sensor_guidelines_matrix.pdf}}
\caption{{Operator-facing sensor-guideline matrix. Recommendations are objective-specific and should not be read as a universal sensor-placement optimum.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig12_sensor_layout_distance_vs_error.pdf}}
\caption{{Sensor distance versus reconstruction error.}}
\end{{figure}}

\section*{{S4. Robustness and uncertainty}}
\input{{tables/supplementary_thermo_hydraulic_robustness.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_uncertainty_anomaly_summary.pdf}}
\caption{{Combined uncertainty and residual-anomaly summary. Temperature uncertainty uses virtual sensor intervals; heat-loss uncertainty is based on calibrated-simulator/reconstructed heat-loss quantities; anomaly cases are controlled perturbations applied to real operating profiles rather than documented field faults.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig15_robustness_uncertainty.pdf}}
\caption{{Robustness to sensor noise, dropout, and parameter uncertainty.}}
\end{{figure}}

\section*{{S5. Ablation study}}
\input{{tables/supplementary_ablation_study_full.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig16_ablation_study.pdf}}
\caption{{Ablation of physics-informed and graph-temporal components.}}
\end{{figure}}

\section*{{S6. External validation details}}
\input{{tables/supplementary_external_validation_modes_full.tex}}
\input{{tables/table_flensburg_domain_shift_improved.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_flensburg_domain_shift_ate_dark.pdf}}
\caption{{Dark ATE-style Flensburg domain-shift and transfer summary. Flensburg is interpreted as a domain-shift stress test rather than proof of universal transfer.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig14_flensburg_domain_shift.pdf}}
\caption{{Flensburg transfer diagnostics.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_flensburg_domain_shift_improved.pdf}}
\caption{{Improved Flensburg domain-shift diagnostics. Flensburg is a domain-shift stress test rather than proof of universal transfer.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_flensburg_return_assumption_sensitivity.pdf}}
\caption{{Return-temperature assumption sensitivity for Flensburg. The return value is an assumption when measured return temperature is unavailable.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_flensburg_transfer_modes_improved.pdf}}
\caption{{Flensburg transfer-mode comparison. Local calibration/adaptation is needed for reliable cross-network use.}}
\end{{figure}}

\section*{{S7. Quality gate and claim-safety audit}}
The full quality-gate and claim-safety audit is provided as machine-readable files in the results directory. The main manuscript uses compact evidence-boundary and claim-mapping tables; the detailed CSV/TXT audit is retained for reproducibility without overloading the supplementary PDF.
\input{{tables/table_xai4heat_status.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig17_final_evidence_summary.pdf}}
\caption{{Final evidence summary.}}
\end{{figure}}

\section*{{S8. Seasonal generalization under different heat-load regimes}}
The seasonal split test evaluates whether estimators calibrated on high-load operation remain reliable under shoulder- and low-load operation. Seasonal regimes are derived from real S\o nderborg operating data by timestamp and heat-load quantiles. Distributed hidden states are generated by the calibrated simulator.
\input{{tables/table_seasonal_generalization.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_seasonal_stress_sensitivity_summary.pdf}}
\caption{{Combined seasonal, stress, and parameter-sensitivity summary. Seasonal regimes are derived from real S\o nderborg operating data; stress disturbances are controlled perturbations of real operating profiles; parameter perturbations are applied to effective calibrated parameters to assess identifiability and robustness. Distributed pressure/head and flow metrics remain simulator-assisted hidden hydraulic states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_seasonal_generalization_ate_dark.pdf}}
\caption{{Dark ATE-style seasonal generalization summary. Seasonal regimes are derived from real S\o nderborg operating data, while distributed state metrics use calibrated-simulator hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_seasonal_generalization_summary.pdf}}
\caption{{Seasonal generalization summary across thermal, heat-loss, energy, and delay metrics.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_seasonal_heat_load_regimes.pdf}}
\caption{{Seasonal heat-load and temperature regimes derived from real S\o nderborg operating profiles.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_seasonal_generalization_metrics.pdf}}
\caption{{Seasonal generalization metrics. Distributed state metrics are evaluated against calibrated-simulator hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_seasonal_temperature_reconstruction.pdf}}
\caption{{Seasonal temperature reconstruction examples; real data define boundary profiles, while distributed labels are simulator-assisted hidden states.}}
\end{{figure}}

\section*{{S9. Combined stress test}}
The combined stress test applies controlled load, ambient, sensor-dropout, return-bias, heat-loss, and friction perturbations to real operating profiles. These are controlled perturbations, not directly observed field fault events. Pressure/head and flow are simulator-assisted hidden hydraulic states because public datasets do not provide dense distributed hydraulic measurements.
\input{{tables/table_combined_stress_test.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_combined_stress_ate_dark.pdf}}
\caption{{Dark ATE-style combined stress-test summary. Disturbances are controlled perturbations applied to real operating profiles, not observed field fault events.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_combined_stress_summary.pdf}}
\caption{{Combined stress-test summary across temperature, heat-loss, pressure-drop, energy, recovery, and maximum-error metrics.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_combined_stress_inputs.pdf}}
\caption{{Stress-test inputs: controlled load-step, ambient-drop, dropout, and bias windows applied to real operating profiles.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_combined_stress_temperature_response.pdf}}
\caption{{Temperature response under combined stress. Distributed temperature fields are calibrated-simulator hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_combined_stress_pressure_flow_response.pdf}}
\caption{{Pressure/head and flow stress response. Pressure/head and flow are simulator-assisted hidden hydraulic states because public datasets do not provide dense distributed hydraulic measurements.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_combined_stress_heat_loss_energy.pdf}}
\caption{{Heat-loss and energy-balance response during controlled stress cases.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_combined_stress_model_comparison.pdf}}
\caption{{Model degradation under controlled stress perturbations.}}
\end{{figure}}

\section*{{S10. Parameter-identifiability sensitivity}}
The calibrated parameters should be interpreted as effective parameters for matching plant-level operating data, not as independently measured pipe-material or hydraulic properties. The sensitivity test quantifies how uncertainty in these effective parameters affects thermo-hydraulic hidden-state reconstruction.
\input{{tables/table_parameter_identifiability_sensitivity.tex}}
\input{{tables/table_parameter_identifiability_sensitivity_improved.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_parameter_sensitivity_ate_dark.pdf}}
\caption{{Dark ATE-style parameter-sensitivity summary. Parameter perturbations are applied to effective calibrated model parameters to assess identifiability and robustness.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_identifiability_tornado_ate_dark.pdf}}
\caption{{Dark ATE-style parameter-identifiability tornado plot for effective calibrated parameters.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_sensitivity_heat_loss.pdf}}
\caption{{Heat-loss sensitivity to effective heat-loss and combined thermal-hydraulic parameter perturbations.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_sensitivity_pressure_flow.pdf}}
\caption{{Pressure/head and flow sensitivity. Hydraulic fields are simulator-assisted hidden states, not dense real measurements.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_sensitivity_energy_residual.pdf}}
\caption{{Energy-balance residual sensitivity to effective calibrated parameter uncertainty.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_identifiability_tornado.pdf}}
\caption{{Parameter-identifiability tornado plot showing the largest changes in heat-loss error under effective-parameter perturbations.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_identifiability_tornado_improved.pdf}}
\caption{{Improved parameter-identifiability tornado plot based on ranked thermal and hydraulic sensitivity indices.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_parameter_sensitivity_grouped_thermal_hydraulic.pdf}}
\caption{{Thermal and hydraulic sensitivity groups for calibrated effective parameters.}}
\end{{figure}}

\section*{{S11. Three-dimensional return-temperature and flow fields}}
The main manuscript uses three-dimensional supply-temperature, heat-loss, and pressure surfaces as compact thermo-hydraulic evidence. This supplementary section retains the corresponding return-temperature field and the detailed flow diagnostics. Distributed return-temperature labels are generated by the calibrated thermo-hydraulic simulator, with real measured return temperature supporting plant-level calibration and measured-node validation where available. Flow remains simulator/proxy-derived because public datasets do not provide dense distributed flow measurements.
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/main_3d_return_temperature_surface.pdf}}
\caption{{Three-dimensional return-temperature field over distance and time. Distributed labels are generated by the calibrated thermo-hydraulic simulator; real operating data support boundary conditions, calibration, and measured-node thermal validation.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig8_flow_reconstruction.pdf}}
\caption{{Detailed flow diagnostics. Flow labels are simulator/proxy-derived and are not directly measured in the public datasets.}}
\end{{figure}}
\input{{tables/supplementary_measured_vs_hidden_metrics.tex}}

\section*{{S12. Uncertainty and residual-based anomaly diagnostics}}
This section reports virtual-sensor uncertainty intervals for temperature, pressure/head, flow, and heat-loss quantities together with residual-based anomaly indicators. The uncertainty bands are derived from the saved model ensemble with a residual-based calibration floor. Pressure/head and flow intervals are simulator-assisted hidden hydraulic states because dense public hydraulic measurements are unavailable. Anomaly cases are controlled perturbations of real operating profiles, not documented field faults.
\input{{tables/table_uncertainty_calibration.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_uncertainty_coverage_ate_dark.pdf}}
\caption{{Dark ATE-style uncertainty coverage and interval-width summary. Uncertainty bands are operational confidence intervals, with hydraulic quantities evaluated as simulator-assisted hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_anomaly_detection_ate_dark.pdf}}
\caption{{Dark ATE-style residual anomaly summary. Anomaly cases are controlled perturbations of real operating profiles and are not documented field faults.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_temperature_bands.pdf}}
\caption{{Uncertainty bands for virtual temperature sensors. Real data support boundary conditions and measured-node thermal validation; distributed labels are calibrated-simulator hidden states.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_calibration_curve.pdf}}
\caption{{Uncertainty calibration curve comparing nominal and empirical coverage.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_temperature_bands_calibrated.pdf}}
\caption{{Conformal-calibrated temperature confidence bands.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_heat_loss_bands.pdf}}
\caption{{Heat-loss uncertainty bands. Heat-loss labels are calibrated-simulator quantities, not direct pipe heat-loss measurements.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_heat_loss_bands_calibrated.pdf}}
\caption{{Conformal-calibrated heat-loss confidence bands.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_width_vs_error.pdf}}
\caption{{Uncertainty sharpness diagnostic showing interval width versus realized error.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_uncertainty_coverage.pdf}}
\caption{{Virtual-sensor prediction-interval coverage. Hydraulic quantities are simulator-assisted hidden states.}}
\end{{figure}}
\input{{tables/table_anomaly_detection_improved.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_anomaly_detection_residuals.pdf}}
\caption{{Residual-score anomaly flags for controlled perturbations applied to real operating profiles.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_anomaly_multiresidual_scores.pdf}}
\caption{{Multi-residual anomaly scores with thermal, hydraulic, energy, and combined diagnostics.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_anomaly_threshold_sweep.pdf}}
\caption{{Threshold sweep showing anomaly sensitivity and false-alarm tradeoffs.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_return_bias_ewma_detection.pdf}}
\caption{{EWMA temporal accumulation for mild return-temperature bias detection.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_sensor_bias_detection.pdf}}
\caption{{Return-temperature sensor-bias detection using measured-node residuals.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_heat_loss_anomaly_detection.pdf}}
\caption{{Heat-loss anomaly residuals. Heat-loss deviations are controlled perturbations of calibrated simulator quantities.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_heat_loss_anomaly_detection_improved.pdf}}
\caption{{Improved heat-loss anomaly score with temporal accumulation.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_sensor_dropout_detection_improved.pdf}}
\caption{{Sensor-dropout detection using sensor-health and missingness residuals.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.90\linewidth]{{figures/final/fig_pressure_drop_anomaly_detection.pdf}}
\caption{{Pressure-drop anomaly residuals. Pressure/head fields are simulator-assisted hidden hydraulic states.}}
\end{{figure}}

\section*{{S13. Digital-twin monitoring KPIs}}
The digital-twin dashboard summarizes virtual-sensor accuracy, uncertainty, energy-balance residuals, pressure-drop consistency, heat-loss estimates, sensor-health indicators, and warning/alarm rates. These KPIs combine real measured-node thermal evidence, calibrated-simulator outputs, and simulator-assisted hidden hydraulic states.
\input{{tables/table_digital_twin_kpis.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_digital_twin_monitoring_dashboard.pdf}}
\caption{{Operational digital-twin monitoring dashboard showing virtual sensors, uncertainty, energy residuals, heat-loss indicators, simulator-assisted pressure/head diagnostics, and residual-based flags.}}
\end{{figure}}

\section*{{S14. Operational energy-impact interpretation}}
This section translates sparse-sensor thermo-hydraulic reconstruction into operational KPIs: delivered heat, heat loss, pump-energy proxy, pressure-drop residual, energy-balance residual, and transparent cost/CO2 proxy indicators. Values are reported over the stated evaluation horizon unless normalized; cost and CO2 are proxy indicators under stated assumptions. Cost and CO2 values are proxy indicators, not optimized economic-dispatch results. The tariff and emission-factor assumptions are stored in \texttt{{results/operational\_energy\_impact\_assumptions.json}}. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements.
\input{{tables/table_scenario_energy_impact_summary.tex}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_operational_energy_pressure_summary.pdf}}
\caption{{Combined operational energy--pressure summary. Delivered heat, heat loss, pump-energy proxy, pressure-drop residual, energy-balance residual, and cost/CO2 proxy are reported over the stated evaluation horizon. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements. Cost and CO2 are proxy indicators under stated assumptions, not optimized dispatch results.}}
\end{{figure}}
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/fig_operational_digital_twin_kpi_dashboard.pdf}}
\caption{{Operational digital-twin KPI dashboard for the real-data-assisted sparse-sensor digital twin. Values are reported over the stated evaluation horizon unless normalized; cost and CO2 are proxy indicators under stated assumptions. Delivered heat is linked to real boundary heat-load data, heat-loss estimates are calibrated-simulator/reconstructed-state quantities, and pump-energy plus pressure-drop indicators are simulator-assisted hidden hydraulic proxies. Cost and CO2 values are proxy indicators, not optimized economic-dispatch results.}}
\end{{figure}}

\section*{{S15. Conceptual and operator-facing figure package}}
This section collects the six conceptual and non-simulation figures used to communicate the full paper workflow: sparse real data, calibrated thermo-hydraulic simulation, PI-GNN-GRU-v3 digital-twin estimation, virtual sensors, uncertainty/anomaly indicators, operational KPIs, and sensor-placement guidance. The figures are schematic or result-summary graphics and do not introduce new result values.
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/contact_sheet_conceptual_non_simulation_figures.png}}
\caption{{Contact sheet of the final conceptual and non-simulation figure package. Schematic figures are labeled as workflow or architecture figures; result-summary figures are generated from existing CSV/table outputs.}}
\end{{figure}}

\section*{{S16. Combined simulation figure package}}
This section collects the compact combined simulation figures that replace many separate state-estimation plots. The main manuscript uses the thermo-hydraulic reconstruction summary, the heat/energy-balance summary, and the operational energy--pressure KPI summary. The uncertainty/anomaly and seasonal/stress/sensitivity summaries are retained in the supplementary material to avoid overloading the main paper.
\begin{{figure}}[H]\centering
\includegraphics[width=0.95\linewidth]{{figures/final/contact_sheet_combined_simulation_figures.png}}
\caption{{Contact sheet of the final combined simulation figure package. Panels are generated from existing CSV/Python result files. Pressure/head and flow are simulator-assisted hidden hydraulic states because public district-heating datasets do not provide dense distributed hydraulic measurements; cost and CO2 indicators are proxy quantities under stated assumptions.}}
\end{{figure}}

\end{{document}}
"""
    _write(PROJECT_ROOT / "paper" / "supplementary_material.tex", text)


def _tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def _fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "--"
    except Exception:
        pass
    try:
        number = float(value)
    except Exception:
        return _tex_escape(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


class RawLatex(str):
    """Marker for table cells that intentionally contain LaTeX syntax."""


def _cell(value: object) -> str:
    if isinstance(value, RawLatex):
        return str(value)
    return _tex_escape(value)


def _public_model_name(value: object) -> str:
    text = str(value)
    text = text.replace("Proposed ", "")
    text = text.replace("PI-GNN-GRU-v3", "PI-GNN-GRU")
    text = text.replace("accuracy_mode", "accuracy")
    text = text.replace("balanced_mode", "balanced")
    text = text.replace("physics_mode", "physics")
    return text


def _write_booktabs_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[object]], resize: bool = False) -> None:
    cols = "l" * len(headers)
    body = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if resize:
        body.append(r"\resizebox{\textwidth}{!}{%")
    body.extend(
        [
            rf"\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            " & ".join(_cell(h) for h in headers) + r" \\",
            r"\midrule",
        ]
    )
    for row in rows:
        body.append(" & ".join(_cell(x) for x in row) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        body.append(r"}")
    body.append(r"\end{table}")
    _write(path, "\n".join(body) + "\n")


def write_reviewer_ready_tables() -> None:
    """Generate compact main-paper tables from saved result CSVs."""
    results = PROJECT_ROOT / "results"
    tables = ensure_dir(PROJECT_ROOT / "paper" / "tables")

    cal = pd.read_csv(results / "calibration_metrics.csv").iloc[0]
    disc = pd.read_csv(results / "discretization_study.csv")
    ver = pd.read_csv(results / "model_verification_summary.csv").iloc[0]
    try:
        import json

        params = json.loads((results / "calibrated_parameters.json").read_text(encoding="utf-8"))
    except Exception:
        params = {}

    rows = [
        ["Calibrated heat-loss coefficient", _fmt(params.get("heat_loss_U_W_m2K"), 3), RawLatex("W m$^{-2}$ K$^{-1}$"), "effective calibrated thermal parameter"],
        ["Effective velocity factor", _fmt(params.get("effective_velocity_factor"), 3), "-", "effective delay/transport calibration"],
        ["Return-temperature offset", _fmt(params.get("return_temperature_offset"), 3), RawLatex("$^\\circ$C"), "effective return-boundary correction"],
        ["Friction factor", _fmt(params.get("friction_factor"), 4), "-", "weakly identifiable without pressure/flow measurements"],
        ["Supply-temperature RMSE", _fmt(cal["RMSE_supply_C"], 3), RawLatex("$^\\circ$C"), "measured-node thermal calibration"],
        ["Return-temperature RMSE", _fmt(cal["RMSE_return_C"], 3), RawLatex("$^\\circ$C"), "measured-node thermal calibration"],
        ["Heat-delivery error", _fmt(cal["heat_delivery_error_percent"], 2), RawLatex("\\%"), "measured heat-load consistency"],
        ["Mean heat loss", _fmt(ver["mean_heat_loss_kW"], 1), "kW", "calibrated simulator"],
        ["Thermal delay", _fmt(ver["thermal_delay_h"], 2), "h", "calibrated simulator"],
        ["Steady-state consistency", _fmt(ver["steady_state_consistency_C_per_step"], 3), RawLatex("$^\\circ$C step$^{-1}$"), "model verification"],
    ]
    for _, row in disc.iterrows():
        rows.append(
            [
                f"Discretization dx={int(row['dx_m'])} m",
                f"outlet Ts {_fmt(row['mean_outlet_supply_C'], 3)}; heat loss {_fmt(row['mean_heat_loss_kW'], 3)}",
                RawLatex("$^\\circ$C; kW"),
                RawLatex(f"delta vs 1000 m: {_fmt(row['outlet_supply_delta_vs_1000m_C'], 4)} $^\\circ$C, {_fmt(row['heat_loss_delta_vs_1000m_kW'], 4)} kW"),
            ]
        )
    _write_booktabs_table(
        tables / "table2_calibration_model_verification.tex",
        "Calibrated effective parameters and verification metrics. Thermal calibration uses real Sønderborg measured-node data; distributed hydraulic fields remain simulator-assisted hidden states.",
        "tab:calibration_verification",
        ["Item", "Value", "Unit", "Interpretation"],
        rows,
        resize=True,
    )

    prep = pd.read_csv(results / "preprocessing_summary.csv")
    prep_s = prep[prep["dataset"].astype(str).eq("sonderborg")].iloc[0]
    availability = pd.read_csv(results / "data_availability_report.csv")
    avail_s = availability[availability["dataset_name"].astype(str).eq("sonderborg")].iloc[0]
    avail_f = availability[availability["dataset_name"].astype(str).eq("flensburg")].iloc[0]
    avail_x = availability[availability["dataset_name"].astype(str).eq("xai4heat")].iloc[0]
    evidence_rows = [
        ["Supply temperature", "Sønderborg measured plant-level data", "real measured-node thermal validation"],
        ["Return temperature", "Sønderborg measured plant-level data", "real measured-node thermal validation"],
        ["Heat delivery", "measured heat load plus calibrated thermal balance", "real-data-assisted validation"],
        ["Distributed pipe temperature", "calibrated thermo-hydraulic simulator", "simulator-assisted hidden state"],
        ["Pressure/head", "calibrated simulator", "simulator-assisted hidden hydraulic state"],
        ["Flow", "heat-load-derived proxy plus simulator", "simulator-assisted hidden hydraulic state"],
        ["Heat loss", "calibrated thermal-loss model and reconstructed states", "simulator-assisted engineering KPI"],
        ["Flensburg transfer", "external public operating dataset", "domain-shift stress test"],
        ["Blind measured-node masking", "not independently available in this run", "not claimed; processed Sønderborg has aggregated all-plants node"],
    ]
    _write_booktabs_table(
        tables / "table_evidence_boundary_summary.tex",
        "Evidence boundary for measured, calibrated, and simulator-assisted quantities. This table defines what is validated with real measurements and what is evaluated as hidden-state reconstruction.",
        "tab:evidence_boundary_summary",
        ["Quantity", "Evidence source", "Validation status"],
        evidence_rows,
        resize=True,
    )

    cal_rows = [
        ["Primary dataset", "Sønderborg", f"{int(avail_s['raw_file_count'])} raw local files; processed CSV available"],
        ["Sampling interval", "15 min preferred", f"{int(prep_s['rows'])} processed rows from {prep_s['start']} to {prep_s['end']}"],
        ["Preprocessing", "duplicates sorted, short gaps interpolated, long gaps flagged/dropped", f"{int(prep_s['short_gap_interpolated_rows'])} short-gap rows; {int(prep_s['long_gap_rows_dropped'])} long-gap rows"],
        ["Calibration period", str(cal["train_period"]), "thermal calibration uses measured supply/return/load variables"],
        ["Held-out period", str(cal["validation_period"]), "held-out data retained for downstream validation/evaluation"],
        ["Calibrated parameters", "heat-loss coefficient, velocity factor, return offset, delay/proxy factors", "hydraulic parameters weakly identifiable without pressure/flow data"],
        ["Flensburg use", "external transfer only", f"available={bool(avail_f['available'])}; not used for Sønderborg tuning"],
        ["XAI4HEAT use", "measured-node validation", "available=True after local package extraction; thermal/energy substation variables only"],
    ]
    _write_booktabs_table(
        tables / "table_calibration_protocol.tex",
        "Calibration and data protocol. The protocol separates Sønderborg calibration/training from Flensburg external transfer and reports preprocessing decisions.",
        "tab:calibration_protocol",
        ["Protocol item", "Value", "Note"],
        cal_rows,
        resize=True,
    )

    blind_status = pd.DataFrame(
        [
            {
                "check": "processed_sonderborg_independent_measured_nodes",
                "status": "not_available",
                "evidence": "processed plant_id has one aggregated value: all_plants",
                "safe_claim": "No independent blind measured-node masking result is claimed for Sønderborg in this run.",
            },
            {
                "check": "current_measured_node_validation",
                "status": "available",
                "evidence": "results/real_measured_node_validation.csv",
                "safe_claim": "Measured-node thermal consistency is reported for available measured/sensor variables.",
            },
            {
                "check": "xai4heat_sparse_substation_masking",
                "status": "not_run",
                "evidence": "XAI4HEAT raw files are not locally available",
                "safe_claim": "Sparse-substation blind masking remains future work unless local raw XAI4HEAT files are provided.",
            },
        ]
    )
    blind_status.to_csv(results / "blind_measured_node_masking_status.csv", index=False)
    _write_booktabs_table(
        tables / "table_blind_measured_node_masking_status.tex",
        "Blind measured-node masking status. The protocol is important for future field validation, but independent measured thermal nodes are not available in the processed Sønderborg file used here.",
        "tab:blind_masking_status",
        ["Check", "Status", "Evidence", "Safe claim"],
        blind_status.values.tolist(),
        resize=True,
    )

    baseline = pd.read_csv(results / "baseline_comparison_final.csv")
    selected = [
        "GRU-MSE",
        "Transformer-MSE",
        "PI-LSTM",
        "Proposed PI-GNN-GRU-v3 accuracy_mode",
        "Proposed PI-GNN-GRU-v3 balanced_mode",
        "Proposed PI-GNN-GRU-v3 physics_mode",
    ]
    rows = []
    for name in selected:
        sub = baseline[baseline["model"].astype(str).eq(name)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            [
                _public_model_name(name),
                _fmt(r["RMSE_Ts_full"], 3),
                _fmt(r["RMSE_Tr_full"], 3),
                _fmt(r["RMSE_H_full"], 3),
                _fmt(r["RMSE_q_full"], 4),
                _fmt(r["heat_loss_error_percent"], 3),
                _fmt(r["energy_balance_residual"], 3),
                _fmt(r["boundary_residual_mean"], 3),
            ]
        )
    _write_booktabs_table(
        tables / "table3_thermo_hydraulic_summary.tex",
        "Thermo-hydraulic benchmark summary for top models. Temperature, pressure/head, and flow full-field metrics are evaluated against calibrated-simulator hidden states; measured-node thermal metrics are reported separately.",
        "tab:thermo_hydraulic_summary",
        [RawLatex("Model"), RawLatex("Ts RMSE ($^\\circ$C)"), RawLatex("Tr RMSE ($^\\circ$C)"), RawLatex("Head RMSE (m)"), RawLatex("Flow RMSE (kg/s)"), RawLatex("Heat-loss err. (\\%)"), RawLatex("Energy resid. (\\%)"), RawLatex("Boundary resid.")],
        rows,
        resize=True,
    )

    ranking = pd.read_csv(results / "model_ranking_summary_final.csv")
    metric_names = {
        "RMSE_Ts_full": "Direct supply-temperature accuracy",
        "RMSE_Tr_full": "Return-temperature reconstruction",
        "heat_loss_error_percent": "Heat-loss reconstruction",
        "energy_balance_residual": "Energy-balance closure",
        "thermal_residual_mean": "Thermal residual",
        "boundary_residual_mean": "Boundary consistency",
    }
    rows = []
    for _, r in ranking.iterrows():
        metric = str(r["metric"])
        rows.append(
            [
                metric_names.get(metric, metric),
                _public_model_name(r["best_model"]),
                _fmt(r["best_value"], 3),
                _public_model_name(r["pignn_gru_v3_best_model"]),
                int(r["pignn_gru_v3_best_rank"]),
                _fmt(r["pignn_gru_v3_best_value"], 3),
                "yes" if bool(r["v3_best_in_metric"]) else "no",
            ]
        )
    th = pd.read_csv(results / "thermo_hydraulic_estimation_metrics.csv")
    for metric, objective in [
        ("pressure_drop_error_percent", "Pressure-drop consistency"),
        ("RMSE_flow_kg_s", "Flow reconstruction"),
        ("pump_head_boundary_error_m", "Pump-head boundary consistency"),
        ("measured_node_temperature_RMSE_C", "Measured-node temperature consistency"),
    ]:
        sub = th[th["metric"].astype(str).eq(metric)].drop_duplicates("metric")
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            [
                objective,
                _public_model_name(r["best_model"]),
                _fmt(r["best_value"], 3),
                "PI-GNN-GRU",
                int(float(r["pignn_gru_v3_rank"])),
                _fmt(r["pignn_gru_v3_value"], 3),
                "yes" if float(r["pignn_gru_v3_rank"]) == 1 else "no",
            ]
        )
    _write_booktabs_table(
        tables / "table4_model_ranking_by_objective.tex",
        "Objective-dependent model ranking. Lower values are better; the table shows why direct RMSE and thermo-hydraulic consistency should be interpreted separately.",
        "tab:model_ranking_objective",
        ["Objective", "Best model", "Best value", "PI-GNN-GRU mode", "PI-GNN rank", "PI-GNN value", "PI-GNN rank 1?"],
        rows,
        resize=True,
    )

    sr = pd.read_csv(results / "sensor_layout_ranking_by_objective.csv")
    rows = []
    for objective in sr["objective"].drop_duplicates():
        top = sr[sr["objective"].eq(objective)].head(2)
        for _, r in top.iterrows():
            rows.append(
                [
                    str(objective),
                    int(r["rank"]),
                    str(r["sensor_layout"]),
                    _fmt(r["score"], 3),
                    str(r["sensor_nodes"]),
                    int(r["sensor_count"]),
                    _fmt(r["max_unobserved_distance_km"], 1),
                ]
            )
    _write_booktabs_table(
        tables / "table5_sensor_layout_recommendation.tex",
        "Sensor-layout recommendations by objective. Scores are objective-specific, so no layout is claimed to be universally optimal.",
        "tab:sensor_layout_recommendation",
        ["Objective", "Rank", "Layout", "Score", "Nodes", "Sensors", "Max gap (km)"],
        rows,
        resize=True,
    )

    defs = [
        ["S1", "inlet only", "0", "lowest hardware count; coarse boundary awareness"],
        ["S2", "inlet + outlet", "0;20", "low-cost boundary consistency"],
        ["S3", "inlet-middle-outlet", "0;10;20", "practical low-sensor thermal-delay monitoring"],
        ["S10", "optimized five sensors", "0;5;10;15;20", "physical consistency and energy-impact monitoring"],
        ["S12", "inlet-two-middle-outlet", "0;7;14;20", "best direct thermal-accuracy score in this run"],
        ["S15/S16", "noisy/dropout layouts", "varies", "robustness stress cases"],
    ]
    _write_booktabs_table(
        tables / "table6_sensor_layout_definitions.tex",
        "Sparse-sensor layouts used in the benchmark.",
        "tab:sensor_layout_definitions",
        ["Layout", "Short name", "Node indices", "Use in study"],
        defs,
        resize=False,
    )

    ext = pd.read_csv(results / "external_validation_flensburg_modes_final.csv")
    rows = []
    for _, r in ext.iterrows():
        rows.append(
            [
                str(r["mode"]).replace("_", " "),
                _fmt(r["RMSE_supply_measured_C"], 2),
                _fmt(r["RMSE_return_measured_C"], 2),
                _fmt(r["heat_load_consistency_error_percent"], 1),
                _fmt(r["heat_loss_error_percent"], 1),
                _fmt(r["energy_balance_residual"], 2),
                str(r["note"])[:86],
            ]
        )
    ds = pd.read_csv(results / "flensburg_domain_shift_analysis_improved.csv").iloc[0]
    caption = (
        "Flensburg transfer and domain-shift metrics. The domain shift is substantial: mean heat load differs by "
        f"{float(ds['heat_load_kw_mean_difference'])/1000:.1f} MW, mean supply temperature by "
        f"{float(ds['supply_temp_C_mean_difference']):.2f} $^\\circ$C, sampling changes from "
        f"{int(ds['sampling_interval_sonderborg_min'])} to {int(ds['sampling_interval_flensburg_min'])} min, "
        "and return temperature is assumed in Flensburg."
    )
    _write_booktabs_table(
        tables / "table6_flensburg_domain_shift.tex",
        caption,
        "tab:flensburg_domain_shift",
        [RawLatex("Transfer mode"), RawLatex("Supply RMSE ($^\\circ$C)"), RawLatex("Return RMSE ($^\\circ$C)"), RawLatex("Heat-load err. (\\%)"), RawLatex("Heat-loss err. (\\%)"), RawLatex("Energy resid. (\\%)"), RawLatex("Interpretation")],
        rows,
        resize=True,
    )

    op = pd.read_csv(results / "operational_kpi_quantification.csv").dropna(subset=["sensor_layout"])
    rows = []
    for _, r in op.iterrows():
        rows.append(
            [
                str(r["scenario"]),
                str(r["sensor_layout"]),
                _fmt(r["heat_loss_error_percent"], 3),
                _fmt(r["pump_energy_proxy_error_percent"], 1),
                _fmt(r["pressure_drop_residual_percent"], 1),
                _fmt(r["energy_balance_residual_percent"], 3),
                _fmt(r["cost_proxy_EUR_per_day"], 0),
                _fmt(r["CO2_proxy_kg_per_day"], 0),
            ]
        )
    _write_booktabs_table(
        tables / "table_operator_sensor_guidelines.tex",
        "Operator sensor-guideline metrics. Cost and CO$_2$ are proxy accounting indicators, not optimized dispatch savings.",
        "tab:operator_sensor_guidelines",
        [RawLatex("Scenario"), RawLatex("Layout"), RawLatex("Heat-loss err. (\\%)"), RawLatex("Pump proxy err. (\\%)"), RawLatex("Pressure-drop resid. (\\%)"), RawLatex("Energy resid. (\\%)"), RawLatex("Cost proxy (EUR/day)"), RawLatex("CO$_2$ proxy (kg/day)")],
        rows,
        resize=True,
    )


def _anchor_values() -> dict[str, str]:
    results = PROJECT_ROOT / "results"
    baseline = pd.read_csv(results / "baseline_comparison_final.csv")
    def row(name: str) -> pd.Series:
        sub = baseline[baseline["model"].astype(str).eq(name)]
        return sub.iloc[0]

    gru = row("GRU-MSE")
    trans = row("Transformer-MSE")
    pilstm = row("PI-LSTM")
    v3a = row("Proposed PI-GNN-GRU-v3 accuracy_mode")
    v3b = row("Proposed PI-GNN-GRU-v3 balanced_mode")
    ext = pd.read_csv(results / "external_validation_flensburg_modes_final.csv")
    direct = ext[ext["mode"].astype(str).eq("direct_transfer")].iloc[0]
    few = ext[ext["mode"].astype(str).str.contains("few_shot", regex=False)].iloc[0]
    op = pd.read_csv(results / "operational_kpi_quantification.csv").dropna(subset=["sensor_layout"])
    s1 = op[op["sensor_layout"].astype(str).eq("S1_inlet_only")].iloc[0]
    s10 = op[op["sensor_layout"].astype(str).eq("S10_optimized_five_sensors")].iloc[0]
    pump_reduction = 100.0 * (float(s1["pump_energy_proxy_error_percent"]) - float(s10["pump_energy_proxy_error_percent"])) / float(s1["pump_energy_proxy_error_percent"])
    pressure_reduction = 100.0 * (float(s1["pressure_drop_residual_percent"]) - float(s10["pressure_drop_residual_percent"])) / float(s1["pressure_drop_residual_percent"])
    pump_kwh_day_delta = float(s1["pump_energy_proxy_kWh_per_day"]) - float(s10["pump_energy_proxy_kWh_per_day"])
    cost_delta = float(s1["cost_proxy_EUR_per_day"]) - float(s10["cost_proxy_EUR_per_day"])
    co2_delta = float(s1["CO2_proxy_kg_per_day"]) - float(s10["CO2_proxy_kg_per_day"])
    return {
        "gru_ts": _fmt(gru["RMSE_Ts_full"], 3),
        "gru_tr": _fmt(gru["RMSE_Tr_full"], 3),
        "gru_heat": _fmt(gru["heat_loss_error_percent"], 3),
        "gru_energy": _fmt(gru["energy_balance_residual"], 3),
        "trans_head": _fmt(trans["RMSE_H_full"], 3),
        "pilstm_head": _fmt(pilstm["RMSE_H_full"], 3),
        "v3a_ts": _fmt(v3a["RMSE_Ts_full"], 3),
        "v3a_tr": _fmt(v3a["RMSE_Tr_full"], 3),
        "v3a_heat": _fmt(v3a["heat_loss_error_percent"], 3),
        "v3a_energy": _fmt(v3a["energy_balance_residual"], 3),
        "v3a_boundary": _fmt(v3a["boundary_residual_mean"], 3),
        "v3b_boundary": _fmt(v3b["boundary_residual_mean"], 3),
        "v3b_flow": _fmt(v3b["RMSE_q_full"], 4),
        "direct_supply": _fmt(direct["RMSE_supply_measured_C"], 2),
        "direct_return": _fmt(direct["RMSE_return_measured_C"], 2),
        "few_supply": _fmt(few["RMSE_supply_measured_C"], 2),
        "few_return": _fmt(few["RMSE_return_measured_C"], 2),
        "pump_reduction": _fmt(pump_reduction, 1),
        "pressure_reduction": _fmt(pressure_reduction, 1),
        "pump_kwh_day_delta": _fmt(pump_kwh_day_delta, 0),
        "cost_delta": _fmt(cost_delta, 0),
        "co2_delta": _fmt(co2_delta, 0),
        "s1_heat": _fmt(s1["heat_loss_error_percent"], 3),
        "s10_heat": _fmt(s10["heat_loss_error_percent"], 3),
    }


def polish_final_manuscript_text() -> None:
    """Add numeric anchors, clean wording, and enforce evidence-boundary captions."""
    paths = [
        PROJECT_ROOT / "paper" / "main_ate_submission_candidate.tex",
        PROJECT_ROOT / "paper" / "main_ate_supervisor_final.tex",
        PROJECT_ROOT / "paper" / "supplementary_material.tex",
    ]
    a = _anchor_values()
    note = "(simulator-assisted hidden states for pressure/head and flow; distributed temperatures evaluated against calibrated simulator outputs)"
    for path in paths:
        if not path.exists():
            continue
        text = _read(path)
        text = text.replace("SÃ¸nderborg", "Sønderborg").replace(r"S\o nderborg", "Sønderborg").replace("Sonderborg", "Sønderborg")
        text = text.replace("Cost and CO2 values", "Cost and CO$_2$ values")
        text = text.replace("CO2", "CO$_2$")
        text = text.replace(
            "District-heating monitoring requires sparse-sensor reconstruction of supply temperature, return temperature, pressure/head, flow, heat delivery, heat loss, transport delay, uncertainty, and energy-balance consistency.",
            "District-heating monitoring requires sparse-sensor reconstruction of supply temperature, return temperature, pressure/head, flow, heat delivery, heat loss, transport delay, uncertainty, and energy-balance consistency. The central claim is that a real-data-calibrated thermo-hydraulic benchmark can evaluate sparse-sensor digital twins beyond direct RMSE by including heat-loss, pressure-drop, flow-balance, energy-balance, uncertainty, robustness, and transfer metrics.",
        )
        text = text.replace(
            "\\input{tables/table1_dataset_roles.tex}\nSønderborg is used",
            "\\input{tables/table1_dataset_roles.tex}\n\\input{tables/table_evidence_boundary_summary.tex}\nSønderborg is used",
        )
        text = text.replace(
            "\\input{tables/table2_calibration_model_verification.tex}\nThe Sønderborg calibration errors",
            "\\input{tables/table2_calibration_model_verification.tex}\n\\input{tables/table_calibration_protocol.tex}\nThe Sønderborg calibration errors",
        )
        text = text.replace(
            "It does not represent dense distributed field validation, because public datasets do not provide full pipe-level measurements.",
            "It does not represent dense distributed field validation, because public datasets do not provide full pipe-level measurements. The calibration protocol also documents the training window, held-out period, 15-minute sampling, preprocessing gap handling, calibrated parameters, and the fact that Flensburg is reserved for transfer rather than tuning.",
        )
        text = text.replace(
            "Public datasets provide plant/substation-level measurements, not complete distributed temperature, pressure/head, and flow fields.",
            "Public datasets provide plant/substation-level measurements, not complete distributed temperature, pressure/head, and flow fields. A blind measured-node masking experiment would be valuable, but the processed Sønderborg file used here contains one aggregated all-plants thermal series rather than multiple independent measured thermal nodes; therefore, no independent blind measured-node masking result is claimed in this run.",
        )
        text = text.replace(
            "\\bibliographystyle{elsarticle-num}",
            "\\section*{Data and Code Availability}\nThe workflow is designed to be reproducible from public operating datasets and generated CSV artifacts. The repository includes data-download and preprocessing scripts, dataset registry files, column-detection reports, calibration scripts, thermo-hydraulic simulator code, model configuration files, train/evaluation workflows, result CSVs, figure-generation scripts, and LaTeX table-generation scripts. Sønderborg and Flensburg source datasets are public and cited in the manuscript. The local XAI4HEAT package is processed in \\texttt{data/processed/xai4heat\\_processed.csv} and used for measured-substation thermal/energy validation; its provenance and measured-node scope are documented in the supplementary material. The repository also includes a repeated-seed runner for a Torch-enabled environment. Simulator-assisted hidden states, model weights, uncertainty artifacts, and quality-gate reports are stored as reproducibility outputs rather than as measured field labels.\n\n\\bibliographystyle{elsarticle-num}",
        )
        text = text.replace(
            "\\end{enumerate}\n\n\\section{Real Operating Datasets and Sparse-Sensor Problem}",
            """\\end{enumerate}

\\subsection*{Literature position and gap}
Fourth-generation district-heating research emphasizes lower supply temperatures, lower return temperatures, integration of variable renewable heat sources, and stronger monitoring of distribution losses \\cite{lund2014fourth}. International district-heating experience also shows that thermal losses, network temperature levels, and operational control remain central barriers to high-efficiency heat supply \\cite{werner2017international}. Broad reviews of district-heating and cooling systems identify monitoring, system-level control, and demand-side uncertainty as persistent operational challenges \\cite{lake2017review}. Early operational optimization studies already treated thermal and hydraulic constraints as coupled rather than separate utility problems \\cite{benonysson1995operational}.

Dynamic simulation has been a long-standing way to recover unmeasured network behavior. Aggregated dynamic models represent thermal inertia and propagation delay at system scale \\cite{larsen2002aggregated}. Detailed temperature-dynamic studies show that pipe delay and return-temperature behavior can be reproduced, but only when calibration is tied to operating data \\cite{gabrielaitiene2007temperature}. Equation-based thermo-hydraulic district-heating models provide clearer physical structure for transport and hydraulic coupling \\cite{van_der_heijde2017dynamic}. Reviews of fast district-heating models explain why reduced-order approximations remain necessary when estimation and operation must be computationally practical \\cite{del_hoyo_arce2018fast}.

More recent work has deepened the thermal-engineering basis of digital-twin-style estimation. Thermal transient analyses show how heat-load changes propagate through network temperatures \\cite{chertkov2019thermal}. Thermo-fluid simulations of district-heating networks illustrate the importance of pressure-drop and flow behavior for interpreting temperature delivery \\cite{guelpa2017thermofluid}. Flexibility reviews argue that district-heating operation increasingly depends on coordinated heat-source, storage, network, and demand response behavior \\cite{vandermeulen2018flexibility}. Data-based reduced-order models show how operating data can accelerate network simulation while retaining thermal structure \\cite{jiang2023rom}. Recent digital-twin work combines hydraulic-resistance identification and load prediction, reinforcing the need to connect thermal and hydraulic evidence \\cite{zheng2024digital}. Physics-guided graph methods are emerging for district-heating transients and fast operational studies \\cite{boussaid2024physics_guided_gnn}. Learning-based thermal power-flow models show the same pressure toward fast but physics-aware surrogates for district-heating operation \\cite{bott2025thermal_power_flow}. Related heat-load forecasting studies demonstrate that graph learning is valuable for networked heat systems, although forecasting demand is different from reconstructing hidden distributed states \\cite{wang2023heat_gnn}.

The broader digital-twin literature frames a twin as a calibrated model-data system, not merely a visualization interface \\cite{tao2019digital}. Modeling-perspective reviews stress that uncertainty, calibration, and evidence boundaries are essential when digital twins are used for decision support \\cite{rasheed2020digital}. Systematic reviews similarly warn that digital-twin claims must specify what is measured, what is inferred, and what remains model-assisted \\cite{semeraro2021digital}. This point is especially important here because public district-heating datasets provide real boundary and node measurements but not dense distributed pressure/head or flow fields.

Graph neural networks provide a natural representation for pipe networks because state estimates can be passed along graph edges while preserving topology \\cite{scarselli2009graph}. Spectral and message-passing graph convolution methods provide practical neural building blocks for graph-structured learning \\cite{kipf2017semi}. Surveys of graph neural networks show that topology-aware learning can improve relational prediction, but they also caution that architecture alone does not guarantee physical consistency \\cite{wu2021gnn_survey}. Physics-informed neural networks introduced the idea of residual-based training with governing equations \\cite{raissi2019physics}. Broader physics-informed machine-learning reviews emphasize that residual constraints must be scaled, weighted, and interpreted carefully when data are noisy or incomplete \\cite{karniadakis2021physics}. For digital-twin deployment, uncertainty calibration is also needed; conformal prediction gives a distribution-free way to construct empirical coverage intervals when exchangeability assumptions are plausible \\cite{angelopoulos2021conformal}. Residual-based anomaly detection should also be treated carefully because threshold-based detectors can be sensitive to drift and missing labels \\cite{chandola2009anomaly}.

The gap addressed by this study is therefore not the absence of another neural architecture. The gap is the absence of a claim-safe, real-data-assisted thermo-hydraulic benchmark that uses public operating data for calibration and measured-node validation, uses a calibrated simulator for hidden distributed state generation, compares strong sequence and graph baselines fairly, reports heat-loss and energy-balance metrics, and explicitly separates real measured evidence from simulator-assisted hidden-state reconstruction.

\\section{Real Operating Datasets and Sparse-Sensor Problem}""",
        )

        text = text.replace(
            "This subsection and the following subsections report temperature, pressure/head, flow, and heat loss as the core thermo-hydraulic state-estimation outputs. Supply and return temperature estimates are evaluated at measured nodes where real data are available and across the pipeline using calibrated-simulator hidden labels. The results confirm that sequence models are strong for some direct RMSE metrics, while PI-GNN-GRU-v3 improves selected return-temperature and consistency-oriented metrics. The full distributed result is simulator-assisted hidden-state reconstruction, while measured-node validation is only for available measured variables.",
            "This subsection and the following subsections report temperature, pressure/head, flow, and heat loss as the core thermo-hydraulic state-estimation outputs. Supply and return temperature estimates are evaluated at measured nodes where real data are available and across the pipeline using calibrated-simulator hidden labels. GRU-MSE gives the lowest full-field supply-temperature RMSE ("
            + a["gru_ts"]
            + " $^\\circ$C), with PI-GNN-GRU-v3 accuracy mode close behind at "
            + a["v3a_ts"]
            + " $^\\circ$C. For return temperature, PI-GNN-GRU-v3 accuracy mode is strongest at "
            + a["v3a_tr"]
            + " $^\\circ$C versus "
            + a["gru_tr"]
            + " $^\\circ$C for GRU-MSE. The full distributed result is simulator-assisted hidden-state reconstruction, while measured-node validation is only for available measured variables.",
        )
        text = text.replace(
            "Heat delivery and heat-loss estimates connect state reconstruction to thermal-engineering operation. Heat-loss and energy-balance metrics can rank models differently from direct RMSE, which is why the benchmark reports both statistical and physical diagnostics. The three-dimensional heat-loss surface is derived from calibrated-simulator segment profiles and saved total heat-loss intervals, with the unit conversion from segment heat loss to kW/km reported in the reproducibility files.",
            "Heat delivery and heat-loss estimates connect state reconstruction to thermal-engineering operation. Heat-loss and energy-balance metrics can rank models differently from direct RMSE, which is why the benchmark reports both statistical and physical diagnostics. PI-GNN-GRU-v3 accuracy mode gives heat-loss error "
            + a["v3a_heat"]
            + "\\% and energy-balance residual "
            + a["v3a_energy"]
            + "\\%, compared with "
            + a["gru_heat"]
            + "\\% and "
            + a["gru_energy"]
            + "\\% for GRU-MSE. The three-dimensional heat-loss surface is derived from calibrated-simulator segment profiles and saved total heat-loss intervals, with the unit conversion from segment heat loss to kW/km reported in the reproducibility files.",
        )
        text = text.replace(
            "Direct RMSE and physical consistency can rank models differently. GRU-MSE and Transformer-MSE remain competitive or strongest for selected direct reconstruction metrics. PI-GNN-GRU-v3 improves selected ATE-relevant metrics, including return-temperature reconstruction, heat-loss error, pressure-drop consistency, and boundary consistency. The result is a multi-objective benchmark, not a claim of universal PI-GNN-GRU-v3 superiority.",
            "Direct RMSE and physical consistency can rank models differently. GRU-MSE is strongest for supply-temperature RMSE ("
            + a["gru_ts"]
            + " $^\\circ$C), while PI-GNN-GRU-v3 accuracy mode ranks second on that metric ("
            + a["v3a_ts"]
            + " $^\\circ$C). PI-GNN-GRU-v3 accuracy mode is rank 1 for return-temperature RMSE ("
            + a["v3a_tr"]
            + " $^\\circ$C), heat-loss error ("
            + a["v3a_heat"]
            + "\\%), and energy-balance residual ("
            + a["v3a_energy"]
            + "\\%). PI-GNN-GRU-v3 balanced mode is rank 1 for boundary residual ("
            + a["v3b_boundary"]
            + ") and flow RMSE ("
            + a["v3b_flow"]
            + " kg/s). The result is a multi-objective benchmark, not a claim of universal PI-GNN-GRU-v3 superiority.",
        )
        text = text.replace(
            "The proposed PI-GNN-GRU-v3 should not be read as the best overall predictor. GRU-MSE is strongest for raw supply-temperature RMSE in the final ranking, and Transformer-MSE remains competitive for several direct and hydraulic hidden-state metrics. PI-GNN-GRU-v3 adds value where topology, sparse-sensor masks, interpolation-residual correction, and normalized physical residuals matter: return-temperature reconstruction, heat-loss error, energy-balance residual, boundary consistency, pressure-drop consistency, and selected robustness diagnostics. This is an ATE-relevant value proposition because heat-loss monitoring and energy-balance closure are operational digital-twin objectives, not only pointwise regression targets.",
            "The proposed PI-GNN-GRU-v3 should not be read as the best overall predictor. GRU-MSE is strongest for raw supply-temperature RMSE in the final ranking ("
            + a["gru_ts"]
            + " $^\\circ$C), and Transformer-MSE remains competitive for several direct and hydraulic hidden-state metrics. PI-GNN-GRU-v3 adds value where topology, sparse-sensor masks, interpolation-residual correction, and normalized physical residuals matter: return-temperature reconstruction ("
            + a["v3a_tr"]
            + " $^\\circ$C), heat-loss error ("
            + a["v3a_heat"]
            + "\\%), energy-balance residual ("
            + a["v3a_energy"]
            + "\\%), boundary consistency ("
            + a["v3b_boundary"]
            + "), pressure-drop consistency, and selected robustness diagnostics. This is an ATE-relevant value proposition because heat-loss monitoring and energy-balance closure are operational digital-twin objectives, not only pointwise regression targets.",
        )
        text = text.replace(
            "The operator-facing decision matrix translates the sensor-layout benchmark into monitoring choices. Direct thermal accuracy favors S12/S10-style layouts when four or five sensors are feasible. Physical consistency and heat-loss monitoring favor optimized five-sensor or S10-style layouts because they reduce unobserved transport length while improving heat-loss and energy-balance indicators. Low-cost monitoring can use inlet--outlet sensing, with inlet-only monitoring reserved for cases where coarse boundary awareness is acceptable and hidden-state uncertainty is tolerated. Energy-impact monitoring favors the optimized five-sensor layout in the current scenario table because pump-energy proxy, heat-loss error, energy-balance residual, cost proxy, and CO$_2$ proxy are all reported transparently under stated assumptions. No layout is universally optimal; the choice depends on whether the utility prioritizes direct thermal accuracy, heat-loss monitoring, capital cost, or operational energy-impact indicators.",
            "The operator-facing decision matrix translates the sensor-layout benchmark into monitoring choices. Direct thermal accuracy favors S12/S10-style layouts when four or five sensors are feasible; S12 has the best direct thermal score (1.570), while S10 is close (1.583) and reduces the maximum unobserved distance to 5.0 km. Physical consistency and heat-loss monitoring favor optimized five-sensor or S10-style layouts because they reduce unobserved transport length while improving heat-loss and energy-balance indicators. Relative to inlet-only sensing, S10 reduces pump-energy proxy error from 53.3\\% to 9.7\\% and pressure-drop residual from 50.4\\% to 6.7\\%. Heat-loss error is similar ("
            + a["s1_heat"]
            + "\\% for inlet-only and "
            + a["s10_heat"]
            + "\\% for S10), so the operator claim is objective-specific rather than universal. Energy-impact monitoring favors optimized five-sensor layouts when pump-energy proxy, pressure-drop residual, and uncertainty are important. No layout is universally optimal; the choice depends on whether the utility prioritizes direct thermal accuracy, heat-loss monitoring, capital cost, or operational energy-impact indicators.",
        )
        text = text.replace(
            "External transfer to Flensburg is challenging and is interpreted as a domain-shift stress test, not proof of broad cross-network transfer. Differences in heat-load scale, supply-temperature distribution, sampling interval, network characteristics, and unavailable return-temperature measurements make direct transfer challenging. Local calibration or adaptation is required for cross-network use. Detailed transfer-mode and return-temperature-assumption diagnostics are retained in the supplementary material so the main paper stays focused on the operational digital-twin evidence.",
            "External transfer to Flensburg is challenging and is interpreted as a domain-shift stress test, not proof of broad cross-network transfer. Direct transfer gives supply-temperature RMSE "
            + a["direct_supply"]
            + " $^\\circ$C and return-temperature RMSE "
            + a["direct_return"]
            + " $^\\circ$C. Few-shot bias adaptation improves the supply-temperature RMSE to "
            + a["few_supply"]
            + " $^\\circ$C and return-temperature RMSE to "
            + a["few_return"]
            + " $^\\circ$C, but heat-load consistency remains difficult. Differences in heat-load scale, supply-temperature distribution, sampling interval, network characteristics, and unavailable return-temperature measurements make direct transfer challenging. Local calibration or adaptation is required for cross-network use. Detailed transfer-mode and return-temperature-assumption diagnostics are retained in the supplementary material so the main paper stays focused on the operational digital-twin evidence.",
        )
        text = text.replace(
            "Values are reported over the stated evaluation horizon unless normalized; cost and CO$_2$ are proxy indicators under stated assumptions. Cost and CO$_2$ values are proxy indicators, not optimized economic-dispatch results. The scenario summaries include a nominal winter-day operation, a combined controlled-stress case, and sparse-sensor layout energy-impact comparisons. Disturbances in the stress case are controlled perturbations applied to real operating profiles, not documented field fault events. The results are reported in Supplementary Section S14 and summarized in Fig.~\\ref{fig:operational_energy_impact}.",
            "Values are reported over the stated evaluation horizon unless normalized; cost and CO$_2$ are proxy indicators under stated assumptions. Cost and CO$_2$ values are proxy indicators, not optimized economic-dispatch results. The sparse-layout KPI table shows that moving from inlet-only to optimized five-sensor monitoring changes pump-energy proxy accounting by about "
            + a["pump_kwh_day_delta"]
            + " kWh/day, cost proxy by about "
            + a["cost_delta"]
            + " EUR/day, and CO$_2$ proxy by about "
            + a["co2_delta"]
            + " kg/day. These numbers indicate monitoring sensitivity to sparse-sensor placement; they are not verified energy or emissions savings. The scenario summaries include a nominal winter-day operation, a combined controlled-stress case, and sparse-sensor layout energy-impact comparisons. Disturbances in the stress case are controlled perturbations applied to real operating profiles, not documented field fault events. The results are reported in Supplementary Section S14 and summarized in Fig.~\\ref{fig:operational_energy_impact}.",
        )
        text = text.replace(
            "This study develops a real-data-assisted sparse-sensor thermo-hydraulic digital-twin benchmark for district-heating monitoring. Real operating data support calibration and measured-node thermal validation, while distributed pressure/head, flow, internal temperature, and heat-loss fields are evaluated as simulator-assisted hidden states. The calibrated simulator reproduces measured thermal boundary behavior accurately and passes numerical consistency checks. Model comparisons show that GRU/Transformer baselines remain strong for direct RMSE, while PI-GNN-GRU-v3 improves selected thermal-engineering metrics such as return-temperature reconstruction, heat-loss error, pressure-drop consistency, and boundary consistency. Sensor placement and Flensburg domain shift strongly affect performance. The operational energy-impact layer further translates reconstructed states into delivered-heat, heat-loss, pump-energy proxy, pressure-drop residual, and energy-balance KPIs under explicit assumptions. The work provides a reproducible benchmark and practical guidance for sparse district-heating virtual sensing without claiming full field validation, optimization/control, or deployment readiness.",
            "This study develops a real-data-assisted sparse-sensor thermo-hydraulic digital-twin benchmark for district-heating monitoring. Real operating data support calibration and measured-node thermal validation, while distributed pressure/head, flow, internal temperature, and heat-loss fields are evaluated as simulator-assisted hidden states. The calibrated simulator reproduces measured thermal boundary behavior with 0.337 $^\\circ$C supply-temperature RMSE, 1.373 $^\\circ$C return-temperature RMSE, and 1.03\\% heat-delivery error, and the 1000 m versus 500 m discretization check changes mean outlet supply temperature by only 0.0004 $^\\circ$C. Model comparisons show that GRU-MSE remains strongest for supply-temperature RMSE ("
            + a["gru_ts"]
            + " $^\\circ$C), while PI-GNN-GRU-v3 accuracy mode is strongest for return-temperature RMSE ("
            + a["v3a_tr"]
            + " $^\\circ$C), heat-loss error ("
            + a["v3a_heat"]
            + "\\%), and energy-balance residual ("
            + a["v3a_energy"]
            + "\\%). PI-GNN-GRU-v3 balanced mode is strongest for boundary residual ("
            + a["v3b_boundary"]
            + "). Sensor placement and Flensburg domain shift strongly affect performance. The operational energy-impact layer further translates reconstructed states into delivered-heat, heat-loss, pump-energy proxy, pressure-drop residual, and energy-balance KPIs under explicit assumptions. The work provides a reproducible benchmark and practical guidance for sparse district-heating virtual sensing without claiming full field validation, optimization/control, or deployment readiness.",
        )

        text = text.replace(
            "The final result package moves the study beyond a pure AI benchmark. It estimates the main engineering quantities required for district-heating monitoring: supply temperature, return temperature, pressure/head, flow, delivered heat, heat loss, cumulative heat loss, energy-balance residual, and thermal delay. Table~\\ref{tab:thermo_hydraulic_summary} summarizes the key thermo-hydraulic estimation results.",
            "The final result package moves the study beyond a pure AI benchmark. It estimates the main engineering quantities required for district-heating monitoring: supply temperature, return temperature, pressure/head, flow, delivered heat, heat loss, cumulative heat loss, energy-balance residual, and thermal delay. Table~\\ref{tab:thermo_hydraulic_summary} summarizes the key thermo-hydraulic estimation results. Quantitatively, GRU-MSE gives the lowest supply-temperature RMSE at "
            + a["gru_ts"]
            + " $^\\circ$C, while PI-GNN-GRU-v3 accuracy mode gives "
            + a["v3a_ts"]
            + " $^\\circ$C and is therefore competitive but not the best on this metric. PI-GNN-GRU-v3 accuracy mode gives the lowest return-temperature RMSE at "
            + a["v3a_tr"]
            + " $^\\circ$C compared with "
            + a["gru_tr"]
            + " $^\\circ$C for GRU-MSE.",
        )
        text = text.replace(
            "Heat loss and energy-balance closure are central ATE-relevant quantities. Table~\\ref{tab:heat_energy_estimation} reports heat and energy metrics. Figure~\\ref{fig:heat_energy} combines delivered heat, segment heat loss, cumulative heat loss, energy-balance residual, heat-loss error, and operational heat-loss ratio. PI-GNN-GRU-v3 is rank 1 for heat-loss error and energy-balance residual in this run, which supports a metric-specific physical-consistency claim.",
            "Heat loss and energy-balance closure are central ATE-relevant quantities. Table~\\ref{tab:heat_energy_estimation} reports heat and energy metrics. Figure~\\ref{fig:heat_energy} combines delivered heat, segment heat loss, cumulative heat loss, energy-balance residual, heat-loss error, and operational heat-loss ratio. In the final benchmark table, PI-GNN-GRU-v3 accuracy mode has heat-loss error "
            + a["v3a_heat"]
            + "\\% and energy-balance residual "
            + a["v3a_energy"]
            + "\\%, compared with "
            + a["gru_heat"]
            + "\\% and "
            + a["gru_energy"]
            + "\\% for GRU-MSE. This supports a metric-specific physical-consistency claim, not a universal accuracy claim.",
        )
        text = text.replace(
            "The central benchmarking result is that no single model dominates every objective. Table~\\ref{tab:model_ranking_objective} and Fig.~\\ref{fig:model_ranking} show metric-dependent ranking. GRU-MSE is strongest for full supply-temperature RMSE and measured-node supply-temperature RMSE. PI-LSTM is strongest for some simulator-assisted hydraulic hidden-state metrics. PI-GNN-GRU-v3 is rank 1 for full return-temperature RMSE, heat-loss error, energy-balance residual, boundary residual, and measured-node return-temperature RMSE.",
            "The central benchmarking result is that no single model dominates every objective. Table~\\ref{tab:model_ranking_objective} and Fig.~\\ref{fig:model_ranking} show metric-dependent ranking. GRU-MSE is strongest for full supply-temperature RMSE ("
            + a["gru_ts"]
            + " $^\\circ$C). Transformer-MSE is strongest for head RMSE in the main benchmark ("
            + a["trans_head"]
            + " m), while PI-LSTM remains competitive for simulator-assisted hydraulic hidden-state reconstruction (head RMSE "
            + a["pilstm_head"]
            + " m). PI-GNN-GRU-v3 is rank 1 for full return-temperature RMSE ("
            + a["v3a_tr"]
            + " $^\\circ$C), heat-loss error ("
            + a["v3a_heat"]
            + "\\%), energy-balance residual ("
            + a["v3a_energy"]
            + "\\%), and boundary residual in balanced mode ("
            + a["v3b_boundary"]
            + ").",
        )
        text = text.replace(
            "GRU-MSE is strongest for raw supply-temperature RMSE in this run, while PI-GNN-GRU-v3 is valuable for selected physical-consistency and heat-loss-oriented objectives.",
            "GRU-MSE is strongest for raw supply-temperature RMSE in this run ("
            + a["gru_ts"]
            + " $^\\circ$C), while PI-GNN-GRU-v3 accuracy mode is within 0.003 $^\\circ$C of that value for supply temperature ("
            + a["v3a_ts"]
            + " $^\\circ$C) and is best for return temperature ("
            + a["v3a_tr"]
            + " $^\\circ$C), heat-loss error ("
            + a["v3a_heat"]
            + "\\%), and energy-balance residual ("
            + a["v3a_energy"]
            + "\\%). PI-GNN-GRU-v3 balanced mode gives the lowest boundary residual ("
            + a["v3b_boundary"]
            + ") and the lowest flow RMSE in the final comparison ("
            + a["v3b_flow"]
            + " kg/s).",
        )
        text = text.replace(
            "The final operator-guideline package adds a second, audit-style ranking by objective. Direct thermal accuracy favors S12 or S10-type layouts in the current calibrated-simulator benchmark, while heat-loss and physical-consistency monitoring favor optimized five-sensor or S10-type layouts. Low-cost monitoring can be based on inlet--outlet sensing when uncertainty is acceptable. Energy-impact monitoring is more demanding because heat-loss ratio, pressure-drop residual, pump-energy proxy, and cost/CO$_2$ proxy are all affected by sensor coverage. These recommendations are planning guidance from the benchmark, not universal deployment rules.",
            "The final operator-guideline package adds a second, audit-style ranking by objective. Direct thermal accuracy favors S12 (score 1.570) or S10 (score 1.583) layouts in the current calibrated-simulator benchmark, while physical-consistency monitoring favors S10 (score 2.563). Low-cost monitoring can be based on inlet--outlet sensing when uncertainty is acceptable. Relative to inlet-only sensing, the optimized five-sensor layout reduces the pump-energy proxy error from 53.3\\% to 9.7\\% and the pressure-drop residual from 50.4\\% to 6.7\\%, reductions of about "
            + a["pump_reduction"]
            + "\\% and "
            + a["pressure_reduction"]
            + "\\%, respectively. Heat-loss error remains similar ("
            + a["s1_heat"]
            + "\\% for inlet-only versus "
            + a["s10_heat"]
            + "\\% for S10), so this is an energy-pressure and robustness argument rather than a claim of universal heat-loss superiority.",
        )
        text = text.replace(
            "Values are reported over the stated evaluation horizon unless normalized. Cost and CO$_2$ values are proxy indicators under stated assumptions, not optimized economic-dispatch results. Pressure/head and flow-based energy indicators are simulator-assisted. The pump-energy proxy, pressure-drop residual, heat-loss ratio, and energy-balance residual are intended as monitoring KPIs rather than optimized dispatch outcomes. The detailed KPI quantification in the supplementary material reports daily delivered heat, daily heat loss, pump-energy proxy, pressure-drop residual, cost proxy, and CO$_2$ proxy for nominal and stressed operating profiles.\nCost and CO2 values are proxy indicators, not optimized economic-dispatch results.",
            "Values are reported over the stated evaluation horizon unless normalized. Cost and CO$_2$ values are proxy indicators under stated assumptions, not optimized economic-dispatch results. Pressure/head and flow-based energy indicators are simulator-assisted. The pump-energy proxy, pressure-drop residual, heat-loss ratio, and energy-balance residual are intended as monitoring KPIs rather than optimized dispatch outcomes. In the operator-guideline KPI table, switching from inlet-only to optimized five-sensor monitoring changes the pump-energy proxy accounting by about "
            + a["pump_kwh_day_delta"]
            + " kWh/day, the cost proxy by about "
            + a["cost_delta"]
            + " EUR/day, and the CO$_2$ proxy by about "
            + a["co2_delta"]
            + " kg/day. These are proxy accounting differences that indicate operational sensitivity to sensor layout; they should not be read as verified savings without dispatch and field-validation studies.",
        )
        text = text.replace(
            "Flensburg is used as an external domain-shift test. It differs from S\\o nderborg in network characteristics, temporal resolution, operating regime, and available variables. Return temperature is unavailable or assumed in the current workflow, which introduces uncertainty. The direct transfer supply-temperature RMSE is approximately 9.58 $^\\circ$C, so Flensburg should not be presented as strong zero-shot generalization. It is better interpreted as evidence that local calibration or few-shot adaptation is needed when a digital twin is moved across district-heating networks.",
            "Flensburg is used as an external domain-shift test. It differs from Sønderborg in network characteristics, temporal resolution, operating regime, and available variables. Return temperature is unavailable or assumed in the current workflow, which introduces uncertainty. Direct transfer gives supply-temperature RMSE "
            + a["direct_supply"]
            + " $^\\circ$C and return-temperature RMSE "
            + a["direct_return"]
            + " $^\\circ$C, so Flensburg should not be presented as strong zero-shot generalization. Few-shot bias adaptation lowers the supply-temperature RMSE to "
            + a["few_supply"]
            + " $^\\circ$C and return-temperature RMSE to "
            + a["few_return"]
            + " $^\\circ$C, supporting the conclusion that local calibration or adaptation is needed when a digital twin is moved across district-heating networks.",
        )
        # Add evidence-boundary note to relevant captions without overloading every figure.
        for stem in [
            "fig_thermo_hydraulic_reconstruction_summary.pdf",
            "fig_heat_energy_balance_summary.pdf",
            "fig_operational_energy_pressure_summary.pdf",
            "fig_model_ranking_ate_dark.pdf",
            "fig_accuracy_physics_tradeoff_ate_dark.pdf",
            "fig_network_sparse_sensor_layout.pdf",
            "fig_uncertainty_anomaly_summary.pdf",
            "fig_digital_twin_dashboard_ate_dark.pdf",
            "fig_flensburg_domain_shift_ate_dark.pdf",
            "fig_seasonal_stress_sensitivity_summary.pdf",
        ]:
            idx = text.find(stem)
            if idx == -1:
                continue
            cap_start = text.find(r"\caption{", idx)
            cap_end = text.find("}", cap_start + 9)
            if cap_start != -1 and cap_end != -1:
                caption = text[cap_start:cap_end]
                if note not in caption:
                    text = text[:cap_end] + " " + note + text[cap_end:]
        text = text.replace("PI-GNN-GRU-v3", "PI-GNN-GRU")
        text = text.replace("Best PI-GNN-v3 mode", "PI-GNN-GRU mode")
        text = text.replace("V3 rank", "PI-GNN rank")
        text = text.replace("V3 value", "PI-GNN value")
        text = text.replace("V3 rank 1?", "PI-GNN rank 1?")
        text = text.replace("v3", "final")
        # Keep stable artifact filenames while avoiding version-style model wording in prose.
        text = text.replace("fig_pignn_gru_final_architecture", "fig_pignn_gru_v3_architecture")
        _write(path, text)
    for folder in [PROJECT_ROOT / "paper" / "tables", PROJECT_ROOT / "paper" / "sections"]:
        if not folder.exists():
            continue
        for path in folder.glob("*.tex"):
            text = _read(path)
            new = text.replace("SÃ¸nderborg", "Sønderborg").replace(r"S\o nderborg", "Sønderborg").replace("Sonderborg", "Sønderborg")
            new = new.replace("CO2", "CO$_2$")
            new = new.replace("PI-GNN-GRU-v3", "PI-GNN-GRU")
            new = new.replace("v3", "final")
            new = new.replace("fig_pignn_gru_final_architecture", "fig_pignn_gru_v3_architecture")
            if new != text:
                _write(path, new)


def polish_submission_package() -> None:
    create_submission_figures()
    _copy_table_aliases()
    fix_latex_unit_symbols()
    write_submission_manuscript()
    write_supplementary_material()
    write_reviewer_ready_tables()
    polish_final_manuscript_text()
    fix_latex_unit_symbols()


if __name__ == "__main__":
    polish_submission_package()
