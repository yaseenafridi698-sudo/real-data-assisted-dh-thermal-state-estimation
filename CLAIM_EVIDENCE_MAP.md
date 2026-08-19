# Claim-to-evidence map

This map is reviewer-facing: it points to the frozen artifacts that support each claim family. Raw values in the CSV/JSON files remain the source of truth.

| Claim family | Primary evidence | Evidence class |
|---|---|---|
| Primary model comparison | `results/baseline_comparison_final.csv`, `results/physics_consistency_comparison_final.csv` | measured-node + calibrated/simulator-assisted |
| Calibration and equifinality | `results/calibration_metrics.csv`, `results/calibrated_parameters.json`, `results/calibration_equifinality_*.csv`, `results/parameter_identifiability_sensitivity.csv` | calibrated simulator |
| Causal/proxy integrity | `results/proxy_causality_audit.csv`, `results/causal_heat_load_input_ablation.csv`, `results/gap_handling_audit.json` | protocol/audit |
| Repeated-seed stability | `results/repeated_seed_statistics.csv`, `results/repeated_seed_raw_metrics.csv`, seed-specific checkpoints | repeated training evidence |
| Multi-window / temporal transfer | `results/multi_window_three_seed_summary.csv`, `results/multi_window_rank_stability.csv`, `results/second_chronological_window_summary.csv` | chronological transfer |
| Dense reconstruction | `results/dense_reconstruction_payloads.npz`, Figures 1--4 | calibrated/simulator-assisted hidden state |
| Heat-loss and energy consistency | `results/heat_loss_profile_metrics.csv`, `results/energy_balance_time_series.csv`, Figure 7 | calibrated simulator + measured boundary |
| Model-value trade-off | `results/concept_model_value_rank_matrix.csv`, `results/baseline_comparison_final.csv`, Figure 8 | mixed, explicitly separated |
| External Flensburg transfer | `results/flensburg_measured_only_validation.csv`, `results/flensburg_domain_shift_analysis.csv`, `results/flensburg_causal_supply_forecast_summary.csv`, frozen Figure 9 | measured external nodes; row-level source external |
| XAI4HEAT measured-node validation | `results/xai4heat_sparse_substation_validation_final.csv` and associated withholding/quality audits | measured-node |
| Uncertainty | `results/uncertainty_quantification_metrics.csv`, `results/uncertainty_conformal_evaluation_locked.csv` | confidence-band evaluation |
| Robustness | `results/thermo_hydraulic_robustness.csv`, `results/noise_dropout_robustness_final.csv`, `results/combined_stress_test.csv` | controlled perturbation / simulator-assisted |
| Anomaly layer | `results/anomaly_detection_metrics_improved.csv`, `results/anomaly_detection_timeseries_improved.csv` | controlled perturbation, not observed faults |
| Statistical uncertainty | `results/moving_block_bootstrap_ci.csv`, `results/moving_block_bootstrap_protocol.json` | blocked chronological inference |
| Figure provenance | `figures/main/figure_provenance.csv` | provenance map |
