# Real-Data-Assisted Thermal State Estimation in District Heating Networks

Reproducibility repository for:

> **Real-Data-Assisted Thermal State Estimation in District Heating Networks: An Evidence-Separated Benchmark with Simulator-Assisted Hydraulics**

This public package contains the paper code, frozen numerical evidence, 56 saved PyTorch state-dict checkpoints, three NPZ evidence payloads, the ten frozen main figures, figure provenance, repeated-seed/multi-window evidence, robustness/uncertainty/anomaly outputs, and integrity checks.

## Evidence boundary

The repository deliberately distinguishes **measured-node evidence** from **calibrated-simulator** and **simulator-assisted hidden-state** evidence. Dense distributed pressure/head and flow are not presented as field measurements. Controlled anomaly/stress cases are not presented as observed field faults. See `PAPER_SCOPE.md` and `CLAIM_EVIDENCE_MAP.md`.

## Quick verification

A fresh clone can verify the public archive without downloading measured district-heating data:

```bash
python scripts/verify_repository.py
```

For checkpoint/NPZ loading and source imports after installing dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/verify_repository.py --full
python -m pytest -q
```

## Canonical Sønderborg input

The exact processed input used by the locked benchmark is **not redistributed** because the upstream record currently does not expose a clear redistribution license in the repository metadata we can rely on. Obtain the source data from the DOI documented in `DATA_LICENSES.md`, reproduce/obtain the canonical processed artifact, then install it with:

```bash
python scripts/install_canonical_data.py /path/to/sonderborg_processed_18703.csv
```

The installer accepts the file only if both locked conditions hold:

- rows: **18,703**
- SHA-256: `35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e`

Once installed, the checkpoint-based frozen-input analysis path can use the canonical file directly; annual raw files are not required for that path.

## Full scientific rerun

After installing the canonical Sønderborg artifact, run:

```bash
python run_real_data_study.py
```

This runs the scientific training/evaluation workflow and skips legacy manuscript-asset generation by default. Add locally obtained Flensburg/XAI4HEAT data under `data/processed/` or their configured raw directories when those external analyses must be regenerated. `python run_quick_demo.py` is only a software smoke test and its fallback-synthetic outputs are **not paper evidence**.

## Figures

`figures/main/` contains the ten frozen main-paper figures in SVG, PDF and PNG plus `figure_provenance.csv`. `scripts/regenerate_main_figures.py` regenerates figures only when the required numerical inputs are present; it skips a figure rather than synthesizing missing evidence. Figure 9 is intentionally retained as a frozen publication artifact because row-level Flensburg inputs are not redistributed.

## Repository layout

- `src/` — model, preprocessing, simulation, evaluation and audit code
- `scripts/` — verification, canonical-data installation, figure generation and figure utilities
- `config/` — frozen configuration and data-source registry
- `results/` — frozen machine-readable paper evidence and checkpoints
- `figures/main/` — frozen main-paper figures and provenance
- `data/external_weather/` — redistributable ERA5-Land/Open-Meteo forcing
- `tests/` — regression/integrity tests
- `CLAIM_EVIDENCE_MAP.md` — reviewer-facing evidence index
- `DATA_LICENSES.md` — data provenance and redistribution policy

## What this repository does not claim

It does not claim dense hydraulic field validation, independent absolute metrology of hidden states, or observed-field validation of the controlled anomaly cases. External datasets remain subject to their original terms.
