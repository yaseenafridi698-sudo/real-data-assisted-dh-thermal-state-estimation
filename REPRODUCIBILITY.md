# Reproducibility levels

## Level A — public archive integrity (no measured-data download)
Run `python scripts/verify_repository.py`. This checks exact paper identity, required evidence, weather hash, checkpoint/NPZ counts, figure set, CSV/JSON readability, path sanitization, measured-data exclusion, and the release manifest.

## Level B — executable-code integrity
Install `requirements.txt`, then run `python scripts/verify_repository.py --full` and `python -m pytest -q`. This loads every state-dict checkpoint and NPZ and imports every public source module.

## Level C — frozen-input checkpoint analysis
Install the canonical 18,703-row Sønderborg artifact with `scripts/install_canonical_data.py`. The canonical hash is enforced. The recovered chronological protocol is regression-tested against the paper audit: 768 retained samples, restart indices 0 and 62, 746 gap-safe candidate windows, and 522/111/91 train/validation/test windows with 11-step embargoes.

## Level D — external-data regeneration
Flensburg and XAI4HEAT row-level regeneration requires obtaining those datasets from their original repositories. They are not redistributed here. Frozen derived evidence and the publication figure are retained where appropriate, with evidence-class limitations stated explicitly.

## Full raw-to-processed boundary
The original annual raw Sønderborg files were not part of the final execution archive from which this public release was reconstructed. Therefore this repository does not claim an independently re-audited raw-annual-files-to-canonical-CSV reproduction. The exact canonical processed artifact is hash-pinned, and all claims that depend on it are scoped accordingly.
