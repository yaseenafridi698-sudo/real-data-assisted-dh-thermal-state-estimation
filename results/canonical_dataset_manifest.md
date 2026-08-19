# Canonical Sonderborg processed dataset

- Path: `data/locked/sonderborg_processed_18703.csv`
- Retained timestamps: **18,703**
- SHA-256: `35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e`
- Preprocessing: 15-min resampling; forward-only interpolation up to 8 samples; long gaps retained as trajectory breaks after missing-row removal; configured 5 C ambient boundary

## Excluded legacy artifact

The excluded legacy artifact retained 19878 timestamps and marked 2648 short-gap rows as interpolated. The frozen causal artifact retains 18703 timestamps and 1473 short-gap interpolations. The 1175-row difference equals the interpolation-count difference (1175); the legacy artifact was produced before forward-only chronological interpolation was enforced and is excluded from active evidence.

The two processed datasets are never pooled or interchanged in the final rebuild.
