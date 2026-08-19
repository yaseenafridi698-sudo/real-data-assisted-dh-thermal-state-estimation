# Data directory

`external_weather/` contains the Open-Meteo ERA5-Land forcing file that may be redistributed with attribution.

`raw/` is for locally downloaded third-party datasets and is ignored by Git.

`locked/` is for the author-held canonical Sønderborg processed input. The exact manuscript file has 18,703 rows and SHA-256 `35ecda536ba73bc82c202a29e160ef0327e611a825aa558d2d304a356ac98d8e`. It is intentionally not bundled in this public release. Use `scripts/install_canonical_data.py` to verify and install an authorized local copy.
