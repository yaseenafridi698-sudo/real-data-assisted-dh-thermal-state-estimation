# Data sources, redistribution, and attribution

This repository separates code licensing from third-party data rights.

## Sønderborg district-heating data

- DOI: https://doi.org/10.5281/zenodo.7972964
- Role: primary measured operating dataset.
- Public-release action: **not redistributed** in this GitHub package, including the frozen processed derivative.
- Reason: the current Zenodo record is open-access but its displayed `License` field is blank. Users should obtain the source dataset from the original record and comply with any rights/terms supplied by the data owner or repository.

## Flensburg district-heating data

- DOI: https://doi.org/10.5281/zenodo.10508280
- Role: external measured validation dataset.
- Public-release action: **not redistributed**.
- Reason: the current Zenodo record is open-access but its displayed `License` field is blank.

## XAI4HEAT SCADA Dataset 2024

- DOI: https://doi.org/10.17632/2mwc6x6kwb.1
- Role: sparse measured substation validation.
- License shown by Mendeley Data: **CC BY 4.0**.
- Public-release action: not bundled; users obtain it from the original record.

## ERA5-Land forcing via Open-Meteo

- Source used by this project: Open-Meteo Historical Weather API, ERA5-Land `temperature_2m`.
- Open-Meteo states that API data are provided under **CC BY 4.0**.
- Bundled file: `data/external_weather/sonderborg_era5_land_2016_2019_hourly.csv`.
- Frozen SHA-256: `36ca9f9d93b92136649536f94885cf4bf049a0dd4fea88ad494030627283f7c5`.
- Attribution: Open-Meteo and the underlying ERA5-Land/ECMWF data source should be credited in downstream reuse.

## Aalborg smart-heat-meter data

- DOI: https://doi.org/10.5281/zenodo.6563114
- Role: optional demand-side realism.
- Public-release action: not bundled. Users should obtain it from the original record and follow the license shown there at the time of access.

## Code vs. data license

The repository `LICENSE` applies to the software authored for this project. It does not relicense third-party datasets or remove attribution/redistribution obligations attached to them.
