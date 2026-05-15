# GraphCast Climatology Analysis

This repository contains a comprehensive Python pipeline (`wholecode.py`) for evaluating the climatological performance of the **GraphCast** AI weather model against observational ground truth (**IMERG**) and reanalysis datasets (**ERA5**).

## Overview

The analysis focuses on surface variables, particularly rainfall, 2m temperature, and 10m winds over the period 2021–2024. It computes a wide array of climatological metrics, scales, and spectra, generating publication-quality figures suitable for high-impact journals such as *Climate Dynamics* and *Journal of Geophysical Research (JGR)*.

## Key Features

- **Spatial Climatology:** Computes and maps Annual and JJAS (June-September) means.
- **Zonal and Meridional Profiles:** Analyzes latitudinal variations and zonal means.
- **Annual Cycles:** Extracts daily/monthly rainfall and temperature curves over global and regional (e.g., Indian Summer Monsoon) domains.
- **Hovmöller Diagrams:** Visualizes latitude-time propagation of variables.
- **Bias and RMSE:** Calculates spatial bias and root-mean-square error against IMERG/ERA5 baselines.
- **Intraseasonal Variability (ISV):** Employs Butterworth bandpass filtering to isolate and map Synoptic (2–10 days) and ISO (10–90 days) variances.
- **Spectral Analysis:** Generates global power spectra (Welch's method) and 2D Wavenumber-Frequency spectra for equatorial regions.
- **Circulation Diagnostics:** Computes and visualizes the regional Hadley Cell mass streamfunction.
- **Automated Skill Metrics:** Outputs statistical summaries (Bias, RMSE, Correlation) directly into LaTeX table format.
- **Performance:** Optimized with `xarray` and `dask` for out-of-core computation, featuring automated caching of heavy intermediate variables.

## Data Requirements

The pipeline expects data in `.zarr` format. Observational baselines include:
- **IMERG** (Precipitation)
- **ERA5** (Precipitation, 2m Temperature, 10m U/V Winds, 3D V-Wind)

Model output:
- **GraphCast** (Lead times: 24hr, 48hr, 72hr)
