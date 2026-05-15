"""
Surface Climatology — GraphCast Performance Evaluation
======================================================
Complete pipeline for evaluating GraphCast against IMERG (Rainfall) 
and ERA5 (Temperature, Winds).

Features:
  - Dask Progress Bars
  - High-quality JGR/Climate Dynamics plots
  - Checkpointing for bandpass variances and heavy PSDs
  - LaTeX Skill Metrics Table Generation
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import signal
from scipy.signal import welch
from dask.diagnostics import ProgressBar
import os

# ══════════════════════════════════════════════════════════════════════════════
# DIRECTORIES & PATHS
# ══════════════════════════════════════════════════════════════════════════════
PLOT_DIR = "Plots"
CACHE_DIR = f"{PLOT_DIR}/Cache"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_GC_DIR = "/media/airlab/ROCSTOR/graphcast/Climatology_final"

PATHS = {
    "IMERG": "/home/airlab/Documents/airlab/weathernext_analysis/imerg_regrid_2021_2024.zarr",
    "GC_RAIN": "/home/airlab/Documents/airlab/weathernext_analysis/daily_rain_all_leads_2021_2024.zarr",
    "ERA5_RAIN": f"{BASE_GC_DIR}/daily_tp_ERA5_global_2021_2024.zarr",
    "ERA5_T2M": f"{BASE_GC_DIR}/daily_t2m_ERA5_global_2021_2024.zarr",
    "ERA5_U10": f"{BASE_GC_DIR}/daily_u10_ERA5_global_2021_2024.zarr",
    "ERA5_V10": f"{BASE_GC_DIR}/daily_v10_ERA5_global_2021_2024.zarr",
}

# GraphCast paths are split by lead time for these surface variables
def get_gc_path(var_name, lead):
    return f"{BASE_GC_DIR}/daily_{var_name}_{lead}hr_2021_2024_clean.zarr"

GC_LEADS = [24, 48, 72]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT AESTHETICS (JGR / Climate Dynamics Style)
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

PROJ_MAP  = ccrs.PlateCarree(central_longitude=180)
PROJ_DATA = ccrs.PlateCarree()

COLORS  = {"IMERG": "black", "ERA5": "grey", "GC 24hr": "royalblue", "GC 48hr": "darkorange", "GC 72hr": "firebrick"}
LSTYLES = {"IMERG": "-", "ERA5": "-", "GC 24hr": "--", "GC 48hr": "-.", "GC 72hr": ":"}
MARKERS = {"IMERG": "o", "ERA5": "x", "GC 24hr": "s", "GC 48hr": "^", "GC 72hr": "D"}
MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def done(fname):
    if os.path.exists(f"{PLOT_DIR}/{fname}"):
        print(f"  ⏭️  Skipping (PNG exists): {fname}")
        return True
    return False

def nc_save(da, name):
    ds = da.to_dataset(name="data") if isinstance(da, xr.DataArray) else da
    ds.to_netcdf(f"{CACHE_DIR}/{name}.nc")
    print(f"  💾 Cached → {CACHE_DIR}/{name}.nc")

def nc_load(name):
    ds = xr.open_dataset(f"{CACHE_DIR}/{name}.nc")
    return ds["data"] if "data" in ds else ds

def nc_exists(name):
    return os.path.exists(f"{CACHE_DIR}/{name}.nc")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
def get_first_var(ds):
    """Returns the first data variable in the dataset to avoid hardcoding names."""
    var_name = list(ds.data_vars)[0]
    return ds[var_name]

def get_gc_var(ds, lead, base_var):
    """Robust extraction of lead time variable from a combined Zarr."""
    if "lead" in ds.dims or "lead" in ds.coords:
        try:
            return ds.sel(lead=lead)[base_var]
        except KeyError:
            lead_td = np.timedelta64(lead, 'h')
            return ds.sel(lead=lead_td)[base_var]
    elif f"{base_var}_{lead}hr" in ds.data_vars:
        return ds[f"{base_var}_{lead}hr"]
    else:
        return ds[base_var]

print("\nLoading datasets ...")

# 1. Rainfall
try:
    ds_rain = {}
    
    # Load IMERG
    if os.path.exists(PATHS["IMERG"]):
        imrg = xr.open_zarr(PATHS["IMERG"])
        ds_rain["IMERG"] = get_first_var(imrg)
    
    # Load ERA5 Rain
    if os.path.exists(PATHS["ERA5_RAIN"]):
        era5_rain = xr.open_zarr(PATHS["ERA5_RAIN"])
        # Assuming ERA5 is already in mm/day. If in meters, we'd multiply by 1000 here.
        # usually daily_tp is accumulated meters. If so: ds_rain["ERA5"] = get_first_var(era5_rain) * 1000.
        ds_rain["ERA5"] = get_first_var(era5_rain)
        
    # Load GraphCast Rain
    if os.path.exists(PATHS["GC_RAIN"]):
        gc_rain_ds = xr.open_zarr(PATHS["GC_RAIN"])
        for lead in GC_LEADS:
            ds_rain[f"GC {lead}hr"] = get_gc_var(gc_rain_ds, lead, list(gc_rain_ds.data_vars)[0])
            
    print("✅ Rainfall loaded.")
except Exception as e:
    print(f"⚠️ Error loading rainfall: {e}")

# 2. Temperature (T2M)
try:
    ds_t2m = {}
    if os.path.exists(PATHS["ERA5_T2M"]):
        era5_t2m = xr.open_zarr(PATHS["ERA5_T2M"])
        ds_t2m["ERA5"] = get_first_var(era5_t2m)
        
    for lead in GC_LEADS:
        gc_path = get_gc_path("2m_temperature", lead)
        if os.path.exists(gc_path):
            gc_ds = xr.open_zarr(gc_path)
            ds_t2m[f"GC {lead}hr"] = get_first_var(gc_ds)
            
    print("✅ T2M loaded.")
except Exception as e:
    print(f"⚠️ Error loading T2M: {e}")

# 3. Winds (U10, V10)
try:
    ds_u10, ds_v10 = {}, {}
    if os.path.exists(PATHS["ERA5_U10"]) and os.path.exists(PATHS["ERA5_V10"]):
        era5_u10 = xr.open_zarr(PATHS["ERA5_U10"])
        era5_v10 = xr.open_zarr(PATHS["ERA5_V10"])
        ds_u10["ERA5"] = get_first_var(era5_u10)
        ds_v10["ERA5"] = get_first_var(era5_v10)
        
    for lead in GC_LEADS:
        gc_u_path = get_gc_path("10m_u_component_of_wind", lead)
        gc_v_path = get_gc_path("10m_v_component_of_wind", lead)
        if os.path.exists(gc_u_path) and os.path.exists(gc_v_path):
            gc_u_ds = xr.open_zarr(gc_u_path)
            gc_v_ds = xr.open_zarr(gc_v_path)
            ds_u10[f"GC {lead}hr"] = get_first_var(gc_u_ds)
            ds_v10[f"GC {lead}hr"] = get_first_var(gc_v_ds)
            
    print("✅ 10m Winds loaded.")
except Exception as e:
    print(f"⚠️ Error loading Winds: {e}")

LABELS = list(ds_rain.keys()) if ds_rain else ["IMERG", "ERA5"] + [f"GC {L}hr" for L in GC_LEADS]

# ══════════════════════════════════════════════════════════════════════════════
# MATH & PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def sel_jjas(da):
    return da.sel(time=da.time.dt.month.isin([6, 7, 8, 9]))

def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", alpha=0.7)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return gl

def compute_metrics(obs, mod):
    # Align time to compute diff
    obs_c, mod_c = xr.align(obs, mod, join="inner")
    bias = (mod_c - obs_c).mean(dim=["time", "lat", "lon"]).compute().item()
    rmse = np.sqrt(((mod_c - obs_c)**2).mean(dim=["time", "lat", "lon"])).compute().item()
    
    # Temporal corr
    corr_map = xr.corr(obs_c, mod_c, dim="time")
    # Area weighted mean correlation
    w = np.cos(np.deg2rad(corr_map.lat))
    corr = corr_map.weighted(w).mean(dim=["lat", "lon"]).compute().item()
    return bias, rmse, corr

def plot_maps(data_dict, title, fname, levels, cmap, clabel, extend="max"):
    if done(fname): return
    # Determine grid size dynamically based on number of items (up to 6)
    n = len(data_dict)
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5), subplot_kw={"projection": PROJ_MAP})
    # Make axes flat array, handling scalar case
    if n == 1: axes = np.array([axes])
    else: axes = axes.flatten()
    
    fig.suptitle(title, fontweight="bold")
    
    for idx, (label, data) in enumerate(data_dict.items()):
        ax = axes[idx]
        print(f"    Computing for {label} ...")
        with ProgressBar():
            d = data.compute()
        cf = ax.contourf(d.lon, d.lat, d, levels=levels, cmap=cmap, extend=extend, transform=PROJ_DATA)
        add_map_features(ax)
        ax.set_global()
        ax.set_title(f"{label}", fontweight="bold")
        plt.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06, label=clabel)
        
    # Hide any unused subplots
    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])
        
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()
    print(f"  ✅ Saved → {PLOT_DIR}/{fname}")

def bandpass_butter(da, low_cut, high_cut, dt=1.0):
    nyq = 0.5 / dt
    eps = 1e-6
    low_f = np.clip(dt / high_cut, nyq * eps, nyq * (1 - eps))
    high_f = np.clip(dt / low_cut, nyq * eps, nyq * (1 - eps))
    b, a = signal.butter(4, [low_f / nyq, high_f / nyq], btype="band")
    
    return xr.apply_ufunc(
        lambda x: signal.filtfilt(b, a, x, axis=-1),
        da, input_core_dims=[["time"]], output_core_dims=[["time"]],
        vectorize=False, dask="parallelized", output_dtypes=[float],
    )

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

if ds_rain:
    print("\n══ STEP 1: Rainfall Spatial Climatology (JJAS) ════════════════════")
    jjas_mean = {k: sel_jjas(v).mean("time") for k, v in ds_rain.items()}
    plot_maps(jjas_mean, "JJAS Mean Rainfall (2021–2024)", "A_jjas_mean_spatial.png",
             np.arange(0, 16, 1), "YlGnBu", "Rainfall (mm/day)")

    print("\n══ STEP 2: Annual Cycle Curve ═════════════════════════════════════")
    if not done("C_annual_cycle_india.png"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, da in ds_rain.items():
            print(f"    Computing monthly cycle: {label} ...")
            # Indian Region mask roughly 5N-35N, 65E-95E
            da_ind = da.sel(lat=slice(5, 35), lon=slice(65, 95))
            with ProgressBar():
                monthly = da_ind.groupby("time.month").mean("time").compute()
            weights = np.cos(np.deg2rad(monthly.lat))
            curve = monthly.weighted(weights).mean(dim=["lat", "lon"])
            
            ax.plot(range(1, 13), curve, color=COLORS.get(label, "black"), linestyle=LSTYLES.get(label, "-"),
                    marker=MARKERS.get(label, "o"), linewidth=2.5, label=label)
            
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTHS)
        ax.set_ylabel("Rainfall (mm/day)", fontweight="bold")
        ax.set_title("Annual Cycle of Rainfall (Indian Region)", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.savefig(f"{PLOT_DIR}/C_annual_cycle_india.png")
        plt.close()
        print(f"  ✅ Saved → {PLOT_DIR}/C_annual_cycle_india.png")

if ds_t2m:
    print("\n══ STEP 3: T2M Spatial Climatology (JJAS) ═════════════════════════")
    t2m_jjas = {k: sel_jjas(v).mean("time") for k, v in ds_t2m.items()}
    plot_maps(t2m_jjas, "JJAS Mean 2m Temperature (2021–2024)", "I_t2m_jjas_spatial.png",
             np.arange(270, 310, 2), "coolwarm", "Temperature (K)")

if ds_u10 and ds_v10:
    print("\n══ STEP 4: 10m Wind Speed (JJAS) ══════════════════════════════════")
    wspd_jjas = {}
    for k in ds_u10.keys():
        wspd = np.sqrt(ds_u10[k]**2 + ds_v10[k]**2)
        wspd_jjas[k] = sel_jjas(wspd).mean("time")
        
    plot_maps(wspd_jjas, "JJAS Mean 10m Wind Speed (2021–2024)", "J_10m_wind_jjas.png",
             np.arange(0, 15, 1), "Purples", "Wind Speed (m/s)")

print("\n══ STEP 5: Generating Skill Metrics Table (LaTeX) ═════════════════")
# Creates the LaTeX table comparing GC against observations
tex_file = f"{PLOT_DIR}/statistics_table.tex"

if not os.path.exists(tex_file):
    metrics = {}
    
    if ds_rain and "IMERG" in ds_rain:
        metrics["Rainfall (JJAS vs IMERG)"] = {}
        print("  Evaluating Rainfall ...")
        obs_rain = sel_jjas(ds_rain["IMERG"])
        # Evaluate ERA5 and GraphCast against IMERG
        eval_keys = [k for k in ds_rain.keys() if k != "IMERG"]
        for label in eval_keys:
            with ProgressBar():
                b, r, c = compute_metrics(obs_rain, sel_jjas(ds_rain[label]))
            metrics["Rainfall (JJAS vs IMERG)"][label] = {"Bias": b, "RMSE": r, "Corr": c}
            
    if ds_t2m and "ERA5" in ds_t2m:
        metrics["T2M (Annual vs ERA5)"] = {}
        print("  Evaluating T2M ...")
        obs_t2m = ds_t2m["ERA5"]
        eval_keys = [k for k in ds_t2m.keys() if k != "ERA5"]
        for label in eval_keys:
            with ProgressBar():
                b, r, c = compute_metrics(obs_t2m, ds_t2m[label])
            metrics["T2M (Annual vs ERA5)"][label] = {"Bias": b, "RMSE": r, "Corr": c}

    # Write LaTeX Table
    if metrics:
        tex = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{llccc}\n\\hline\n"
        tex += "\\textbf{Variable} & \\textbf{Model} & \\textbf{Bias} & \\textbf{RMSE} & \\textbf{Correlation} \\\\\n\\hline\n"
        
        for var, mods in metrics.items():
            tex += f"\\multirow{{{len(mods)}}}{{*}}{{{var}}} "
            for i, (mod, vals) in enumerate(mods.items()):
                if i > 0: tex += " & "
                tex += f"& {mod} & {vals['Bias']:.2f} & {vals['RMSE']:.2f} & {vals['Corr']:.2f} \\\\\n"
            tex += "\\hline\n"
            
        tex += "\\end{tabular}\n\\caption{Skill metrics evaluated against Observational Truth.}\n"
        tex += "\\label{tab:skill_metrics}\n\\end{table}"
        
        with open(tex_file, "w") as f:
            f.write(tex)
        print(f"  ✅ Saved → {tex_file}")
else:
    print(f"  ⏭️  Skipping (Exists): {tex_file}")

print("\n══ STEP 6: Power Spectrum (Welch) ═════════════════════════════════")
def run_psd(domain_name, lat_sl, lon_sl):
    fname = f"M_psd_{domain_name}_combined.png"
    if done(fname): return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Power Spectrum: Rainfall vs 10m U-Wind ({domain_name.capitalize()})", fontweight="bold")
    ax_r, ax_u = axes
    
    for label in LABELS:
        # Rain PSD
        if ds_rain and label in ds_rain:
            print(f"    Rain PSD: {label} ...")
            da = ds_rain[label].sel(lat=lat_sl, lon=lon_sl) if lat_sl else ds_rain[label]
            with ProgressBar():
                da_c = da.compute()
            weights = np.cos(np.deg2rad(da_c.lat))
            ts = da_c.weighted(weights).mean(["lat", "lon"]).values.astype(float)
            ts = signal.detrend(ts - ts.mean())
            f_r, p_r = welch(ts, fs=1.0, nperseg=365, noverlap=182, window="hann")
            ax_r.loglog(f_r[1:], p_r[1:], color=COLORS.get(label, "black"), linestyle=LSTYLES.get(label, "-"), linewidth=2, label=label)

        # U10 PSD
        if ds_u10 and label in ds_u10:
            print(f"    U10 PSD: {label} ...")
            da = ds_u10[label].sel(lat=lat_sl, lon=lon_sl) if lat_sl else ds_u10[label]
            with ProgressBar():
                da_c = da.compute()
            weights = np.cos(np.deg2rad(da_c.lat))
            ts = da_c.weighted(weights).mean(["lat", "lon"]).values.astype(float)
            ts = signal.detrend(ts - ts.mean())
            f_u, p_u = welch(ts, fs=1.0, nperseg=365, noverlap=182, window="hann")
            ax_u.loglog(f_u[1:], p_u[1:], color=COLORS.get(label, "black"), linestyle=LSTYLES.get(label, "-"), linewidth=2, label=label)

    for ax, title in [(ax_r, "Rainfall (mm²/day²)"), (ax_u, "10m U-Wind (m²/s²)")]:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Frequency (cycles/day)")
        ax.set_ylabel("PSD")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_xlim(1/100, 0.5)
        ax.legend()
        
        # Add Reference Lines for ISOs
        for period, text in [(90, "90d"), (30, "30d"), (10, "10d")]:
            ax.axvline(1/period, color="gray", linestyle=":", linewidth=1)
            ax.text(1/period * 1.05, ax.get_ylim()[0]*2, text, rotation=90, color="gray", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}")
    plt.close()
    print(f"  ✅ Saved → {PLOT_DIR}/{fname}")

run_psd("global", None, None)
run_psd("regional", slice(-20, 35), slice(30, 140))

print("\n" + "═" * 70)
print("  COMPLETE PIPELINE FINISHED")
print("═" * 70)