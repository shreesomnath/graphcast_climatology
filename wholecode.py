"""
Comprehensive Surface Climatology Pipeline — GraphCast Evaluation
=================================================================
Includes all 24+ analysis steps: Spatial maps (Annual/JJAS), Zonal/Meridional
Profiles, Hovmoller diagrams, Bias/RMSE maps, Variance filtering (Synoptic/ISO),
Wavenumber-Frequency spectra, Power spectra, Hadley cell circulation, and Skill Tables.
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
    
    # 3D Variables for Hadley Cell
    "ERA5_V_3D": f"{BASE_GC_DIR}/daily_v_component_of_wind_ERA5_2021_2024_clean.zarr",
}

def get_gc_path(var_name, lead):
    return f"{BASE_GC_DIR}/daily_{var_name}_{lead}hr_2021_2024_clean.zarr"

GC_LEADS = [24, 48, 72]

# ══════════════════════════════════════════════════════════════════════════════
# PLOT AESTHETICS (JGR / Climate Dynamics Style)
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight"
})

PROJ_MAP  = ccrs.PlateCarree(central_longitude=180)
PROJ_DATA = ccrs.PlateCarree()

COLORS  = {"IMERG": "black", "ERA5": "dimgrey", "GC 24hr": "royalblue", "GC 48hr": "darkorange", "GC 72hr": "firebrick"}
LSTYLES = {"IMERG": "-", "ERA5": "-", "GC 24hr": "--", "GC 48hr": "-.", "GC 72hr": ":"}
MARKERS = {"IMERG": "o", "ERA5": "x", "GC 24hr": "s", "GC 48hr": "^", "GC 72hr": "D"}
MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def done(fname):
    if os.path.exists(f"{PLOT_DIR}/{fname}"):
        print(f"  ⏭️  Skipping {fname}")
        return True
    return False

def nc_save(da, name):
    da.to_dataset(name="data").to_netcdf(f"{CACHE_DIR}/{name}.nc")

def nc_load(name):
    return xr.open_dataset(f"{CACHE_DIR}/{name}.nc")["data"]

def nc_exists(name):
    return os.path.exists(f"{CACHE_DIR}/{name}.nc")

def get_first_var(ds):
    return ds[list(ds.data_vars)[0]]

def sel_jjas(da):
    return da.sel(time=da.time.dt.month.isin([6, 7, 8, 9]))

def add_map_features(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", alpha=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    return gl

def bandpass_butter(da, low_cut, high_cut, dt=1.0):
    nyq = 0.5 / dt
    eps = 1e-6
    low_f = np.clip(dt / high_cut, nyq * eps, nyq * (1 - eps))
    high_f = np.clip(dt / low_cut, nyq * eps, nyq * (1 - eps))
    b, a = signal.butter(4, [low_f / nyq, high_f / nyq], btype="band")
    return xr.apply_ufunc(
        lambda x: signal.filtfilt(b, a, x, axis=-1),
        da, input_core_dims=[["time"]], output_core_dims=[["time"]],
        vectorize=False, dask="parallelized", output_dtypes=[float]
    )

def plot_multi_map(data_dict, title, fname, levels, cmap, clabel, extend="max"):
    if done(fname): return
    n = len(data_dict)
    cols = 3 if n > 4 else (2 if n > 1 else 1)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6.5, rows*4), subplot_kw={"projection": PROJ_MAP})
    if n == 1: axes = [axes]
    else: axes = axes.flatten()
    
    fig.suptitle(title, fontweight="bold")
    for idx, (label, data) in enumerate(data_dict.items()):
        ax = axes[idx]
        print(f"    Computing map: {label} ...")
        with ProgressBar():
            d = data.compute()
        cf = ax.contourf(d.lon, d.lat, d, levels=levels, cmap=cmap, extend=extend, transform=PROJ_DATA)
        add_map_features(ax)
        ax.set_global()
        ax.set_title(f"{label} | Mean={float(d.mean()):.2f}", fontweight="bold", fontsize=10)
        plt.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06, label=clabel)
        
    for i in range(n, len(axes)): fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close()
    print(f"  ✅ Saved {fname}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading Datasets...")
ds_rain, ds_t2m, ds_u10, ds_v10, ds_v3d = {}, {}, {}, {}, {}

# 1. Rain
try:
    if os.path.exists(PATHS["IMERG"]): ds_rain["IMERG"] = get_first_var(xr.open_zarr(PATHS["IMERG"]))
    if os.path.exists(PATHS["ERA5_RAIN"]): ds_rain["ERA5"] = get_first_var(xr.open_zarr(PATHS["ERA5_RAIN"])) * 1000 # Assume m to mm
    if os.path.exists(PATHS["GC_RAIN"]):
        gc_r = xr.open_zarr(PATHS["GC_RAIN"])
        for lead in GC_LEADS:
            ds_rain[f"GC {lead}hr"] = gc_r.sel(lead=np.timedelta64(lead, 'h'))["daily_rain"] if "lead" in gc_r.coords else gc_r[f"daily_rain_{lead}hr"]
except Exception as e: print(f"⚠️ Rain load error: {e}")

# 2. T2M
try:
    if os.path.exists(PATHS["ERA5_T2M"]): ds_t2m["ERA5"] = get_first_var(xr.open_zarr(PATHS["ERA5_T2M"]))
    for lead in GC_LEADS:
        p = get_gc_path("2m_temperature", lead)
        if os.path.exists(p): ds_t2m[f"GC {lead}hr"] = get_first_var(xr.open_zarr(p))
except Exception as e: print(f"⚠️ T2M load error: {e}")

# 3. U10, V10
try:
    if os.path.exists(PATHS["ERA5_U10"]): ds_u10["ERA5"] = get_first_var(xr.open_zarr(PATHS["ERA5_U10"]))
    if os.path.exists(PATHS["ERA5_V10"]): ds_v10["ERA5"] = get_first_var(xr.open_zarr(PATHS["ERA5_V10"]))
    for lead in GC_LEADS:
        pu = get_gc_path("10m_u_component_of_wind", lead)
        pv = get_gc_path("10m_v_component_of_wind", lead)
        if os.path.exists(pu): ds_u10[f"GC {lead}hr"] = get_first_var(xr.open_zarr(pu))
        if os.path.exists(pv): ds_v10[f"GC {lead}hr"] = get_first_var(xr.open_zarr(pv))
except Exception as e: print(f"⚠️ Wind load error: {e}")

# 4. 3D V Wind (for Hadley)
try:
    if os.path.exists(PATHS["ERA5_V_3D"]): ds_v3d["ERA5"] = get_first_var(xr.open_zarr(PATHS["ERA5_V_3D"]))
    for lead in GC_LEADS:
        pv3 = get_gc_path("v_component_of_wind", lead)
        if os.path.exists(pv3): ds_v3d[f"GC {lead}hr"] = get_first_var(xr.open_zarr(pv3))
except Exception as e: print(f"⚠️ 3D V Wind load error: {e}")

LABELS_R = list(ds_rain.keys())

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nRUNNING COMPREHENSIVE PIPELINE\n" + "="*60)

# --- 1. SPATIAL MEANS ---
if ds_rain:
    plot_multi_map({k: v.mean("time") for k,v in ds_rain.items()}, "Annual Mean Rainfall (2021-2024)", "1_annual_mean_spatial.png", np.arange(0,16,1), "YlGnBu", "mm/day")
    plot_multi_map({k: sel_jjas(v).mean("time") for k,v in ds_rain.items()}, "JJAS Mean Rainfall (2021-2024)", "2_jjas_mean_spatial.png", np.arange(0,16,1), "YlGnBu", "mm/day")

if ds_t2m:
    plot_multi_map({k: sel_jjas(v).mean("time") for k,v in ds_t2m.items()}, "JJAS Mean 2m Temperature (2021-2024)", "3_t2m_jjas_spatial.png", np.arange(270,310,2), "coolwarm", "K", extend="both")

# --- 2. ZONAL / MERIDIONAL PROFILES ---
if ds_rain and not done("4_jjas_meridional_profile.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, da in ds_rain.items():
        with ProgressBar():
            zonal = sel_jjas(da).mean(["time", "lon"]).compute()
        ax.plot(zonal.lat, zonal, color=COLORS.get(lbl, "k"), linestyle=LSTYLES.get(lbl, "-"), label=lbl, lw=2)
    ax.axvline(0, color="k", ls="--"); ax.set_xlim(-60, 60); ax.set_title("JJAS Meridional Variation of Rainfall")
    ax.set_xlabel("Latitude"); ax.set_ylabel("Rainfall (mm/day)"); ax.legend(); ax.grid(True, ls="--")
    plt.savefig(f"{PLOT_DIR}/4_jjas_meridional_profile.png"); plt.close()

if ds_rain and not done("5_annual_zonal_profile.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, da in ds_rain.items():
        with ProgressBar():
            zonal = da.mean(["time", "lon"]).compute()
        ax.plot(zonal.lat, zonal, color=COLORS.get(lbl, "k"), linestyle=LSTYLES.get(lbl, "-"), label=lbl, lw=2)
    ax.axvline(0, color="k", ls="--"); ax.set_xlim(-90, 90); ax.set_title("Annual Zonal Mean Profile of Rainfall")
    ax.set_xlabel("Latitude"); ax.set_ylabel("Rainfall (mm/day)"); ax.legend(); ax.grid(True, ls="--")
    plt.savefig(f"{PLOT_DIR}/5_annual_zonal_profile.png"); plt.close()

# --- 3. ANNUAL CYCLES ---
def plot_annual_cycle(data_dict, title, fname, ylabel, lat_slice=None, lon_slice=None):
    if done(fname): return
    fig, ax = plt.subplots(figsize=(9, 5))
    for lbl, da in data_dict.items():
        sub = da.sel(lat=lat_slice, lon=lon_slice) if lat_slice and lon_slice else (da.sel(lat=lat_slice) if lat_slice else da)
        with ProgressBar():
            clim = sub.groupby("time.month").mean("time").compute()
        w = np.cos(np.deg2rad(clim.lat))
        curve = clim.weighted(w).mean(["lat", "lon"])
        ax.plot(range(1, 13), curve, color=COLORS.get(lbl, "k"), linestyle=LSTYLES.get(lbl, "-"), marker=MARKERS.get(lbl, "o"), label=lbl, lw=2)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS); ax.set_title(title); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True, ls="--"); plt.savefig(f"{PLOT_DIR}/{fname}"); plt.close()

if ds_rain:
    plot_annual_cycle(ds_rain, "Global Mean Annual Cycle (Rainfall)", "6a_annual_cycle_global_rain.png", "mm/day")
    plot_annual_cycle(ds_rain, "Tropical Belt (30S-30N) Annual Cycle (Rain)", "6b_annual_cycle_tropics_rain.png", "mm/day", slice(-30,30))
    plot_annual_cycle(ds_rain, "Indian Region (5N-35N, 65E-95E) Annual Cycle", "6c_annual_cycle_india_rain.png", "mm/day", slice(5,35), slice(65,95))

if ds_t2m:
    plot_annual_cycle(ds_t2m, "Regional (India) Annual Cycle 2m Temp", "6d_annual_cycle_india_t2m.png", "K", slice(5,35), slice(65,95))

# --- 4. HOVMOLLER DIAGRAM ---
if ds_rain and not done("7_hovmoller_lat_month.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Hovmöller Diagram: Zonal Mean Annual Cycle", fontweight="bold")
    for idx, (lbl, da) in enumerate(list(ds_rain.items())[:4]): # Max 4 panels
        ax = axes[idx]
        with ProgressBar():
            clim = da.groupby("time.month").mean("time").mean("lon").compute()
        M, L = np.meshgrid(range(1,13), clim.lat)
        cf = ax.contourf(M, L, clim.values.T, levels=np.arange(0,16,1), cmap="YlGnBu", extend="max")
        ax.axhline(0, color="k", ls="--")
        ax.set_title(lbl); ax.set_xlabel("Month"); ax.set_ylabel("Latitude"); ax.set_ylim(-60, 60)
        ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS, rotation=45)
        plt.colorbar(cf, ax=ax, label="mm/day")
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/7_hovmoller_lat_month.png"); plt.close()

# --- 5. BIAS & RMSE ---
if ds_rain and "IMERG" in ds_rain:
    bias_ann = {k: v.mean("time") - ds_rain["IMERG"].mean("time") for k,v in ds_rain.items() if k != "IMERG"}
    plot_multi_map(bias_ann, "Annual Mean Bias vs IMERG", "8a_bias_annual_rain.png", np.arange(-5, 5.5, 0.5), "RdBu_r", "Bias (mm/day)", extend="both")
    
    bias_jjas = {k: sel_jjas(v).mean("time") - sel_jjas(ds_rain["IMERG"]).mean("time") for k,v in ds_rain.items() if k != "IMERG"}
    plot_multi_map(bias_jjas, "JJAS Mean Bias vs IMERG", "8b_bias_jjas_rain.png", np.arange(-5, 5.5, 0.5), "RdBu_r", "Bias (mm/day)", extend="both")

    rmse_maps = {}
    for k, v in ds_rain.items():
        if k != "IMERG":
            o, m = xr.align(ds_rain["IMERG"], v, join="inner")
            rmse_maps[k] = np.sqrt(((m - o)**2).mean("time"))
    plot_multi_map(rmse_maps, "RMSE vs IMERG (Rainfall)", "9_rmse_spatial_rain.png", np.arange(0, 15, 1), "Reds", "RMSE (mm/day)")

# --- 6. VARIABILITY & EXTREMES ---
if ds_rain:
    plot_multi_map({k: v.std("time") for k,v in ds_rain.items()}, "Total Daily Rainfall SD", "10_total_sd_rain.png", np.arange(0, 20, 1), "OrRd", "SD (mm/day)")

if ds_rain and not done("11_rainfall_pdf.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (reg, lat_sl) in zip(axes, [("Global", slice(-90, 90)), ("Tropics 30S-30N", slice(-30, 30))]):
        for lbl, da in ds_rain.items():
            with ProgressBar():
                vals = da.sel(lat=lat_sl).values.flatten()
            vals = vals[vals > 0.1] # Wet days
            ax.hist(vals, bins=np.arange(0, 100, 1), density=True, histtype="step", color=COLORS.get(lbl,"k"), ls=LSTYLES.get(lbl,"-"), lw=2, label=lbl)
        ax.set_yscale("log"); ax.set_xlabel("Daily Rainfall (mm/day)"); ax.set_ylabel("Probability Density")
        ax.set_title(f"PDF - {reg} (Wet Days > 0.1 mm)"); ax.legend(); ax.grid(True, ls="--")
    plt.savefig(f"{PLOT_DIR}/11_rainfall_pdf.png"); plt.close()

# --- 7. FILTERING (Synoptic & ISO) ---
if ds_rain:
    var_tot, var_syn, var_iso = {}, {}, {}
    for lbl, da in ds_rain.items():
        lkey = lbl.replace(" ", "_")
        if nc_exists(f"vt_{lkey}") and nc_exists(f"vs_{lkey}") and nc_exists(f"vi_{lkey}"):
            var_tot[lbl] = nc_load(f"vt_{lkey}"); var_syn[lbl] = nc_load(f"vs_{lkey}"); var_iso[lbl] = nc_load(f"vi_{lkey}")
        else:
            print(f"    Bandpass filtering {lbl} ... (Heavy!)")
            with ProgressBar():
                d = da.compute()
            vt = d.var("time").compute(); nc_save(vt, f"vt_{lkey}"); var_tot[lbl] = vt
            syn = bandpass_butter(d, 2, 10); vs = syn.var("time").compute(); nc_save(vs, f"vs_{lkey}"); var_syn[lbl] = vs
            iso = bandpass_butter(d, 10, 90); vi = iso.var("time").compute(); nc_save(vi, f"vi_{lkey}"); var_iso[lbl] = vi

    plot_multi_map({k: var_syn[k]/var_tot[k] for k in var_tot}, "Synoptic (2-10d) / Total Variance", "12a_ratio_synoptic_tot.png", np.arange(0,1.1,0.1), "RdYlBu_r", "Ratio", "neither")
    plot_multi_map({k: var_iso[k]/var_tot[k] for k in var_tot}, "ISO (10-90d) / Total Variance", "12b_ratio_iso_tot.png", np.arange(0,1.1,0.1), "RdYlBu_r", "Ratio", "neither")
    plot_multi_map({k: np.sqrt(var_syn[k]) for k in var_syn}, "Synoptic (2-10d) Filtered SD", "13a_synoptic_sd.png", np.arange(0,10,0.5), "OrRd", "SD")
    plot_multi_map({k: np.sqrt(var_iso[k]) for k in var_iso}, "ISO (10-90d) Filtered SD", "13b_iso_sd.png", np.arange(0,10,0.5), "OrRd", "SD")

# --- 8. POWER SPECTRA ---
if ds_rain and not done("14_power_spectrum_global.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl, da in ds_rain.items():
        print(f"    Spectra {lbl} ...")
        with ProgressBar(): d = da.compute()
        w = np.cos(np.deg2rad(d.lat))
        ts = d.weighted(w).mean(["lat", "lon"]).values
        f, p = welch(signal.detrend(ts - ts.mean()), fs=1.0, nperseg=365, noverlap=182, window="hann")
        ax.loglog(f[1:], p[1:], color=COLORS.get(lbl,"k"), ls=LSTYLES.get(lbl,"-"), label=lbl, lw=2)
    ax.set_title("Global Area-Weighted Power Spectrum (Welch)"); ax.set_xlabel("Frequency (cpd)"); ax.set_ylabel("PSD")
    for pd in [90, 30, 10, 5, 2]: ax.axvline(1/pd, color="gray", ls=":"); ax.text(1/pd, ax.get_ylim()[0]*2, f"{pd}d", rotation=90)
    ax.legend(); ax.grid(True, which="both", ls="--"); plt.savefig(f"{PLOT_DIR}/14_power_spectrum_global.png"); plt.close()

# --- 9. WAVENUMBER FREQUENCY ---
if ds_rain and not done("15_wk_spectra_eq.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Wavenumber-Frequency Spectra (JJAS, 5S-5N)", fontweight="bold")
    for idx, (lbl, da) in enumerate(list(ds_rain.items())[:4]):
        ax = axes[idx]
        print(f"    WK Spectra {lbl} ...")
        with ProgressBar():
            eq = sel_jjas(da).sel(lat=slice(-5,5)).mean("lat").compute().values
        eq_dt = signal.detrend(eq, axis=0)
        Nt, Nlon = eq.shape
        power2d = np.fft.fftshift((np.abs(np.fft.fft2(eq_dt))**2)/(Nt*Nlon))
        f_sh = np.fft.fftshift(np.fft.fftfreq(Nt, d=1.0))
        w_sh = np.fft.fftshift(np.fft.fftfreq(Nlon, d=1.0)*Nlon)
        
        f_mask = (np.abs(f_sh) >= 1/90) & (np.abs(f_sh) <= 0.5); w_mask = (w_sh >= -20) & (w_sh <= 20)
        P = power2d[np.ix_(f_mask, w_mask)]; W, F = np.meshgrid(w_sh[w_mask], f_sh[f_mask])
        
        cf = ax.contourf(W, F, np.log10(P + 1e-12), levels=20, cmap="Spectral_r")
        ax.set_title(lbl); ax.set_xlabel("Zonal Wavenumber"); ax.set_ylabel("Frequency (cpd)")
        ax.axvline(0, color="k", ls="--"); ax.axhline(0, color="k", ls="--")
        plt.colorbar(cf, ax=ax, label="log10(Power)")
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/15_wk_spectra_eq.png"); plt.close()

# --- 10. HADLEY CELL ---
if ds_v3d and not done("16_hadley_cell.png"):
    fig, axes = plt.subplots(1, min(len(ds_v3d), 3), figsize=(18, 6))
    if len(ds_v3d) == 1: axes = [axes]
    for idx, (lbl, da) in enumerate(list(ds_v3d.items())[:3]):
        ax = axes[idx]
        print(f"    Hadley Cell {lbl} ...")
        with ProgressBar():
            v_mean = sel_jjas(da).mean(["time", "lon"]).compute()
        # Streamfunction integration
        a = 6.371e6; g = 9.81
        dp = np.gradient(v_mean.level.values * 100) # Assuming level is hPa
        psi = np.zeros_like(v_mean.values)
        cos_lat = np.cos(np.deg2rad(v_mean.lat.values))
        for k in range(1, len(v_mean.level)):
            psi[k, :] = psi[k-1, :] + v_mean.values[k, :] * dp[k] * 2 * np.pi * a * cos_lat / g
        psi_scale = psi / 1e10
        
        cf = ax.contourf(v_mean.lat, v_mean.level, psi_scale, levels=np.arange(-10, 11, 1), cmap="RdBu_r", extend="both")
        ax.contour(v_mean.lat, v_mean.level, psi_scale, levels=np.arange(-10, 11, 2), colors="k", linewidths=0.5)
        ax.set_ylim(1000, 100); ax.set_xlim(-40, 40); ax.set_title(f"Mass Streamfunction - {lbl}")
        ax.set_xlabel("Latitude"); ax.set_ylabel("Pressure (hPa)"); ax.axvline(0, color="k", ls="--")
        plt.colorbar(cf, ax=ax, label="10^10 kg/s")
    plt.tight_layout(); plt.savefig(f"{PLOT_DIR}/16_hadley_cell.png"); plt.close()

# --- 11. SKILL METRICS TABLE ---
tex_file = f"{PLOT_DIR}/17_statistics_table.tex"
if not os.path.exists(tex_file):
    print("    Generating LaTeX Skill Table ...")
    tex = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{llccc}\n\\hline\n"
    tex += "\\textbf{Variable} & \\textbf{Model} & \\textbf{Bias} & \\textbf{RMSE} & \\textbf{Correlation} \\\\\n\\hline\n"
    
    def add_metric_rows(obs, mods, var_name):
        res = ""
        res += f"\\multirow{{{len(mods)}}}{{*}}{{{var_name}}} "
        for i, (m, da) in enumerate(mods.items()):
            o, d = xr.align(obs, da, join="inner")
            with ProgressBar():
                b = (d - o).mean().compute().item()
                r = np.sqrt(((d - o)**2).mean()).compute().item()
                c = xr.corr(o, d, dim="time").mean().compute().item()
            if i > 0: res += " & "
            res += f"& {m} & {b:.2f} & {r:.2f} & {c:.2f} \\\\\n"
        res += "\\hline\n"
        return res

    if ds_rain and "IMERG" in ds_rain:
        tex += add_metric_rows(sel_jjas(ds_rain["IMERG"]), {k:sel_jjas(v) for k,v in ds_rain.items() if k!="IMERG"}, "Rainfall (JJAS)")
    if ds_t2m and "ERA5" in ds_t2m:
        tex += add_metric_rows(ds_t2m["ERA5"], {k:v for k,v in ds_t2m.items() if k!="ERA5"}, "T2M (Annual)")

    tex += "\\end{tabular}\n\\caption{Skill metrics.}\n\\end{table}"
    with open(tex_file, "w") as f: f.write(tex)

print("\n" + "="*60 + "\n✅ ALL TASKS COMPLETE!\n" + "="*60)
