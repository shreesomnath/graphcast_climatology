"""
Q1/Q2 Analysis Pipeline — GraphCast Master (Publication Ready)
==============================================================
Upgrades: 
- Fixed Cp Unit Conversion (K/day)
- Fixed Year-Boundary Spikes (Continuous Gradient)
- Fixed Seasonal Convolution Edge Distortion (mode="valid")
- Fixed Regional Wrap-Around Boundary Artifacts (isel slice 1,-1)
- Headless Environment Safe
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import time
import warnings
from datetime import datetime
from dask.diagnostics import ProgressBar

# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION QUALITY SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'legend.title_fontsize': 11,
    'legend.frameon': False, 'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 2.0, 'lines.markersize': 6,
    'axes.grid': False, 'grid.alpha': 0.3, 'grid.linestyle': '--', 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.transparent': False
})

VAR_DIR  = "/media/airlab/ROCSTOR/graphcast/Climatology_final"
PLOT_DIR = "Plots/Analysis"
CACHE_DIR= "Plots/Analysis/Cache"
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def log_msg(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def plot_done(fname_base):
    if os.path.exists(f"{PLOT_DIR}/{fname_base}.png") and os.path.exists(f"{PLOT_DIR}/{fname_base}.pdf"):
        log_msg(f"  ⏭️  Skipping (exists): {fname_base}")
        return True
    return False

def nc_done(name): return os.path.exists(f"{CACHE_DIR}/{name}.nc")
def nc_save(da, name): da.to_dataset(name="data").to_netcdf(f"{CACHE_DIR}/{name}.nc")
def nc_load(name): return xr.open_dataset(f"{CACHE_DIR}/{name}.nc")["data"]

def save_fig(fig, fname_base):
    png_path, pdf_path = f"{PLOT_DIR}/{fname_base}.png", f"{PLOT_DIR}/{fname_base}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    
    # Try to display locally, silently pass if running headless (Apptainer/HPC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: plt.show()
        except: pass
        
    plt.close(fig)
    log_msg(f"  ✅ Saved → {png_path} & .pdf")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
Cp = 1004.0; R = 287.0; k = R / Cp; L = 2.5e6; p0 = 1000.0; g = 9.81; a = 6.371e6

Q1Q2_PLEVS = np.array([100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=float)
VINTEG_PLEVS = np.array([150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=float)

LEADS   = ["24hr", "48hr", "72hr"] 
COLORS  = {"24hr":"royalblue", "48hr":"darkorange", "72hr":"firebrick"}
LSTYLES = {"24hr":"-", "48hr":"--", "72hr":"-."}
MARKERS = {"24hr":"o", "48hr":"s", "72hr":"^"}
MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun", "Jul","Aug","Sep","Oct","Nov","Dec"]

ISM_LAT = slice(5,  35); ISM_LON = slice(60, 100); MERID_LON = slice(65, 95)
PROJ_MAP = ccrs.PlateCarree(central_longitude=80); PROJ_DATA = ccrs.PlateCarree()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_var(varname, lead):
    if "10m" in varname or "2m" in varname:
        raise ValueError(f"CRITICAL ERROR: Attempting to load surface variable '{varname}' for a 3D calculation.")
    path = f"{VAR_DIR}/daily_{varname}_{lead}_2021_2024_clean.zarr"
    if not os.path.exists(path): raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_zarr(path, consolidated=False)[varname]

def sel_jjas(da): return da.sel(time=da.time.dt.month.isin([6, 7, 8, 9]))
def sel_mjjaso(da): return da.sel(time=da.time.dt.month.isin([5, 6, 7, 8, 9, 10]))
def area_weights(da): return np.cos(np.deg2rad(da.lat))

def to_theta(T_da, plevs):
    p_da = xr.DataArray(plevs, dims=["level"], coords={"level": T_da.level.sel(level=list(plevs)).values})
    return T_da * (p0 / p_da) ** k

def add_map_features(ax, extent=None):
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, linestyle=":", edgecolor='dimgray', alpha=0.7)
    if extent: ax.set_extent(extent, crs=PROJ_DATA)
    else: ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3, linestyle="--")
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {"size": 10, "color": "black"}; gl.ylabel_style = {"size": 10, "color": "black"}

def build_gradients(T_full, q_full, u_full, v_full, om_full, plevs):
    theta = T_full * (p0 / xr.DataArray(plevs, dims=["level"], coords={"level": T_full.level.values})) ** k
    dp_da = xr.DataArray(np.gradient(plevs * 100.0), dims=["level"], coords={"level": T_full.level.values})
    dx_da = xr.DataArray(a * np.cos(np.deg2rad(T_full.lat.values)) * np.deg2rad(0.25), dims=["lat"], coords={"lat": T_full.lat})
    dy = a * np.deg2rad(0.25)

    dtheta_dt = theta.diff("time") / 86400.0
    dq_dt     = q_full.diff("time") / 86400.0

    tm = theta.isel(time=slice(1, None)); qm = q_full.isel(time=slice(1, None))
    um = u_full.isel(time=slice(1, None)); vm = v_full.isel(time=slice(1, None)); om_m = om_full.isel(time=slice(1, None))

    def cdx(da): return (da.roll(lon=-1,roll_coords=False) - da.roll(lon=1,roll_coords=False)) / (2*dx_da)
    def cdy(da): return (da.roll(lat=-1,roll_coords=False) - da.roll(lat=1,roll_coords=False)) / (2*dy)
    def cdp(da): return (da.roll(level=-1,roll_coords=False) - da.roll(level=1,roll_coords=False)) / (2*dp_da)

    pfact = (xr.DataArray(plevs, dims=["level"], coords={"level": T_full.level.values}) / p0) ** k
    
    Q1_4d = pfact * (dtheta_dt + um*cdx(tm) + vm*cdy(tm) + om_m*cdp(tm))
    Q2_4d = -(L/Cp) * (dq_dt + um*cdx(qm) + vm*cdy(qm) + om_m*cdp(qm))
    return Q1_4d, Q2_4d

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_time = time.time()
    log_msg("🚀 STARTING CLIMATE DYNAMICS ANALYSIS PIPELINE")
    print("═" * 70)

    def compute_Q1_Q2_profile(lead):
        # Using v3 cache keys to guarantee we don't load data containing edge wrap-around artifacts
        ck_q1, ck_q2 = f"Q1_profile_v3_{lead}", f"Q2_profile_v3_{lead}" 
        if nc_done(ck_q1) and nc_done(ck_q2):
            log_msg(f"  ⏭️  Loading cached Q1/Q2: {lead}")
            return nc_load(ck_q1), nc_load(ck_q2)

        log_msg(f"  Computing Q1/Q2 profile: {lead} ...")
        def load_ism(varname):
            return load_var(varname, lead).sel(lat=ISM_LAT, lon=ISM_LON, level=list(Q1Q2_PLEVS))

        T, q = load_ism("temperature"), load_ism("specific_humidity")
        u, v, om = load_ism("u_component_of_wind"), load_ism("v_component_of_wind"), load_ism("vertical_velocity")

        Q1_4d, Q2_4d = build_gradients(T, q, u, v, om, Q1Q2_PLEVS)
        
        # Slicing (1, -1) to remove wrap-around artifacts from the roll() function
        Q1_jjas = sel_jjas(Q1_4d).isel(lat=slice(1,-1), lon=slice(1,-1))
        Q2_jjas = sel_jjas(Q2_4d).isel(lat=slice(1,-1), lon=slice(1,-1))
        w = area_weights(Q1_jjas)
        
        log_msg("    Triggering heavy disk read and computation for Q1/Q2...")
        with ProgressBar():
            Q1_lev = Q1_jjas.weighted(w).mean(dim=["lat","lon","time"]).compute() * 86400.0
            Q2_lev = Q2_jjas.weighted(w).mean(dim=["lat","lon","time"]).compute() * 86400.0

        nc_save(Q1_lev, ck_q1); nc_save(Q2_lev, ck_q2)
        return Q1_lev, Q2_lev

    # STEP 1: Q1 VERTICAL PROFILE
    log_msg("STEP 1: Q1 Vertical Profile (WAF Fig 13)")
    if not plot_done("1_Q1_vertical_profile"):
        fig, ax = plt.subplots(figsize=(7, 9))
        for lead in LEADS:
            Q1, _ = compute_Q1_Q2_profile(lead)
            ax.plot(Q1.values, Q1Q2_PLEVS, color=COLORS[lead], linestyle=LSTYLES[lead], marker=MARKERS[lead], label=f"GC {lead}")
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_ylim(1000, 100); ax.set_xlabel("Q1 (K day⁻¹)"); ax.set_ylabel("Pressure (hPa)")
        ax.set_title("JJAS Apparent Heat Source Q1\n(5°–35°N, 60°–100°E)")
        ax.legend(); ax.grid(True)
        ax.set_yticks(Q1Q2_PLEVS); ax.set_yticklabels([str(int(p)) for p in Q1Q2_PLEVS])
        save_fig(fig, "1_Q1_vertical_profile")

    # STEP 2: Q2 VERTICAL PROFILE
    log_msg("STEP 2: Q2 Vertical Profile")
    if not plot_done("2_Q2_vertical_profile"):
        fig, ax = plt.subplots(figsize=(7, 9))
        for lead in LEADS:
            _, Q2 = compute_Q1_Q2_profile(lead)
            ax.plot(Q2.values, Q1Q2_PLEVS, color=COLORS[lead], linestyle=LSTYLES[lead], marker=MARKERS[lead], label=f"GC {lead}")
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_ylim(1000, 100); ax.set_xlabel("Q2 (K day⁻¹)"); ax.set_ylabel("Pressure (hPa)")
        ax.set_title("JJAS Moisture Sink Q2\n(5°–35°N, 60°–100°E)")
        ax.legend(); ax.grid(True)
        ax.set_yticks(Q1Q2_PLEVS); ax.set_yticklabels([str(int(p)) for p in Q1Q2_PLEVS])
        save_fig(fig, "2_Q2_vertical_profile")

    # STEP 3: Q1 + Q2 COMBINED
    log_msg("STEP 3: Q1 & Q2 Combined")
    if not plot_done("3_Q1_Q2_combined"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 9), sharey=True)
        fig.suptitle("JJAS Q1 (solid) and Q2 (dashed)\n5°–35°N, 60°–100°E, 100–1000 hPa", fontweight="bold")
        for ax, lead in zip(axes, LEADS):
            Q1, Q2 = compute_Q1_Q2_profile(lead)
            ax.plot(Q1.values, Q1Q2_PLEVS, color=COLORS[lead], linestyle="-", label="Q1 (heat source)")
            ax.plot(Q2.values, Q1Q2_PLEVS, color=COLORS[lead], linestyle="--", label="Q2 (moisture sink)")
            ax.axvline(0, color="black", linewidth=1.0, linestyle=":")
            ax.set_ylim(1000, 100); ax.set_xlabel("K day⁻¹"); ax.set_title(f"GC {lead}")
            ax.legend(); ax.grid(True); ax.set_yticks(Q1Q2_PLEVS); ax.set_yticklabels([str(int(p)) for p in Q1Q2_PLEVS])
        axes[0].set_ylabel("Pressure (hPa)")
        save_fig(fig, "3_Q1_Q2_combined")

    # STEP 4: Q1/Q2 SEASONAL EVOLUTION
    log_msg("STEP 4: Q1/Q2 Seasonal Evolution (WAF Fig 14)")
    def compute_Q1_Q2_seasonal(lead):
        ck_q1, ck_q2 = f"Q1_seasonal_v3_{lead}", f"Q2_seasonal_v3_{lead}"
        if nc_done(ck_q1) and nc_done(ck_q2): return nc_load(ck_q1), nc_load(ck_q2)

        log_msg(f"  Computing Q1/Q2 seasonal: {lead} ...")
        def load_ism_full(varname):
            return load_var(varname, lead).sel(lat=ISM_LAT, lon=ISM_LON, level=list(VINTEG_PLEVS))

        T, q = load_ism_full("temperature"), load_ism_full("specific_humidity")
        u, v, om = load_ism_full("u_component_of_wind"), load_ism_full("v_component_of_wind"), load_ism_full("vertical_velocity")

        Q1_4d, Q2_4d = build_gradients(T, q, u, v, om, VINTEG_PLEVS)
        
        # Slicing (1, -1) to remove wrap-around artifacts
        Q1_mjjaso = sel_mjjaso(Q1_4d).isel(lat=slice(1,-1), lon=slice(1,-1))
        Q2_mjjaso = sel_mjjaso(Q2_4d).isel(lat=slice(1,-1), lon=slice(1,-1))

        dp_da = xr.DataArray(np.gradient(VINTEG_PLEVS * 100.0), dims=["level"], coords={"level": Q1_mjjaso.level.values})
        Q1_col = (Q1_mjjaso * dp_da).sum("level") / dp_da.sum()
        Q2_col = (Q2_mjjaso * dp_da).sum("level") / dp_da.sum()
        
        w = area_weights(Q1_col)
        log_msg("    Triggering heavy disk read and computation for Seasonal Data...")
        with ProgressBar():
            Q1_ts = Q1_col.weighted(w).mean(dim=["lat","lon"]).compute()
            Q2_ts = Q2_col.weighted(w).mean(dim=["lat","lon"]).compute()

        Q1_clim = Q1_ts.groupby("time.dayofyear").mean("time") * 86400.0
        Q2_clim = Q2_ts.groupby("time.dayofyear").mean("time") * 86400.0
        nc_save(Q1_clim, ck_q1); nc_save(Q2_clim, ck_q2)
        return Q1_clim, Q2_clim

    if not plot_done("4_Q1_Q2_seasonal"):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        fig.suptitle("Seasonal Evolution of Q1 (solid) and Q2 (dashed) — May to October\nVertically integrated 150–1000 hPa | 5°–35°N, 60°–100°E")
        for ax, lead in zip(axes, LEADS):
            Q1_clim, Q2_clim = compute_Q1_Q2_seasonal(lead)
            
            kernel  = np.ones(5) / 5
            Q1_sm   = np.convolve(Q1_clim.values, kernel, mode="valid")
            Q2_sm   = np.convolve(Q2_clim.values, kernel, mode="valid")
            days_sm = Q1_clim.dayofyear.values[2:-2]
            
            ax.plot(days_sm, Q1_sm, color=COLORS[lead], linestyle="-", label="Q1")
            ax.plot(days_sm, Q2_sm, color=COLORS[lead], linestyle="--", label="Q2")
            ax.axhline(0, color="black", linestyle=":"); ax.axvline(153, color="gray", linestyle="--")
            ax.set_title(f"GC {lead}"); ax.set_xlabel("Day of Year (May–Oct)")
            ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
        axes[0].set_ylabel("K day⁻¹")
        save_fig(fig, "4_Q1_Q2_seasonal")

    # STEP 5: MERIDIONAL PROFILES
    log_msg("STEP 5: Meridional Profiles (Bidyut Fig 8)")
    def compute_meridional_diag(lead):
        ck = f"merid_diag_v3_{lead}"
        if nc_done(ck):
            ds = xr.open_dataset(f"{CACHE_DIR}/{ck}.nc")
            return {v: ds[v] for v in ds.data_vars}

        log_msg(f"  Computing meridional diagnostics: {lead} ...")
        log_msg("    Calculating Wind Shear...")
        u = load_var("u_component_of_wind", lead).sel(lon=MERID_LON)
        with ProgressBar():
            u200 = sel_jjas(u).sel(level=200).mean(dim=["time","lon"]).compute()
            u850 = sel_jjas(u).sel(level=850).mean(dim=["time","lon"]).compute()
        shear = u200 - u850

        log_msg("    Calculating Specific Humidity...")
        q = load_var("specific_humidity", lead).sel(lon=MERID_LON)
        with ProgressBar():
            q_ll = (sel_jjas(q).sel(level=[850, 925, 1000]).mean(dim="level").mean(dim=["time","lon"]).compute()) * 1000.0

        log_msg("    Calculating Theta E...")
        T, qv = load_var("temperature", lead).sel(lon=MERID_LON), load_var("specific_humidity", lead).sel(lon=MERID_LON)
        T_ll, qv_ll = sel_jjas(T).sel(level=[850, 925, 1000]), sel_jjas(qv).sel(level=[850, 925, 1000])
        theta_ll = to_theta(T_ll, np.array([850, 925, 1000]))
        with ProgressBar():
            theta_e = (theta_ll * np.exp(L * qv_ll / (Cp * T_ll))).mean(dim="level").mean(dim=["time","lon"]).compute()

        ds_out = xr.Dataset({"wind_shear": shear, "spec_humidity_ll": q_ll, "theta_e": theta_e})
        ds_out.to_netcdf(f"{CACHE_DIR}/{ck}.nc")
        return {"wind_shear": shear, "spec_humidity_ll": q_ll, "theta_e": theta_e}

    if not plot_done("5_meridional_fig8"):
        fig, axes = plt.subplots(1, 4, figsize=(22, 8))
        fig.suptitle("Climatological JJAS Meridional Profiles (65°–95°E average)", fontweight="bold")
        
        for lead in LEADS:
            diag = compute_meridional_diag(lead)
            lats = diag["wind_shear"].lat.values
            kw = dict(color=COLORS[lead], linestyle=LSTYLES[lead], marker=MARKERS[lead], markevery=10, label=f"GC {lead}")
            axes[0].plot(lats, diag["wind_shear"].values, **kw)
            axes[1].plot(lats, diag["spec_humidity_ll"].values, **kw)
            axes[2].plot(lats, diag["theta_e"].values, **kw)
            Q1, _ = compute_Q1_Q2_profile(lead)
            axes[3].plot(Q1.values, Q1Q2_PLEVS, color=COLORS[lead], linestyle=LSTYLES[lead], label=f"GC {lead}")

        titles = ["(a) Easterly Wind Shear\nU200−U850 (m/s)", "(b) Low-Level Specific Humidity\n850/925/1000 hPa (g/kg)",
                  "(c) Equivalent Potential Temp.\nθe at 850/925/1000 hPa (K)", "(d) Q1 Vertical Profile\n0–30°N, 60–100°E (K/day)"]
        ylabels = ["Wind Shear (m/s)", "Specific Humidity (g/kg)", "θe (K)", "Pressure (hPa)"]

        for i, ax in enumerate(axes[:3]):
            ax.set_title(titles[i]); ax.set_xlabel("Latitude"); ax.set_ylabel(ylabels[i]); ax.set_xlim(-20, 40)
            ax.axvline(0, color="gray", linewidth=1.0, linestyle="--"); ax.axvline(23.5, color="gray", linewidth=1.0, linestyle=":")
            ax.legend(); ax.grid(True)
        axes[3].set_title(titles[3]); axes[3].set_xlabel("Q1 (K day⁻¹)"); axes[3].set_ylabel("Pressure (hPa)")
        axes[3].set_ylim(1000, 100); axes[3].axvline(0, color="black", linewidth=1.0, linestyle="--")
        axes[3].set_yticks(Q1Q2_PLEVS); axes[3].set_yticklabels([str(int(p)) for p in Q1Q2_PLEVS]); axes[3].legend(); axes[3].grid(True)
        save_fig(fig, "5_meridional_fig8")

    # STEP 6: DTT ANNUAL CYCLE
    log_msg("STEP 6: DTT Annual Cycle (Bidyut Fig 6)")
    def compute_DTT(lead):
        ck = f"DTT_{lead}"
        if nc_done(ck): return nc_load(ck)
        log_msg(f"  Computing DTT: {lead} ...")
        T = load_var("temperature", lead).sel(level=[200, 250, 300, 400, 500, 600])

        def tt_box(lat_sl, lon_sl):
            da_b = T.sel(lat=lat_sl, lon=lon_sl)
            return da_b.weighted(np.cos(np.deg2rad(da_b.lat))).mean(dim=["level","lat","lon"]).groupby("time.month").mean("time")

        log_msg("    Triggering heavy disk read and computation for DTT...")
        with ProgressBar(): dtt = (tt_box(slice(5,35), slice(40,100)) - tt_box(slice(-15,5), slice(40,100))).compute()
        nc_save(dtt, ck); return dtt

    if not plot_done("6_DTT_annual_cycle"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for lead in LEADS:
            dtt = compute_DTT(lead)
            ax.plot(range(1,13), dtt.values, color=COLORS[lead], linestyle=LSTYLES[lead], marker=MARKERS[lead], label=f"GC {lead}")
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTHS); ax.set_xlabel("Month"); ax.set_ylabel("DTT (K)")
        ax.set_title("Annual Cycle of Tropospheric Temperature Gradient\nDTT = TT(5°–35°N) − TT(15°S–5°N) | 200–600 hPa")
        ax.legend(); ax.grid(True); ax.set_xlim(1, 12)
        save_fig(fig, "6_DTT_annual_cycle")

    # STEP 7: TROPOSPHERIC TEMPERATURE SPATIAL MAP
    log_msg("STEP 7: TT JJAS Spatial Map (Bidyut Fig 7)")
    def compute_TT_map(lead):
        ck = f"TT_map_{lead}"
        if nc_done(ck): return nc_load(ck)
        log_msg(f"  Computing TT map: {lead} ...")
        T = load_var("temperature", lead).sel(level=[200, 250, 300, 400, 500, 600])
        with ProgressBar(): tt = sel_jjas(T).mean(dim=["level","time"]).compute()
        nc_save(tt, ck); return tt

    if not plot_done("7_TT_jjas_spatial"):
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={"projection": PROJ_MAP})
        fig.suptitle("JJAS Mean Tropospheric Temperature TT (200–600 hPa)", fontweight="bold")
        tt_ref = compute_TT_map("24hr")

        for ax, lead in zip(axes, LEADS):
            tt = compute_TT_map(lead)
            if lead == "24hr":
                data, title, cmap, levels = tt, "GC 24hr — Absolute TT (K)", "RdYlBu_r", np.arange(248, 272, 2)
            else:
                data, title, cmap, levels = tt - tt_ref, f"GC {lead} − GC 24hr (K)", "RdBu_r", np.arange(-2, 2.25, 0.25)
            cf = ax.contourf(tt.lon, tt.lat, data, levels=levels, cmap=cmap, extend="both", transform=PROJ_DATA)
            add_map_features(ax); ax.set_title(title)
            fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06, label="K").ax.tick_params(labelsize=10)
        save_fig(fig, "7_TT_jjas_spatial")

    # STEP 8: AUTOMATED LaTeX TABLE GENERATION
    log_msg("STEP 8: Generate LaTeX Statistics Table")
    tex_path = f"{PLOT_DIR}/statistics_table.tex"
    mean_dtt, mean_tt = [], []

    for lead in LEADS:
        dtt = compute_DTT(lead)
        mean_dtt.append(float(dtt.sel(month=[6, 7, 8, 9]).mean().values))
        tt = compute_TT_map(lead)
        mean_tt.append(float(tt.weighted(np.cos(np.deg2rad(tt.lat))).mean(dim=["lat", "lon"]).values))

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Domain-Averaged JJAS Climatological Statistics across Forecast Leads}\n")
        f.write("\\label{tab:gc_stats}\n\\begin{tabular}{lccc}\n\\hline\\hline\n")
        f.write("\\textbf{Statistic} & \\textbf{GC 24hr} & \\textbf{GC 48hr} & \\textbf{GC 72hr} \\\\\n\\hline\n")
        f.write(f"Mean DTT (K) & {mean_dtt[0]:.2f} & {mean_dtt[1]:.2f} & {mean_dtt[2]:.2f} \\\\\n")
        f.write(f"Mean TT (200--600 hPa) (K) & {mean_tt[0]:.2f} & {mean_tt[1]:.2f} & {mean_tt[2]:.2f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    log_msg(f"  ✅ Saved LaTeX table → {tex_path}")

    # FINAL LOGGING
    elapsed = (time.time() - start_time) / 60
    print("\n" + "═"*70)
    log_msg(f"🎉 PIPELINE COMPLETE! Total Execution Time: {elapsed:.2f} minutes")
    print("═"*70)