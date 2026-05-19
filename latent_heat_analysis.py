"""
Q1/Q2 Analysis Pipeline — GraphCast vs ERA5
=======================================================================
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
# PUBLICATION STYLE 
# ══════════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            10,
    "axes.labelsize":       10,
    "axes.titlesize":       11,
    "axes.titleweight":     "bold",
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      7.5,
    "legend.title_fontsize":8,
    "legend.borderpad":     0.3,
    "legend.labelspacing":  0.3,
    "lines.linewidth":      1.6,
    "lines.markersize":     4.5,
    "lines.markeredgewidth":0.4,
    "axes.linewidth":       1.0,
    "xtick.major.width":    1.0,
    "ytick.major.width":    1.0,
    "xtick.major.size":     5.0,
    "ytick.major.size":     5.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "axes.grid":            False,
    "grid.alpha":           0.30,
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "grid.color":           "#888888",
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "legend.frameon":       False,
})

COLORS  = {"24hr": "#0072B2", "48hr": "#D55E00", "72hr": "#009E73", "ERA5": "#000000"}
LSTYLES = {"24hr": "-",  "48hr": "--", "72hr": "-.", "ERA5": "-"}
MARKERS = {"24hr": "o",  "48hr": "s",  "72hr": "^",  "ERA5": "D"}
LW      = {"24hr": 1.6,  "48hr": 1.6,  "72hr": 1.6,  "ERA5": 1.3}

VAR_DIR   = "/media/airlab/ROCSTOR/graphcast/Climatology_final"
PLOT_DIR  = "Plots/Analysis"
CACHE_DIR = "Plots/Analysis/Cache"
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def log_msg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def plot_done(fname_base):
    if (os.path.exists(f"{PLOT_DIR}/{fname_base}.png") and
            os.path.exists(f"{PLOT_DIR}/{fname_base}.pdf")):
        log_msg(f"  ⏭️  Skipping (exists): {fname_base}")
        return True
    return False

def nc_done(name): return os.path.exists(f"{CACHE_DIR}/{name}.nc")
def nc_save(da, name): da.to_dataset(name="data").to_netcdf(f"{CACHE_DIR}/{name}.nc")
def nc_load(name):    return xr.open_dataset(f"{CACHE_DIR}/{name}.nc")["data"]

def save_fig(fig, fname_base):
    fig.savefig(f"{PLOT_DIR}/{fname_base}.png", dpi=300)
    fig.savefig(f"{PLOT_DIR}/{fname_base}.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: plt.show()
        except: pass
    plt.close(fig)
    log_msg(f"  ✅ Saved → {PLOT_DIR}/{fname_base}.png + .pdf")

def set_pressure_axis(ax, plevs):
    ax.set_ylim(1000, 100) 
    ax.set_yticks(plevs)
    ax.set_yticklabels([str(int(p)) for p in plevs])
    ax.set_ylabel("Pressure (hPa)")

def add_grid(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.30, color="#888888")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
Cp = 1004.0; R = 287.0; k = R / Cp; L = 2.5e6; p0 = 1000.0; g = 9.81; a = 6.371e6

Q1Q2_PLEVS   = np.array([100,150,200,250,300,400,500,600,700,850,925,1000], dtype=float)
VINTEG_PLEVS = np.array([150,200,250,300,400,500,600,700,850,925,1000],     dtype=float)

LEADS  = ["24hr", "48hr", "72hr", "ERA5"]
ISM_LAT  = slice(5,  35)
ISM_LON  = slice(60, 100)
MERID_LON= slice(65, 95)

PROJ_MAP  = ccrs.PlateCarree(central_longitude=80)
PROJ_DATA = ccrs.PlateCarree()

# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_var(varname, lead):
    if "10m" in varname or "2m" in varname:
        raise ValueError(f"Surface variable '{varname}' not valid for 3D Q1/Q2.")
    path = f"{VAR_DIR}/daily_{varname}_{lead}_2021_2024_clean.zarr"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_zarr(path, consolidated=False)[varname]

def sel_jjas(da):   return da.sel(time=da.time.dt.month.isin([6,7,8,9]))
def sel_mjjaso(da): return da.sel(time=da.time.dt.month.isin([5,6,7,8,9,10]))
def area_weights(da): return np.cos(np.deg2rad(da.lat))

def to_theta(T_da, plevs):
    p_da = xr.DataArray(plevs, dims=["level"],
                        coords={"level": T_da.level.sel(level=list(plevs)).values})
    return T_da * (p0 / p_da) ** k

def build_gradients(T_full, q_full, u_full, v_full, om_full, plevs):
    theta = T_full * (p0 / xr.DataArray(
        plevs, dims=["level"], coords={"level": T_full.level.values})) ** k

    dx_da = xr.DataArray(
        a * np.cos(np.deg2rad(T_full.lat.values)) * np.deg2rad(0.25),
        dims=["lat"], coords={"lat": T_full.lat})
    dy = a * np.deg2rad(0.25)

    dtheta_dt = theta.diff("time") / 86400.0
    dq_dt     = q_full.diff("time") / 86400.0

    tm   = theta.isel(time=slice(1, None))
    qm   = q_full.isel(time=slice(1, None))
    um   = u_full.isel(time=slice(1, None))
    vm   = v_full.isel(time=slice(1, None))
    om_m = om_full.isel(time=slice(1, None))

    def cdx(da): return (da.roll(lon=-1, roll_coords=False) -
                          da.roll(lon=1,  roll_coords=False)) / (2 * dx_da)
    def cdy(da): return (da.roll(lat=-1, roll_coords=False) -
                          da.roll(lat=1,  roll_coords=False)) / (2 * dy)
    def cdp(da): return da.chunk({"level": -1}).differentiate("level") / 100.0

    pfact = (xr.DataArray(plevs, dims=["level"],
                          coords={"level": T_full.level.values}) / p0) ** k

    Q1_4d = pfact * (dtheta_dt + um*cdx(tm) + vm*cdy(tm) + om_m*cdp(tm))
    Q2_4d = -(L/Cp) * (dq_dt   + um*cdx(qm) + vm*cdy(qm) + om_m*cdp(qm))
    return Q1_4d, Q2_4d

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_time = time.time()
    log_msg("🚀 STARTING PIPELINE — Clean publication output")
    print("═" * 70)

    def compute_Q1_Q2_profile(lead):
        ck_q1, ck_q2 = f"Q1_profile_v4_{lead}", f"Q2_profile_v4_{lead}"
        if nc_done(ck_q1) and nc_done(ck_q2):
            log_msg(f"  ⏭️  Loading cached Q1/Q2: {lead}")
            return nc_load(ck_q1), nc_load(ck_q2)

        log_msg(f"  Computing Q1/Q2 profile: {lead} ...")
        def load_ism(varname):
            return load_var(varname, lead).sel(
                lat=ISM_LAT, lon=ISM_LON, level=list(Q1Q2_PLEVS))

        T, q = load_ism("temperature"), load_ism("specific_humidity")
        u, v, om = (load_ism("u_component_of_wind"),
                    load_ism("v_component_of_wind"),
                    load_ism("vertical_velocity"))

        Q1_4d, Q2_4d = build_gradients(T, q, u, v, om, Q1Q2_PLEVS)

        Q1_jjas = sel_jjas(Q1_4d).isel(lat=slice(1,-1), lon=slice(1,-1))
        Q2_jjas = sel_jjas(Q2_4d).isel(lat=slice(1,-1), lon=slice(1,-1))
        w = area_weights(Q1_jjas)

        log_msg("    Computing domain-mean Q1/Q2 ...")
        with ProgressBar():
            Q1_lev = Q1_jjas.weighted(w).mean(dim=["lat","lon","time"]).compute() * 86400.0
            Q2_lev = Q2_jjas.weighted(w).mean(dim=["lat","lon","time"]).compute() * 86400.0

        nc_save(Q1_lev, ck_q1); nc_save(Q2_lev, ck_q2)
        return Q1_lev, Q2_lev

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Q1 VERTICAL PROFILE
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 1: Q1 Vertical Profile")
    if not plot_done("1_Q1_vertical_profile"):
        fig, ax = plt.subplots(figsize=(4.0, 5.5), layout="constrained")

        for lead in LEADS:
            Q1, _ = compute_Q1_Q2_profile(lead)
            lbl   = "ERA5" if lead == "ERA5" else f"GC {lead}"
            ax.plot(Q1.values, Q1Q2_PLEVS,
                    color=COLORS[lead], linestyle=LSTYLES[lead],
                    linewidth=LW[lead], marker=MARKERS[lead],
                    markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.5, label=lbl, zorder=3)

        ax.axvline(0, color="#333333", linewidth=1.0, linestyle=":", zorder=2)
        set_pressure_axis(ax, Q1Q2_PLEVS)
        ax.set_xlabel(r"$Q_1$ (K day$^{-1}$)")
        ax.set_title("JJAS Apparent Heat Source $Q_1$", loc="center")
        ax.legend(loc="lower right")
        add_grid(ax)
        save_fig(fig, "1_Q1_vertical_profile")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Q2 VERTICAL PROFILE
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 2: Q2 Vertical Profile")
    if not plot_done("2_Q2_vertical_profile"):
        fig, ax = plt.subplots(figsize=(4.0, 5.5), layout="constrained")

        for lead in LEADS:
            _, Q2 = compute_Q1_Q2_profile(lead)
            lbl   = "ERA5" if lead == "ERA5" else f"GC {lead}"
            ax.plot(Q2.values, Q1Q2_PLEVS,
                    color=COLORS[lead], linestyle=LSTYLES[lead],
                    linewidth=LW[lead], marker=MARKERS[lead],
                    markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.5, label=lbl, zorder=3)

        ax.axvline(0, color="#333333", linewidth=1.0, linestyle=":", zorder=2)
        set_pressure_axis(ax, Q1Q2_PLEVS)
        ax.set_xlabel(r"$Q_2$ (K day$^{-1}$)")
        ax.set_title("JJAS Moisture Sink $Q_2$", loc="center")
        ax.legend(loc="lower right")
        add_grid(ax)
        save_fig(fig, "2_Q2_vertical_profile")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Q1 + Q2 COMBINED (4-panel)
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 3: Q1 & Q2 Combined (4-panel)")
    if not plot_done("3_Q1_Q2_combined"):
        fig, axes = plt.subplots(1, 4, figsize=(8.5, 5.0), sharey=True, layout="constrained")

        for idx, (ax, lead) in enumerate(zip(axes, LEADS)):
            Q1, Q2 = compute_Q1_Q2_profile(lead)
            lbl    = "ERA5" if lead == "ERA5" else f"GC {lead}"

            ax.plot(Q1.values, Q1Q2_PLEVS,
                    color=COLORS[lead], linestyle="-",
                    linewidth=LW[lead], label=r"$Q_1$", zorder=3)
            ax.plot(Q2.values, Q1Q2_PLEVS,
                    color=COLORS[lead], linestyle="--",
                    linewidth=LW[lead], label=r"$Q_2$", zorder=3)
            ax.axvline(0, color="#444444", linewidth=1.0, linestyle=":", zorder=2)

            if idx == 0:
                set_pressure_axis(ax, Q1Q2_PLEVS)
            else:
                ax.tick_params(axis='y', which='both', length=0)

            ax.set_xlabel(r"K day$^{-1}$")
            ax.set_title(lbl, loc="center")
            ax.legend(loc="lower right")
            add_grid(ax)

        fig.suptitle(r"JJAS Climatological $Q_1$ (solid) and $Q_2$ (dashed)", fontsize=13, fontweight="bold")
        save_fig(fig, "3_Q1_Q2_combined")
        
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: SEASONAL EVOLUTION (4-panel) -> ⚠️ UPDATED FOR COLUMN ENERGETICS
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 4: Q1/Q2 Seasonal Evolution")

    def compute_Q1_Q2_seasonal(lead):
        ck_q1, ck_q2 = f"Q1_seasonal_v5_{lead}", f"Q2_seasonal_v5_{lead}"
        if nc_done(ck_q1) and nc_done(ck_q2):
            return nc_load(ck_q1), nc_load(ck_q2)

        log_msg(f"  Computing Q1/Q2 seasonal energetics: {lead} ...")
        def load_ism_full(varname):
            return load_var(varname, lead).sel(
                lat=ISM_LAT, lon=ISM_LON, level=list(VINTEG_PLEVS))

        T, q = load_ism_full("temperature"), load_ism_full("specific_humidity")
        u, v, om = (load_ism_full("u_component_of_wind"),
                    load_ism_full("v_component_of_wind"),
                    load_ism_full("vertical_velocity"))

        Q1_4d, Q2_4d = build_gradients(T, q, u, v, om, VINTEG_PLEVS)

        Q1_m = sel_mjjaso(Q1_4d).isel(lat=slice(1,-1), lon=slice(1,-1))
        Q2_m = sel_mjjaso(Q2_4d).isel(lat=slice(1,-1), lon=slice(1,-1))

        dp_da = xr.DataArray(
            np.gradient(VINTEG_PLEVS * 100.0), dims=["level"],
            coords={"level": Q1_m.level.values})
            
        # ======================================================================
        # FIX: True Column Energetics (W m^-2) following Yanai et al. (1973)
        # 1. Integrate over pressure (summing Q * dp) without normalizing.
        # 2. Multiply by (Cp / g) to convert from [K*Pa/s] to [W m^-2].
        # ======================================================================
        Q1_col = (Q1_m * dp_da).sum("level") * (Cp / g)
        Q2_col = (Q2_m * dp_da).sum("level") * (Cp / g)

        w = area_weights(Q1_col)
        log_msg("    Computing seasonal time series ...")
        with ProgressBar():
            Q1_ts = Q1_col.weighted(w).mean(dim=["lat","lon"]).compute()
            Q2_ts = Q2_col.weighted(w).mean(dim=["lat","lon"]).compute()

        # REMOVED the * 86400.0 scaling. The variables are now in W m^-2.
        Q1_clim = Q1_ts.groupby("time.dayofyear").mean("time") 
        Q2_clim = Q2_ts.groupby("time.dayofyear").mean("time") 
        nc_save(Q1_clim, ck_q1); nc_save(Q2_clim, ck_q2)
        return Q1_clim, Q2_clim

    _MOY_TICKS = [121, 152, 182, 213, 244, 274]   
    _MOY_LBLS  = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]

    if not plot_done("4_Q1_Q2_seasonal"):
        fig, axes = plt.subplots(1, 4, figsize=(8.5, 4.0), sharey=True, layout="constrained")

        for idx, (ax, lead) in enumerate(zip(axes, LEADS)):
            Q1_clim, Q2_clim = compute_Q1_Q2_seasonal(lead)
            lbl = "ERA5" if lead == "ERA5" else f"GC {lead}"

            kernel  = np.ones(5) / 5
            Q1_sm   = np.convolve(Q1_clim.values, kernel, mode="valid")
            Q2_sm   = np.convolve(Q2_clim.values, kernel, mode="valid")
            days_sm = Q1_clim.dayofyear.values[2:-2]

            ax.plot(days_sm, Q1_sm, color=COLORS[lead], linestyle="-",
                    linewidth=LW[lead], label=r"$\langle Q_1 \rangle$", zorder=3)
            ax.plot(days_sm, Q2_sm, color=COLORS[lead], linestyle="--",
                    linewidth=LW[lead], label=r"$\langle Q_2 \rangle$", zorder=3)
            
            ax.axhline(0, color="#444444", linewidth=1.0, linestyle=":", zorder=2)

            ax.set_xlim(121, 274)
            ax.set_xticks(_MOY_TICKS)
            ax.set_xticklabels(_MOY_LBLS, rotation=30, ha="right")
            ax.set_xlabel("Month")
            
            if idx == 0:
                # UPDATED Label to reflect Watt per square meter energetics
                ax.set_ylabel(r"$\langle Q_1 \rangle$, $\langle Q_2 \rangle$ (W m$^{-2}$)")
            else:
                ax.tick_params(axis='y', which='both', length=0)
                
            ax.set_title(lbl, loc="center")
            ax.legend(loc="upper left")
            add_grid(ax)

        fig.suptitle(r"Seasonal Evolution of Vertically Integrated $\langle Q_1 \rangle$ and $\langle Q_2 \rangle$", fontsize=13, fontweight="bold")
        save_fig(fig, "4_Q1_Q2_seasonal")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: MERIDIONAL PROFILES (4-panel)
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 5: Meridional Profiles")

    def compute_meridional_diag(lead):
        ck = f"merid_diag_v4_{lead}"
        if nc_done(ck):
            ds = xr.open_dataset(f"{CACHE_DIR}/{ck}.nc")
            return {v: ds[v] for v in ds.data_vars}

        log_msg(f"  Computing meridional diagnostics: {lead} ...")

        u = load_var("u_component_of_wind", lead).sel(lon=MERID_LON)
        with ProgressBar():
            u200 = sel_jjas(u).sel(level=200).mean(dim=["time","lon"]).compute()
            u850 = sel_jjas(u).sel(level=850).mean(dim=["time","lon"]).compute()
        shear = u200 - u850

        q = load_var("specific_humidity", lead).sel(lon=MERID_LON)
        with ProgressBar():
            q_ll = (sel_jjas(q).sel(level=[850,925,1000])
                    .mean(dim="level").mean(dim=["time","lon"]).compute()) * 1000.0

        T  = load_var("temperature",       lead).sel(lon=MERID_LON)
        qv = load_var("specific_humidity", lead).sel(lon=MERID_LON)
        T_ll  = sel_jjas(T).sel(level=[850,925,1000])
        qv_ll = sel_jjas(qv).sel(level=[850,925,1000])
        theta_ll = to_theta(T_ll, np.array([850,925,1000]))
        with ProgressBar():
            theta_e = (theta_ll * np.exp(L * qv_ll / (Cp * T_ll))
                       ).mean(dim="level").mean(dim=["time","lon"]).compute()

        ds_out = xr.Dataset({"wind_shear":       shear,
                              "spec_humidity_ll": q_ll,
                              "theta_e":          theta_e})
        ds_out.to_netcdf(f"{CACHE_DIR}/{ck}.nc")
        return {"wind_shear": shear, "spec_humidity_ll": q_ll, "theta_e": theta_e}

    if not plot_done("5_meridional_fig8"):
        fig, axes = plt.subplots(1, 4, figsize=(9.5, 4.5), layout="constrained")

        LAT_XLIM = (5, 35)

        for lead in LEADS:
            diag = compute_meridional_diag(lead)
            lats = diag["wind_shear"].lat.values
            lbl  = "ERA5" if lead == "ERA5" else f"GC {lead}"
            kw   = dict(color=COLORS[lead], linestyle=LSTYLES[lead],
                        linewidth=LW[lead], marker=MARKERS[lead],
                        markersize=4, markevery=8,
                        markeredgecolor="white", markeredgewidth=0.3,
                        label=lbl, zorder=3)
            
            axes[0].plot(lats, diag["wind_shear"].values,       **kw)
            axes[1].plot(lats, diag["spec_humidity_ll"].values, **kw)
            axes[2].plot(lats, diag["theta_e"].values,          **kw)

            Q1, _ = compute_Q1_Q2_profile(lead)
            axes[3].plot(Q1.values, Q1Q2_PLEVS,
                         color=COLORS[lead], linestyle=LSTYLES[lead],
                         linewidth=LW[lead], marker=MARKERS[lead],
                         markersize=4, markeredgecolor="white", 
                         markeredgewidth=0.3, label=lbl, zorder=3)

        panel_titles = [
            "Wind Shear ($U_{200}-U_{850}$)",
            "Low-level $q$ (850-1000 hPa)",
            r"$\theta_e$ (850-1000 hPa)",
            "$Q_1$ Profile",
        ]
        ylabels = [
            "m s$^{-1}$",
            "g kg$^{-1}$",
            "K",
            "Pressure (hPa)",
        ]

        for i, ax in enumerate(axes[:3]):
            ax.set_title(panel_titles[i], loc="center")
            ax.set_xlabel("Latitude (°N)")
            ax.set_ylabel(ylabels[i])
            ax.set_xlim(*LAT_XLIM)
            ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
            ax.axvline(23.5, color="#888888", linewidth=1.0, linestyle=":", alpha=0.7)
            ax.legend(loc="best")
            add_grid(ax)

        axes[3].set_title(panel_titles[3], loc="center")
        axes[3].set_xlabel(r"$Q_1$ (K day$^{-1}$)")
        axes[3].axvline(0, color="#333333", linewidth=1.0, linestyle=":", zorder=2)
        set_pressure_axis(axes[3], Q1Q2_PLEVS)
        axes[3].legend(loc="lower right")
        add_grid(axes[3])

        fig.suptitle("JJAS Climatological Meridional Profiles (65°–95°E)", fontsize=13, fontweight="bold")
        save_fig(fig, "5_meridional_fig8")

   # ──────────────────────────────────────────────────────────────────────────
    # STEP 8: LaTeX TABLE (Expanded with Meridional Extrema)
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 8: LaTeX Statistics Table")
    tex_path = f"{PLOT_DIR}/statistics_table.tex"

    rows = {}   
    for lead in LEADS:
        # 1. Load Vertical Q1/Q2 Profile Data
        Q1, Q2 = compute_Q1_Q2_profile(lead)
        q1_mid = float(Q1.sel(level=slice(400, 500)).mean().values)
        q2_mid = float(Q2.sel(level=slice(400, 500)).mean().values)
        q1_peak_lev = float(Q1Q2_PLEVS[int(np.argmax(Q1.values))])
        q2_peak_lev = float(Q1Q2_PLEVS[int(np.argmax(Q2.values))])
        
        # 2. Load Meridional Data to extract physical extremes
        diag = compute_meridional_diag(lead)
        max_easterly_shear = float(diag["wind_shear"].min().values) 
        max_q_ll = float(diag["spec_humidity_ll"].max().values)
        max_theta_e = float(diag["theta_e"].max().values)

        rows[lead] = (q1_mid, q2_mid, q1_peak_lev, q2_peak_lev, 
                      max_easterly_shear, max_q_ll, max_theta_e)

    col_order  = ["ERA5", "24hr", "48hr", "72hr"]
    col_labels = ["ERA5", "GC 24hr", "GC 48hr", "GC 72hr"]

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Domain-averaged JJAS thermodynamic and dynamic statistics. "
                "Peak values for wind shear, specific humidity, and $\\theta_e$ are "
                "extracted from the 65\\textdegree--95\\textdegree E meridional cross-section.}\n")
        f.write("\\label{tab:gc_era5_stats}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")

        hdr = " & ".join([f"\\textbf{{{l}}}" for l in col_labels])
        f.write(f"\\textbf{{Statistic}} & {hdr} \\\\\n\\hline\n")

        def fmt_row(label, key_idx):
            vals = " & ".join([f"{rows[l][key_idx]:.2f}" for l in col_order])
            return f"{label} & {vals} \\\\\n"
            
        def fmt_row_int(label, key_idx):
            vals = " & ".join([f"{int(rows[l][key_idx])}" for l in col_order])
            return f"{label} & {vals} \\\\\n"

        f.write(fmt_row("$Q_1$ at 400--500 hPa (K day$^{-1}$)", 0))
        f.write(fmt_row("$Q_2$ at 400--500 hPa (K day$^{-1}$)", 1))
        f.write(fmt_row_int("Level of peak $Q_1$ (hPa)",        2))
        f.write(fmt_row_int("Level of peak $Q_2$ (hPa)",        3))
        f.write("\\hline\n")
        
        f.write(fmt_row("Peak Easterly Wind Shear (m s$^{-1}$)", 4))
        f.write(fmt_row("Peak Low-Level $q$ (g kg$^{-1}$)",      5))
        f.write(fmt_row("Peak Low-Level $\\theta_e$ (K)",        6))

        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    log_msg(f"  ✅ Saved expanded LaTeX table → {tex_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 9: MODEL SKILL METRICS (PCC, RMSE, Bias vs ERA5)
    # ──────────────────────────────────────────────────────────────────────────
    log_msg("STEP 9: Generating Model Skill Metrics Table")
    skill_tex_path = f"{PLOT_DIR}/skill_metrics_table.tex"

    q1_era5, _ = compute_Q1_Q2_profile("ERA5")
    diag_era5 = compute_meridional_diag("ERA5")
    shear_era5 = diag_era5["wind_shear"].sel(lat=slice(5, 35))

    def calc_metrics(model, obs):
        m_vals, o_vals = model.values.flatten(), obs.values.flatten()
        pcc = np.corrcoef(m_vals, o_vals)[0, 1]
        rmse = np.sqrt(np.mean((m_vals - o_vals)**2))
        bias = np.mean(m_vals - o_vals)
        return pcc, rmse, bias

    skill_rows = {}
    forecast_leads = ["24hr", "48hr", "72hr"]
    
    for lead in forecast_leads:
        q1_mod, _ = compute_Q1_Q2_profile(lead)
        diag_mod = compute_meridional_diag(lead)
        shear_mod = diag_mod["wind_shear"].sel(lat=shear_era5.lat)
        
        q1_pcc, q1_rmse, q1_bias = calc_metrics(q1_mod, q1_era5)
        ws_pcc, ws_rmse, ws_bias = calc_metrics(shear_mod, shear_era5)
        
        skill_rows[lead] = {
            "Q1_PCC": q1_pcc, "Q1_RMSE": q1_rmse, "Q1_Bias": q1_bias,
            "WS_PCC": ws_pcc, "WS_RMSE": ws_rmse, "WS_Bias": ws_bias
        }

    with open(skill_tex_path, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Statistical skill metrics of GraphCast forecasts relative to ERA5 reanalysis. "
                "Metrics include Pattern Correlation Coefficient (PCC), Root Mean Square Error (RMSE), "
                "and Mean Bias evaluated over the vertical $Q_1$ profile and the meridional wind shear profile "
                "(5\\textdegree--35\\textdegree N).}\n")
        f.write("\\label{tab:model_skill}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\\hline\n")
        
        f.write("& \\multicolumn{3}{c}{\\textbf{Apparent Heat Source ($Q_1$)}} & \\multicolumn{3}{c}{\\textbf{Easterly Wind Shear ($U_{200}-U_{850}$)}} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write("\\textbf{Forecast Lead} & \\textbf{PCC} & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{PCC} & \\textbf{RMSE} & \\textbf{Bias} \\\\\n\\hline\n")

        for lead in forecast_leads:
            d = skill_rows[lead]
            f.write(f"GC {lead} & {d['Q1_PCC']:.3f} & {d['Q1_RMSE']:.2f} & {d['Q1_Bias']:+.2f} & "
                    f"{d['WS_PCC']:.3f} & {d['WS_RMSE']:.2f} & {d['WS_Bias']:+.2f} \\\\\n")

        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    log_msg(f"  ✅ Saved Skill Metrics table → {skill_tex_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL STATUS
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print("\n" + "═"*70)
    log_msg(f"🎉 PIPELINE COMPLETE — {elapsed:.1f} min")
    print("═"*70)