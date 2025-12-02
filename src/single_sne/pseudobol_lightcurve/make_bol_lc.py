from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from scipy.interpolate import interp1d, UnivariateSpline
from single_sne.pseudobol_lightcurve.aux import rd_lcbol_data, al_av, build_time_grid, build_time_grid_all, estimate_ni_mass, estimate_56ni_alt
from single_sne.pseudobol_lightcurve.dataclasses import LightCurveHeader, FilterLightCurve, PassbandInfo
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF



#--------------------------------------------------
# Attempt at translating Stéphane's IDL code into Python, started 19.11.25
#----------------------------------------------------------------


def read_pbinfo(pbinfo_path: str | Path) -> List[PassbandInfo]:
    """
    Read pbinfo.dat equivalent to IDL READCOL with format 'a,d,d,d'.
    """
    pbinfo_path = Path(pbinfo_path).expanduser()
    names: list[str] = []
    lambda_eff: list[float] = []
    ew: list[float] = []
    zpt: list[float] = []

    with pbinfo_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 4:
                # too short to be valid: skip
                continue

            try:
                lam = float(parts[1])
                ew_val = float(parts[2])
                zpt_val = float(parts[3])
            except ValueError:
                # probably a header row like: "Name lambda_eff(A) EW(A) ..."
                # skip quietly, or log if you like
                # print(f"Skipping non-numeric pbinfo line: {line}")
                continue

            names.append(parts[0])
            lambda_eff.append(lam)
            ew.append(ew_val)
            zpt.append(zpt_val)

    if not names:
        raise RuntimeError(f"No valid passband lines found in {pbinfo_path}")

    return [
        PassbandInfo(name=n, lambda_eff=le, ew=w, zpt=z)
        for n, le, w, z in zip(names, lambda_eff, ew, zpt)
    ]


# -------------------------------------------------------------------
# Core function: Python version of mklcbol.pro (without GUI)
# -------------------------------------------------------------------
EPS_TIME = 1e-2  # IDL: 1d-2

def _filter_shortname(name: str) -> str:
    """
    Translation of the big CASE block that maps filter full names to short names.
    """
    mapping = {
        "uvw2_Swift": "uvw2",
        "uvm2_Swift": "uvm2",
        "uvw1_Swift": "uvw1",
        "uu_Swift": "u",
        "bb_Swift": "b",
        "vv_Swift": "v",
        "U_Bessell": "U",
        "B_Bessell": "B",
        "V_Bessell": "V",
        "R_Bessell": "R",
        "I_Bessell": "I",
        "u_prime": "u'",
        "g_prime": "g'",
        "r_prime": "r'",
        "i_prime": "i'",
        "z_prime": "z'",
        "J_MKO": "J",
        "H_MKO": "H",
        "K_MKO": "Ks",   # CHECK! as in IDL comment
        "Kprime_MKO": "Kp",
        "J_2MASS": "J",
        "H_2MASS": "H",
        "Ks_2MASS": "Ks",
    }
    return mapping.get(name, name)

def _apply_band_preset(bands: str | Sequence[str]) -> List[str]:
    """
    Handles the 'bands' keyword logic from IDL (all/opt/opt_nou/nir or string like 'UBVRI').
    Returns a list of *pbinfo names* to select, e.g. 'U_Bessell', 'J_2MASS', etc.
    """
    if isinstance(bands, str):
        b = bands.lower()
        if b == "all":
            return [
                "U_Bessell", "B_Bessell", "V_Bessell", "R_Bessell", "I_Bessell",
                "J_2MASS", "H_2MASS", "Ks_2MASS",
            ]
        elif b == "opt":
            return [f"{x}_Bessell" for x in "UBVRI"]
        elif b == "opt_nou":
            return [f"{x}_Bessell" for x in "BVRI"]
        elif b == "nir":
            return [f"{x}_2MASS" for x in ["J", "H", "Ks"]]
        else:
            # interpret as something like 'UBVRI' -> ['U','B','V','R','I']
            return list(bands)
    else:
        return list(bands)
    
def _select_filters(
    filt_names: List[str],
    filt_short: List[str],
    bands: Optional[str | Sequence[str]],
) -> np.ndarray:
    """
    Returns a boolean mask of which filters are selected (like idxselect in IDL).
    IDL logic: try full name first, then shortname.
    """
    nf = len(filt_names)
    idxselect = np.zeros(nf, dtype=bool)

    if bands is None:
        # If nothing specified, default to *all* filters
        idxselect[:] = True
        return idxselect

    band_list = _apply_band_preset(bands)

    for band in band_list:
        # full name
        matches = [i for i, n in enumerate(filt_names) if n == band]
        if len(matches) == 1:
            idxselect[matches[0]] = True
            continue

        # short name
        matches = [i for i, s in enumerate(filt_short) if s == band]
        if len(matches) == 1:
            idxselect[matches[0]] = True

    return idxselect


def _interp_mag_linear_with_error(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    time_interp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified version of the IDL 'l' branch:
    - if |t - measurement time| < EPS_TIME -> use that point directly
    - else: bracket and do *simple* linear interpolation for value
            and linear interpolation of errors (approximation)
    """
    nt = len(time_interp)
    y_out = np.empty(nt, dtype=float)
    yerr_out = np.empty(nt, dtype=float)

    for ii, t in enumerate(time_interp):
        # exact (or nearly exact) data point?
        diff = np.abs(time - t)
        j = np.argmin(diff)
        if diff[j] < EPS_TIME:
            y_out[ii] = mag[j]
            yerr_out[ii] = magerr[j]
            continue

        # bracket
        # indices with time <= t and >= t
        left_candidates = np.where(time <= t)[0]
        right_candidates = np.where(time >= t)[0]
        if len(left_candidates) == 0 or len(right_candidates) == 0:
            # out-of-range -> extrapolate using np.interp (same as nearest bracket)
            y_out[ii] = np.interp(t, time, mag)
            yerr_out[ii] = np.interp(t, time, magerr)
            continue

        rri = left_candidates.max()
        rrs = right_candidates.min()
        x1, x2 = time[rri], time[rrs]
        y1, y2 = mag[rri], mag[rrs]
        e1, e2 = magerr[rri], magerr[rrs]

        if x2 == x1:
            y_out[ii] = y1
            yerr_out[ii] = e1
        else:
            w = (t - x1) / (x2 - x1)
            y_out[ii] = (1 - w) * y1 + w * y2
            # very simple error propagation (linear interp of errors)
            yerr_out[ii] = np.sqrt((1 - w) ** 2 * e1 ** 2 + w ** 2 * e2 ** 2)

    return y_out, yerr_out

def _interp_mag_gp(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    time_interp: np.ndarray,
    explosion_time: Optional[float] = None,
    frac_scale: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate magnitudes with a Gaussian process.

    Parameters
    ----------
    time : array
        Observation times (MJD).
    mag : array
        Magnitudes at those times.
    magerr : array
        1σ errors on the magnitudes.
    time_interp : array
        Times at which to predict.
    length_scale : float
        Typical correlation timescale (days) of the light curve.
        Smaller => more wiggles, larger => smoother.

    Returns
    -------
    y : array
        GP mean magnitudes at time_interp.
    yerr : array
        GP 1σ uncertainty at time_interp.
    """
    # sort by time
    order = np.argsort(time)
    t = np.asarray(time[order], dtype=float)
    y = np.asarray(mag[order], dtype=float)
    e = np.asarray(magerr[order], dtype=float)
    x_star = np.asarray(time_interp, dtype=float)

    
    # choose reference "explosion" time
    if explosion_time is None:
        # time of maximum light in this band = time of minimum magnitude
        j_max = np.argmin(y)
        t_max = t[j_max]
        explosion_time = t_max - 20.0  # your heuristic
        
    # time since explosion; avoid log(0)
    t_rel = np.maximum(t - explosion_time, 0.5)
    x_rel = np.maximum(x_star - explosion_time, 0.5)


    # log-time warp => stationary in log t_rel => fractional scale in t_rel
    tau = np.log(t_rel)
    tau_star = np.log(x_rel)

    # normalize tau a bit
    tau0 = tau.mean()
    tau_norm = (tau - tau0)[:, None]
    tau_star_norm = (tau_star - tau0)[:, None]

    # RBF in log-time: constant length_scale in log-space
    # Choose *fixed* kernel hyperparameters
    length_scale = frac_scale          # e.g. 0.1 in log-time
    const_value = 1.0                  # relative amplitude; not too important

    kernel = ConstantKernel(const_value, constant_value_bounds="fixed") * RBF(
        length_scale=length_scale,
        length_scale_bounds="fixed",
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=magerr**2,
        normalize_y=True,
        optimizer=None,     # <- DO NOT fit hyperparameters
    )

    gpr.fit(tau_norm, y)
    mu, sigma = gpr.predict(tau_star_norm, return_std=True)
    return mu, sigma 

def _interp_mag_any(
    time: np.ndarray,
    mag: np.ndarray,
    magerr: np.array,
    time_interp: np.ndarray,
    interpmeth: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SciPy-based interpolation for interpmeth in {'c', 'u', 's', ...}.

    Parameters
    ----------
    time : array
        Observation times.
    mag : array
        Magnitudes at those times.
    magerr : array
        1σ uncertainties on mag (same length as time).
    time_interp : array
        Times at which to interpolate.
    interpmeth : {'l', 'g', 'c','u','s',...}
        'l': linear
        'g': Gaussian process
        'c' : global least-squares quadratic (like /lsquadratic)
        'u' : piecewise quadratic (interp1d(kind='quadratic'))
        's' : cubic spline (UnivariateSpline)
        else : linear interpolation fallback

    Returns
    -------
    y : array
        Interpolated magnitudes at time_interp.
    yerr : array
        Interpolated magnitude errors at time_interp (approximate for 'u'/'s').
    """

    # Ensure sorted, unique x for interpolation
    idx = np.argsort(time)
    t_sorted = time[idx]
    y_sorted = mag[idx]
    yerr_sorted = magerr[idx]

    # Drop exact duplicates in t if they exist (optional but safer for splines)
    # (keep last occurrence)
    _, unique_idx = np.unique(t_sorted, return_index=True)
    t = t_sorted[unique_idx]
    y = y_sorted[unique_idx]
    e = yerr_sorted[unique_idx]

    m = interpmeth.lower()
    
    if m =="c": m =="u"
    
    # --- 'c' : global least-squares quadratic (lsquadratic in IDL) -------------
    '''if m == "c": #  TO DO: FIGURE OUT WHY THIS IS NOT WORKING ----
        # Shift time axis to improve conditioning of the quadratic fit
        t0 = np.mean(t)
        tt = t - t0                  # centred times for fitting
        x = time_interp - t0         # centred times for evaluation

        # Avoid zero/NaN weights
        w = np.where(e > 0, 1.0 / e, 0.0)

        # Fit y = a0 + a1 * tt + a2 * tt^2
        coeffs, cov = np.polyfit(tt, y, deg=2, w=w, cov=True)
        a0, a1, a2 = coeffs

        # Interpolated values at x = (time_interp - t0)
        y_out = a0 + a1 * x + a2 * x**2

        # Error propagation: var(y) = v^T C v, v = [1, x, x^2]
        v = np.vstack([np.ones_like(x), x, x**2])  # (3, N)
        Cv = cov @ v                               # (3,3) @ (3,N) -> (3,N)
        var = np.sum(v * Cv, axis=0)               # (N,)
        var = np.clip(var, 0.0, np.inf)
        yerr_out = np.sqrt(var)

        return y_out, yerr_out'''
    # -------- 'l': linear -----------------------------------------------------
    if m == "l":
        y_out, yerr_out = _interp_mag_linear_with_error(t, y, e, time_interp)
        return y_out, yerr_out
    
    # ----'g': Gaussian Process ------------------------------------------------
    if m =="g":
        y_out, yerr_out = _interp_mag_gp(t, y, e, time_interp)
        return y_out, yerr_out
    # --- 'u' : quadratic (piecewise) -------------------------------------------

    elif m == "u":
        # /quadratic: piecewise quadratic interpolation
        # interp1d(kind='quadratic') uses quadratic splines segment-wise.
        # Quadratic interpolation of the magnitudes
        f_mag = interp1d(
            t, y,
            kind="quadratic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        y_out = f_mag(time_interp)

        # Approximate error interpolation using the same scheme
        # (this is not a strict propagation, but close in spirit to IDL's "ignore"
        # and still gives you some time structure in the uncertainties)
        f_err = interp1d(
            t, e,
            kind="quadratic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        yerr_out = np.abs(f_err(time_interp))

        return y_out, yerr_out

    # --- 's' : cubic spline ----------------------------------------------------
    elif m == "s":
        # Spline through the points; you can optionally use weights 1/e
        # but set s=0 to pass exactly through the data (like an interpolating spline)
        # If some errors are zero, avoid infinite weights
        w = np.where(e > 0, 1.0 / e, 1.0)
        spline = UnivariateSpline(t, y, w=w, k=3, s=0)
        y_out = spline(time_interp)

        # For the errors, use a separate spline on e (or just linear interp if you prefer)
        spline_err = UnivariateSpline(t, e, k=3, s=0)
        yerr_out = np.abs(spline_err(time_interp))

        return y_out, yerr_out

    # --- default: linear interpolation ----------------------------------------
    else:
        f_mag = interp1d(
            t, y,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        y_out = f_mag(time_interp)

        f_err = interp1d(
            t, e,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        yerr_out = np.abs(f_err(time_interp))

        return y_out, yerr_out

def mklcbol(
    infile: str | Path,
    pbinfo: str | Path = "~/idl/lcbol/pbinfo.dat",
    bands: Optional[str | Sequence[str]] = None,
    interpmeth: str = "g",   # 'l', 'g', 'u', 's' (linear, gaussian process, quad, spline)
    dtinterp: float = 3.0,   # currently not used here (same as IDL)
    batch: bool = True,      # True = non-interactive
    fout: Optional[str | Path] = None,
) -> Path:
    """
    Python port of mklcbol.pro *without* interactive GUI.

    Returns the path of the created bolometric light curve file.
    """

    infile = Path(infile)
    pbinfo = Path(pbinfo).expanduser()

    # ---------------------------------------------------------------
    # Read in input data (hdr, lcdata)
    # ---------------------------------------------------------------
    hdr, lcdata = rd_lcbol_data(infile)
    print(f"\n\n\n\n\n-----------------SANITY CHECK---------------------")
    print("Distance modulus:", hdr.dmod)
    d_cm = 3.085677e18 * 10 ** ((hdr.dmod + 5) / 5)
    print("Distance in cm:", d_cm)
    print(f"### generating pseudo-bolometric LC for {hdr.name}")

    # ---------------------------------------------------------------
    # Read filter info from pbinfo.dat and match to lcdata filters
    # ---------------------------------------------------------------
    pb_list = read_pbinfo(pbinfo)

    # arrays of passband names, etc.
    pb_names = [p.name for p in pb_list]
    pb_lambda = np.array([p.lambda_eff for p in pb_list])
    pb_ew = np.array([p.ew for p in pb_list])
    pb_zpt = np.array([p.zpt for p in pb_list])

    filt_name = []
    filt_short = []
    filt_lambda = []
    filt_ew = []
    filt_zpt = []
    new_lcdata: List[FilterLightCurve] = []

    for lc in lcdata:
        if lc.filt not in pb_names:
            print(f" WARNING - filter {lc.filt} is not part of {pbinfo}")
            continue

        j = pb_names.index(lc.filt)
        new_lcdata.append(lc)
        filt_name.append(lc.filt)
        filt_short.append(_filter_shortname(lc.filt))
        filt_lambda.append(pb_lambda[j])
        filt_ew.append(pb_ew[j])
        filt_zpt.append(pb_zpt[j])

    if len(new_lcdata) < 2:
        raise RuntimeError("Need at least 2 filters!")

    lcdata = new_lcdata
    hdr.nfilt = len(lcdata)

    filt_lambda = np.array(filt_lambda)
    filt_ew = np.array(filt_ew)
    filt_zpt = np.array(filt_zpt)

    # sort by increasing effective wavelength
    order = np.argsort(filt_lambda)
    lcdata = [lcdata[i] for i in order]
    filt_name = [filt_name[i] for i in order]
    filt_short = [filt_short[i] for i in order]
    filt_lambda = filt_lambda[order]
    filt_ew = filt_ew[order]
    filt_zpt = filt_zpt[order]

    # ---------------------------------------------------------------
    # Filter selection (no GUI; use bands + batch behavior)
    # ---------------------------------------------------------------
    idxselect = _select_filters(filt_name, filt_short, bands)
    rrselect = np.where(idxselect)[0]
    nselect = len(rrselect)

    if nselect < 2:
        raise RuntimeError("Need to select at least 2 filters")

    print(
        f"{nselect} selected filters: "
        + ", ".join(filt_name[i] for i in rrselect)
    )

    # ---------------------------------------------------------------
    # Build reference time grid from all selected filters
    # ---------------------------------------------------------------
    time_interp = build_time_grid(lcdata, idxselect)
    nt = len(time_interp)

    interpmag = np.zeros((nselect, nt), dtype=float)
    interpmagerr = np.zeros_like(interpmag)

    # ---------------------------------------------------------------
    # Interpolate magnitudes for each selected filter
    # ---------------------------------------------------------------
    for i, idx in enumerate(rrselect):
        lc = lcdata[idx]
        time = lc.time
        mag = lc.mag
        magerr = lc.magerr

        y, yerr = _interp_mag_any(time, mag, magerr, time_interp, interpmeth)

        interpmag[i, :] = y
        interpmagerr[i, :] = yerr

    # ---------------------------------------------------------------
    # Convert de-reddened magnitudes to fluxes (includes extinction)
    # ---------------------------------------------------------------
    nselect = len(rrselect)  # in case we dropped any filters (we didn't here)
    interpflux = np.zeros((nselect, nt), dtype=float)
    interpfluxerr = np.zeros_like(interpflux)
    
    LN10 = np.log(10.0)


    for i, idx in enumerate(rrselect):
        lam = float(filt_lambda[idx])
        zpt = float(filt_zpt[idx])

        AlAv_host, AlAv_host_err = al_av(lam, r_v=hdr.rvhost, rverr=hdr.rvhosterr)
        AlAv_MW, AlAv_MW_err = al_av(lam, r_v=hdr.rvmw, rverr=hdr.rvmwerr)

        Al_host = AlAv_host * hdr.avhost
        Al_MW = AlAv_MW * hdr.avmw
        Al_tot = Al_host + Al_MW

        # flux from magnitude:
        # F = 10^[-0.4 (m - A_tot - zpt)]
        # flux from magnitude:
        #   F = 10^[-0.4 (m - A_tot - zpt)]
        mag_i    = interpmag[i, :]
        magerr_i = interpmagerr[i, :]
        flux     = 10.0 ** (-0.4 * (mag_i - Al_tot - zpt))
        
        # --- photometric component of the flux error -------------------
        # dF/dm = -0.4 ln(10) * F  =>  sigma_F,phot = |dF/dm| * sigma_m
        flux_err_phot = np.abs(0.4 * LN10 * flux * magerr_i)


        # --- extinction component (same structure as IDL) --------------
        if hdr.avhost > 0.0:
            Al_host_err = Al_host * np.sqrt(
                (AlAv_host_err / AlAv_host) ** 2
                + (hdr.avhosterr / hdr.avhost) ** 2
            )
        else:
            Al_host_err = 0.0

        if hdr.avmw > 0.0:
            Al_MW_err = Al_MW * np.sqrt(
                (AlAv_MW_err / AlAv_MW) ** 2
                + (hdr.avmwerr / hdr.avmw) ** 2
            )
        else:
            Al_MW_err = 0.0

        Al_tot_err = np.sqrt(Al_host_err**2 + Al_MW_err**2)

        # IDL-style extinction term (kept for consistency)
        # NOTE: if you want the "pure" derivative, you'd drop the /Al_tot factor.
        with np.errstate(divide="ignore", invalid="ignore"):
            flux_err_ext = flux * (0.4 * LN10 * Al_tot_err / np.where(Al_tot == 0, 1.0, Al_tot))

        # --- combine in quadrature -------------------------------------
        flux_err = np.sqrt(flux_err_phot**2 + flux_err_ext**2)

        interpflux[i, :]    = flux
        interpfluxerr[i, :] = flux_err

    # ---------------------------------------------------------------
    # Integrate fluxes over wavelength, handling gaps and overlaps
    # (translation of the "correct treatment of gaps and overlaps" block)
    # ---------------------------------------------------------------
    flux_int = np.zeros(nt, dtype=float)
    flux_int_err = np.zeros_like(flux_int)

    idxlap = 0  # 1 if current filter overlapped with previous one
    wred = None  # just to appease type checkers

    for ii in range(nselect):
        idx = rrselect[ii]
        lam = float(filt_lambda[idx])
        ew = float(filt_ew[idx])

        if ii > 0 and idxlap == 1:
            wblue = wred  # reuse previous wred
        else:
            wblue = lam - ew / 2.0

        if ii < nselect - 1:
            # all but last filter
            wred = lam + ew / 2.0
            idxnext = rrselect[ii + 1]
            lam_next = float(filt_lambda[idxnext])
            ew_next = float(filt_ew[idxnext])
            wbluenext = lam_next - ew_next / 2.0

            f_i = interpflux[ii, :]
            ferr_i = interpfluxerr[ii, :]

            f_next = interpflux[ii + 1, :]
            ferr_next = interpfluxerr[ii + 1, :]

            if wred <= wbluenext:
                # isolated filter => gap between this and next filter
                # FILTER core
                flux_int += (wred - wblue) * f_i
                flux_int_err += (wred - wblue) * ferr_i

                # GAP: mean flux in the gap
                flux_int += (wbluenext - wred) * 0.5 * (f_i + f_next)
                flux_int_err += (wbluenext - wred) * 0.5 * np.sqrt(
                    ferr_i**2 + ferr_next**2
                )
                idxlap = 0
            else:
                # OVERLAP with next filter
                # FILTER non-overlapping part
                flux_int += (wbluenext - wblue) * f_i
                flux_int_err += (wbluenext - wblue) * ferr_i

                # OVERLAP: mean flux
                flux_int += (wred - wbluenext) * 0.5 * (f_i + f_next)
                flux_int_err += (wred - wbluenext) * 0.5 * np.sqrt(
                    ferr_i**2 + ferr_next**2
                )
                idxlap = 1
        else:
            # last filter
            wred = lam + ew / 2.0
            f_i = interpflux[ii, :]
            ferr_i = interpfluxerr[ii, :]
            flux_int += (wred - wblue) * f_i
            flux_int_err += (wred - wblue) * ferr_i

    # ---------------------------------------------------------------
    # Convert integrated flux to luminosity
    # L = 4π d^2 F
    # ---------------------------------------------------------------
    # distance in cm: 1 pc = 3.085677e18 cm
    dist_cm = 3.085677e18 * 10.0 ** ((hdr.dmod + 5.0) / 5.0)
    dist_cm_err = dist_cm * np.log(10.0) * hdr.dmoderr / 5.0

    lum_int = 4.0 * np.pi * dist_cm**2 * flux_int

    # avoid division by zero in error term
    flux_ratio = np.where(flux_int > 0, flux_int_err / flux_int, 0.0)
    lum_int_err = lum_int * np.sqrt(
        2.0 * (dist_cm_err / dist_cm) ** 2 + flux_ratio**2
    )
    
    #---------------------------------------------------------------
    # Estimate peak luminosity and nickel mass
    #----------------------------------------------------------------
    L_peak = lum_int.max()
    t_peak = time_interp[np.argmax(lum_int)]
    t0 = 60689.1589 #from Lindsey's data, TODO: make this actually from scratch
    t0_err = 0.0085
    M_ni, M_ni_err, t_rise = estimate_ni_mass(
    L_peak=L_peak,
    t_peak=t_peak,
    t0=60689.1589,
    t0_err=0.0085
    )
    
    M_ni_alt = estimate_56ni_alt(L_peak)
    # ---------------------------------------------------------------
    # Write output file
    # ---------------------------------------------------------------
    if fout is None:
        shortnames_selected = "".join(filt_short[i] for i in rrselect)
        outname = f"{hdr.name}_lcbol_{shortnames_selected}.dat"
        fout = infile.parent / outname   # <— save next to the input file

    fout = Path(fout)
    
    
    
    with fout.open("w") as f:
        from datetime import datetime, UTC

        f.write(f"#generated on {datetime.now(UTC).isoformat()} using mklcbol.py\n")
        f.write(f"# Peak Luminosity: {L_peak}\n#Peak time: {t_peak}\n#M(56Ni):{M_ni}\n#M(56Ni) error:{M_ni_err}\n#Rise time:{t_rise}\n#Alt. M(56Ni) (no integration): {M_ni_alt}\n")
        f.write(f"#INFILE   {infile.name}\n")
        f.write(f"#NAME     {hdr.name}\n")
        f.write(
            f"#AV_HOST  {hdr.avhost:6.3f} +/- {hdr.avhosterr:6.3f}\n"
        )
        f.write(
            f"#RV_HOST  {hdr.rvhost:6.3f} +/- {hdr.rvhosterr:6.3f}\n"
        )
        f.write(
            f"#AV_MW    {hdr.avmw:6.3f} +/- {hdr.avmwerr:6.3f}\n"
        )
        f.write(
            f"#RV_MW    {hdr.rvmw:6.3f} +/- {hdr.rvmwerr:6.3f}\n"
        )
        f.write(
            f"#DIST_MOD {hdr.dmod:6.3f} +/- {hdr.dmoderr:6.3f}\n"
        )
        filt_list = ", ".join(filt_name[i] for i in rrselect)
        f.write(f"#NFILT    {nselect:2d} ({filt_list})\n")
        f.write("#\n")
        f.write(
            "#time[d]    lbol[erg/s]      lbolerr[erg/s]\n"
        )
        for t, L, Lerr in zip(time_interp, lum_int, lum_int_err):
            f.write(f"{t:10.3f}  {L:15.8E}  {Lerr:15.8E}\n")

    print(f"Created file {fout}")
    return fout

