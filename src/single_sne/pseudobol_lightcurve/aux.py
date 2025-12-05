from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import pathlib
from typing import Dict
from typing import List, Optional, Sequence, Tuple
from single_sne.pseudobol_lightcurve.dataclasses import LightCurveHeader, FilterLightCurve, PassbandInfo
import requests
from bs4 import BeautifulSoup
from astropy.constants import c 
import astropy.units as u
from astropy.units import Unit
from astropy.cosmology import FlatLambdaCDM


KM_PER_S = Unit("km / s")
C_KM_S = c.to(KM_PER_S).value

def rd_lcbol_data(infile: str | pathlib.Path, debug = False) -> Tuple[LightCurveHeader, List[FilterLightCurve]]:
    """
    Python port of rd_lcbol_data.pro.

    Parameters
    ----------
    infile : str or Path
        Path to an ASCII file with the lcbol input format
        (header + per-filter blocks).

    Returns
    -------
    hdr : LightCurveHeader
    lcdata : list[FilterLightCurve]
    """
    infile = pathlib.Path(infile)
    with infile.open() as f:
        lines = [ln.rstrip("\n") for ln in f]

    # -----------------------------
    # Parse header
    # -----------------------------
    name = None
    avhost = avhosterr = 0.0
    rvhost = rvhosterr = 0.0
    avmw = avmwerr = 0.0
    rvmw = rvmwerr = 0.0
    dmod = dmoderr = 0.0
    nfilt = None

    i = 0
    nlines = len(lines)

    while i < nlines:
        line = lines[i].strip()
        i += 1
        if not line.startswith("#"):
            continue

        parts = line.split()
        tag = parts[0]

        if tag == "#NAME":
            # e.g. "#NAME SN2005cf"
            if len(parts) >= 2:
                name = parts[1]

        elif tag == "#AV_HOST":
            # "#AV_HOST 0.123 +/- 0.010" or just "#AV_HOST 0.123"
            if len(parts) >= 2:
                avhost = float(parts[1])
            if len(parts) >= 4:
                avhosterr = float(parts[3])

        elif tag == "#RV_HOST":
            if len(parts) >= 2:
                rvhost = float(parts[1])
            if len(parts) >= 4:
                rvhosterr = float(parts[3])

        elif tag == "#AV_MW":
            if len(parts) >= 2:
                avmw = float(parts[1])
            if len(parts) >= 4:
                avmwerr = float(parts[3])

        elif tag == "#RV_MW":
            if len(parts) >= 2:
                rvmw = float(parts[1])
            if len(parts) >= 4:
                rvmwerr = float(parts[3])

        elif tag == "#DIST_MOD":
            if len(parts) >= 2:
                dmod = float(parts[1])
            if len(parts) >= 4:
                dmoderr = float(parts[3])

        elif tag == "#NFILT":
            if len(parts) >= 2:
                nfilt = int(parts[1])
            # in the IDL version, reaching NFILT ends the header loop
            break

    if nfilt is None:
        raise ValueError("Header did not contain #NFILT")

    if name is None:
        name = infile.stem  # fallback

    hdr = LightCurveHeader(
        name=name,
        nfilt=nfilt,
        avhost=avhost,
        avhosterr=avhosterr,
        rvhost=rvhost,
        rvhosterr=rvhosterr,
        avmw=avmw,
        avmwerr=avmwerr,
        rvmw=rvmw,
        rvmwerr=rvmwerr,
        dmod=dmod,
        dmoderr=dmoderr,
    )

    # -----------------------------
    # Parse per-filter blocks
    # -----------------------------
    lcdata: List[FilterLightCurve] = []
    # i is at the line right after "#NFILT"
    for _ in range(nfilt):
        # find next "#FILTER" line
        while i < nlines and "#FILTER" not in lines[i]:
            i += 1
        if i >= nlines:
            raise ValueError("Unexpected end of file while looking for #FILTER blocks")

        header_line = lines[i].strip()
        i += 1
        parts = header_line.split()
        # Expect something like: "#FILTER B_Bessell NMEAS 23"
        if len(parts) < 4 or parts[0] != "#FILTER":
            raise ValueError(f"Malformed filter header line: {header_line!r}")

        filt_name = parts[1]
        try:
            nmeas = int(parts[3])
        except ValueError as e:
            raise ValueError(f"Cannot parse NMEAS from filter header: {header_line}") from e

        if debug: print(f"[ INFO  ] reading filter {filt_name}")

        time = []
        mag = []
        magerr = []

        for _j in range(nmeas):
            if i >= nlines:
                raise ValueError("Unexpected end of file while reading measurements")

            l = lines[i].strip()
            i += 1
            if not l or l.startswith("#"):
                # This would be odd, but let's be a bit defensive:
                continue

            cols = l.split()
            if len(cols) < 3:
                raise ValueError(f"Expected 'time mag magerr' line, got: {l!r}")

            time.append(float(cols[0]))
            mag.append(float(cols[1]))
            magerr.append(float(cols[2]))

        lcdata.append(
            FilterLightCurve(
                filt=filt_name,
                time=np.array(time, dtype=float),
                mag=np.array(mag, dtype=float),
                magerr=np.array(magerr, dtype=float),
            )
        )

    return hdr, lcdata

def _poly(y: float, coeffs: np.ndarray) -> float:
    """
    IDL POLY(y, c) uses c[0] + c[1]*y + c[2]*y^2 + ...
    This is the opposite order of np.polyval, so we implement it directly.
    """
    y = np.asarray(y)
    out = np.zeros_like(y, dtype=float)
    for i, c in enumerate(coeffs):
        out = out + c * y**i
    return out


def al_av(
    lambda_angstrom: float,
    r_v: float = 3.1,
    rverr: float | None = None,
    debug = False,
) -> Tuple[float, float]:
    """
    Python port of IDL al_av.pro

    Given wavelength (Å) and R_V, return (A_lambda / A_V, sigma_A_lAv),
    assuming the CCM89 extinction law with O'Donnell (1994) optical
    coefficients by default.

    Parameters
    ----------
    lambda_angstrom : float
        Wavelength in Angstrom.
    r_v : float, optional
        Total-to-selective extinction R_V. Default 3.1.
    rverr : float or None, optional
        Uncertainty on R_V. If provided, the function returns
        (AlAv, AlAv_err) where AlAv_err = AlAv * rverr / r_v
    
    Returns
    -------
    AlAv : float
        A_lambda / A_V at the given wavelength.
    AlAv_err : float
        Error on A_lambda / A_V due to uncertainty in R_V.
        0.0 if rverr is None.
    """
    # IDL: x = 1e4 / lambda   ; convert to inverse microns
    x = 1e4 / float(lambda_angstrom)

    a = 0.0
    b = 0.0

    # IR: 0.3 < x < 1.1
    if (x > 0.3) and (x < 1.1):
        a = 0.574 * x**1.61
        b = -0.527 * x**1.61

    # optical / NIR: 1.1 <= x < 3.3
    elif (x >= 1.1) and (x < 3.3):
        y = x - 1.82

        # O'Donnell (1994) revised coefficients
        c1 = np.array([1.0, 0.104,   -0.609,    0.701,  1.137, -1.718, -0.827,  1.647, -0.505])
        c2 = np.array([0.0, 1.952,   2.908,   -3.989, -7.985, 11.102,  5.491, -10.805,  3.347])
        a = _poly(y, c1)
        b = _poly(y, c2)

    # mid-UV: 3.3 <= x < 8.0
    elif (x >= 3.3) and (x < 8.0):
        f_a = 0.0
        f_b = 0.0
        if x > 5.9:
            y = x - 5.9
            f_a = -0.04473 * y**2 - 0.009779 * y**3
            f_b =  0.2130 * y**2 + 0.1207   * y**3
        a = 1.752 - 0.316 * x - (0.104 / ((x - 4.67) ** 2 + 0.341)) + f_a
        b = -3.090 + 1.825 * x + (1.206 / ((x - 4.62) ** 2 + 0.263)) + f_b

    # far-UV: 8.0 <= x <= 11.0
    elif (x >= 8.0) and (x <= 11.0):
        y = x - 8.0
        c1 = np.array([-1.073, -0.628,  0.137, -0.070])
        c2 = np.array([13.670,  4.257, -0.420,  0.374])
        a = _poly(y, c1)
        b = _poly(y, c2)

    # Else: a and b remain 0.0 → AlAv = 0 (same as IDL default fall-through)

    AlAv = a + b / r_v

    if rverr is not None:
        AlAv_err = AlAv * (rverr / r_v)
    else:
        AlAv_err = 0.0

    return AlAv, AlAv_err

def read_pbinfo(pbinfo_path: str | pathlib.Path, debug = False,) -> Dict[str, PassbandInfo]:
    """
    Python equivalent of the IDL 'readcol' on pbinfo.dat:
    reads columns: name, lambda_eff, ew, zpt (skipping '#' comments).

    Returns a dict mapping filter name -> FilterMeta.
    """
    pbinfo_path = pathlib.Path(pbinfo_path)
    names: list[str] = []
    lam: list[float] = []
    ew: list[float] = []
    zpt: list[float] = []

    with pbinfo_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed pbinfo line: {line!r}")
            names.append(parts[0])
            lam.append(float(parts[1]))
            ew.append(float(parts[2]))
            zpt.append(float(parts[3]))

    filt_meta: Dict[str, PassbandInfo] = {}
    for n, l, w, zp in zip(names, lam, ew, zpt):
        filt_meta[n] = PassbandInfo(
            name=n,
            lambda_eff=l,
            ew=w,
            zpt=zp,
        )
    return filt_meta


def merge_close_times(time, mag, magerr, tol=0.5):
    """
    Merge measurements whose times differ by < tol.
    Weighted mean for mag; errors combined accordingly.
    """
    time = np.asarray(time)
    mag = np.asarray(mag)
    magerr = np.asarray(magerr)

    idx_sort = np.argsort(time)
    time = time[idx_sort]
    mag = mag[idx_sort]
    magerr = magerr[idx_sort]

    merged_time = []
    merged_mag = []
    merged_magerr = []

    group = [0]  # start index of first group

    for i in range(1, len(time)):
        if abs(time[i] - time[group[-1]]) <= tol:
            group.append(i)
        else:
            # process previous group
            subset = np.array(group)
            w = 1.0 / magerr[subset]**2
            m = np.sum(w * mag[subset]) / np.sum(w)
            merr = np.sqrt(1.0 / np.sum(w))

            merged_time.append(np.mean(time[subset]))
            merged_mag.append(m)
            merged_magerr.append(merr)

            group = [i]

    # process last group
    subset = np.array(group)
    w = 1.0 / magerr[subset]**2
    m = np.sum(w * mag[subset]) / np.sum(w)
    merr = np.sqrt(1.0 / np.sum(w))

    merged_time.append(np.mean(time[subset]))
    merged_mag.append(m)
    merged_magerr.append(merr)

    return (
        np.array(merged_time),
        np.array(merged_mag),
        np.array(merged_magerr)
    )

def build_time_grid(
    lcdata: List[FilterLightCurve],
    selected_idx: np.ndarray,
) -> np.ndarray:
    """
    IDL logic for building time_interp:
      - find largest tmin and smallest tmax among selected filters
      - collect all times between tmin and tmax from selected filters
      - unique + sort
    """
    times = []
    tmin = -np.inf
    tmax = np.inf
    mags = []
    magerrs = []
    print(f"Check time inputs:")
    for i, sel in enumerate(selected_idx):
        if sel:
            formatted = [f"{t:.4f}" for t in lcdata[i].time[:10]]
            print(i, formatted)
        
    rrselect = np.where(selected_idx)[0]
    
    print(f"\n\n\n-------SANITY CHECK build_time_grid-------")
    print(f"selected_idx: {selected_idx}")
    print(f"Length of lightcurve data:", len(lcdata))
    
    for idx in rrselect:
        tt = lcdata[idx].time
        mm = lcdata[idx].mag
        merr = lcdata[idx].magerr

        tmin = max(tmin, np.min(tt))
        tmax = min(tmax, np.max(tt))

        times.append(tt)
        mags.append(mm)
        magerrs.append(merr)


    if not times:
        raise RuntimeError("No selected filters with time data")

    tarr = np.concatenate(times)
    marr = np.concatenate(mags)
    earr = np.concatenate(magerrs)
    mask = (tarr >= tmin) & (tarr <= tmax)
    tt = tarr[mask]
    mm = marr[mask]
    ee = earr[mask]
    
    #tol = 0.2
    #merged_time,_, _ = merge_close_times(tt, mm, ee, tol) 
    
    #time_interp = merged_time
    time_interp = np.unique(np.sort(tt))
    print(f"tmin: {tmin}, tmax: {tmax}")
    print(f"time_interp;",[f"{t:.3f}" for t in time_interp])
    print(f"\n\n\n\nLength of time array: {len(time_interp)}")
    #time_interp = np.linspace(tmin, tmax, len(tt))

    return time_interp


def sort_filters_by_lambda(
    lcdata: List[FilterLightCurve],
    pbinfo: dict[str, PassbandInfo],
) -> Tuple[List[FilterLightCurve], List[PassbandInfo]]:
    """
    Return lcdata and passband info sorted by increasing lambda_eff.
    Only reorders entries; does not drop any filters.
    """
    # build list of (lc, pb) pairs
    pairs: list[tuple[FilterLightCurve, PassbandInfo]] = []
    for lc in lcdata:
        if lc.filt not in pbinfo:
            raise KeyError(f"Filter {lc.filt} missing in pbinfo")
        pairs.append((lc, pbinfo[lc.filt]))

    # sort by lambda_eff
    pairs_sorted = sorted(pairs, key=lambda p: p[1].lambda_eff)

    lc_sorted  = [p[0] for p in pairs_sorted]
    pb_sorted  = [p[1] for p in pairs_sorted]
    return lc_sorted, pb_sorted

def build_time_grid_all(lcdata: List[FilterLightCurve]) -> np.ndarray:
    selected_idx = np.ones(len(lcdata), dtype=bool)
    return build_time_grid(lcdata, selected_idx)

def get_ned_ebv(name):
    url = f"https://ned.ipac.caltech.edu/cgi-bin/objsearch?objname={name}&extend=no..."
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    
    for row in soup.text.splitlines():
        if "E(B-V)" in row and "Galactic" in row:
            val = float(row.split()[-1])
            return val
    return None

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

def distmod_from_z_astropy(z: float,
                           H0: float = 73.0,
                           Om0: float = 0.27) -> float:
    """
    Distance modulus μ from redshift z using a flat ΛCDM cosmology
    via astropy.cosmology.

    Parameters
    ----------
    z : float
        Redshift.
    H0 : float
        Hubble constant [km/s/Mpc].
    Om0 : float
        Matter density parameter Ω_m.

    Returns
    -------
    mu : float
        Distance modulus in magnitudes.
    """
    cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0)
    return cosmo.distmod(z).value


def distmod_from_z_with_pecvel(
    z: float,
    H0: float = 73.0,
    sigma_v: float = 300.0,
) -> tuple[float, float]:
    """
    Distance modulus μ and its uncertainty from redshift z, including a
    simple peculiar-velocity error model (sigma_v).

    Returns
    -------
    mu : float
        Distance modulus.
    sigma_mu : float
        Uncertainty on μ from peculiar velocity only.
    """
    mu = distmod_from_z_astropy(z, H0=H0)

    v = C_KM_S * z
    frac = sigma_v / v  # fractional distance error
    sigma_mu = 5.0 / np.log(10.0) * frac

    return mu, sigma_mu


import numpy as np
from dataclasses import dataclass

def estimate_ni_mass(L_peak, t_peak, t0, t0_err=0.0, alpha=1.0):
    """
    Estimate 56Ni mass using Arnett's rule.

    Parameters
    ----------
    L_peak : float
        Peak bolometric luminosity in erg/s.
    t_peak : float
        Time of bolometric maximum (MJD).
    t0 : float
        Explosion epoch from SALT2 (MJD).
    t0_err : float, optional
        Uncertainty on t0 (MJD). Default: 0.
    alpha : float, optional
        Arnett factor (default = 1). Values 0.8–1.3 are common.

    Returns
    -------
    M_ni : float
        Estimated Nickel mass in solar masses (Msun).
    M_ni_err : float
        Error propagated from t0_err (Msun).
    t_rise : float
        Rise time in days.
    """

    # --- Compute rise time ---
    t_rise = 17.0
    t_rise_err = 2.0
    '''
    t_peak - t0
    if t_rise <= 0:
        raise ValueError(f"Computed rise time {t_rise:.2f} is non-physical. Check t_peak and t0.")'''

    # --- Radioactive energy deposition per Msun of 56Ni ---
    # From Nadyozhin (1994) and standard Arnett practice:
    term_Ni = 6.45e43 * np.exp(-t_rise / 8.8)
    term_Co = 1.45e43 * np.exp(-t_rise / 111.3)
    denom = term_Ni + term_Co   # erg/s produced by 1 Msun of Ni at t_rise

    # --- Nickel mass ---
    M_ni = L_peak / (alpha * denom)

    # --- Error propagation from t0 uncertainty ---
    # dM/dt_rise via numerical derivative
    dt = 0.05  # small step (days)
    def M_of_trise(tr):
        tN = 6.45e43 * np.exp(-tr / 8.8)
        tC = 1.45e43 * np.exp(-tr / 111.3)
        return L_peak / (alpha * (tN + tC))

    dMdt = (M_of_trise(t_rise + dt) - M_of_trise(t_rise - dt)) / (2 * dt)

    M_ni_err = abs(dMdt) * t_rise_err  # since t_rise = t_peak - t0

    return M_ni, M_ni_err, t_rise


def estimate_56ni_alt(L_max):
    return L_max/(2.21e43)