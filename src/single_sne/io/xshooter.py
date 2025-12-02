from __future__ import annotations
import os
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pdb import set_trace as stop
import scienceplots
plt.style.use(['science'])
from pathlib import Path
import astropy.units as u
from typing import Literal, Tuple, Optional
from single_sne.units import XSH_FLUX_UNIT
from single_sne.units import INSTRUMENT_UNITS
from single_sne.io.clean_data import clean_data
from single_sne.spectra.spectra import is_strictly_increasing
WAVE_UNIT = INSTRUMENT_UNITS["XSHOOTER"][0]
FLUX_UNIT = INSTRUMENT_UNITS["XSHOOTER"][1]

__all__ = [
    "discover_merge1d_files",
    "read_primary_linear_wcs",
    "object_and_date_from_header",
]

def _get_header_value(hdr, keys, default=""):
    for k in keys:
        if k in hdr and str(hdr[k]).strip():
            return str(hdr[k]).strip()
    return default


def discover_merge1d_files(
    root: str | Path = ".",
    *,
    product_mode: str = "SCI",         # "SCI" | "TELL" | "ANY"
    prefer_end_products: bool = True,
    allow_tmp: bool = False,
) -> dict[str, Path]:
    paths = sorted(Path(root).glob("**/*_FLUX_MERGE1D_*.fits"))
    if not paths:
        return {}

    arms: dict[str, list[Path]] = {"UVB": [], "VIS": [], "NIR": []}

    for p in paths:
        up = p.name.upper()

        if not allow_tmp and ("reflex_tmp_products" in str(p) or "xsh_respon" in str(p).lower()):
            continue

        if   "SCI_SLIT_FLUX_MERGE1D_" in up: prod = "SCI"
        elif "TELL_SLIT_FLUX_MERGE1D_" in up or "TELLURIC" in up: prod = "TELL"
        else: prod = "UNK"

        if product_mode != "ANY" and prod != product_mode:
            continue

        arm = "UVB" if "_UVB" in up else "VIS" if "_VIS" in up else "NIR" if "_NIR" in up else None
        if arm:
            arms[arm].append(p)

    def prefer_end(arr: list[Path]) -> list[Path]:
        if prefer_end_products:
            end = [p for p in arr if "reflex_end_products" in str(p)]
            return end or arr
        return arr

    final: dict[str, Path] = {}
    if prod == "SCI":
        for arm, arr in arms.items():
            arr = prefer_end(arr)
            if arr:
                final[arm] = sorted(arr, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        return final
    elif prod == "TELL":
        for arm, arr in arms.items():
            arr = prefer_end(arr)
            if arr:
                final[arm] = sorted(arr, key=lambda x: x.stat().st_mtime, reverse=True)
        return final


def make_quality_mask(qual, bad_bits=(1, 2, 4, 8, 16)):
    """Return boolean mask: True = good pixel."""
    bad_mask = np.zeros_like(qual, dtype=bool)
    for bit in bad_bits:
        bad_mask |= (qual & bit) != 0
    return ~bad_mask

def read_primary_linear_wcs(path: str | Path, ext: int = 0, debug = False, clean = False, show=False, etc = False) -> tuple[u.Quantity, u.Quantity, fits.Header]:
    """
    Read MERGE1D PrimaryHDU with linear WCS (CRVAL1, CDELT1, CRPIX1).
    Returns wavelength in nm and flux in native units (no unit in header → dimensionless).
    """
    #if debug: print(path.name)
    with fits.open(path, memmap=False) as hdul:
        # ---- arrays ----
        flux = np.asarray(hdul[0].data, dtype=float)   # 'FLUX'
        err  = np.asarray(hdul['ERRS'].data, dtype=float) if 'ERRS' in hdul else None
        if debug:print(f"\n\n\n\n ------------------READING IN XSHOOTER DATA----------------");print(hdul.info())#; hdr = hdul['ERRS'].header; print(f"\n\nError header:",hdr)
        qual = np.asarray(hdul['QUAL'].data, dtype=int) if 'QUAL' in hdul else None
            

        # ---- wavelength from linear WCS in primary ----
        hdr   = hdul[0].header
        #if debug: print(repr(hdr))
        n     = hdr['NAXIS1']
        crval = float(hdr['CRVAL1'])
        cdelt = float(hdr.get('CDELT1', hdr.get('CD1_1')))
        crpix = float(hdr.get('CRPIX1', 1.0))
        pix   = np.arange(n, dtype=float) + 1.0
        wave  = (pix - crpix) * cdelt + crval

        arm = hdr['HIERARCH ESO SEQ ARM']
        
        cunit = (str(hdr.get('CUNIT1', '')).strip() or str(hdr.get('WCAX1U', '')).strip()).lower()
        if cunit in ('angstrom', 'angstroem', 'a', 'ang'):
            wave_nm = wave * 0.1
        elif cunit in ('nm', 'nanometer', 'nanometre', ''):
            wave_nm = wave
        elif cunit in ('um', 'micron', 'micrometer', 'micrometre', 'µm'):
            wave_nm = wave * 1000.0
        else:
            if debug: print(f"[xsh] unrecognized wavelength unit '{cunit}', leaving as-is")
            wave_nm = wave
            
        if etc:#-------------Calculate Error per spectral bin as done in the ETC:--------
            snr_pix = flux/err
            if arm == 'UVB':
                res = 5400
            elif arm == 'VIS':
                res = 8900
            elif arm =='NIR':
                res = 5600
            else:
                arm = None
            #calculate no. pixels per resolution element:
            N_pix_per_res = (wave/res)/cdelt
            snr_res = snr_pix*np.sqrt(N_pix_per_res)
            

        # ---- sanity: align lengths ----
        m = len(flux)
        if err  is not None and len(err)  != m: err  = err[:m]
        if qual is not None and len(qual) != m: qual = qual[:m]

        #Clean pixels based on quality array:
        if clean and qual is not None:
            q = np.asarray(qual).astype(np.int64, copy=False)
            good = (q == 0)

            if debug:
                print(f"Lengths before cleaning: {len(wave), len(flux), len(err)}")

            wave_clean = wave[good]
            flux_clean = flux[good]
            err_clean  = err[good] if err is not None else None

            if debug:
                print(f"Lengths after cleaning: ({len(wave_clean)}, {len(flux_clean)}, {len(err_clean) if err is not None else None})")

            return wave_clean, flux_clean, err_clean, hdr
        
        # Flux: MERGE1D typically 
        # F_lambda; header rarely carries unit → keep dimensionless here.
        if etc: 
            return wave_nm, flux, err, hdr, snr_pix, snr_res
        else:
            return wave_nm, flux, err, hdr

def _get_object_date(hdr):
    obj = _get_header_value(hdr, ["OBJECT", "HIERARCH ESO OBS TARG NAME", "HIERARCH ESO OBS NAME"], "TARGET")
    date = _get_header_value(hdr, ["DATE-OBS", "HIERARCH ESO TPL START", "MJD-OBS"], "")
    # Compact date for filenames if DATE-OBS like '2025-05-20T00:23:04'
    date_for_title =  _get_header_value(hdr, ["DATE-OBS"], "")
    date_for_file = re.sub(r'[:T\- ]', '', date.split('.')[0]) if date else ""
    return obj, date_for_title, date_for_file

def read_xshooter_dat(
    path,
    *,
    as_quantity: bool = True,
    require_increasing: Literal["strict", "sort", "warn"] = "strict",
    dedup_tol: float = 0.0,   # in Å; e.g., 1e-6 drops exact dupes
    zero_tol: float = 0.0,    # drop rows with flux <= zero_tol
    debug: bool = False,
) -> tuple[u.Quantity, u.Quantity, bool]:
    """
    Read an XSHOOTER .dat combined spectrum(UVB, VIS, NIR) with 2 columns:
        wavelength [nm], flux [erg/s/cm²/Å]

    Drops rows with NaN and with flux <= zero_tol.
    Enforces (or fixes) strictly increasing wavelength.

    Returns
    -------
    wave, flux, fixed : (Quantity, Quantity, bool)
        `fixed` is True if sorting/dedup was applied.
    """
    arr = np.genfromtxt(path, comments="#", dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: need 2 columns (wavelength, flux)")

    w = arr[:, 0]
    f = arr[:, 1]
    ferr = arr[:, 2] if arr.ndim>2 else None
    fixed = False

    # Drop NaNs
    m = np.isfinite(w) & np.isfinite(f) 
    if ferr is not None:
        m &= np.isfinite(ferr)
    # Drop zeros / near-zeros
    m &= f > zero_tol
    w, f = w[m], f[m]
    if ferr is not None:
        ferr = ferr[m]

    if w.size == 0:
        raise ValueError(f"{path}: no valid rows after cleaning")

    if not is_strictly_increasing(w):
        if require_increasing == "strict":
            raise ValueError(
                "Wavelength is not strictly increasing; set require_increasing='sort' to auto-fix."
            )
        elif require_increasing == "warn":
            if debug:
                print("[WARN] wavelength not strictly increasing (kept as-is)")
        elif require_increasing == "sort":
            idx = np.argsort(w, kind="mergesort")
            w, f = w[idx], f[idx]
            if ferr is not None: ferr = ferr[idx]

            if dedup_tol >= 0.0:
                keep = [0]
                for i in range(1, w.size):
                    if (w[i] - w[keep[-1]]) > dedup_tol:
                        keep.append(i)
                    else:
                        keep[-1] = i  # keep the last occurrence
                keep = np.asarray(keep, int)
                w, f = w[keep], f[keep]
                if ferr is not None: ferr[keep]

            if not is_strictly_increasing(w):
                raise ValueError("Failed to enforce strictly increasing wavelength after sort/dedup.")
            fixed = True

    if as_quantity:
        w = w * WAVE_UNIT
        f = f * FLUX_UNIT
        if ferr is not None:
            ferr = ferr * FLUX_UNIT
    
    if ferr is not None:
        w, f, ferr = clean_data(w, f, ferr)
        return w, f, ferr, fixed
    else:
        w, f = clean_data(w, f)
        return w, f, fixed
    
    
def group_tellurics_by_star(arms: dict[str, list[Path]]):
    from pathlib import Path
    from collections import defaultdict 
    """
    arms: {"VIS": [Path(...), ...], "NIR": [...], ...}
    → {"HD89461": {"VIS":[...], "NIR":[...]}, "Hip102108": {...}, ...}
    """
    by_star: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for arm, paths in arms.items():
        for p in paths:
            # e.g. "Hip102108_TELL_SLT_TELL_SLIT_FLUX_MERGE1D_VIS.fits"
            fname = p.name
            star = fname.split("_", 1)[0].strip()   # "Hip102108" or "HD89461"
            by_star[star][arm].append(p)

    # turn nested default dicts into plain dicts (optional, just for cleanliness)
    return {star: dict(arm_dict) for star, arm_dict in by_star.items()}

def _obs_mjd(path: Path) -> float:
    """Get observation time in MJD from a MERGE1D FITS file."""
    from pathlib import Path
    import numpy as np
    from astropy.io import fits
    from astropy.time import Time
    
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        if "MJD-OBS" in hdr:
            return float(hdr["MJD-OBS"])
        elif "DATE-OBS" in hdr:
            return Time(hdr["DATE-OBS"]).mjd
        else:
            raise KeyError(f"No MJD-OBS/DATE-OBS in {path.name}")

def choose_vis_closest_to_nir(arms_for_star: dict, debug: bool = False) -> dict:
    """
    For a given star's arms dict, keep only the VIS file that is
    closest in time to the NIR observation.
    """
    vis_list = arms_for_star.get("VIS") or []
    nir_list = arms_for_star.get("NIR") or []

    # If we don't have both, nothing to do
    if not vis_list or not nir_list:
        return arms_for_star

    # For now: assume 1 NIR file, use it as reference
    t_nir = _obs_mjd(nir_list[0])

    # Compute times for all VIS files
    t_vis = np.array([_obs_mjd(p) for p in vis_list], dtype=float)
    if debug: print(f"[choose_vis_closest_to_nir] t_vis:",t_vis)

    # Index of VIS file closest in time to NIR
    idx = int(np.argmin(np.abs(t_vis - t_nir)))
    best_vis = vis_list[idx]

    # Optionally also pick the NIR closest to that VIS if you ever have >1 NIR
    # (currently just keep all NIRs, or the first one)
    arms_for_star["VIS"] = [best_vis]

    return arms_for_star

import inspect as _inspect

__all__ = [
    name
    for name, obj in globals().items()
    if not name.startswith("_")
    and (
        _inspect.isfunction(obj)
        or _inspect.isclass(obj)
        # or _inspect.ismodule(obj)  # include submodules if you want
    )
]