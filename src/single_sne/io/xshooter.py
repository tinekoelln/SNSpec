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
from single_sne.units import XSH_FLUX_UNIT

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

def clean_arm(w, f, interp_small_gaps=True, max_gap=5):
    """
    Ensure finite wavelengths/flux. Optionally linearly fill small internal NaN gaps (<= max_gap samples).
    Edge NaNs are trimmed, not filled.
    """
    w = np.asarray(w, float)
    f = np.asarray(f, float)

    m = np.isfinite(w) & np.isfinite(f)
    w, f = w[m], f[m]

    if w.size and w[0] > w[-1]:
        idx = np.argsort(w); w, f = w[idx], f[idx]

    if not interp_small_gaps or w.size == 0:
        return w, f


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
    for arm, arr in arms.items():
        arr = prefer_end(arr)
        if arr:
            final[arm] = sorted(arr, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return final

def read_primary_linear_wcs(path: str | Path, ext: int = 0) -> tuple[u.Quantity, u.Quantity, fits.Header]:
    """
    Read MERGE1D PrimaryHDU with linear WCS (CRVAL1, CDELT1, CRPIX1).
    Returns wavelength in nm and flux in native units (no unit in header → dimensionless).
    """
    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[ext]
        hdr = hdu.header
        y = np.array(hdu.data, dtype=float) * XSH_FLUX_UNIT
        n = hdr["NAXIS1"]
        crval = float(hdr["CRVAL1"])
        cdelt = float(hdr.get("CDELT1", hdr.get("CD1_1")))
        crpix = float(hdr.get("CRPIX1", 1.0))
        pix = np.arange(n, dtype=float) + 1.0
        w = (pix - crpix) * cdelt + crval  # wavelength in header unit

        cunit = (str(hdr.get("CUNIT1", "")).strip() or str(hdr.get("WCAX1U", "")).strip()).lower()
        if cunit in ("angstrom", "angstroem", "a", "ang"):
            w_nm = (w * u.AA).to(u.nm)
        elif cunit in ("nm", "nanometer", "nanometre", ""):
            w_nm = w * u.nm
        elif cunit in ("um", "micron", "micrometer", "micrometre", "µm"):
            w_nm = (w * u.micron).to(u.nm)
        else:
            w_nm = w * u.nm  # default
        # Flux: MERGE1D typically F_lambda; header rarely carries unit → keep dimensionless here.
        return w_nm, y, hdr

def _get_object_date(hdr):
    obj = _get_header_value(hdr, ["OBJECT", "HIERARCH ESO OBS TARG NAME", "HIERARCH ESO OBS NAME"], "TARGET")
    date = _get_header_value(hdr, ["DATE-OBS", "HIERARCH ESO TPL START", "MJD-OBS"], "")
    # Compact date for filenames if DATE-OBS like '2025-05-20T00:23:04'
    date_for_title =  _get_header_value(hdr, ["DATE-OBS"], "")
    date_for_file = re.sub(r'[:T\- ]', '', date.split('.')[0]) if date else ""
    return obj, date_for_title, date_for_file
