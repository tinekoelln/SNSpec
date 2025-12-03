from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from typing import Literal, Tuple, Optional, Union
from single_sne.units import INSTRUMENT_UNITS
from single_sne.spectra.spectra import is_strictly_increasing
from single_sne.io.clean_data import clean_data

WAVE_UNIT = INSTRUMENT_UNITS["FLAMINGOS2"][0]
FLUX_UNIT = INSTRUMENT_UNITS["FLAMINGOS2"][1]
__all__ = [
    "discover_merge1d_files",
    "read_primary_linear_wcs",
    "object_and_date_from_header",
]

PathLike = Union[str, Path]
def read_salt_dat(
    path: PathLike,
    *,
    as_quantity: bool = True,
    require_increasing: Literal["strict", "sort", "warn"] = "strict",
    dedup_tol: float = 0.0,   # in Å; e.g., 1e-6 drops exact dupes
    zero_tol: float = 0.0,    # drop rows with flux <= zero_tol
    debug: bool = False,
) -> Tuple[u.Quantity, u.Quantity, bool]:
    """
    Read a SALT .dat spectrum with 2 columns:
        wavelength [Å], flux [erg/s/cm²/Å]

    Drops rows with NaN and with flux <= zero_tol.
    Enforces (or fixes) strictly increasing wavelength.

    Returns
    -------
    wave, flux, fixed : (Quantity, Quantity, bool)
        `fixed` is True if sorting/dedup was applied.
    """
    if debug:
        print(f"          [read_salt_dat] Starting...")
    arr = np.genfromtxt(path, comments="#", dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: need 2 columns (wavelength, flux)")
    if debug: print(f"[read_salt_dat]: Successfully read in SALT array {path.name}")
    w = arr[:, 0]
    f = arr[:, 1]

    # Drop NaNs
    m = np.isfinite(w) & np.isfinite(f)
    # Drop zeros / near-zeros
    m &= f > zero_tol
    w, f = w[m], f[m]

    if w.size == 0:
        raise ValueError(f"{path}: no valid rows after cleaning")


    fixed = False
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

            if dedup_tol >= 0.0:
                keep = [0]
                for i in range(1, w.size):
                    if (w[i] - w[keep[-1]]) > dedup_tol:
                        keep.append(i)
                    else:
                        keep[-1] = i  # keep the last occurrence
                keep = np.asarray(keep, int)
                w, f = w[keep], f[keep]

            if not is_strictly_increasing(w):
                raise ValueError("Failed to enforce strictly increasing wavelength after sort/dedup.")
            fixed = True

    if as_quantity:
        w = w * WAVE_UNIT
        f = f * FLUX_UNIT
        #fe = fe * FLUX_UNIT if fe is not None else None

    #w, f, fe = clean_data(w, f, fe)
    return (w, f, fixed) #if fe is not None else (w, f, fixed)

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