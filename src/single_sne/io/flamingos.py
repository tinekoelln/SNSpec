from __future__ import annotations
import pathlib
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from typing import Literal, Tuple, Optional
from astropy.units import Quantity
from single_sne.units import INSTRUMENT_UNITS
from single_sne.spectra.spectra import is_strictly_increasing
from single_sne.io.clean_data import clean_data, fix_negative_flux, drop_high_variance_spikes

WAVE_UNIT = INSTRUMENT_UNITS["FLAMINGOS2"][0]
FLUX_UNIT = INSTRUMENT_UNITS["FLAMINGOS2"][1]
__all__ = [
    "discover_merge1d_files",
    "read_primary_linear_wcs",
    "object_and_date_from_header",
]
import numpy as np
import astropy.units as u
from typing import Optional, Tuple

PathLike = Union[str, pathlib.Path]

def read_flamingos_dat(
    path:PathLike,
    *,
    as_quantity: bool = True,
    require_increasing: Literal["strict", "sort", "warn"] = "strict",
    dedup_tol: float = 0.0,     # in same unit as wavelength (Å); e.g. 1e-6 to drop exact dupes
    debug: bool = False,
) -> Tuple[u.Quantity, u.Quantity, Optional[u.Quantity], bool]:
    """
    Read a FLAMINGOS .dat spectrum: λ[Å], Fλ[erg/s/cm²/Å], (optional) σ_Fλ.
    Ensures (or fixes) strictly increasing wavelength.

    Returns
    -------
    wave, flux, flux_err, fixed : (Quantity, Quantity, Quantity|None, bool)
        `fixed` is True if sorting/dedup was applied.
    """
    if debug:print(f"[read_flamingos_dat] Reading flamingos data... ")
    arr = np.genfromtxt(path, comments="#", dtype=float)
    arr_trimmed = arr[:-50]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: need ≥2 columns (wavelength, flux[, flux_err])")

    w = arr_trimmed[:, 0]
    f = arr_trimmed[:, 1]
    fe = arr_trimmed[:, 2] if arr.shape[1] >= 3 else None

    # Drop rows with NaN in required cols
    m = np.isfinite(w) & np.isfinite(f)
    m &= (w > 13300) #Cutting off at 1.3um
    if fe is not None:
        m &= np.isfinite(fe)
    w, f = w[m], f[m]
    if fe is not None:
        fe = fe[m]
        
    

    if w.size == 0:
        raise ValueError(f"{path}: no valid rows after cleaning")

    # Monotonic control

    fixed = False
    if not is_strictly_increasing(w):
        if require_increasing == "strict":
            raise ValueError("Wavelength is not strictly increasing; set require_increasing='sort' to auto-fix.")
        elif require_increasing == "warn":
            if debug:
                print("[WARN] wavelength not strictly increasing (kept as-is)")
        elif require_increasing == "sort":
            # sort by wavelength
            idx = np.argsort(w, kind="mergesort")
            w, f = w[idx], f[idx]
            if fe is not None:
                fe = fe[idx]
            if debug:
                print("[WARN] wavelength not strictly increasing, sorted file")

            # optional de-duplication within tolerance
            if dedup_tol >= 0.0:
                keep = [0]
                for i in range(1, w.size):
                    if (w[i] - w[keep[-1]]) > dedup_tol:
                        keep.append(i)
                    else:
                        # if duplicate/near-duplicate, keep the last sample (or average if you prefer)
                        keep[-1] = i
                keep = np.asarray(keep, int)
                w, f = w[keep], f[keep]
                if fe is not None:
                    fe = fe[keep]
            # re-check strictness
            if not is_strictly_increasing(w):
                raise ValueError("Failed to enforce strictly increasing wavelength after sort/dedup.")
            fixed = True

    if as_quantity:
        w = w * WAVE_UNIT
        f = f * FLUX_UNIT
        fe = fe * FLUX_UNIT if fe is not None else None
        
    w, f, fe = clean_data(w, f, fe, drop_spikes=True)
    if debug: 
            print(f"[read_flamingos_dat]After clean_data")
            print(f"[read_flamingos_dat] Unit check for wave: {isinstance(w, Quantity)}")
            print(f"[read_flamingos_dat] Unit check for flux: {isinstance(f, Quantity)}")
            print(f"[read_flamingos_dat] Unit check for flux_error: {isinstance(fe, Quantity)}")
    w, f, fe = fix_negative_flux(w, f, fe, mode="interp", debug=debug)
    
    if debug: 
            print(f"\n[read_flamingos_dat] After fixing negative flux")
            print(f"[read_flamingos_dat] Unit check for wave: {isinstance(w, Quantity)}")
            print(f"[read_flamingos_dat] Unit check for flux: {isinstance(f, Quantity)}")
            print(f"[read_flamingos_dat] Unit check for flux_error: {isinstance(fe, Quantity)}")

    return w, f, fe, fixed

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