from __future__ import annotations
import numpy as np
import warnings
import astropy.units as u
from specutils import Spectrum



__all__ = ["scale_by_overlap", 
            "stitch_two_arms", 
            ]

def _as_quantity(x, unit=None):
    """Return x as a Quantity if it has no unit and `unit` is given; else pass through."""
    if isinstance(x, u.Quantity):
        return x
    return x * unit if unit is not None else x

def _unit_str(q):
    return str(q.unit) if isinstance(q, u.Quantity) else "(dimensionless)"
    

def _ensure_compatible_units(w1, f1, w2, f2, *, warn_convert=True):
        """
        If inputs are Quantities, convert w2/f2 to match w1/f1 units.
        If unitless arrays, pass through untouched.
        Raises on non-convertible units.
        """
        # Wavelength units
        if isinstance(w1, u.Quantity) and isinstance(w2, u.Quantity):
            if w1.unit != w2.unit:
                if warn_convert:
                    warnings.warn(f"⚠️Converting wavelength units: {w2.unit} → {w1.unit}", RuntimeWarning)
                try:
                    w2 = w2.to(w1.unit)
                except u.UnitConversionError:
                    raise ValueError(f"‼️Wavelength units not convertible: {w1.unit} vs {w2.unit}")
    
        # Flux(-density) units
        if isinstance(f1, u.Quantity) and isinstance(f2, u.Quantity):
            if f1.unit != f2.unit:
                if warn_convert:
                    warnings.warn(f"⚠️Converting flux units: {f2.unit} → {f1.unit}", RuntimeWarning)
                try:
                    f2 = f2.to(f1.unit)
                except u.UnitConversionError:
                    raise ValueError(f"‼️Flux units not convertible: {f1.unit} vs {f2.unit}")
    
        return w1, f1, w2, f2

def _clean(w, f):
    """Sort by wavelength, drop NaNs, ensure strictly increasing."""
    w = np.asarray(w)
    f = np.asarray(f)
    if w.shape != f.shape:
            raise ValueError(f"‼️Shape mismatch: wavelength {w.shape} vs flux {f.shape}")
    
    m = np.isfinite(w) & np.isfinite(f)
    w, f = w[m], f[m]
    if w.size == 0:
        return w, f
    order = np.argsort(w)
    w, f = w[order], f[order]
    # drop exact duplicates in w to avoid join weirdness
    if w.size > 1:
        keep = np.concatenate([[True], np.diff(w) > 0])
        w, f = w[keep], f[keep]
    return w, f

def _window_mask(w, lo, hi):
    return (w >= lo) & (w <= hi)


def _scale_in_overlap(
    w1, f1, w2, f2,
    ov_lo, ov_hi,
    stat: str = "median",
    require_min_points: int = 5,
    eps: float = 1e-30,
    return_info: bool = False,
    # clip: float | None = None,  # e.g. 3.0 for 3-sigma clipping (needs astropy.stats)
):
    """
    Scale factor s so that s * f2 ≈ f1 over [ov_lo, ov_hi].

    Parameters
    ----------
    w1, f1, w2, f2 : array-like OR astropy Quantities
        If Quantities are provided, w1/w2 must be convertible, and f1/f2 must be convertible.
    ov_lo, ov_hi   : scalars or Quantities
        Overlap window. If Quantities, convertible to wavelength unit.
    stat           : {'median','mean'}
        Robust center for the ratio distribution.
    require_min_points : int
        Minimum number of valid (finite) pairs used.
    eps            : float
        Tiny floor to avoid division by exactly/near zero y2.
    return_info    : bool
        If True, return (s, n_used, info_dict) with window actually used.

    Returns
    -------
    s : float or dimensionless Quantity
        Dimensionless scale (Quantity if inputs were Quantities).
    n_used : int
        Number of valid ratio points used.
    info : dict (optional)
        {'window_used': (lo, hi), 'n_inside': int}
    """

    # --- 1) Normalize to common units if Quantities are passed
    # Track whether we had units to decide the return type.
    had_units = hasattr(w1, "unit") or hasattr(w2, "unit") or hasattr(f1, "unit") or hasattr(f2, "unit")

    if had_units:
        # Wavelength unit reference
        wl_unit = (w1.unit if hasattr(w1, "unit") else w2.unit)
        W1 = np.asarray(u.Quantity(w1, wl_unit).to_value(wl_unit))
        W2 = np.asarray(u.Quantity(w2, wl_unit).to_value(wl_unit))
        # Flux unit reference
        fl_unit = (f1.unit if hasattr(f1, "unit") else f2.unit)
        F1 = np.asarray(u.Quantity(f1, fl_unit).to_value(fl_unit))
        F2 = np.asarray(u.Quantity(f2, fl_unit).to_value(fl_unit))
        # Window to wavelength unit
        LO = u.Quantity(ov_lo, wl_unit).to_value(wl_unit) if hasattr(ov_lo, "unit") else float(ov_lo)
        HI = u.Quantity(ov_hi, wl_unit).to_value(wl_unit) if hasattr(ov_hi, "unit") else float(ov_hi)
    else:
        W1, F1, W2, F2 = map(lambda a: np.asarray(a, float), (w1, f1, w2, f2))
        LO, HI = float(ov_lo), float(ov_hi)

    # Ensure LO < HI and finite
    if not np.isfinite(LO) or not np.isfinite(HI) or HI <= LO:
        s = 1.0 * u.dimensionless_unscaled if had_units else 1.0
        return (s, 0, {'window_used': (LO, HI), 'n_inside': 0}) if return_info else (s, 0)

    # --- 2) Sort w2 and drop duplicate wavelengths (interp requires ascending xp)
    o2 = np.argsort(W2)
    W2s, F2s = W2[o2], F2[o2]
    if W2s.size > 1:
        keep = np.concatenate([[True], np.diff(W2s) > 0])
        W2s, F2s = W2s[keep], F2s[keep]

    if W1.size == 0 or W2s.size == 0:
        s = 1.0 * u.dimensionless_unscaled if had_units else 1.0
        return (s, 0, {'window_used': (LO, HI), 'n_inside': 0}) if return_info else (s, 0)

    # --- 3) Clip to mutual usable window (no extrapolation)
    lo_eff = max(LO, W1.min(), W2s.min())
    hi_eff = min(HI, W1.max(), W2s.max())
    if not np.isfinite(lo_eff) or not np.isfinite(hi_eff) or hi_eff <= lo_eff:
        s = 1.0 * u.dimensionless_unscaled if had_units else 1.0
        return (s, 0, {'window_used': (lo_eff, hi_eff), 'n_inside': 0}) if return_info else (s, 0)

    m1 = (W1 >= lo_eff) & (W1 <= hi_eff)
    if not np.any(m1):
        s = 1.0 * u.dimensionless_unscaled if had_units else 1.0
        return (s, 0, {'window_used': (lo_eff, hi_eff), 'n_inside': 0}) if return_info else (s, 0)

    X  = W1[m1]
    Y1 = F1[m1]

    # Interpolate F2 onto X (strictly inside W2s range due to lo_eff/hi_eff)
    Y2 = np.interp(X, W2s, F2s)

    # --- 4) Build robust mask: finite, denom not tiny
    good = np.isfinite(Y1) & np.isfinite(Y2) & (np.abs(Y2) > eps)
    n_good = int(np.count_nonzero(good))
    if n_good < require_min_points:
        s = 1.0 * u.dimensionless_unscaled if had_units else 1.0
        return (s, n_good, {'window_used': (lo_eff, hi_eff), 'n_inside': n_good}) if return_info else (s, n_good)

    ratios = Y1[good] / Y2[good]

    # Optional: sigma clipping for extra robustness
    # if clip is not None:
    #     ratios = sigma_clip(ratios, sigma=float(clip), maxiters=3, masked=False)

    if stat == "mean":
        sval = float(np.nanmean(ratios))
    else:
        sval = float(np.nanmedian(ratios))

    if not np.isfinite(sval) or sval == 0.0:
        sval = 1.0

    s = sval * u.dimensionless_unscaled if had_units else sval

    if return_info:
        return s, n_good, {'window_used': (lo_eff, hi_eff), 'n_inside': n_good}
    return s, n_good

def _maybe_to(q: u.Quantity, target: Optional[u.Unit]) -> u.Quantity:
    return q if target is None else q.to(u.Unit(target))

def _is_quantity(x: Any) -> bool:
    return hasattr(x, "unit") and isinstance(x, u.Quantity)
    
def _as_quantity(
    x: Any,
    y: Any | None = None,
    *,
    wave_unit: str | u.Unit | None = None,
    flux_unit: str | u.Unit | None = None,
    debug: bool = False,
) -> Quantities:
    """
    Normalize a variety of spectrum-like inputs to (wavelength, flux) Quantities.

    Accepted inputs
    ---------------
    1) specutils.Spectrum  (preferred)  -> uses .spectral_axis, .flux
    3) (Quantity, Quantity)             -> optional unit conversion
    4) (Quantity, array) or (array, Quantity) -> supply missing unit
    5) (array, array)                   -> wave_unit and flux_unit REQUIRED

    Parameters
    ----------
    x, y
        See above. If `x` is a Spectrum/Spectrum1D, `y` is ignored.
    wave_unit, flux_unit
        Optional target units to convert to. If a plain array is given, these
        are REQUIRED for that component.
    debug : bool
        Print what path was taken and the resulting units.

    Returns
    -------
    (wavelength, flux) as astropy.units.Quantity
    """
    if _is_quantity(x) and _is_quantity(y):
        wq = _maybe_to(x, wave_unit)
        fq = _maybe_to(y, flux_unit)
        if debug:
            print("[_as_quantity] from Quantities:", wq.unit, fq.unit)
        # Basic length check
        if wq.shape != fq.shape:
            raise ValueError(f"‼️Shape mismatch: wavelength {wq.shape} vs flux {fq.shape}")
        return wq, fq
    
    #Spectra inputs
    if isinstance(x, Spectrum):
        wq = x.spectral_axis
        fq = x.flux
        # Optional conversion
        wq = _maybe_to(wq, wave_unit)
        fq = _maybe_to(fq, flux_unit)
        if debug:
            print("[_as_quantity] from Spectrum:", wq.unit, fq.unit)
        return wq, fq

    #Plain arrays
    if wave_unit is None or flux_unit is None:
        raise ValueError("‼️For plain arrays, wave_unit and flux_unit must be provided.")
    wq = np.asarray(x, dtype=float) * u.Unit(wave_unit)
    fq = np.asarray(y, dtype=float) * u.Unit(flux_unit)
    if debug:
        print("[_as_quantity] from arrays:", wq.unit, fq.unit)
    if wq.shape != fq.shape:
        raise ValueError(f"‼️Shape mismatch: wavelength {wq.shape} vs flux {fq.shape}")
    return wq, fq


def _style_of_inputs(x, y):
    """Return 'spectrum' | 'quantity' | 'ndarray' to mirror output style."""
    if isinstance(x, Spectrum) or isinstance(y, Spectrum):
        return "spectrum"
    if hasattr(x, "unit") or hasattr(y, "unit"):
        return "quantity"
    return "ndarray"


def _return_like(style: str, spec: Spectrum):
    """Convert Spectrum result back to the requested style."""
    w = spec.spectral_axis            # Quantity
    f = spec.flux                     # Quantity
    if style == "spectrum":
        return spec
    if style == "quantity":
        return w, f
    # ndarray
    return w.value, f.value



def stitch_arms(
        wave_left,
        flux_left,
        wave_right,
        flux_right,
        *,
        overlap=(550, 555) * u.nm,
        stitch_edge=555 * u.nm,
        scale_stat="median",
        debug=False,
    ):
    """
    Stitch two spectra: (left) + (right).
    1) Scale RIGHT to LEFT using overlap window.
    2) Concatenate LEFT[<= edge] + (scaled RIGHT)[> edge].

    Parameters
    ----------
    wave_left, flux_left : Quantity[float]
    wave_right, flux_right : Quantity[float]
        Wavelength and flux arrays with units (must be convertible).
    overlap : tuple Quantity
        (lo, hi) window to compute the scale factor.
    stitch_edge : Quantity
        Boundary wavelength where you switch from LEFT to RIGHT.
    scale_stat : {"median","mean"}
        Statistic to compute scale.
    debug : bool
        Print details.

    Returns
    -------
    wave_comb : Quantity
    flux_comb : Quantity
    scale_right : float
        Multiplicative factor applied to RIGHT.
    """

    #--- Test input style:
    wave_left, flux_left = _as_quantity(wave_left, flux_left)
    wave_right, flux_right = _as_quantity(wave_right, flux_right)
    

    # --- unit checks & convert to common units
    wl_unit = wave_left.unit
    fl_unit = flux_left.unit


    
    try:
        if not wave_right.unit.is_equivalent(wl_unit):
            raise u.UnitConversionError(f"Wavelength units not convertible: {wave_right.unit} ↔ {wl_unit}")
        wave_right = wave_right.to(wl_unit)
    except Exception:
        raise u.UnitConversionError("Left and right wavelength units are not convertible.")

    try:
        if not flux_right.unit.is_equivalent(fl_unit):
            raise u.UnitConversionError(f"Flux units not convertible: {flux_right.unit} ↔ {fl_unit}")
        flux_right = flux_right.to(fl_unit)
    except Exception:
        raise u.UnitConversionError(
            "Left and right flux units are not convertible. "
            "If one is Fν and the other is Fλ, convert explicitly (e.g., fν↔fλ) before stitching."
        )    
    ov_lo = overlap[0].to(wl_unit).value
    ov_hi = overlap[1].to(wl_unit).value
    if not (ov_lo < ov_hi):
        raise ValueError("Invalid overlap bounds: need overlap[0] < overlap[1].")
    edge = u.Quantity(stitch_edge).to(wl_unit).value

    # --- strip units for internal ops
    w1, f1 = _clean(wave_left.to_value(wl_unit), flux_left.to_value(fl_unit))
    w2, f2 = _clean(wave_right.to_value(wl_unit), flux_right.to_value(fl_unit))

    if debug:
        def _rng(w): 
            return f"{w[0]:.6g}–{w[-1]:.6g}" if w.size else "∅"
        print(f"[stitch] left  range: { _rng(w1) }  (n={w1.size})")
        print(f"[stitch] right range: { _rng(w2) }  (n={w2.size})")
        print(f"[stitch] overlap: [{ov_lo:.6g}, {ov_hi:.6g}]  edge={edge:.6g} ({wl_unit})")

    if w1.size == 0 and w2.size == 0:
        return (np.array([]) * wl_unit, np.array([]) * fl_unit, 1.0)
    if w1.size == 0:
        return (wave_right, flux_right, 1.0)
    if w2.size == 0:
        return (wave_left, flux_left, 1.0)

    # --- scale right to left in overlap, falls back to s = 1 if not enough points
    s, n  = _scale_in_overlap(w1, f1, w2, f2, ov_lo, ov_hi, stat=scale_stat)
    if debug:
        print(f"[stitch] scale factor (right→left) = {s}")
    f2s = f2 * s

    # --- select sides around the stitch edge
    left_mask  = w1 <= edge
    right_mask = w2 >  edge

    w_comb = np.concatenate([w1[left_mask], w2[right_mask]])
    f_comb = np.concatenate([f1[left_mask], f2s[right_mask]])

    # final clean (just in case)
    w_comb, f_comb = _clean(w_comb, f_comb)

    return w_comb * wl_unit, f_comb * fl_unit, float(s)
