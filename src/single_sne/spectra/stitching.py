from __future__ import annotations
import numpy as np
import warnings
import astropy.units as u
from specutils import Spectrum
from pathlib import Path
from astropy.units import Quantity
from typing import Tuple, Optional, Union

from single_sne.units import INSTRUMENT_UNITS
from single_sne.io.flamingos import read_flamingos_dat
from single_sne.io.salt import read_salt_dat
from single_sne.spectra.spectra import to_fnu_mjy



__all__ = ["scale_by_overlap", 
            "stitch_two_arms", 
            ]

def _as_quantity(x, unit=None):
    """Return x as a Quantity if it has no unit and `unit` is given; else pass through."""
    if isinstance(x, u.Quantity):
        return x
    return x * unit if unit is not None else x

    

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

def _clean(w, f, err):
    """Sort by wavelength, drop NaNs, ensure strictly increasing."""
    w = np.asarray(w)
    f = np.asarray(f)
    err = np.asarray(err)
    if w.shape != f.shape:
            raise ValueError(f"‼️Shape mismatch: wavelength {w.shape} vs flux {f.shape}")
    
    m = np.isfinite(w) & np.isfinite(f) & np.isfinite(err)
    w, f, err = w[m], f[m], err[m]
    if w.size == 0:
        return w, f, err
    order = np.argsort(w)
    w, f, err = w[order], f[order], err[order]
    # drop exact duplicates in w to avoid join weirdness
    if w.size > 1:
        keep = np.concatenate([[True], np.diff(w) > 0])
        w, f, err = w[keep], f[keep], err[keep]
    return w, f, err

def ensure_left_right_by_wmin(w1, f1, w2, f2, *, debug=False):
    """
    Ensure (w_left,f_left) is the lower-wavelength arm and (w_right,f_right) the higher.
    Works with ndarray or Quantity. NaNs ignored in min/max tests.
    Returns: w_left, f_left, w_right, f_right, swapped(bool)
    """
    # normalize to numeric for comparisons (preserve units in outputs)
    if hasattr(w1, "to_value"):
        wu = w1.unit
        a1 = np.asarray(w1.to_value(wu))
        a2 = np.asarray(w2.to_value(wu))
    else:
        a1 = np.asarray(w1, float)
        a2 = np.asarray(w2, float)

    # robust mins/max (ignore NaN)
    m1 = np.nanmin(a1) if a1.size else np.inf
    m2 = np.nanmin(a2) if a2.size else np.inf
    M1 = np.nanmax(a1) if a1.size else -np.inf
    M2 = np.nanmax(a2) if a2.size else -np.inf

    # decide if we need to swap: prefer the one with smaller min as "left".
    # if mins equal, use max as tiebreaker.
    swap = False
    if (m1 > m2) or (m1 == m2 and M1 > M2):
        w1, w2 = w2, w1
        f1, f2 = f2, f1
        swap = True
        if debug:
            print(f"[order] swapped: left min={m2:.6g}, right min={m1:.6g}")
    else:
        if debug:
            print(f"[order] kept: left min={m1:.6g}, right min={m2:.6g}")

    return w1, f1, w2, f2, swap

def _scale_in_overlap(
    w1, f1, w2, f2,
    ov_lo, ov_hi,
    stat: str = "median",
    require_min_points: int = 5,
    eps: float = 1e-30,
    return_info: bool = False,
    scale_to_right: bool = False,
    debug = False,
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
    #Guarantee that w1, f1 refer to the smaller wavelengths, w2, f2 to the bigger ones. 
    #If scale to right is actived, scales to the bigger wavelength no matter what order they were given in:
    w1,f1,w2,f2, _ = ensure_left_right_by_wmin(w1,f1,w2,f2, debug=debug)
            
    
    # --- 1) Normalize to common units if Quantities are passed
    # Track whether we had units to decide the return type.
    had_units = hasattr(w1, "unit") or hasattr(w2, "unit") or hasattr(f1, "unit") or hasattr(f2, "unit")
    if debug: print(f"[_scale_in_overlap] Had units: {had_units}")

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
        if debug: print(f"[_scale_in_overlap] LO or HI value not finite. Returning s=1")
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
        if debug: print(f"[_scale_in_overlap] Inside sort w2: Returning s={s}")
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
        if debug: print(f"[_scale_in_overlap] Inside clip to manual usable window. Returning s={s}")
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

    if scale_to_right:
        ratios = Y2[good] / Y1[good]
    else:
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
    if debug: print(f"[_scale_in_overlap] End of function. Returning s={s}")
    return s, n_good

def _maybe_to(q: u.Quantity, target: Optional[u.Unit]) -> u.Quantity:
    return q if target is None else q.to(u.Unit(target))

def _is_quantity(x: Any) -> bool:
    return hasattr(x, "unit") and isinstance(x, u.Quantity)
    
def _as_quantity(
    x: Any,
    y: Any | None = None,
    z: Any | None = None,
    *,
    wave_unit: str | u.Unit | None = None,
    flux_unit: str | u.Unit | None = None,
    debug: bool = False,) -> Tuple[Quantity, Quantity]:
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
    x, y, z
        See above. If `x` is a Spectrum/Spectrum1D, `y`, z is ignored.
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
        eq = _maybe_to(z, flux_unit)
        if debug:
            print("[_as_quantity] from Quantities:", wq.unit, fq.unit)
        # Basic length check
        if wq.shape != fq.shape != eq.shape:
            raise ValueError(f"‼️Shape mismatch: wavelength {wq.shape} vs flux {fq.shape}")
        if eq is not None:
            if wq.shape!= eq.shape:
                raise ValueError(f"‼️Shape mismatch: wavelength {wq.shape} vs flux_error{eq.shape}")
            else:
                return wq, fq, eq
                
        else:
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



def stitch_arms(
        wave_left,
        flux_left,
        err_left,
        wave_right,
        flux_right,
        err_right,
        *,
        overlap=(550, 555) * u.nm,
        stitch_edge=555 * u.nm,
        scale_stat="median",
        scale_to_right=False,
        return_scaled = False, 
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
    wave_left, flux_left, err_left = _as_quantity(wave_left, flux_left, err_left)
    wave_right, flux_right, err_right = _as_quantity(wave_right, flux_right, err_right)
    

    # --- unit checks & convert to common units
    wl_unit = wave_left.unit
    fl_unit = flux_left.unit
    el_unit = err_left.unit


    
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
        
    try:
        if not err_right.unit.is_equivalent(el_unit):
            raise u.UnitConversionError(f"Flux units not convertible: {err_right.unit} ↔ {el_unit}")
        err_right = err_right.to(el_unit)
    except Exception:
        raise u.UnitConversionError(
            "Left and right flux error units are not convertible. "
            "If one is Fν and the other is Fλ, convert explicitly (e.g., fν↔fλ) before stitching."
        )     
    ov_lo = overlap[0].to(wl_unit).value
    ov_hi = overlap[1].to(wl_unit).value
    if not (ov_lo < ov_hi):
        raise ValueError("Invalid overlap bounds: need overlap[0] < overlap[1].")
    edge = u.Quantity(stitch_edge).to(wl_unit).value

    # --- strip units for internal ops
    w1, f1, e1 = _clean(wave_left.to_value(wl_unit), flux_left.to_value(fl_unit), err_left.to_value(el_unit))
    w2, f2, e2 = _clean(wave_right.to_value(wl_unit), flux_right.to_value(fl_unit), err_right.to_value(el_unit))

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
    s, n  = _scale_in_overlap(w1, f1, w2, f2, ov_lo, ov_hi, stat=scale_stat, scale_to_right=scale_to_right)
    if scale_to_right:
        if debug:
            print(f"[stitch] scale factor (right→left) = {s}")
        f1s = f1 * s
        e1s = e1 * s
        f2s = f2
        e2s = e2
    else:
        if debug:
            print(f"[stitch] scale factor (left→right) = {s}")
        f1s = f1
        e1s = e1
        f2s = f2 * s
        e2s = e2 * s
        

    # --- select sides around the stitch edge
    left_mask  = w1 <= edge
    right_mask = w2 >  edge

    w_comb = np.concatenate([w1[left_mask], w2[right_mask]])
    f_comb = np.concatenate([f1s[left_mask], f2s[right_mask]])
    err_comb = np.concatenate([e1s[left_mask], e2s[right_mask]])

    # final clean (just in case)
    w_comb, f_comb, err_comb = _clean(w_comb, f_comb, err_comb)
    #sanity check: are there units?

    if debug: print(f"[stitch_arms]Wave has units after scaling:{isinstance(w_comb, Quantity)}");print(f"[stitch_arms]Flux has units after scaling:{isinstance(f_comb, Quantity)}")
    
    if return_scaled:
        scaled_right_wave = w2 * wl_unit
        scaled_right_flux = f2s * fl_unit
        scaled_right_error = e2s * fl_unit
        return w_comb * wl_unit, f_comb * fl_unit, err_comb*el_unit, float(s), (scaled_right_wave, scaled_right_flux, scaled_right_error)
    else:
        return w_comb * wl_unit, f_comb * fl_unit, err_comb * el_unit, float(s)


def combine_salt_dir(sa_dir: Path, *, jwst_units: bool = False, debug: bool = False):
    """
    Read all *.dat in a SALT directory, scale to a common reference using
    overlap, stitch into one spectrum, and (optionally) convert to JWST units.
    
    Returns
    -------
    tag, w_out, f_out
    where tag == "SALT"
    """
    if debug:print(f"\n\n[combine_salt_dir]---------------------------")
    if debug: print(f"Convert to JWST units:{jwst_units}")
    salt_files = sorted(sa_dir.glob("*.dat"))
    if debug: print(f"Files found: {list(salt_files)}")
    if not salt_files:
        raise FileNotFoundError("No SALT .dat files found")

    # 1) read all SALT pieces (Å, erg/s/cm^2/Å)
    segs = []
    for dat in salt_files:
        if debug: print(f"[combine_salt_dir] Trying to read dat: {dat.name}")
        w_s, f_s, _fixed = read_salt_dat(dat, require_increasing="sort", zero_tol=0.0)
        if debug: print(f"[combine_salt_dir] Read salt dat: Length: {len(w_s)}, {len(f_s)}")
        segs.append((w_s, f_s))
        if debug:
            print(f"[SALT] {dat.name}: {w_s.min():.1f}–{w_s.max():.1f} Å (N={w_s.size})")

    # 2) scale+stitch all segments
    w_comb, f_comb = segs[0]
    if len(segs)>1:
        for (w_next, f_next) in segs[1:]:

            # compute overlap window in wavelength
            lo = np.maximum(w_comb.min(), w_next.min())
            hi = np.minimum(w_comb.max(), w_next.max())

            if (hi - lo) > (1.0 * u.AA):  # require at least ~1 Å overlap to compute a scale
                # scale right->left (next -> current combined)
                s, _ = _scale_in_overlap(
                    w_comb.to_value(u.AA), f_comb.to_value(FU),
                    w_next.to_value(u.AA), f_next.to_value(FU),
                    lo.to_value(u.AA), hi.to_value(u.AA),
                    stat="median",
                )
                f_next = f_next * s
                if debug:
                    print(f"[SALT] scale(next→comb)={s:.6g} using overlap {lo:.2f}–{hi:.2f} Å")
                # join near the upper 80% of the overlap
                edge = (lo + 0.8*(hi - lo))
                w_comb, f_comb, _ = stitch_arms(
                    w_comb, f_comb, w_next, f_next,
                    overlap=(lo, hi), stitch_edge=edge, scale_stat="median", debug=debug
                )
            else:
                # no overlap: just append by simple concatenation respecting edge
                edge = w_comb.max()
                w_comb, f_comb, _ = stitch_arms(
                    w_comb, f_comb, w_next, f_next,
                    overlap=(edge, edge), stitch_edge=edge, scale_stat="median", debug=debug
                )
        else:
            w_comb = w_s
            f_comb = f_s
        
    if debug: print(f"[combine_salt_dir] Returning array of length {len(w_comb)}")
    return "SALT", w_comb, f_comb   


def combine_flamingos_dir(fl_dir: Path, *, jwst_units: bool = False, debug: bool = False):
    """
    Read all FLAMINGOS *.dat in an epoch directory, scale to a common reference via
    overlap, stitch into one spectrum, and (optionally) convert to JWST units.

    Returns
    -------
    tag, w_out, f_out, fe_out
        tag == "FLAMINGOS"
        w_out : Quantity (Å by default, or µm if jwst_units=True)
        f_out : Quantity (erg s^-1 cm^-2 Å^-1 by default, or mJy if jwst_units=True)
        fe_out: Quantity (same unit as f_out), or None if no errors were present anywhere
    """
    flam_files = sorted(fl_dir.glob("*.dat"))
    if debug: 
        print(f"\n\n\n\n\n\n\n\n\n\n[combine_flamingos_dir]Files found: {list(flam_files)}")
        print(f"Convert to JWST units: {jwst_units}")
    if not flam_files:
        raise FileNotFoundError("No FLAMINGOS .dat files found")
    WU = INSTRUMENT_UNITS["FLAMINGOS2"][0]
    FU = INSTRUMENT_UNITS["FLAMINGOS2"][1]
    if debug: 
        print(f"Instrument units---------------")
        print(f"Wavelength unit: {WU}")
        print(f"Flux unit: {FU}")
        print(f"--------------------------------")

    # 1) Read all segments (λ[Å], Fλ, σFλ)
    segs = []
    any_errors = False
    for dat in flam_files:
        w, f, fe, _ = read_flamingos_dat(dat, require_increasing="sort", debug=debug)
        if fe is not None:
            any_errors = True
        segs.append((w, f, fe))
        if debug:
            rng = f"[{w.min():.1f}–{w.max():.1f}] Å"
            print(f"[FLAM] {dat.name}: N={w.size}  λ∈{rng}  has_err={fe is not None}")
            print(f"[FLAM]Units present? Wave: {w.unit}, Flux: {f.unit}, Flux error:{fe.unit}")
    
    # If no errors anywhere, we'll keep fe arrays as None
    def _zero_like(q, debug = False):
        if debug:print(f"[_zero_like]{q.name} is a Quantity: {isinstance(q, Quantity)}")
        
        if isinstance(q, Quantity):
            if debug:print(f"Given units for {q.name}: {q.unit}")
            return np.zeros_like(q.value) * q.unit
        else:
            return np.zeros_like(q)

    # 2) scale + stitch all segments
    w_comb, f_comb, fe_comb = segs[0]
    if fe_comb is None and any_errors:
        fe_comb = _zero_like(f_comb)  # start an uncertainty track if some files have it

    for (w_next, f_next, fe_next) in segs[1:]:
        if any_errors and fe_next is None:
            fe_next = _zero_like(f_next)
            
        if not isinstance(w_comb, Quantity):
            w_comb = w_comb*WU
            #sanity check: Did it become a Quantity?
            f_comb = f_comb*FU
            if fe_comb is not None: fe_comb = fe_comb* FU
        
        if isinstance(w_next, Quantity):
            if debug: print(f"[combine_flamingos_dir]Read in w_next is a Quantity. Units: {w_next.unit}")
        else:
            if debug: print(f"[combine_flamingos_dir]Read in wavelength is unitless. Adding {WU}")
            w_next = w_next*WU
            #sanity check: Did it become a Quantity?
            f_next = f_next*FU
            if fe_next is not None: fe_next = fe_next* FU
            

        # compute overlap window
        lo = np.maximum(w_comb.min(), w_next.min())
        hi = np.minimum(w_comb.max(), w_next.max())
        
        if debug: print(f"---------------OVERLAP for FLAMINGOS STITCHING:\n {lo} to {hi}")
        if (hi - lo) > (1.0 * WU):
            edge = hi - 0.95 * (hi - lo)
            #edge = lo + 0.8 * (hi - lo)
            if debug: print(f"Edge for stitching: {edge}")
        else:
            edge = w_comb.max()
            
    
        # stitch flux with your utility…
        w_new, f_new, fe_new, _ = stitch_arms(
            w_comb, f_comb, fe_comb, w_next, f_next, fe_next, 
            overlap=(lo, hi), stitch_edge=edge, scale_stat="median", scale_to_right=True, debug=debug
        )
        if debug:print(f"[FLAM] Unit check after stitch_arms: {w_new.unit, f_new.unit}")
        # stitch flux with your utility…
        

        # …and stitch uncertainties using the same masks/edge rule
        # replicate stitch_arms masks to splice fe arrays consistently
        left_mask  = (w_comb <= edge)
        right_mask = (w_next >  edge)

        if fe_comb is None and fe_next is None:
            if debug: print(f"[FLAM] No flux errors in dataset")
            fe_new = None
        else:
            if debug: print(f"[FLAM] Dataset contains errors column")
            if fe_comb is None:  # start track if missing
                fe_comb = _zero_like(f_comb, debug = debug)
                
            if fe_next is None:
                fe_next = _zero_like(f_next, debug = debug)
            fe_new = u.Quantity(
                np.concatenate([fe_comb[left_mask].to_value(FU),
                                fe_next[right_mask].to_value(FU)]),
                FU
            )

        # update combined
        w_comb, f_comb, fe_comb = w_new, f_new, fe_new
        if debug:print(f"[FLAM] Unit check after everything: {w_new.unit, f_new.unit, fe_new.unit}")
        
    return "FLAMINGOS", w_comb, f_comb, fe_comb

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