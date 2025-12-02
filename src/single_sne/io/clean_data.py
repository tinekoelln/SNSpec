import numpy as np
from astropy import units as u 
from scipy.signal import medfilt
from astropy.units import Quantity


def fix_negative_flux(w, f, f_err=None, *, mode="interp", replace_value=0.0, debug=False):
    """
    Handle negative flux values.

    Parameters
    ----------
    w : array-like
        Wavelength array (assumed strictly increasing).
    f : array-like
        Flux array.
    f_err : array-like or None
        Optional flux error array; kept in sync if values are dropped.
    mode : {"interp", "replace", "drop"}
        How to treat negative fluxes:
        - "interp"   : replace by linear interpolation between nearest valid neighbours
        - "replace"  : set to a constant value (default: 0.0)
        - "drop"     : remove those samples entirely
    replace_value : float
        Only used if mode="replace". Default = 0.0
    debug : bool
        Print diagnostics.

    Returns
    -------
    w_out, f_out, f_err_out
        Cleaned arrays. If f_err was None, it will return only (w_out, f_out).
    """
    w_unit = getattr(w, "unit", None)
    if debug: print(f"[fix_neg_flux] Wave units:", w_unit)
    f_unit = getattr(f, "unit", None)
    
    w = np.asarray(w, float)
    f = np.asarray(f, float)
    if f_err is not None:
        f_err_unit = getattr(f_err, "unit", None)
        f_err = np.asarray(f_err, float)

    neg_mask = f < 0
    n_bad = neg_mask.sum()

    if n_bad == 0:
        return (w*w_unit if w_unit else w,
                f*f_unit if f_unit else f, 
                f_err*f_err_unit if f_err_unit else f_err) if f_err is not None else (w*w_unit if w_unit else w,
                f*f_unit if f_unit else f)

    if debug:
        print(f"[fix_negative_flux] found {n_bad} negative flux samples")

    if mode == "drop":
        keep = ~neg_mask
        w2, f2 = w[keep], f[keep]
        if f_err is not None:
            f_err2 = f_err[keep]
        return (w*w_unit if w_unit else w,
                f2*f_unit if f_unit else f2, 
                f_err2*f_err_unit if f_err_unit else f_err2) if f_err is not None else (w*w_unit if w_unit else w,
                f2*f_unit if f_unit else f2)

    if mode == "replace":
        f2 = f.copy()
        f2[neg_mask] = replace_value
        if debug:
            print(f"[fix_negative_flux] replaced negatives with {replace_value}")
        if f_err is not None:
            f_err2 = f.copy()
            f_err2[neg_mask] = replace_value
            return w2, f2, f_err2
        return (w*w_unit if w_unit else w,
                f2*f_unit if f_unit else f2, 
                f_err2*f_err_unit if f_err_unit else f_err2) if f_err is not None else (w*w_unit if w_unit else w,
                f2*f_unit if f_unit else f2)

    if mode == "interp":
        f2 = f.copy()
        good = ~neg_mask
        if good.sum() < 2:
            raise ValueError("Not enough valid points to interpolate negative fluxes")
        f2[neg_mask] = np.interp(w[neg_mask], w[good], f[good])
        if f_err is not None:
            f_err2 = f_err.copy()
            f_err2[neg_mask] = np.interp(w[neg_mask], w[good], f_err[good])
        if debug:
            print("[fix_negative_flux] applied linear interpolation to negative fluxes")
        return (w*w_unit if w_unit else w,
                f2*f_unit if f_unit else f2, 
                f_err2*f_err_unit if f_err_unit else f_err2) if f_err is not None else (w*w_unit if w_unit else w,
                f2*f_unit if f_unit else f2)

    raise ValueError(f"Unknown mode: {mode!r} (use 'interp', 'replace', or 'drop')")

def drop_high_variance_spikes(
    w,
    f,
    *,
    window=11,
    sigma=6.0,
    replace=False,
    replace_with="interp",   # "median" or "interp"
    keep_edges=True,
    debug=False,
):
    """
    Detect and optionally remove 'spiky' flux outliers relative to a rolling-median baseline.

    Parameters
    ----------
    w, f : array-like or Quantity
        Wavelength and flux arrays (same length). Units preserved if given as Quantity.
    window : int
        Kernel size for median filter (must be odd). Controls baseline smoothness.
    sigma : float
        Robust threshold in units of MAD (≈σ for Gaussian): |resid| > sigma * 1.4826*MAD -> spike.
    replace : bool
        If True, replace spikes in-place; if False, drop them entirely.
    replace_with : {"median","interp"}
        - "median": set spike flux to rolling-median baseline value.
        - "interp": linearly interpolate between nearest non-spike neighbors.
    keep_edges : bool
        If False and replace=False, edge points that cannot be safely interpolated are dropped.
        If replace=True and "interp" is chosen but neighbors are missing, falls back to "median".
    debug : bool
        Print summary stats.

    Returns
    -------
    w_out, f_out : same type as inputs
        Cleaned arrays (Quantity if inputs had units).
    mask_keep : np.ndarray[bool]
        Boolean mask of kept points (True = kept in output). If replace=True, all True.
    spike_idx : np.ndarray[int]
        Indices identified as spikes.

    Notes
    -----
    - Uses a robust global scale estimate from residuals: MAD * 1.4826.
    - Choose 'replace' when you want to preserve sampling; choose drop when you want a pure mask.
    - Make `window` larger to avoid overfitting broad features; it must be odd.
    """
    # ---- preserve units
    w_unit = getattr(w, "unit", None)
    f_unit = getattr(f, "unit", None)
    wf = np.asarray(w, float), np.asarray(f, float)
    w_arr, f_arr = wf

    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd integer >= 3")

    if w_arr.size != f_arr.size:
        raise ValueError("w and f must have the same length")

    # ---- rolling-median baseline (robust to spikes)
    baseline = medfilt(f_arr, kernel_size=window)

    # ---- robust residual scale via MAD
    resid = f_arr - baseline
    mad = np.median(np.abs(resid[np.isfinite(resid)])) if resid.size else 0.0
    scale = 1.4826 * mad if mad > 0 else (np.std(resid) if np.any(np.isfinite(resid)) else 0.0)

    if scale == 0.0:
        # Nothing to do; no dispersion detectable
        mask_keep = np.ones_like(f_arr, dtype=bool)
        spike_idx = np.array([], dtype=int)
        w_out = w * w_unit if w_unit else w_arr
        f_out = f * f_unit if f_unit else f_arr
        if debug:
            print("[spikes] zero scale -> no spikes flagged.")
        return w_out, f_out, mask_keep, spike_idx

    spikes = np.abs(resid) > sigma * scale
    spike_idx = np.flatnonzero(spikes)

    if debug:
        frac = 100.0 * spike_idx.size / max(1, f_arr.size)
        print(f"[spikes] window={window}, sigma={sigma}, MAD={mad:.3g}, "
              f"scale={scale:.3g}, flagged={spike_idx.size} ({frac:.2f}%)")

    # ---- act on spikes
    if not replace:
        # Drop spikes
        mask_keep = ~spikes
        # handle edges: optionally force-keep edges if requested
        if keep_edges and mask_keep.size:
            mask_keep[0] = True
            mask_keep[-1] = True
        w_clean = w_arr[mask_keep]
        f_clean = f_arr[mask_keep]
    else:
        # Replace spikes
        f_clean = f_arr.copy()
        if replace_with == "median":
            f_clean[spikes] = baseline[spikes]
        elif replace_with == "interp":
            # linear interpolation over spikes using nearest non-spike neighbors
            keep = ~spikes
            if keep.sum() >= 2:
                f_clean[spikes] = np.interp(
                    w_arr[spikes],
                    w_arr[keep],
                    f_arr[keep],
                )
            else:
                # Fallback if not enough good points
                f_clean[spikes] = baseline[spikes]
        else:
            raise ValueError("replace_with must be 'median' or 'interp'")

        mask_keep = np.ones_like(f_arr, dtype=bool)
        w_clean = w_arr

    # ---- reattach units
    w_out = (w_clean * w_unit) if w_unit else w_clean
    f_out = (f_clean * f_unit) if f_unit else f_clean
    
    if debug: print(f"[drop_high_variance_spikes] Quantities? Wave: {isinstance(w_out, Quantity)}; Flux: {isinstance(f_out, Quantity)}")

    return w_out, f_out, mask_keep, spike_idx

def clean_data(w, f,f_err = None, *, interp_small_gaps=True, max_gap=5, drop_spikes = False, debug=False):
    """
    Ensure finite wavelength/flux pairs.
    Optionally linearly fill **internal** NaN gaps up to `max_gap` samples.
    Edge NaNs are trimmed, not filled.

    Parameters
    ----------
    w, f : array-like
        Wavelength and flux arrays (float or Quantity with unit).
    interp_small_gaps : bool
        If True, linearly interpolate internal NaNs no longer than `max_gap`.
    max_gap : int
        Maximum number of consecutive NaNs to interpolate.
    """

    # -- Handle Quantity safely --
    w_unit = getattr(w, "unit", None)
    f_unit = getattr(f, "unit", None)

    w = np.asarray(w, float)
    f = np.asarray(f, float)
    if f_err is not None:
        f_err = np.asarray(f_err, float)

    # -- Remove global NaNs first (but do not erase internal gaps yet) --
    bad = ~np.isfinite(w) | (w == 0)
    w[bad] = np.nan
    f[bad] = np.nan
    if f_err is not None:
        f_err[bad] = np.nan

    # -- Trim leading / trailing NaNs only --
    good = np.isfinite(w)
    if not np.any(good):
        if f_err is not None:
            return w[:0], f[:0], f_err[:0]
        else:
            return w[:0], f[:0]   # all bad

    first, last = np.where(good)[0][[0, -1]]
    w = w[first:last+1]
    f = f[first:last+1]
    if f_err is not None:
        f_err = f_err[first:last+1]

    # -- Interpolate internal short NaN gaps --
    if interp_small_gaps:
        isn = ~np.isfinite(f)
        if np.any(isn):
            # find consecutive NaN runs
            idx = np.where(isn)[0]
            splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
            for block in splits:
                if len(block) <= max_gap:     # only fill short gaps
                    i0, i1 = block[0]-1, block[-1]+1
                    if i0 >= 0 and i1 < len(f):
                        f[block] = np.interp(w[block], [w[i0], w[i1]], [f[i0], f[i1]])
                        if f_err is not None:
                            f_err[block] = np.interp(w[block], [w[i0], w[i1]], [f_err[i0], f_err[i1]])

    # -- Reattach units if present --
    if w_unit is not None: w = w * w_unit
    if f_unit is not None: f = f * f_unit
    if f_err is not None: f_err = f_err*f_unit
    
    
    if drop_spikes:
        w, f, keep_mask, _= drop_high_variance_spikes(w, f, replace = False)
        if f_err is not None:
            f_err = f_err[keep_mask]

    # -- Final sort, just in case --
    if w[0] > w[-1]:
        idx = np.argsort(w)
        if f_err is not None:
            w, f, f_err = w[idx], f[idx], f_err[idx]
        else:
            w, f = w[idx], f[idx]

    if debug:
        print(f"[clean_arm] size={len(w)}, finite={np.isfinite(f).sum()}, unit={w_unit}")
    if f_err is not None:
        return w, f, f_err
    else:
        return w, f