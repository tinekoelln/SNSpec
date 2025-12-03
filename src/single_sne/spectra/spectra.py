from __future__ import annotations
import numpy as np
from typing import Tuple
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.nddata import StdDevUncertainty, VarianceUncertainty
try:
    from astropy.nddata import InverseVariance  # specutils/astropy variant name
except Exception:
    InverseVariance = None  # not all versions expose thisfrom astropy import units as u
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
import matplotlib.pyplot as plt
from astropy import constants as const




__all__ = [
    "is_strictly_increasing"
    "to_fnu_mjy",
    "bin_spectrum",
]

# Monotonic control
def is_strictly_increasing(w, debug=False) -> bool:
    """
    True if array is strictly increasing (no equal neighbors), ignoring NaNs.
    Works for numpy arrays and astropy Quantities.
    """
    if debug: print(f"[is_strictly_increasing] Begin debugging...")
    # Convert to plain float values (preserve ordering) no matter if Quantity or ndarray
    if hasattr(w, "to_value"):
        a = np.asarray(w.to_value(getattr(w, "unit", 1.0)), dtype=float)
        if debug: print(f"[is_strictly_increasing]: quantity converted to numpy array"); print(a)
        
    else:
        a = np.asarray(w, dtype=float)
        if debug: print(f"numpy array given, no conversion needed")

    # Drop non-finite
    m = np.isfinite(a)
    b = a[m]
    if debug:
        if b.size!=a.size:
            print(f"Nonfinite values dropped")
    if b.size < 2:
        return True
    ok = np.all(np.diff(b) > 0.0)
    if debug:
        print(f"[mono] size={b.size}, min/max={b.min():.6g}/{b.max():.6g}, strictly_inc={bool(ok)}")
    return bool(ok)

def to_fnu_mjy(
    wavelength,
    f_lambda,
    f_lambda_err=None,
    wavelength_err=None,
    *,
    debug=False,
):
    """
    Convert F_lambda (erg s^-1 cm^-2 Å^-1) at wavelength (Å) to F_nu in mJy.
    Accepts numpy arrays or astropy Quantities for all inputs.

    Parameters
    ----------
    wavelength : array-like or Quantity
        Wavelength. If unitless, assumed Å.
    f_lambda : array-like or Quantity
        Flux density per wavelength. If unitless, assumed erg/(s cm^2 Å).
    f_lambda_err : array-like or Quantity, optional
        1σ uncertainty on F_lambda. If unitless, assumed erg/(s cm^2 Å).
    wavelength_err : array-like or Quantity, optional
        1σ uncertainty on wavelength. If unitless, assumed Å.
    debug : bool

    Returns
    -------
    Fnu_mjy : Quantity
        Flux density per frequency in mJy.
    Fnu_err_mjy : Quantity or None
        1σ uncertainty in mJy (None if no uncertainties provided).
    """
    if debug: print(f"Entered the function")
    
    if debug:
        print(f"Length wavelength vector: {len(wavelength)}")
        print(f"Length flux vector: {len(f_lambda)}")
        if f_lambda_err is not None: print(f"Length flux error vector: {len(f_lambda_err)}")
    
    
    # Normalize wavelength to Å
    if isinstance(wavelength, u.Quantity):
        lam = wavelength.to(u.AA)
        if debug: print(f"Quantity for wavelength found")
    else:
        if debug: print(f"No wavelength units provided; assuming Angstrom.")
        lam = np.asarray(wavelength, float) * u.AA

    # Normalize F_lambda to erg s^-1 cm^-2 Å^-1
    flam_unit = u.erg / (u.s * u.cm**2 * u.AA)
    if isinstance(f_lambda, u.Quantity):
        Flam = f_lambda.to(flam_unit)
        if debug: print(r"Quantity for $F_{\lambda}$ found")
    else:
        if debug: print(f"No F_lambda units provided; assuming erg/(s cm^2 Å).")
        Flam = np.asarray(f_lambda, float) * flam_unit

    # Optional errors
    if f_lambda_err is not None:
        if isinstance(f_lambda_err, u.Quantity):
            sFlam = f_lambda_err.to(flam_unit)
        else:
            sFlam = np.asarray(f_lambda_err, float) * flam_unit
    else:
        sFlam = None

    if wavelength_err is not None:
        if isinstance(wavelength_err, u.Quantity):
            slam = wavelength_err.to(u.AA)
        else:
            slam = np.asarray(wavelength_err, float) * u.AA
    else:
        slam = None

    # Speed of light in Å/s
    c_A_per_s = const.c.to(u.AA / u.s)

    # F_nu conversion
    if debug: print(f"Conversion to fnu started")
    Fnu = (Flam * lam**2 / c_A_per_s).to(u.mJy)
    if debug: print(f"Conversion to fnu done")
    # Uncertainty propagation
    if f_lambda_err is not None:
        if (sFlam is None) and (slam is None):
            return Fnu, None

        term_flux = (lam**2 / c_A_per_s) * (sFlam if sFlam is not None else 0 * flam_unit)
        term_wave = (2 * lam / c_A_per_s) * (Flam if slam is not None else 0 * flam_unit) * (slam if slam is not None else 0 * u.AA)

        Fnu_err = np.hypot(term_flux, term_wave).to(u.mJy)
    else: Fnu_err = None

    if debug:
        print(f"[to_fnu_mjy] λ range: {lam.min():.3g}–{lam.max():.3g} {lam.unit}")
        print(f"[to_fnu_mjy] Fλ unit: {Flam.unit}")
        print(f"SANITY CHECK--------------------------------------------")
        print(f"Length wavelength vector: {len(wavelength)}")
        print(f"Length flux vector: {len(Fnu)}")
        if Fnu_err is not None: print(f"Length flux error vector: {len(Fnu_err)}")

        if sFlam is not None:
            print(f"[to_fnu_mjy] σ(Fλ) provided, unit: {sFlam.unit}")
            

    return Fnu, Fnu_err



def bin_spectrum(
    wavelength,
    flux,
    bin_size,
    wave_unit="um",
    flux_unit="mJy",
    flux_err=None,
    out_wave_unit=None,
    out_flux_unit=None,
    debug = False
):
    """
    Flux-conserving binning to a regular wavelength grid.
    
    Parameters
    ----------
    wavelength : array-like
        Wavelength values (numeric). Interpreted in `wave_unit`.
    flux : array-like
        Flux values (numeric). Interpreted in `flux_unit`.
    bin_size : float
        Desired bin width, in units of `wave_unit`. Must be > 0.
    wave_unit, flux_unit : str
        Input units (e.g. "um", "Angstrom", "mJy", "Jy", "erg/(s cm2 Angstrom)").
    out_wave_unit, out_flux_unit : str | None
        If given, convert outputs to these units. 
    flux_err   : array-like or None
            Per-sample 1σ flux uncertainty in the same unit as `flux`.
    
    Returns
    -------
    wave_out : Quantity
    flux_out : Quantity
    flux_err_out : Quantity | None
        Binned 1σ uncertainty; None if `flux_err` was not provided.
    """
    
    # basic checks
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    
    if debug:
        print(f"Length wavelength vector: {len(wavelength)}")
        print(f"Length flux vector: {len(flux)}")
        print(f"Length flux error vector: {len(flux_err)}")

    w = np.asarray(wavelength, dtype=float)
    f = np.asarray(flux, dtype=float)

    if w.size == 0 or f.size == 0:
        raise ValueError("wavelength/flux arrays are empty")
    if w.shape != f.shape:
        raise ValueError(f"Shape mismatch: wavelength {w.shape} vs flux {f.shape}")
        
    # drop NaNs
    m = np.isfinite(w) & np.isfinite(f)
    if flux_err is not None:
        fe = np.asarray(flux_err, float)
        m &= np.isfinite(fe)
    else:
        fe = None
    if not np.all(m):
        w, f = w[m], f[m]
        if fe is not None:
            fe = fe[m]

    if w.size < 2:
        raise ValueError("Not enough finite points after cleaning to bin")

    # ensure increasing wavelength (specutils expects monotonic spectral_axis)
    if w[0] > w[-1]:
        w = w[::-1]
        f = f[::-1]
        if fe is not None:
            fe = fe[::-1]

    u_wave = u.Unit(wave_unit)
    if debug: print(f"Wave unit: {u_wave}")
    u_flux = u.Unit(flux_unit)
    if debug: print(f"Flux unit: {u_flux}")

    # Attach optional uncertainty
    unc = StdDevUncertainty(fe, unit = u_flux) if fe is not None else None
    if debug:
        print('Debug: uncertainty in flux:', unc)
    
    # convert to Spectrum1D with units
    spectrum = Spectrum(spectral_axis=w * u_wave, flux=f * u_flux, uncertainty = unc)
    if debug:
        print('Debug: Spectrum: ', spectrum)

    # compute bin edges
    wmin = float(w.min())
    wmax =float(w.max())
    #Ensures at least 2 bins:
    nw = max(int((wmax - wmin) / float(bin_size) + 1), 2)
    warr = np.linspace(wmin,wmax, nw)*u_wave

    #----resample
    if debug:
        print("Running resampling...")
    resampler = FluxConservingResampler()
    rebinned = resampler(spectrum, warr)
    if debug:
        print("Resample successful.")

    #----debug: plot input and binned spectrum:
    if debug:
        try:
            import matplotlib.pyplot as plt
            plt.figure(); plt.plot(w, f, lw=1.0, label = "pre-bin spectrum");plt.title("Comparison between binned and unbinned spectrum")
            plt.plot(rebinned.spectral_axis, rebinned.flux, lw = 0.8, label = 'binned spectrum'); plt.legend(); plt.show();plt.close()
        except Exception:
            pass

    if debug: print(f"out_wave_unit:{out_wave_unit}\nout_flux_unit:{out_flux_unit}")
    #----outputs as Quantities, not bare arrays
    wave_out = rebinned.spectral_axis.to(u.Unit(out_wave_unit) if out_wave_unit else u_wave)
    flux_out = rebinned.flux.to(u.Unit(out_flux_unit) if out_flux_unit else u_flux)
    if debug:
        print("DEBUG: Flux out:", flux_out)

    #make sure flux_err_out is handled:
    flux_err_out = None
    unc_reb  = getattr(rebinned, "uncertainty", None)
    
    if unc_reb is not None:
        try:
            unc_std = unc_reb.represent_as(StdDevUncertainty)
            if debug:
                print("rebinned_uncertainty:", unc_reb)
                print("Converted uncertainty:", unc_std)
                print("Uncertainty array", unc_std.array)
            if hasattr(unc_std, "quantity"):
                err_q = unc_std.quantity
                # Convert to requested output flux unit (or flux_out.unit)
                target_fu = u.Unit(out_flux_unit) if out_flux_unit else flux_out.unit
                flux_err_out = err_q.to(target_fu)
            else:
                # Fallback: wrap raw array in flux unit
                err_q = u.Quantity(getattr(unc_std, "array", None), flux_out.unit)
                # Convert to requested output flux unit (or flux_out.unit)
                target_fu = u.Unit(out_flux_unit) if out_flux_unit else flux_out.unit
                flux_err_out = err_q.to(target_fu)

        except Exception:
            unc_std = None
    
    return wave_out, flux_out, flux_err_out



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