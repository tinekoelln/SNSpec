from __future__ import annotations
import numpy as np
from pathlib import Path
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
    "to_fnu_mjy", 
    "nm_to_angstrom",
    "angstrom_to_micron",
    "nm_to_micron",
    "bin_spectrum",
]



def to_fnu_mjy(f_lambda_erg_s_cm2_A, wavelength_A, debug = False):
    """
    Convert f_lambda (erg/s/cm^2/Å) at wavelength (Å) to f_nu in mJy.
    If no units are present, the function assumes the passed wavelength is in A

    Accepts numpy arrays or astropy Quantities for both inputs.
    If no units are provided:
      - wavelength is assumed Å
      - f_lambda is assumed erg s^-1 cm^-2 Å^-1
    Returns an astropy Quantity in mJy.
    """
     # Normalize wavelength to Å
    if isinstance(wavelength_A, u.Quantity):
        lam = wavelength.to(u.AA)
        if debug:
            print("Units present:", wavelength_A.unit)

    else:
        if debug:
            print("No units provided, assumed Angstrom")
        lam = wavelength.to(u.AA)

    # Normalize F_lambda to erg s^-1 cm^-2 Å^-1
    flam_unit = u.erg / (u.s * u.cm**2 * u.AA)
    if isinstance(f_lambda, u.Quantity):
        Flam = f_lambda.to(flam_unit)
    else:
        Flam = np.asarray(f_lambda, float) * flam_unit
    
    if debug:
        print(f"[to_fnu_mjy] λ range: {lam.min():.3g}–{lam.max():.3g} {lam.unit}")
        print(f"[to_fnu_mjy] Fλ unit: {Flam.unit}")
    # c in Å/s
    c_A_per_s = const.c.to(u.AA / u.s)  
    
    # F_nu = F_lambda * λ^2 / c
    Fnu = (Flam * lam**2 / c_A_per_s).to(u.mJy)  # 1 mJy = 1e-26 erg s^-1 cm^-2 Hz^-1
    return Fnu

    
def nm_to_angstrom(wavelength_nm: float | np.ndarray, debug = False) -> float | np.ndarray:
    """
    Convert wavelength from nanometers (nm) to Angstrom (Å).

    Parameters
    ----------
    wavelength_nm : float or array-like
        Wavelength in nanometers.

    Returns
    -------
    float or array-like
        Wavelength in Angstrom.

    Notes
    -----
    1 nm = 10 Å
    """
    if isinstance(wavelength_nm, u.Quantity):
        if debug:
            print("Units present:", wavelength_nm.unit)
        if wavelength_nm.unit.is_equivalent(u.AA):
            return wavelength_nm
        else:
            return wavelength_nm.to(u.AA)
    else:
        if debug:
            print("No units pr")
        return (np.asarray(wavelength_nm) * u.nm).to(u.AA)
            

def angstrom_to_micron(wavelength_A: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength from Angstrom (Å) to micrometers (µm).

    Parameters
    ----------
    wavelength_A : float or array-like
        Wavelength in Angstrom.

    Returns
    -------
    float or array-like
        Wavelength in micrometers.

    Notes
    -----
    1 Å = 1e-4 µm
    """
    if isinstance(wavelength_A, u.Quantity):
        if debug:
            print("Units present:", wavelength_A.unit)
        if wavelength_A.unit.is_equivalent(u.um):
            return wavelength_A
        else:
            return wavelength_A.to(u.um)
    else:
        if debug:
            print("No units pr")
        return (np.asarray(wavelength_A) * u.AA).to(u.um)

def nm_to_micron(wavelength_nm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert wavelength from nanometers (nm) to micrometers (µm).

    Parameters
    ----------
    wavelength_nm : float or array-like
        Wavelength in nanometers.

    Returns
    -------
    float or array-like
        Wavelength in micrometers.

    Notes
    -----
    1 nm = 1e-3 µm
    """
    if isinstance(wavelength_nm, u.Quantity):
        if debug:
            print("Units present:", wavelength_nm.unit)
        if wavelength_nm.unit.is_equivalent(u.um):
            return wavelength_nm
        else:
            return wavelength_nm.to(u.um)
    else:
        if debug:
            print("No units pr")
        return (np.asarray(wavelength_nm) * u.nm).to(u.um)    


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
    u_flux = u.Unit(flux_unit)

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
            plt.figure(); plt.plot(w, f, lw=0.8, label = "pre-bin spectrum")
            plt.plot(rebinned.spectral_axis, rebinned.flux, lw = 0.6, label = 'binned spectrum'); plt.show()
        except Exception:
            pass


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
                print("rebinned_uncertainty:", unc_rebinned)
                print("Converted uncertainty:", unc_std)
                print("Uncertainty array", unc_std.array)
            if hasattr(unc_std, "quantity"):
                err_q = unc_std.quantity
            else:
                # Fallback: wrap raw array in flux unit
                err_q = u.Quantity(getattr(unc_std, "array", None), flux_out.unit)
                # Convert to requested output flux unit (or flux_out.unit)
                target_fu = u.Unit(out_flux_unit) if out_flux_unit else flux_out.unit
                flux_err_out = err_q.to(target_fu)
                     
         except Exception:
             unc_std = None
    
    return wave_out, flux_out, flux_err_out



