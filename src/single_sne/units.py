from __future__ import annotations

import astropy.units as u
from astropy.units import Unit  # use the string parser

import numpy as np

# House wavelength unit we use on grids for X-shooter steps
XSH_WAVE_UNIT = u.nm

# Native XShooter flux (F_lambda)
# This is equivalent to u.erg / (u.s * u.cm**2 * u.AA),
# but uses the string parser to avoid any cm**2 / cm*cm operator issues.
XSH_FLUX_UNIT = Unit("erg / (s cm2 Angstrom)")

INSTRUMENT_UNITS = {
    "JWST":       (u.um, u.mJy),
    "FLAMINGOS2": (u.AA, XSH_FLUX_UNIT),
    "XSHOOTER":   (u.nm, XSH_FLUX_UNIT),
    "SALT":       (u.AA, XSH_FLUX_UNIT),
}

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
            print("[nm_to_angstrom] Units present:", wavelength_nm.unit)
        if wavelength_nm.unit == u.AA:
            return wavelength_nm
        else:
            return wavelength_nm.to(u.AA)
    else:
        if debug:
            print("[nm_to_angstrom] No units present")
        return (np.asarray(wavelength_nm) * u.nm).to(u.AA)
            

def angstrom_to_micron(wavelength_A: float | np.ndarray, debug = False) -> float | np.ndarray:
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
            print("[angstrom_to_micron]: Units present:", wavelength_A.unit)
        if wavelength_A.unit == u.um:
            if debug: print(f"[angstrom_to_micron]: provided unit {wavelength_A.unit} is already microns, returning as is")
            return wavelength_A
        else:
            if debug: print(f"[angstrom_to_micron]:Units given: {wavelength_A.unit}")
            wavelength_um = wavelength_A.to(u.um)
            if debug: print(f"[angstrom_to_micron]:Units after conversion: {wavelength_um.unit}")
            return wavelength_um
    else:
        if debug:
            print("[angstrom_to_micron]: No units present")
        return (np.asarray(wavelength_A) * u.AA).to(u.um)

def nm_to_micron(wavelength_nm: float | np.ndarray, debug= False) -> float | np.ndarray:
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
            print("[nm_to_micron]: Units present:", wavelength_nm.unit)
        if wavelength_nm.unit == u.um:
            return wavelength_nm
        else:
            return wavelength_nm.to(u.um)
    else:
        if debug:
            print("[nm_to_micron]: No units present")
        return (np.asarray(wavelength_nm) * u.nm).to(u.um)    