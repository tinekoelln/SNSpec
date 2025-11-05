from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import astropy.units as u

@dataclass(frozen=True)
class InstrumentUnits:
    wavelength: u.UnitBase
    flux: u.UnitBase

# Central registry (add more as you go)
_INSTRUMENT_UNITS: Dict[str, InstrumentUnits] = {
    # JWST x1d spectra are typically in F_nu (Jy); wavelengths in microns.
    "jwst": InstrumentUnits(wavelength=u.um, flux=u.mJy),

    # ESO X-shooter: most reductions export λ in Å and F_λ in erg s^-1 cm^-2 Å^-1
    "xshooter": InstrumentUnits(wavelength=u.nm, flux=u.Unit("erg / (s cm2 Angstrom)")),

    # Gemini FLAMINGOS-2 (spectra often delivered similarly to optical/NIR F_λ; adjust if your pipeline differs)
    "flamingos-2": InstrumentUnits(wavelength=u.AA, flux=u.Unit("erg / (s cm2 Angstrom)")),
}

# Alias map so users can type variants
_ALIASES: Dict[str, str] = {
    "jwst": "jwst",
    "JWST": "jwst",
    "XSHOOTER": "xshooter",
    "nirspec": "jwst",
    "miri": "jwst",
    "xsh": "xshooter",
    "x-shooter": "xshooter",
    "xshooter": "xshooter",
    "flamingo": "flamingos-2",
    "flamingos": "flamingos-2",
    "flamingos2": "flamingos-2",
    "flamingos-2": "flamingos-2",
}

def list_supported_instruments() -> Iterable[str]:
    """Return canonical instrument keys supported by the registry."""
    return sorted(set(_INSTRUMENT_UNITS.keys()))

def resolve_instrument(name: str) -> str:
    """Normalize user input to a canonical instrument key or raise ValueError."""
    key = _ALIASES.get(name.strip().lower())
    if key is None or key not in _INSTRUMENT_UNITS:
        supported = ", ".join(list_supported_instruments())
        raise ValueError(f"Unknown instrument '{name}'. Supported: {supported}")
    return key

def which_kind_of_spectrum(telescope: str) -> Tuple[u.UnitBase, u.UnitBase]:
    """
    Return default (wavelength_unit, flux_unit) for a given telescope/instrument.

    Parameters
    ----------
    telescope : str
        Examples: "JWST", "XSHOOTER", "FLAMINGOS-2" (aliases like "xsh", "miri", "nirspec" work).

    Returns
    -------
    (wavelength_unit, flux_unit) : tuple of astropy.units.UnitBase
        e.g. (u.um, u.mJy) for JWST, or (u.AA, erg/(s cm2 Å)) for XSHOOTER.

    Notes
    -----
    - This is a policy decision for *your* package. If you ingest files that
      carry different units, convert them on read to these defaults so the rest
      of your code can assume a consistent convention.
    """
    key = resolve_instrument(telescope)
    spec = _INSTRUMENT_UNITS[key]
    return spec.wavelength, spec.flux

def register_instrument(name: str, wavelength_unit: u.UnitBase, flux_unit: u.UnitBase, *aliases: str) -> None:
    """
    Extend the registry at runtime (e.g., from site config or a plugin).
    """
    canon = name.strip().lower()
    _INSTRUMENT_UNITS[canon] = InstrumentUnits(wavelength=wavelength_unit, flux=flux_unit)
    _ALIASES[canon] = canon
    for a in aliases:
        _ALIASES[a.strip().lower()] = canon
