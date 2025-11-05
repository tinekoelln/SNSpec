import astropy.units as u

# House wavelength unit we use on grids for X-shooter steps
XSH_WAVE_UNIT = u.nm

# Native XShooter flux (F_lambda)
XSH_FLUX_UNIT = u.Unit("erg / (cm**2 s Angstrom)")

INSTRUMENT_UNITS = {
    "JWST":      (u.um, u.mJy),
    "FLAMINGOS2":(u.AA,  u.erg / (u.s * u.cm**2 * u.AA)),
    "XSHOOTER":  (u.nm,  u.erg / (u.s * u.cm**2 * u.AA)),
}
