from single_sne.spectra.spectra import to_fnu_mjy, is_strictly_increasing
from astropy import units as u
from numpy.testing import assert_allclose

import numpy as np
from astropy.constants import c

#Test sorting epochs for both quantities and numpy arrays:
def test_is_increasing():
    w_numpy = [1.1, 1.2, 1.4, 1.3, 1.5, np.nan]
    w_quantity = w_numpy*u.um
    
    assert is_strictly_increasing(w_numpy, debug = True) == False
    assert is_strictly_increasing(w_quantity, debug = True) == False
    
def test_to_fnu_mjy_unitless():
    w_AA = np.array([4000, 4200, 4400, 4600, 4800, 5000])
    f_lambda = np.array([3.2e-17,3.0e-17,2.85e-17,2.78e-17,2.60e-17,2.50e-17])  
    flux_err = np.array([1.2e-18,1.1e-18,1.1e-18,1.0e-18,1.0e-18,1.0e-18])
    
    # Run function
    f_mjy, ferr_mjy = to_fnu_mjy(w_AA, f_lambda, flux_err)

    # --- Assert units
    assert f_mjy.unit == u.mJy
    assert ferr_mjy.unit == u.mJy

    # --- Compute reference manually (same formula as function)
    lam = w_AA * u.AA
    flam = f_lambda * (u.erg / (u.s * u.cm**2 * u.AA))
    ferr = flux_err * (u.erg / (u.s * u.cm**2 * u.AA))

    c_A_per_s = c.to(u.AA / u.s)
    fnu_ref = (flam * lam**2 / c_A_per_s).to(u.mJy)
    ferr_ref = (ferr * lam**2 / c_A_per_s).to(u.mJy)

    # --- Assert numerical accuracy
    assert_allclose(f_mjy.value, fnu_ref.value, rtol=1e-6)
    assert_allclose(ferr_mjy.value, ferr_ref.value, rtol=1e-6)
    
def test_to_fnu_mjy_AA():
    w_AA = np.array([4000, 4200, 4400, 4600, 4800, 5000])*u.AA
    f_lambda = np.array([3.2e-17,3.0e-17,2.85e-17,2.78e-17,2.60e-17,2.50e-17])* (u.erg / (u.s * u.cm**2 * u.AA))
    flux_err = np.array([1.2e-18,1.1e-18,1.1e-18,1.0e-18,1.0e-18,1.0e-18])* (u.erg / (u.s * u.cm**2 * u.AA))
    
    # Run function
    f_mjy, ferr_mjy = to_fnu_mjy(w_AA, f_lambda, flux_err)

    # --- Assert units
    assert f_mjy.unit == u.mJy
    assert ferr_mjy.unit == u.mJy

    # --- Compute reference manually (same formula as function)
    lam = w_AA 
    flam = f_lambda
    ferr = flux_err

    c_A_per_s = c.to(u.AA / u.s)
    fnu_ref = (flam * lam**2 / c_A_per_s).to(u.mJy)
    ferr_ref = (ferr * lam**2 / c_A_per_s).to(u.mJy)

    # --- Assert numerical accuracy
    assert_allclose(f_mjy.value, fnu_ref.value, rtol=1e-6)
    assert_allclose(ferr_mjy.value, ferr_ref.value, rtol=1e-6)
    

def test_to_fnu_mjy_nm():
    w_AA = np.array([4000, 4200, 4400, 4600, 4800, 5000])*u.AA
    w_nm = np.array([4000, 4200, 4400, 4600, 4800, 5000])*0.1*u.nm
    f_lambda = np.array([3.2e-17,3.0e-17,2.85e-17,2.78e-17,2.60e-17,2.50e-17])* (u.erg / (u.s * u.cm**2 * u.AA))
    flux_err = np.array([1.2e-18,1.1e-18,1.1e-18,1.0e-18,1.0e-18,1.0e-18])* (u.erg / (u.s * u.cm**2 * u.AA))
    
    # Run function
    f_mjy, ferr_mjy = to_fnu_mjy(w_nm, f_lambda, flux_err)

    # --- Assert units
    assert f_mjy.unit == u.mJy
    assert ferr_mjy.unit == u.mJy

    # --- Compute reference manually (same formula as function)
    lam = w_AA 
    flam = f_lambda
    ferr = flux_err

    c_A_per_s = c.to(u.AA / u.s)
    fnu_ref = (flam * lam**2 / c_A_per_s).to(u.mJy)
    ferr_ref = (ferr * lam**2 / c_A_per_s).to(u.mJy)

    # --- Assert numerical accuracy
    assert_allclose(f_mjy.value, fnu_ref.value, rtol=1e-6)
    assert_allclose(ferr_mjy.value, ferr_ref.value, rtol=1e-6)
    