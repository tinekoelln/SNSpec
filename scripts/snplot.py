import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
from astropy import units as u
import scienceplots
plt.style.use(['science'])

def to_fnu(flux, wavelength):
    """
    Convert flux from erg/s/cm^2/Angstrom to Jansky.

    Parameters:
    flux : float or array-like
        Flux in erg/s/cm^2/Angstrom.
    wavelength : float or array-like
        Wavelength in Angstrom.

    Returns:
    float or array-like
        Flux in miliJansky.
    """
    c = 2.998e18  # Speed of light in Angstrom/s
    f_nu = (flux * wavelength**2) / c * 1e26  # Convert to miliJansky
    return f_nu

def read_jwst(filename):
    """
    Read JWST data from a file.

    Parameters:
    filename : str
        Path to the JWST data file.

    Returns:
    tuple of arrays
        Wavelength in micrometers and flux in mJy.
    """
    jwst = pd.read_csv(filename, sep=r'\s+', header=None, comment='#', index_col=False)
    jwst.dropna(inplace=True)
    jwst.columns = ['wavelength(um)', 'flux(mJy)']
    
    return jwst

def nm_to_angstrom(wavelength_nm):
    """
    Convert wavelength from nanometers to Angstrom.

    Parameters: 
    wavelength_nm : float or array-like
        Wavelength in nanometers.   
    Returns:
    float or array-like
        Wavelength in Angstrom.
    """
    return wavelength_nm * 10.0 

def angstrom_to_micron(wavelength_A):
    """
    Convert wavelength from Angstrom to micrometers.

    Parameters:
    wavelength_A : float or array-like
        Wavelength in Angstrom.

    Returns:
    float or array-like
        Wavelength in micrometers.
    """
    return wavelength_A * 1e-4

def bin_spectrum(wavelength, flux, bin_size):
    """
    Bin the spectrum data.

    Parameters:
    wavelength : array-like
        Wavelength data, in micrometers.
    flux : array-like
        Flux data, in mJy.
    bin_size : bin length
        length of binning in micrometers.

    Returns:
    tuple of arrays
        Binned wavelength and flux.
    """

    # bin optical spectrum
    dw = bin_size
    wmin = wavelength.min()
    wmax = wavelength.max()
    nw = int((wmax - wmin)/dw + 1)
    warr = np.linspace(wmin, wmax, nw)
    spectrum = Spectrum(spectral_axis=wavelength*u.um, flux=flux*u.Unit("mJy"))
    resampler = FluxConservingResampler()
    rebinned_spectrum = resampler(spectrum, warr*u.um)
    wavelength = warr
    flux = rebinned_spectrum.flux.value
    
    return wavelength, flux


def combine_data(optical, jwst, output_filename):
    """
    Combine optical and JWST data and save to a file.

    Parameters:
    optical : tuple of arrays
        Optical data (wavelength in micrometers, flux in mJy).
    jwst : tuple of arrays
        JWST data (wavelength in micrometers, flux in mJy).
    output_filename : str
        Output filename to save the combined data.
    """
    opt_wavelength, opt_flux = optical
    jwst_wavelength, jwst_flux = jwst

    combined_wavelength = np.concatenate((opt_wavelength, jwst_wavelength))
    combined_flux = np.concatenate((opt_flux, jwst_flux))

    # Sort by wavelength
    sorted_indices = np.argsort(combined_wavelength)
    combined_wavelength = combined_wavelength[sorted_indices]
    combined_flux = combined_flux[sorted_indices]

    # Save to file
    df = pd.DataFrame({
        'Wavelength_um': combined_wavelength,
        'Flux_mJy': combined_flux
    })
    df.to_csv(output_filename, index=False)
    
    return combined_wavelength, combined_flux


def xshooter_jwst_compatible(xshooter_wave, xshooter_flux):
    #takes in an xshooter dat (wavelength: nm, flux: erg/s/cm^2/Angstrom) and
    #return a file in jwst compatible units (wavelength: microns, flux: mJy)
    
    #from nm to Angstrom
    xshooter_wave_A = nm_to_angstrom(xshooter_wave)
    
    flux_mJy = to_fnu(xshooter_flux, xshooter_wave_A)
    wave_micron = angstrom_to_micron(xshooter_wave_A)
    
    return wave_micron, flux_mJy


def xshooter_jwst_compatible(xshooter_wave, xshooter_flux):
    #takes in an xshooter dat (wavelength: nm, flux: erg/s/cm^2/Angstrom) and
    #return a file in jwst compatible units (wavelength: microns, flux: mJy)
    
    #from nm to Angstrom
    xshooter_wave_A = nm_to_angstrom(xshooter_wave)
    
    flux_mJy = to_fnu(xshooter_flux, xshooter_wave_A)
    wave_micron = angstrom_to_micron(xshooter_wave_A)
    
    return wave_micron, flux_mJy

def salt_jwst_compatible(salt_wave, salt_flux):
    #takes in a salt dat (wavelength: Angstrom, flux: erg/s/cm^2/Angstrom) and
    #return a file in jwst compatible units (wavelength: microns, flux: mJy)
    
    flux_mJy = to_fnu(salt_flux, salt_wave)
    wave_micron = angstrom_to_micron(salt_wave)
    
    return wave_micron, flux_mJy
    
    
def convert_xshooter(infile, bin_size=0.001):
    #reads in xshooter file, converts to jwst units, bins spectrum, and saves to outfile
    xshooter = pd.read_csv(infile, sep=r'\s+', header=None, comment='#', index_col=False)
    xshooter_dropna = xshooter.dropna()
    xshooter_dropna.columns = ['wavelength(nm)', 'flux(erg/s/cm2/ang)']

    
    xshooter_wave, xshooter_flux = xshooter_jwst_compatible(xshooter_dropna['wavelength(nm)'].values, xshooter_dropna['flux(erg/s/cm2/ang)'].values)
    #bin the data to match jwst resolution
    xshooter_wave, xshooter_flux = bin_spectrum(xshooter_wave, xshooter_flux, bin_size=bin_size)

    #save converted xshooter data
    xshooter_converted = pd.DataFrame({'wavelength(um)': xshooter_wave, 'flux(mJy)': xshooter_flux})
    outfile = infile.replace(".dat", "_converted.dat")
    xshooter_converted.to_csv(outfile, index=False, sep=' ' )
    return xshooter_converted

def read_salt(infile):
    #reads in salt file and returns wavelength and flux arrays
    salt = pd.read_csv(infile, sep=r'\s+', header=None, comment='#', index_col=False)
    salt_dropna = salt.dropna()
    salt_dropna.columns = ['wavelength(ang)', 'flux(erg/s/cm2/ang)']
    salt_dropna = salt_dropna[salt_dropna['flux(erg/s/cm2/ang)'] != 0]
    
    salt_wave = salt_dropna['wavelength(ang)'].values
    salt_flux = salt_dropna['flux(erg/s/cm2/ang)'].values
    
    return salt_wave, salt_flux

def convert_salt(infile, bin_size=0.001):
    #reads in salt file, converts to jwst units, bins spectrum, and saves to outfile
    salt_wave, salt_flux = read_salt(infile)
    
    salt_wave_jwst, salt_flux_jwst = salt_jwst_compatible(salt_wave, salt_flux)
    #bin the data to match jwst resolution
    salt_wave_jwst, salt_flux_jwst = bin_spectrum(salt_wave_jwst, salt_flux_jwst, bin_size=bin_size)

    #save converted salt data
    salt_converted = pd.DataFrame({'wavelength(um)': salt_wave_jwst, 'flux(mJy)': salt_flux_jwst})
    outfile = infile.replace(".dat", "_converted.dat")
    salt_converted.to_csv(outfile, index=False, sep=' ' )
    return salt_converted

def read_flamingos(infile):
    #reads in flamingos file and returns wavelength and flux arrays
    flamingos = pd.read_csv(infile, sep=r'\s+', header=None, comment='#', index_col=False)
    flamingos_dropna = flamingos.dropna()
    flamingos_dropna.columns = ['wavelength(ang)', 'flux(erg/s/cm2/ang)', 'flux_error(erg/s/cm2/ang)']
    flamingos_dropna = flamingos_dropna[flamingos_dropna['flux(erg/s/cm2/ang)'] != 0]
    
    flamingos_wave = flamingos_dropna['wavelength(ang)'].values
    flamingos_flux = flamingos_dropna['flux(erg/s/cm2/ang)'].values
    flamingos_error = flamingos_dropna['flux_error(erg/s/cm2/ang)'].values
    
    #drop pixels for which the flux error is more than 2x the flux value
    mask = np.abs(flamingos_error) <= 2 * np.abs(flamingos_flux)
    flamingos_wave = flamingos_wave[mask]
    flamingos_flux = flamingos_flux[mask]
    flamingos_error = flamingos_error[mask]
    
    mask2 = flamingos_flux > 0
    flamingos_wave = flamingos_wave[mask2]
    flamingos_flux = flamingos_flux[mask2]
    flamingos_error = flamingos_error[mask2]
    
    return flamingos_wave, flamingos_flux, flamingos_error

def convert_flamingos(infile, bin_size=0.001):
    #reads in flamingos file, converts to jwst units, bins spectrum, and saves to outfile
    flamingos_wave, flamingos_flux, flamingos_error = read_flamingos(infile)
    
    flamingos_wave_jwst, flamingos_flux_jwst = salt_jwst_compatible(flamingos_wave, flamingos_flux)
    flamingos_wave_jwst, flamingos_error_jwst = salt_jwst_compatible(flamingos_wave, flamingos_error)
    plt.figure(figsize=(8,6))
    plt.plot(flamingos_wave_jwst, flamingos_flux_jwst)
    plt.xlim(1, 1.8)
    plt.ylim(0, 5)
    #bin the data to match jwst resolution
    flamingos_wave_jwst_binned, flamingos_flux_jwst_binned = bin_spectrum(flamingos_wave_jwst, flamingos_flux_jwst, bin_size=bin_size)
    flamingos_wave_jwst_binned, flamingos_error_jwst_binned = bin_spectrum(flamingos_wave_jwst, flamingos_error_jwst, bin_size=bin_size)

    #save converted flamingos data
    flamingos_converted = pd.DataFrame({'wavelength(um)': flamingos_wave_jwst_binned, 'flux(mJy)': flamingos_flux_jwst_binned, 'flux_error(mJy)': flamingos_error_jwst_binned})
    outfile = infile.replace(".dat", "_converted.dat")
    flamingos_converted.to_csv(outfile, index=False, sep=' ' )
    return flamingos_converted 