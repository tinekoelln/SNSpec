from __future__ import annotations


import pandas as pd
import numpy as np
from astropy import units as u
from single_sne.spectra.spectra import is_strictly_increasing
from single_sne.io.clean_data import clean_data
from pathlib import Path
from typing import Tuple, Optional, Union

PathLike = Union[str, Path]


def read_jwst_df(path: PathLike, as_quantity: bool = True, debug = False) -> pd.DataFrame:
    """
    Read a 2-column JWST ASCII file into a pandas DataFrame.

    Expected format:
    - two whitespace-separated columns: wavelength[um], flux[mJy]
    - lines starting with '#' are ignored
    Returns
    -------
    Read JWST ASCII spectrum (2 or 3 columns).
    Returns a DataFrame whose columns are either plain floats (default µm/mJy)
    or Quantity columns if as_quantity=True.
    """
    #p = Path(path).expanduser().resolve()
    #if debug: print(p)
    #if not p.exists():
        #raise FileNotFoundError(f"JWST file not found: {p}")

    if debug:
        print(f"Path obtained: {path}")

    
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",      # needed for regex sep
        comment="#",
        header=None,
        names=["wavelength_um", "flux_mJy","flux_err_mJy"],
        usecols=[0, 1],
        dtype=float,
    ).dropna()
    df = df.sort_values(by="wavelength_um")
    df.reset_index(inplace = True, drop=True)

    if debug: 
        print(f"[read_jwst_df]: DataFrame read")
    # If only 2 columns present, the third will be NaN; drop it if entirely NaN
    if "flux_err_mJy" in df and df["flux_err_mJy"].isna().all():
        df = df.drop(columns=["flux_err_mJy"])

    # Basic sanity: at least 2 columns remaining
    if not {"wavelength_um", "flux_mJy"}.issubset(df.columns):
                raise ValueError(f"Expected at least two numeric columns in {p}")

    if as_quantity:
            # Replace float columns by Quantity columns (pandas Series of Quantity)
            df["wavelength_um"] = df["wavelength_um"].to_numpy() * u.um
            if debug: print(f"[read_jwst_df]: converted wavelength column to array of Quantities")
            df["flux_mJy"]      = df["flux_mJy"].to_numpy() * u.mJy
            if debug: print(f"[read_jwst_df]: converted flux column to array of Quantities")
            if "flux_err_mJy" in df.columns:
                df["flux_err_mJy"] = df["flux_err_mJy"].to_numpy() * u.mJy
    if debug: print(f"[read_jwst_df] Sending DF back to function...")   
    return df

def read_jwst_arrays(path: PathLike, as_quantity: bool = True, debug = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Same data as read_jwst_df, but returns arrays:
    (wavelength, flux, flux_err_or_None)
    If as_quantity=True, these are Quantities with units (µm, mJy).
    """
    #if debug: print(path)
    df = read_jwst_df(path, as_quantity=True, debug=debug)
    #if debug: print(f"[read_jwst_arrays] dataframe acquired successfully")
    
    if as_quantity:
        w = df["wavelength_um"].values * u.um if not hasattr(df["wavelength_um"].iloc[0], "unit") else df["wavelength_um"].to_numpy()
        f = df["flux_mJy"].values * u.mJy     if not hasattr(df["flux_mJy"].iloc[0], "unit")     else df["flux_mJy"].to_numpy()
        if debug:
            print(f"✅[OK]: Wave and flux extracted, added to dataframe")
        if "flux_err_mJy" in df.columns:
            fe = df["flux_err_mJy"].values * u.mJy if not hasattr(df["flux_err_mJy"].iloc[0], "unit") else df["flux_err_mJy"].to_numpy()
            if debug:
                print(f"✅[OK]: Flux error present, added to dataframe")
        else:
            fe = None
    else:
        w = df["wavelength_um"].to_numpy()
        f = df["flux_mJy"].to_numpy()
        fe = df["flux_err_mJy"].to_numpy() if "flux_err_mJy" in df.columns else None
    #print(f"[read_jwst_arrays]: current w & f {w} {f}")
    
    w, f = clean_data(w, f)
    return w, f, fe
