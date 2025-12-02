from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
from pathlib import Path
from typing import Iterable


def _iter_files(d: Path, patterns: Iterable[str] | str = ("*.dat", "*.fits")):
    # Coerce a single string into a tuple of one pattern
    if isinstance(patterns, (str, Path)):
        patterns = (str(patterns),)
    for pat in patterns:
        for p in d.glob(pat):
            if p.is_file():
                yield p
                
                
def find_lightcurve(root, debug=False):
    #receives root folder, finds light_curve folder, returns list of light_curves
    lcdir = root/ "lightcurves"
    if not lcdir.is_dir(): lcdir = root/ "light_curves" 
    
    curves_list = []
    if lcdir.is_dir():
        for datafile in _iter_files(lcdir, patterns =( "sn*.txt")):
            curves_list.append(datafile)
            
    return lcdir, curves_list
    

def read_lightcurve(path, debug=False):
    # Receives path to the lightcurve, outputs dataframe with mjd, mag, err, filter, substracted
    lightcurve_df = pd.read_csv(path, sep = ' ')
    return lightcurve_df

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