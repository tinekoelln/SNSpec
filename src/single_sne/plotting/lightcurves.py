from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import astropy.units as u
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scienceplots
plt.style.use(['science'])

import numpy as np
import pandas as pd
from typing import Optional, Tuple

#TO DO: BUILD IN INTERPOLATION POSSIBILITY

def plot_lightcurves(lightcurve_df, title, interp=False):
    #separate by filter:
    
    fig, ax = plt.subplots(figsize=(8.3*2, 4.5*2))
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=plt.cm.tab20.colors
    )
    
    
    for filt, subdf in lightcurve_df.groupby("filter"):
        print(f"Filter: {filt}")
        #get the next color from the current SciencePlots cycle
        color = ax._get_lines.get_next_color()        
        date = subdf["mjd"]
        mag = subdf["mag"]
        err = subdf["err"]
        if filt == "v":
            color = '#9E0059'
        
        ax.errorbar(date, mag, err, fmt='o', label = filt, color = color)
        
        if interp:
            interp_curve = interp1d(date, mag, kind='cubic', bounds_error=False, fill_value=np.nan)
            #date, mag, err = prepare_for_interp(date, mag, err)
            ax.plot(mag, interp_curve, color = color)

    ax.legend()
    ax.invert_yaxis()
    ax.set_title(title)
        
    plt.show()
    plt.close
    
    return fig