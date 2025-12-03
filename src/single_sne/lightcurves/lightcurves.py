from __future__ import annotations
import pathlib
from single_sne.io.lightcurves import find_lightcurve, read_lightcurve
from single_sne.plotting.lightcurves import plot_lightcurves


#To DO: create function fo construct bolometric light curve
#create function to estimate nickel mass from bolometric lightcurve

def get_and_plot_lightcurves(root):
    #obtain lightcurve list:
    
    lcdir, lc_list = find_lightcurve(root)
    for file in lc_list:
        full_path = pathlib.Path(lcdir)/file
        print(file.stem)
        snname = file.stem
        
        lc_df = read_lightcurve(full_path)
        
        plot_lightcurves(lc_df, title=f"Light Curve for {snname}")
    
    

