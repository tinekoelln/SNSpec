import numpy as np
import pandas as pd
from single_sne.io.lightcurves import read_lightcurve, find_lightcurve
from single_sne.plotting.lightcurves import plot_lightcurves
from single_sne.lightcurves.lightcurves import get_and_plot_lightcurves
from pathlib import Path


def test_lightcurve():
    path = '/Users/ckoelln/Documents/Single_Objects/light_curves/sn2025ifq.txt'
    lightcurve = read_lightcurve(path)
    
    assert isinstance(lightcurve, pd.DataFrame)
    assert lightcurve.shape[1]==5

def test_find_lightcurve():
    root = '~/Documents/Single_Objects/'
    root = Path(root).expanduser().resolve() 
    lcdir, lc_list = find_lightcurve(root)
    
    #print("TESTING GET AND PLOT LIGHTCURVES")
    #get_and_plot_lightcurves(root)
    
    assert isinstance(lc_list, list)
    
def test_plot_lightcurve():
    lightcurve_file = '/Users/ckoelln/Documents/Single_Objects/light_curves/sn2025cy_SWIFT+BG.dat'
    lightcurve_df = pd.read_csv(lightcurve_file, comment = '#', sep = ' ')
    
    fig = plot_lightcurves(lightcurve_df, title = 'Lightcurve of SN2025cy')
    fig.savefig('/Users/ckoelln/Documents/Single_Objects/light_curves/SN2025cy_lighcurveSWIFT+BG.pdf', dpi = 600)