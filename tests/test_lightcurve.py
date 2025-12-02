import numpy as np
import pandas as pd
from single_sne.io.lightcurves import read_lightcurve, find_lightcurve
from single_sne.plotting.lightcurves import plot_lightcurves
from single_sne.lightcurves.lightcurves import get_and_plot_lightcurves
from pathlib import Path

# Base directory for test data (relative to THIS file)
DATA_DIR = Path(__file__).parent / "tests_data"
def test_lightcurve():
    path = DATA_DIR / "/sn2025ifq.txt"
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
    
def test_plot_lightcurve(tmp_path):
    lightcurve_file = DATA_DIR / "sn2025cy_SWIFT+BG.dat"
    lightcurve_df = pd.read_csv(lightcurve_file, comment = '#', sep = ' ')
    
    tmp_path.mkdir(parents=True, exist_ok=True)
    fig = plot_lightcurves(lightcurve_df, title = 'Lightcurve of SN2025cy')
    fig.savefig(tmp_path /'SN2025cy_lighcurveSWIFT+BG.pdf', dpi = 600)