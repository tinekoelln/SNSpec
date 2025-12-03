from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import astropy.units as u
import scienceplots
from astropy.units import Quantity
plt.style.use(['science'])
__all__ = ["plot_epoch"] 
from single_sne.plotting.plot_helpers import setup_science_style  # wherever you put it
setup_science_style()

def load_combined_spectrum(path, debug = False):
    # column 0: instrument (string)
    try:
        spectrum_table = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        engine="python",
        header=None,                 # no header in file
        names=["instrument", "wavelength", "flux", "flux_error"],
        usecols=[0, 1, 2, 3],        # read first four cols
        na_values=["NaN", "nan", ""]
        )
        return spectrum_table
    except Exception as e:
        print(f"[load_combined_spectrum]ERROR: {e}")

def plot_epoch(epoch_dir: str | Path,
    outdir: str | Path,
    *,
    debug=False,
    show = False,
):
    """
    Plot per-epoch: each part + combined.
    - Use steps (horizontal/vertical) via ax.step(..., where='mid')
    - Plot combined in black
    """
    
    epoch_dir = Path(epoch_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    
    #read in epoch combined spectrum
    parent = epoch_dir.parent.name   # e.g. "SN2025cy"
    epoch  = epoch_dir.name          # e.g. "Epoch1"
    inst_order = []
    if debug: print(f"      [plot_epoch] Begin function...")
    for p in epoch_dir.iterdir():
        if p.is_dir():
            inst_order.append(p.name.upper())
    inst_label = "+".join(inst_order)
    combined_spectrum = epoch_dir / f"{parent}_{epoch}_combined_{inst_label}.dat"
    spec_table = load_combined_spectrum(combined_spectrum, debug=debug)    
    inst_comb = spec_table["instrument"].astype(str)
    w_comb = np.asarray(spec_table["wavelength"], dtype=float)
    f_comb = np.asarray(spec_table["flux"], dtype = float)
    jwst_units = inst_comb.str.upper().str.contains("JWST").any()
    
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    if jwst_units:
        ax.set_xlabel("Wavelength [µm]")
        ax.set_ylabel(r"$F_{\nu} [mJy]$")
    else:
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"$\lambda^2 F_\lambda$ [erg/s/cm$^2$/Å]")
    
    for inst in np.unique(inst_comb):
        m = inst_comb == inst
        m &= inst != 'nan'
        ax.step(w_comb[m], f_comb[m], where = 'post', label=inst, lw = 2.0)
        
    ax.step(w_comb, f_comb, where = 'post', lw=1.0, color="k", label="Combined")
    ax.set_title(f"{parent} {epoch} combined spectrum")
    ax.legend()
    ax.set_xscale('asinh')
    fig.tight_layout()
    
    
    #Save figure:
    png = outdir / f"{parent}_{epoch}_combined_{inst_label}.png"
    pdf = outdir / f"{parent}_{epoch}_combined_{inst_label}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    if show: plt.show();plt.close(fig)
    #save copy in epoch dir too
    if outdir != epoch_dir:
        png = epoch_dir / png.name
        pdf = epoch_dir / pdf.name
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
    print(f"✅Plots for {epoch} saved to {outdir}")
    if debug: print(f"  saved {png.name}, {pdf.name}")