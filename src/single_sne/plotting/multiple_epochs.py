import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import astropy.units as u
from astropy.units import Quantity
import scienceplots
plt.style.use(['science'])
__all__ = ["_plot_epoch"]   


def load_combined_spectrum(path, debug = False):
    if debug: print(f"\n\n\n[load_combine_spectrum]")
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

def plot_all_epochs(root: str | Path,
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
    
    root = Path(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spectra = []
    # epoch directories: Epoch1, Epoch2, Epoch3...
    labels = []
    for epoch_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("epoch")):
        #read in spectrum for this epoch
        parent = epoch_dir.parent.name   # e.g. "SN2025cy"
        epoch  = epoch_dir.name          # e.g. "Epoch1"
        inst_order = []
        for p in epoch_dir.iterdir():
            if p.is_dir():
                inst_order.append(p.name.upper())
        inst_label = "+".join(inst_order)
        labels.append(f"{epoch}-{inst_label}")
        # find the single combined file saved earlier
        matches = sorted(epoch_dir.glob("*_combined_*.dat"))
        if not matches:
            if debug: print(f"[skip] no combined file in {epoch_dir.name}")
            continue
        combined = matches[0]
        spectra.append(combined.resolve())
        
    
    jwst_units = np.any(np.char.find(np.char.upper(inst_label), "JWST") >= 0)
    
    fig, ax = plt.subplots(figsize=(8.3, 4.5))
    if jwst_units:
        ax.set_xlabel("Wavelength [µm]")
        ax.set_ylabel(r"$F_{\nu} [mJy]$")
        ax.set_xlim((0.3, 12.5))
    else:
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"$\lambda^2 F_\lambda$ [erg/s/cm$^2$/Å]")
    # helper

    for i in range(0, len(spectra)):
        epoch_spectrum = spectra[i]

        epoch_table = load_combined_spectrum(epoch_spectrum)
        w_comb = epoch_table["wavelength"]
        f_comb = epoch_table["flux"]
        ax.step(w_comb, f_comb, where = 'post', label=f"Epoch {i+1}", zorder = -i)
        
    ax.set_title(f"{parent} - All epochs")
    ax.set_xscale('asinh')
    ax.legend()
    fig.tight_layout()
    
    
    #Save figure:
    png = outdir / f"{parent}_all_epochs_combined.png"
    pdf = outdir / f"{parent}_all_epochs_combined.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    if show: plt.show();plt.close()
    #save copy in parent dir too
    if outdir != root:
        png = root / png.name
        pdf = root / pdf.name
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
    print(f"✅Plot for all epochs combined saved to {outdir}")
    if debug: print(f"  saved {png.name}, {pdf.name}")