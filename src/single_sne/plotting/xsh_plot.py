from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
import scienceplots
plt.style.use(['science'])


__all__ = ["plot_signal_to_noise_combined",
            "plot_xsh_arms_and_combined"]


def plot_signal_to_noise_per_arm(
    arm_data: tuple[np.ndarray, np.ndarray] | None,
    *, 
    outfile_base,
    label: str,
    title: str,
    debug = False,
    show = False
):
    w, snr_pix = arm_data[0], arm_data[1]
    #calculate signal to noise:
    figsize = (16.7/2.54, 8.0/2.54)
    left, right, bottom, top = .06, .98, .12, .93
    

    fig2 = plt.figure(figsize=figsize)
    fig2.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig2.add_subplot(1,1,1)
    ax.set_title(title)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"SNR")
    ax.plot(w, snr_pix, lw=1.3, label='SNR per pixel', color='k', alpha = 0.5)
    ax.legend()
    
    if label == 'UVB':
        outfile_base = Path(outfile_base)
        pdf = f"{outfile_base}_SNR_UVB.pdf"
        png = f"{outfile_base}_SNR_UVB.png"
        txt = f"{outfile_base}_SNR_UVB.txt"
    elif label=='VIS':
        outfile_base = Path(outfile_base)
        pdf = f"{outfile_base}_SNR_VIS.pdf"
        png = f"{outfile_base}_SNR_VIS.png"
        txt = f"{outfile_base}_SNR_VIS.txt"
    elif label =='NIR':
        outfile_base = Path(outfile_base)
        pdf = f"{outfile_base}_SNR_NIR.pdf"
        png = f"{outfile_base}_SNR_NIR.png"
        txt = f"{outfile_base}_SNR_NIR.txt"
        
    fig2.savefig(pdf, dpi=500, bbox_inches="tight")
    fig2.savefig(png, dpi=500, bbox_inches="tight") 
    np.savetxt(
        txt,
        np.column_stack([w, snr_pix]),
        fmt="%.6f",
        header="wavelength[nm] SNR",
        comments=''
        )   
    
    
    
    
def plot_signal_to_noise_combined(
    uvb: tuple[np.ndarray, np.ndarray] | None,
    vis: tuple[np.ndarray, np.ndarray] | None,
    nir: tuple[np.ndarray, np.ndarray] | None,
    combined: tuple[np.ndarray, np.ndarray] | None,
    *,
    title: str,
    outfile_base: str,
    debug=False,
    show = False,
):
    figsize = (16.7/2.54, 8.0/2.54)
    left, right, bottom, top = .06, .98, .12, .93

    fig3 = plt.figure(figsize=figsize)
    fig3.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig3.add_subplot(1,1,1)
    ax.set_title(title)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"SNR")
    
    def plot_l2fl(w, f, label, lw=2.0, *, color=None, cut=None, alpha = 1.0):
        if w is None or w.size == 0: return
        ww, ff = w, f
        if cut is not None:
            m = (ww >= cut[0]) & (ww <= cut[1])
            ww, ff = ww[m], ff[m]
        ax.plot(ww, ff, lw=lw, label=label, color=color, alpha=alpha)
        return w, f

    if uvb:        
        w_uvb, snr_uvb_pix = plot_l2fl(uvb[0], (uvb[1]), "UVB", alpha = 0.5)
        plot_signal_to_noise_per_arm((w_uvb, snr_uvb_pix), outfile_base= outfile_base, label = 'UVB', title='Signal to Noise Ratio in UVB Arm', debug = debug)
        
    if vis:
        w_vis, snr_vis_pix = plot_l2fl(vis[0], vis[1], "VIS", alpha = 0.5, cut=(545, 1015))

        plot_signal_to_noise_per_arm((w_vis, snr_vis_pix), outfile_base= outfile_base, label = 'VIS', title='Signal to Noise Ratio in VIS Arm', debug = debug)

    if nir: 
        w_nir, snr_nir_pix = plot_l2fl(nir[0], nir[1], "NIR",  alpha = 0.5, cut=(1000, 1e9))
        
        
        plot_signal_to_noise_per_arm((w_nir, snr_nir_pix), outfile_base= outfile_base,  label = 'NIR', title='Signal to Noise Ratio in NIR Arm', debug = debug)
    
    if combined:
        plot_l2fl(combined[0], combined[1], "combined", lw=1.0, color="black")
    
    # x-range auto
    xs = []
    if uvb is None:
        for pair in (vis, nir, combined):
            if pair and pair[0] is not None and np.size(pair[0]):
                w = pair[0]
                if hasattr(w, "unit"):
                    xs.append(np.asarray(w.to_value(u.nm), float))
                else:
                    xs.append(np.asarray(w, float))

    else:   
        for pair in (uvb, vis, nir, combined):
            print(pair)

            if pair and pair[0] is not None and np.size(pair[0]):
                w = pair[0]
                print(w)
                if hasattr(w, "unit"):
                    xs.append(np.asarray(w.to_value(u.nm), float))
                else:
                    xs.append(np.asarray(w, float))

            
    if xs:
        xmin = max(320, min(float(np.nanmin(x)) for x in xs))
        xmax = min(2500, max(float(np.nanmax(x)) for x in xs))
        ax.set_xlim(xmin, xmax)

    # y-range auto
    '''ys = []
    for pair in (uvb, vis, nir, combined):
        if pair and pair[0].size:
            ys.append(pair[1] * (pair[0]**2))
    if ys:
        yall = np.concatenate(ys)
        ylow = np.nanpercentile(yall, 0.5)
        yhigh = np.nanpercentile(yall, 99.5)
        pad = 0.1 * (yhigh - ylow) if np.isfinite(yhigh - ylow) else 1.0
        ax.set_ylim(ylow - pad, yhigh + pad)'''

    ax.legend()
    outfile_base = Path(outfile_base)
    pdf = f"{outfile_base}_SNR_combined.pdf"
    png = f"{outfile_base}_SNR_combined.png"
    txt = f"{outfile_base}_SNR_combined.txt"
    fig3.savefig(pdf, dpi=300, bbox_inches="tight")
    fig3.savefig(png, dpi=300, bbox_inches="tight")
    
    np.savetxt(
        txt,
        np.column_stack([combined[0], combined[1]]),
        fmt="%.6f",
        header="wavelength[nm]  SNR",
        comments=''
        )  
    if show: plt.show();plt.close(fig3)
    if debug:
        print(f"[OK] saved {pdf}")
        print(f"[OK] saved {png}")

    

def plot_xsh_arms_and_combined(
    uvb: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    vis: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    nir: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    combined: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    title: str,
    outfile_base: str,
    debug=False,
    show = False,
    snr = False
):
    fontsize = 10
    if show: 
        figsize = (16.7/2.54, 8.0*1.5/2.54)
    else:
        figsize = (16.7/2.54, 8.0/2.54)
    left, right, bottom, top = .06, .98, .12, .93
    
    def plot_l2fl(w, f, label, lw=2.0, snr = False, ferr = None, color=None, cut=None):
        if w is None or w.size == 0: return
        ww, ff = w, f
        if ferr is not None:
            fferr = ferr
        if cut is not None:
            m = (ww >= cut[0]) & (ww <= cut[1])
            ww, ff = ww[m], ff[m]
            if ferr is not None: fferr = fferr[m]
        if snr: 
            ax["big"].step(ww, ff * (ww**2), where="mid", lw=lw, label=label, color=color)
            ax["small"].step(ww, ff/fferr, where="mid", lw=lw, label=label, color=color)
        else:
            ax.step(ww, ff * (ww**2), where="mid", lw=lw, label=label, color=color)
    
    if snr: 
        fig1, ax = plt.subplot_mosaic(
        [
            ["big"],      # top large panel
            ["small"],   # small bottom panel
        ],
        height_ratios=[2, 1],  # control relative size
        figsize=figsize,
        constrained_layout=True
        )
        ax["big"].set_title(title)
        ax["small"].set_title("Signal to Noise Ratio")
        ax["small"].set_xlabel("Wavelength [nm]")
        ax["small"].set_ylabel("SNR")
        ax["big"].set_xlabel("Wavelength [nm]")
        ax["big"].set_ylabel(r"$\lambda^2 F_\lambda$")
        
    else:
        fig1, ax = plt.subplots(1,1,
                                figsize=figsize,
                                constrained_layout=True)
        ax.set_title(title)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"$\lambda^2 F_\lambda$")


    if uvb: plot_l2fl(uvb[0], uvb[1], "UVB", snr=snr, ferr = uvb[2])
    if vis: plot_l2fl(vis[0], vis[1], "VIS",snr=snr, ferr = vis[2], cut=(545, 1015))
    if nir: plot_l2fl(nir[0], nir[1], "NIR", snr=snr, ferr=nir[2], cut=(1000, 1e9))
    if combined:
        plot_l2fl(combined[0], combined[1], "combined", snr=snr, ferr=combined[2], lw=1.0, color="black")

    # x-range auto
    xs = []
    for pair in (uvb, vis, nir, combined):
        if pair and pair[0] is not None and np.size(pair[0]):
                w = pair[0]
                if hasattr(w, "unit"):
                    xs.append(np.asarray(w.to_value(u.nm), float))
                else:
                    xs.append(np.asarray(w, float))
    if xs:
        xmin = max(320, min(float(np.nanmin(x)) for x in xs))
        xmax = min(2500, max(float(np.nanmax(x)) for x in xs))
        if snr:
            ax["big"].set_xlim(xmin, xmax)
            ax["small"].set_xlim(xmin, xmax)
        else: 
            ax.set_xlim(xmin, xmax)

    # y-range auto
    ys = []
    for pair in (uvb, vis, nir, combined):
        if pair and pair[0] is not None and np.size(pair[0]):
                w = pair[0]
                f= pair[1]
                if hasattr(w, "unit"):
                    if hasattr(f, "unit"):
                        ys.append((f.to_value(f.unit)*(w.to_value(w.unit))**2).astype(float))
                else:
                    ys.append(np.asarray(f*w**2, float))
    if ys:
        yall = np.concatenate(ys)
        ylow = np.nanpercentile(yall, 0.5)
        yhigh = np.nanpercentile(yall, 99.5)
        pad = 0.1 * (yhigh - ylow) if np.isfinite(yhigh - ylow) else 1.0
        if snr:
            ax["big"].set_ylim(ylow - pad, yhigh + pad)
            ax["big"].legend()
        else:
            ax.set_ylim(ylow - pad, yhigh + pad)
            ax.legend()


    outfile_base = Path(outfile_base)
    if snr:
        pdf = f"{outfile_base}_with_SNR.pdf"
        png = f"{outfile_base}_with_SNR.png"
    else:
        pdf = f"{outfile_base}.pdf"
        png = f"{outfile_base}.png"
    fig1.savefig(pdf, dpi=600, bbox_inches="tight")
    fig1.savefig(png, dpi=600, bbox_inches="tight")
    if show: plt.show();plt.close(fig1)
    if debug:
        print(f"[OK] saved {pdf}")
        print(f"[OK] saved {png}")
        
