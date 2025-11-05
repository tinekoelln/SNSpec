from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
import scienceplots
plt.style.use(['science'])


__all__ = ["plot_xsh_arms_and_combined"]

def plot_xsh_arms_and_combined(
    uvb: tuple[np.ndarray, np.ndarray] | None,
    vis: tuple[np.ndarray, np.ndarray] | None,
    nir: tuple[np.ndarray, np.ndarray] | None,
    combined: tuple[np.ndarray, np.ndarray] | None,
    *,
    title: str,
    outfile_base: str,
    debug=False,
):
    fontsize = 10
    figsize = (16.7/2.54, 8.0/2.54)
    left, right, bottom, top = .06, .98, .12, .93

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(r"$\lambda^2 F_\lambda$")

    def plot_l2fl(w, f, label, lw=2.0, color=None, cut=None):
        if w is None or w.size == 0: return
        ww, ff = w, f
        if cut is not None:
            m = (ww >= cut[0]) & (ww <= cut[1])
            ww, ff = ww[m], ff[m]
        ax.step(ww, ff * (ww**2), where="mid", lw=lw, label=label, color=color)

    if uvb: plot_l2fl(uvb[0], uvb[1], "UVB")
    if vis: plot_l2fl(vis[0], vis[1], "VIS", cut=(545, 1015))
    if nir: plot_l2fl(nir[0], nir[1], "NIR", cut=(1000, 1e9))
    if combined:
        plot_l2fl(combined[0], combined[1], "combined", lw=1.0, color="black")

    # x-range auto
    xs = []
    for pair in (uvb, vis, nir, combined):
        if pair and pair[0].size: xs.append(pair[0])
    if xs:
        xmin = max(320, min(float(np.nanmin(x)) for x in xs))
        xmax = min(2500, max(float(np.nanmax(x)) for x in xs))
        ax.set_xlim(xmin, xmax)

    # y-range auto
    ys = []
    for pair in (uvb, vis, nir, combined):
        if pair and pair[0].size:
            ys.append(pair[1] * (pair[0]**2))
    if ys:
        yall = np.concatenate(ys)
        ylow = np.nanpercentile(yall, 0.5)
        yhigh = np.nanpercentile(yall, 99.5)
        pad = 0.1 * (yhigh - ylow) if np.isfinite(yhigh - ylow) else 1.0
        ax.set_ylim(ylow - pad, yhigh + pad)

    ax.legend()
    outfile_base = Path(outfile_base)
    pdf = f"{outfile_base}.pdf"
    png = f"{outfile_base}.png"
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    if debug:
        print(f"[OK] saved {pdf}")
        print(f"[OK] saved {png}")
