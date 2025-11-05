# single_sne/spectra/xsh_merge.py
from __future__ import annotations
import numpy as np
import astropy.units as u
from .stitching import stitch_arms

__all__ = ["merge_uvb_vis_nir"]

def merge_uvb_vis_nir(
    uvb: tuple[u.Quantity, u.Quantity] | None,
    vis: tuple[u.Quantity, u.Quantity] | None,
    nir: tuple[u.Quantity, u.Quantity] | None,
    *,
    uvb_vis_overlap=(550, 555) * u.nm,
    uvb_vis_edge=555 * u.nm,
    vis_nir_overlap=(1010, 1020) * u.nm,
    vis_nir_edge=1019 * u.nm,
    scale_stat="median",
    debug=False,
) -> tuple[u.Quantity, u.Quantity]:
    """
    Pairwise stitch: (UVB + VIS) → Combo, then (Combo + NIR) → final.
    Fluxes must already be F_lambda in the same units.
    """
    w, f = None, None
    if uvb and vis:
        w, f, _ = stitch_arms(
            uvb[0], uvb[1], vis[0], vis[1],
            overlap=uvb_vis_overlap, stitch_edge=uvb_vis_edge,
            scale_stat=scale_stat, debug=debug
        )
    elif uvb:
        w, f = uvb
    elif vis:
        w, f = vis

    if nir:
        if (w is None) or (f is None):
            w, f = nir
        else:
            w, f, _ = stitch_arms(
                w, f, nir[0], nir[1],
                overlap=vis_nir_overlap, stitch_edge=vis_nir_edge,
                scale_stat=scale_stat, debug=debug
            )
    if w is None:
        return (u.Quantity([]), u.Quantity([]))
    return w, f
