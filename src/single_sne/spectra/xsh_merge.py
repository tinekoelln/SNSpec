# single_sne/spectra/xsh_merge.py
from __future__ import annotations
import numpy as np
import astropy.units as u, Quantity
from .stitching import stitch_arms

__all__ = ["merge_uvb_vis_nir",
           "merge_vis_nir"]

def merge_uvb_vis_nir(
    uvb: tuple[u.Quantity, u.Quantity, u.Quantity] | None,
    vis: tuple[u.Quantity, u.Quantity, u.Quantity] | None,
    nir: tuple[u.Quantity, u.Quantity, u.Quantity] | None,
    *,
    uvb_vis_overlap=(Quantity(550, u.nm), Quantity(555, u.nm)),
    uvb_vis_edge=Quantity(555, u.nm),
    vis_nir_overlap=(Quantity(1010, u.nm), Quantity(1020, u.nm)),
    vis_nir_edge=Quantity(1019,u.nm),
    scale_stat="median",
    debug=False,
) -> tuple[u.Quantity, u.Quantity, u.Quantity, float, float] :
    """
    Pairwise stitch: (UVB + VIS) → Combo, then (Combo + NIR) → final.
    Fluxes must already be F_lambda in the same units.
    returns: (wavelength, flux, error, vis_scale, nir_scale)
    """
    
    w, f, err = None, None, None
    if uvb and vis:
        w, f, err, s_vis = stitch_arms(
            uvb[0], uvb[1], uvb[2],  vis[0], vis[1], vis[2],
            overlap=uvb_vis_overlap, stitch_edge=uvb_vis_edge,
            scale_stat=scale_stat, debug=debug,
        )
    elif uvb:
        w, f, err = uvb
    elif vis:
        w, f, err = vis

    if nir:
        if (w is None) or (f is None):
            w, f, err = nir
        else:
            w, f, err, s_nir = stitch_arms(
                w, f, err, nir[0], nir[1], nir[2],
                overlap=vis_nir_overlap, stitch_edge=vis_nir_edge,
                scale_stat=scale_stat, debug=debug,
            )
            
    if w is None:
        return (u.Quantity([]), u.Quantity([]))
    return w, f, err, s_vis if uvb and vis else 1.0, s_nir if nir and ((uvb and vis) or vis) else 1.0



def merge_vis_nir(
    vis: tuple[u.Quantity, u.Quantity, u.Quantity] | None,
    nir: tuple[u.Quantity, u.Quantity, u.Quantity] | None,
    *,
    vis_nir_overlap=(Quantity(1010,u.nm), Quantity(1020, u.nm)),
    vis_nir_edge=Quantity(1019,u.nm),
    scale_stat="median",
    debug=False,
    return_scaled = False
) -> tuple[u.Quantity, u.Quantity, u.Quantity, float, float] :
    """
    Pairwise stitch: (UVB + VIS) → Combo, then (Combo + NIR) → final.
    Fluxes must already be F_lambda in the same units.
    returns: (wavelength, flux, error, vis_scale, nir_scale)
    """
    
    w, f, err = None, None, None
    if vis and nir:
        if return_scaled:
            w, f, err, s_nir, nir_scaled = stitch_arms(
            vis[0], vis[1], vis[2], nir[0], nir[1], nir[2],
            overlap=vis_nir_overlap, stitch_edge=vis_nir_edge,
            scale_stat=scale_stat, debug=debug, return_scaled=return_scaled
            )
        else:
            w, f, err, s_nir = stitch_arms(
                vis[0], vis[1], vis[2], nir[0], nir[1], nir[2],
                overlap=vis_nir_overlap, stitch_edge=vis_nir_edge,
                scale_stat=scale_stat, debug=debug, return_scaled=return_scaled
            )
    elif vis:
        w, f, err = vis
    elif nir:
        w, f, err = nir
            
    if w is None:
        return (u.Quantity([]), u.Quantity([]))
    
    if return_scaled:
        if not (vis and nir):s_nir = 1.0
        return w, f, err, s_nir, nir_scaled 
    else:
        return w, f, err, s_nir if vis and nir else 1.0
    

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