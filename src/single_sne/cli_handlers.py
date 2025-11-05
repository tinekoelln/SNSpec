from __future__ import annotations
import numpy as np
import astropy.units as u

from single_sne.io.xshooter import discover_merge1d_files, read_primary_linear_wcs, _get_object_date
from single_sne.spectra.spectra import bin_spectrum
from single_sne.spectra.xsh_merge import merge_uvb_vis_nir
from single_sne.plotting.xsh_plot import plot_xsh_arms_and_combined
from single_sne.units import INSTRUMENT_UNITS
import logging
import argparse
import pathlib
import sys
from datetime import date
import numpy as np
from pathlib import Path
from astropy import units as u
import click
from .spectra.instruments import which_kind_of_spectrum, list_supported_instruments

def cmd_convert_fnu(args):
    infile = Path(args.infile).expanduser()
    outfile = Path(args.outfile).expanduser()

    # load 2-column ASCII: wavelength, flux
    data = np.loadtxt(infile)
    wave = data[:, 0]
    flam = data[:, 1]

    fnu = to_fnu_mjy(flam, wave)
    out = np.column_stack([wave, fnu])
    np.savetxt(outfile, out, header="wavelength_A   fnu_mJy")

    print(f"✅ Wrote converted file: {outfile}")
    return 0

def cmd_bin_spectrum(args, debug=False):
    import sys
    import numpy as np
    from pathlib import Path
    import pandas as pd
    from astropy import units as u
    from .spectra import bin_spectrum

    if getattr(args, "verbose", 0):
        debug = True
        print("--------DEBUG MODE ACTIVATED--------")

    # validate
    if args.bin_size <= 0:
        print("ERROR: --bin-size must be > 0", file=sys.stderr)
        return 2

    # Handle stdout early
    to_stdout = getattr(args, "outfile", None) in (None, "-", "/dev/stdout")

    # resolve paths
    p_in = Path(args.infile).expanduser()
    if not p_in.exists():
        print(f"ERROR: input file not found: {p_in}", file=sys.stderr)
        return 2

    if not to_stdout:
        p_out = Path(args.outfile).expanduser()
        if p_out.exists() and not getattr(args, "overwrite", False):
            print(f"ERROR: {p_out} exists. Use --overwrite to replace.", file=sys.stderr)
            return 2

    # --- load 2–3 col ASCII robustly with pandas ---
    src = sys.stdin if args.infile in (None, "-", "/dev/stdin") else p_in
    try:
        df = pd.read_csv(
            src,
            sep=r"[,\s]+",             # commas OR any whitespace
            engine="python",
            comment="#",
            header=None,
            skiprows=getattr(args, "skip_header", 0),
            dtype=str,                 # read as strings first
        )
        if getattr(args, "verbose", 0):
            print(f"DEBUG raw df shape: {df.shape}")

        df = df.apply(pd.to_numeric, errors="coerce")
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
        elif df.shape[1] == 2:
            pass
        else:
            print("ERROR: need at least 2 numeric columns (wavelength, flux).", file=sys.stderr)
            return 2

        df = df.dropna(subset=[0, 1])
        if df.empty:
            print("ERROR: no valid numeric rows after cleaning.", file=sys.stderr)
            return 2

        wavelength = df.iloc[:, 0].to_numpy(dtype=float)
        flux       = df.iloc[:, 1].to_numpy(dtype=float)
        flux_err   = df.iloc[:, 2].to_numpy(dtype=float) if df.shape[1] >= 3 else None

        if getattr(args, "verbose", 0):
            ferr_shape = "None" if flux_err is None else flux_err.shape
            print(f"DEBUG cleaned columns: wave={wavelength.shape}, flux={flux.shape}, ferr={ferr_shape}")

    except Exception as e:
        print(f"ERROR loading/cleaning input: {e}", file=sys.stderr)
        return 2

    # --- compute ---
    try:
        wave_b, flux_b, ferr_b = bin_spectrum(
            wavelength=wavelength,
            flux=flux,
            bin_size=args.bin_size,
            wave_unit=args.wave_unit,
            flux_unit=args.flux_unit,
            out_wave_unit=args.out_wave_unit,
            out_flux_unit=args.out_flux_unit,
            flux_err=flux_err,  # may be None
            debug=debug,
        )
    except Exception as e:
        print(f"ERROR during binning: {e}", file=sys.stderr)
        return 2

    # Decide target units for writing
    out_wu = u.Unit(args.out_wave_unit) if args.out_wave_unit else u.Unit(args.wave_unit)
    out_fu = u.Unit(args.out_flux_unit) if args.out_flux_unit else u.Unit(args.flux_unit)

    # bin_spectrum already returned Quantities in these (or convertible) units
    try:
        w_val  = wave_b.to_value(out_wu)
        f_val  = flux_b.to_value(out_fu)
        fe_val = ferr_b.to_value(out_fu) if ferr_b is not None else None
    except Exception as e:
        print(f"ERROR converting output units: {e}", file=sys.stderr)
        return 2

    cols = [w_val, f_val] + ([fe_val] if fe_val is not None else [])
    out  = np.column_stack(cols)

    if debug:
        print("DEBUG output shape (rows, cols):", out.shape)

    hdr = f"wavelength[{out_wu}]  flux[{out_fu}]" + (f"  flux_err[{out_fu}]" if fe_val is not None else "")

    # Write (stdout or file). Use scientific format for flux columns.
    if to_stdout:
        fmt = ["%.8f", "%.6e"] + (["%.6e"] if fe_val is not None else [])
        np.savetxt(sys.stdout.buffer, out, header=hdr, fmt=fmt)
    else:
        fmt = ["%.8f", "%.6e"] + (["%.6e"] if fe_val is not None else [])
        np.savetxt(str(p_out), out, header=hdr, fmt=fmt)

    print(f"✅ wrote binned spectrum → {'stdout' if to_stdout else p_out}")
    return 0

        
def cmd_read_jwst(args):
    from astropy import units as u
    from .io.jwst import read_jwst_arrays
    w, f, fe = read_jwst_arrays(args.infile, as_quantity=args.as_quantity)
    if args.as_quantity:
        wv, fv = w.to_value(u.um), f.to_value(u.mJy)
        fev = fe.to_value(u.mJy) if fe is not None else None
    else:
        wv, fv, fev = w, f, fe
    out = np.column_stack([wv, fv] + ([fev] if fev is not None else []))
    np.savetxt(args.outfile, out, header="wavelength[um] flux[mJy] [flux_err[mJy]]")


def cmd_units(args, debug=False) -> int:
    """
    Print default (wavelength, flux) units for an instrument.
    """
    try:
        w, f = which_kind_of_spectrum(args.instrument)
    except ValueError as e:
        print(f"ERROR: {e}")
        if args.list_on_error:
            print("\nSupported instruments:")
            for name in list_supported_instruments():
                print(f"  - {name}")
        return 2

    if args.output == "latex":
        w_str = u.Unit(w).to_string("latex_inline")
        f_str = u.Unit(f).to_string("latex_inline")
    else:
        w_str = str(w)
        f_str = str(f)

    print(f"Instrument: {args.instrument}")
    print(f"Wavelength unit: {w_str}")
    print(f"Flux unit:       {f_str}")
    return 0

def cmd_list_instruments(args, debug=False) -> int:
    for name in list_supported_instruments():
        print(name)
    return 0

def cmd_xsh_merge_plot(args, debug=False) -> int:
    sel = discover_merge1d_files(
        args.root,
        product_mode=args.product_mode,
        prefer_end_products=not args.no_prefer_end,
        allow_tmp=args.allow_tmp,
    )
    if not sel:
        print("ERROR: no MERGE1D files found")
        return 2

    if getattr(args, "verbose", 0):
            debug = True
            print("--------DEBUG MODE ACTIVATED--------")

    # read present arms (nm + dimensionless → we assign your house flux unit)
    data = {}
    first_hdr = None
    for arm in ("UVB","VIS","NIR"):
        if arm in sel:
            w_nm, f_native, hdr = read_primary_linear_wcs(sel[arm], ext=0)
            if first_hdr is None:
                first_hdr = hdr
            data[arm] = (w_nm, f_native)

    # metadata / output base
    obj, date_title, date_file = _get_object_date(first_hdr) if first_hdr else ("TARGET", "", "")
    base_out = args.out if args.out else (f"{obj}_{date_file}_xshooter" if date_file else f"{obj}_xshooter")

    # rebin per arm (nm grids)
    uvb = vis = nir = None
    wave_unit, flux_unit = INSTRUMENT_UNITS["XSHOOTER"]   # your chosen house unit

    if "UVB" in data:
        w, f = data["UVB"]
        dw = args.dw_uvb
        wgrid, f_out, ferr_out = bin_spectrum(w, f, dw,wave_unit, flux_unit)
        uvb = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit))

    if "VIS" in data:
        w, f = data["VIS"]
        dw = args.dw_vis
        wgrid, f_out, ferr_out = bin_spectrum(w, f, dw,wave_unit, flux_unit)
        vis = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit))

    if "NIR" in data:
        w, f = data["NIR"]
        dw = args.dw_nir
        wgrid, f_out, ferr_out = bin_spectrum(w, f, dw,wave_unit, flux_unit)
        nir = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit))
    
    # merge
    uvb_vis_overlap = (args.uvb_vis_overlap[0] * u.nm,
                       args.uvb_vis_overlap[1] * u.nm)
    vis_nir_overlap = (args.vis_nir_overlap[0] * u.nm,
                       args.vis_nir_overlap[1] * u.nm)
    
    
    w_comb, f_comb = merge_uvb_vis_nir(
        uvb=(uvb[0]*wave_unit, uvb[1]*flux_unit) if uvb else None,
        vis=(vis[0]*wave_unit, vis[1]*flux_unit) if vis else None,
        nir=(nir[0]*wave_unit, nir[1]*flux_unit) if nir else None,
        uvb_vis_overlap=uvb_vis_overlap,
        uvb_vis_edge=args.uvb_vis_edge * wave_unit,
        vis_nir_overlap=vis_nir_overlap,
        vis_nir_edge=args.vis_nir_edge * wave_unit,
        scale_stat=args.scale_stat,
        debug=debug,
    )

    # write ASCII (nm, Flam)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dat = out_dir / f"{base_out}.dat"
    outfile_base = out_dir / base_out
    
    if w_comb.size:
        w = w_comb.to_value(wave_unit)
        f = f_comb.to_value(flux_unit)
        m = np.isfinite(w) & np.isfinite(f)
        w, f = w[m], f[m]
        order = np.argsort(w)
        w, f = w[order], f[order]
        with open(out_dat, "w") as fh:
            fh.write("# wavelength(nm)  flux(erg/s/cm2/Ang)\n")
            for wi, fi in zip(w, f):
                fh.write(f"{wi:.8f} {fi:.16e}\n")
        if debug:
            print(f"[OK] wrote {out_dat}")

    # plot
    plot_xsh_arms_and_combined(
        uvb=(uvb[0], uvb[1]) if uvb else None,
        vis=(vis[0], vis[1]) if vis else None,
        nir=(nir[0], nir[1]) if nir else None,
        combined=(w_comb.to_value(wave_unit), f_comb.to_value(flux_unit)) if w_comb.size else None,
        title=f"{obj} ({date_title}) - VLT/X-shooter" if date_title else f"{obj} - VLT/X-shooter",
        outfile_base=outfile_base,
        debug=debug,
    )
    return 0
