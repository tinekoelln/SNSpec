from __future__ import annotations
import numpy as np

from single_sne.io.xshooter import discover_merge1d_files, read_primary_linear_wcs, _get_object_date
from single_sne.spectra.spectra import bin_spectrum, to_fnu_mjy
from single_sne.spectra.xsh_merge import merge_uvb_vis_nir
from single_sne.plotting.xsh_plot import plot_xsh_arms_and_combined, plot_signal_to_noise_combined
from single_sne.units import INSTRUMENT_UNITS
from single_sne.spectra.process_epochs import process_all_epochs
from single_sne.plotting.single_epoch import plot_epoch
from single_sne.plotting.multiple_epochs import plot_all_epochs
import sys
import numpy as np
from pathlib import Path
from astropy import units as u
from .spectra.instruments import which_kind_of_spectrum, list_supported_instruments

def cmd_convert_fnu(args, debug = False):
    
    if getattr(args, "verbose", 0):
            debug = True
            print("--------DEBUG MODE ACTIVATED--------")
            
    infile = Path(args.infile).expanduser()
    outfile = Path(args.outfile).expanduser()

    # load 2-column ASCII: wavelength, flux
    data = np.loadtxt(infile)
    wave = data[:, 0]
    flam = data[:, 1]

    fnu = to_fnu_mjy(flam, wave, debug)
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


def cmd_read_jwst(args, debug = False):
    if getattr(args, "verbose", 0):
            debug = True
            print("--------DEBUG MODE ACTIVATED--------")
    from astropy import units as u
    from .io.jwst import read_jwst_arrays
    w, f, fe = read_jwst_arrays(args.infile, as_quantity=args.as_quantity, debug = debug)
    if args.as_quantity:
        wv, fv = w.to_value(u.um), f.to_value(u.mJy)
        fev = fe.to_value(u.mJy) if fe is not None else None
    else:
        wv, fv, fev = w, f, fe
    out = np.column_stack([wv, fv] + ([fev] if fev is not None else []))
    np.savetxt(args.outfile, out, header="wavelength[um] flux[mJy] [flux_err[mJy]]")
    
def cmd_process_salt(args, debug = False):
    if getattr(args, "verbose", 0):
            debug = True
            print("--------DEBUG MODE ACTIVATED--------")
            from astropy import units as u


def cmd_units(args, debug=False) -> int:
    """
    Print default (wavelength, flux) units for an instrument.
    """
    if getattr(args, "verbose", 0):
            debug = True
            print("--------DEBUG MODE ACTIVATED--------")
            
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


    root = Path(args.root).expanduser()
    if args.outdir is None:
        outdir = root
        if debug:
            print(f"[info] No --outdir given → using root: {outdir}")
    else:
        outdir = Path(args.outdir).expanduser()

    outdir.mkdir(parents=True, exist_ok=True)
    
    if getattr(args, "verbose", 0):
            debug = True
            print("--------DEBUG MODE ACTIVATED--------")
            
    if args.clean:
        print("Cleaning enabled — will filter out bad QUAL pixels")
        clean = True
    else:
        clean = False

    if args.show:
        show = True
    else:
        show = False
        
    if args.snr:
        snr = True
    else:
        snr = False
        
    if args.unbinned:
        unbinned = True
    else:
        unbinned = False
        
    if args.etc:
        etc = True
        if debug: print(f"-------ETC CALCULATION ONGOING----------")
    else:
        etc = False

    # read present arms (nm + dimensionless → we assign your house flux unit)
    data = {}
    first_hdr = None
    for arm in ("UVB","VIS","NIR"):
        if arm in sel:
            path = sel[arm]
            if isinstance(path, (list, tuple)):
                if len(path) == 0:
                    continue  # or raise, depending on how you handle “no file”
                path = path[0]   # take the single selected file
            if etc: 
                w_nm, f_native, err_native, hdr, snr_pix, snr_res = read_primary_linear_wcs(path, ext=0, debug = debug, clean=clean, etc=etc)
                if first_hdr is None:
                    first_hdr = hdr
                data[arm] = (w_nm, f_native, err_native, snr_pix, snr_res)
                if debug: print(f"LENGTH OF DATA IN {arm}: {len(w_nm), len(snr_pix), len(snr_res)}")
            else:
                w_nm, f_native, err_native, hdr = read_primary_linear_wcs(path, ext=0, debug = debug, clean=clean)
                if first_hdr is None:
                    first_hdr = hdr
                data[arm] = (w_nm, f_native, err_native)

    # metadata / output base
    obj, date_title, date_file = _get_object_date(first_hdr) if first_hdr else ("TARGET", "", "")
    if clean:
        base_out = args.out if args.out else (f"{obj}_{date_file}_xshooter_clean" if date_file else f"{obj}_xshooter_clean")
    else: 
        base_out = args.out if args.out else (f"{obj}_{date_file}_xshooter" if date_file else f"{obj}_xshooter")
    # rebin per arm (nm grids)
    uvb = vis = nir = None
    wave_unit, flux_unit = INSTRUMENT_UNITS["XSHOOTER"]   # your chosen house unit
    
    def clean_arm(w, f, err, snr_pix = None, snr_res = None):
        mask = np.isfinite(f) & np.isfinite(err) & (err != 0)
        w, f, err = w[mask], f[mask], err[mask]
        if snr_pix is not None:
            snr_pix = snr_pix[mask]
            if snr_res is not None:
                snr_res = snr_res[mask]
                return w, f, err, snr_pix, snr_res
        else:
            return w, f, err
                

    uvb_unbinned, vis_unbinned, nir_unbinned = None, None, None
    if "UVB" in data:
        if etc:
            w, f, err, snr_pix, snr_res = data["UVB"]
            w, f, err, snr_pix, snr_res = clean_arm(w, f, err, snr_pix, snr_res)
        else:
            w, f, err  = data["UVB"]
            w, f, err = clean_arm(w, f, err)
        dw = args.dw_uvb
        wgrid, f_out, ferr_out = bin_spectrum(w, f, dw, wave_unit, flux_unit, flux_err= err)
        uvb = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit), ferr_out.to_value(flux_unit))
        if unbinned:   
            uvb_unbinned = (w, f, err)
        else: 
            uvb_unbinned = None
        if etc: uvb_etc = (w, f, err, snr_pix, snr_res)

    if "VIS" in data:
        if etc:
            w, f, err, snr_pix, snr_res = data["VIS"]
            w, f, err, snr_pix, snr_res = clean_arm(w, f, err, snr_pix, snr_res)

        else:
            w, f, err  = data["VIS"]
            w, f, err = clean_arm(w, f, err)

        dw = args.dw_vis
        wgrid, f_out, ferr_out = bin_spectrum(w, f, dw, wave_unit, flux_unit, flux_err= err)
        vis = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit), ferr_out.to_value(flux_unit))
        if unbinned: vis_unbinned = (w, f, err)
        if etc: vis_etc = (w, f, err, snr_pix, snr_res)


    if "NIR" in data:
        if etc:
            w, f, err, snr_pix, snr_res = data["NIR"]
            w, f, err, snr_pix, snr_res = clean_arm(w, f, err, snr_pix, snr_res)
        else:
            w, f, err  = data["NIR"]
            w, f, err = clean_arm(w, f, err)
        dw = args.dw_nir
        wgrid, f_out, ferr_out = bin_spectrum(w, f, dw, wave_unit, flux_unit, flux_err= err)
        nir = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit), ferr_out.to_value(flux_unit))
        if unbinned: nir_unbinned = (w, f, err)
        if etc: 
            nir_etc = (w, f, err, snr_pix, snr_res)
            print("Length of NIR data")
            print(len(w), len(snr_pix), len(snr_res))

    
    
    # merge
    uvb_vis_overlap = (args.uvb_vis_overlap[0] * u.nm,
                       args.uvb_vis_overlap[1] * u.nm)
    vis_nir_overlap = (args.vis_nir_overlap[0] * u.nm,
                       args.vis_nir_overlap[1] * u.nm)
    

    def comb_spectrum(uvb, vis, nir, uvb_vis_overlap, uvb_vis_edge, vis_nir_overlap, vis_nir_edge, scale_stat, debug = debug):
        w_comb, f_comb, err_comb, s_vis, s_nir = merge_uvb_vis_nir(
            uvb=(uvb[0]*wave_unit, uvb[1]*flux_unit, uvb[2]*flux_unit ) if uvb else None,
            vis=(vis[0]*wave_unit, vis[1]*flux_unit, vis[2]*flux_unit) if vis else None,
            nir=(nir[0]*wave_unit, nir[1]*flux_unit, nir[2]*flux_unit) if nir else None,
            uvb_vis_overlap=uvb_vis_overlap,
            uvb_vis_edge=args.uvb_vis_edge * wave_unit,
            vis_nir_overlap=vis_nir_overlap,
            vis_nir_edge=args.vis_nir_edge * wave_unit,
            scale_stat=args.scale_stat,
            debug=debug,
        )
        return w_comb, f_comb, err_comb, s_vis, s_nir
    
    w_comb, f_comb, err_comb, s_vis, s_nir = comb_spectrum(
            uvb if uvb else None, 
            vis if vis else None, 
            nir if nir else None, 
            uvb_vis_overlap=uvb_vis_overlap,
            uvb_vis_edge=args.uvb_vis_edge * wave_unit,
            vis_nir_overlap=vis_nir_overlap,
            vis_nir_edge=args.vis_nir_edge * wave_unit,
            scale_stat=args.scale_stat,
            debug=debug,
        )
        

    # write ASCII (nm, Flam)
    out_dat = outdir / f"{base_out}.dat"
    outfile_base = outdir / base_out
        
    if w_comb.size:
        w = w_comb.to_value(wave_unit)
        f = f_comb.to_value(flux_unit)
        err = err_comb.to_value(flux_unit)
        m = np.isfinite(w) & np.isfinite(f) & np.isfinite(err)
        w, f, err = w[m], f[m], err[m]
        order = np.argsort(w)
        w, f, err = w[order], f[order], err[order]
        with open(out_dat, "w") as fh:
            fh.write("# wavelength(nm)  flux(erg/s/cm2/Ang) flux_error(erg/s/cm2/Ang)\n")
            for wi, fi, erri in zip(w, f, err):
                fh.write(f"{wi:.8f} {fi:.16e} {erri:.16e}\n")
        if debug:
            print(f"[OK] wrote {out_dat}")
            
    if unbinned:
        w_comb_unbinned, f_comb_unbinned, err_comb_unbinned, s_vis_unbinned, s_nir_unbinned = comb_spectrum(
            uvb_unbinned if uvb_unbinned else None, 
            vis_unbinned if vis_unbinned else None, 
            nir_unbinned if nir_unbinned else None, 
            uvb_vis_overlap=uvb_vis_overlap,
            uvb_vis_edge=args.uvb_vis_edge * wave_unit,
            vis_nir_overlap=vis_nir_overlap,
            vis_nir_edge=args.vis_nir_edge * wave_unit,
            scale_stat=args.scale_stat,
            debug=debug,
        )
        
        # write ASCII (nm, Flam)
        out_dat = outdir / f"{base_out}_unbinned.dat"
        outfile_base = outdir / base_out
        
        if w_comb_unbinned.size:
            w = w_comb_unbinned.to_value(wave_unit)
            f = f_comb_unbinned.to_value(flux_unit)
            err = err_comb_unbinned.to_value(flux_unit)
            m = np.isfinite(w) & np.isfinite(f) & np.isfinite(err)
            w, f, err = w[m], f[m], err[m]
            order = np.argsort(w)
            w, f, err = w[order], f[order], err[order]
            with open(out_dat, "w") as fh:
                fh.write("# wavelength(nm)  flux(erg/s/cm2/Ang) flux_error(erg/s/cm2/Ang)\n")
                for wi, fi, erri in zip(w, f, err):
                    fh.write(f"{wi:.8f} {fi:.11e} {erri:.16e}\n")
            if debug:
                print(f"[OK] wrote {out_dat}")
    print(outfile_base)
    # plot
    if etc:
        plot_signal_to_noise_combined(
        uvb=(uvb_etc[0], uvb_etc[3], uvb_etc[4]) if uvb else None,
        vis=(vis_etc[0], vis_etc[3], vis_etc[4]) if vis else None,
        nir=(nir_etc[0], nir_etc[3], nir_etc[4]) if nir else None,
        combined=(w_comb.to_value(wave_unit), f_comb.to_value(flux_unit), err_comb.to_value(flux_unit)) if w_comb.size else None,
        title=f"{obj} ({date_title}) - Signal to Noise Ratio - VLT/X-shooter" if date_title else f"{obj} - VLT/X-shooter",
        outfile_base=outfile_base,
        debug=debug,
        show = show
    )
    
    plot_xsh_arms_and_combined(
            uvb=(uvb[0], uvb[1], uvb[2]) if uvb else None,
            vis=(vis[0], vis[1]*s_vis, vis[2]*s_vis) if vis else None,
            nir=(nir[0], nir[1]*s_nir, nir[2]*s_nir) if nir else None,
            combined=(w_comb.to_value(wave_unit), f_comb.to_value(flux_unit), err_comb.to_value(flux_unit)) if w_comb.size else None,
            title=f"{obj} ({date_title}) - VLT/X-shooter" if date_title else f"{obj} - VLT/X-shooter",
            outfile_base=outfile_base,
            debug=debug,
            show = show,
            snr=snr,
            )
    print(f"✅ Wrote merged spectrum data: {out_dat}, plot: {outfile_base}.png")
    return 0


def cmd_process_epochs(args, debug=False) -> int:
    """
    CLI handler:
    1) process all epochs (stitch & save combined files),
    2) plot all epochs overlay,
    3) plot each epoch individually.
    """
    if getattr(args, "verbose", 0):
                debug = True
                print(f"\n\n--------DEBUG MODE ACTIVATED--------")
                print(f"Initiating processing of all SN epochs...")
            
            
    if args.show: 
        show = True
    else:
        show = False
    
    
    root = Path(args.root).expanduser()
    # If no outdir specified → use root
    if args.outdir is None:
        outdir = root
        if debug:
            print(f"[info] No --outdir given → using root: {outdir}")
    else:
        outdir = Path(args.outdir).expanduser()

    outdir.mkdir(parents=True, exist_ok=True)
    
    if not root.is_dir():
        print(f"‼️[ERROR]: root not found: {root}", file=sys.stderr)
        return 2
    
    # 1. Process epochs, creates combined spectra files for each epoch
    try:
        process_all_epochs(root=root, outdir=outdir, debug=debug)
        if debug: print("✅[OK]: process_all_epochs completed")
    except Exception as e:
        print(f"‼️[ERROR]: process_all_epochs failed: {e}", file=sys.stderr)
        return 2
    
    #2. Create figures per epoch
    if debug: 
        print(f"Beginning single epoch figure creation")
        print(f"{sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("epoch"))}")
    for epoch_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("epoch")):
        if debug: print(f"[DEBUG]: creating figures for {epoch_dir.name}")
        try:
            plot_epoch(epoch_dir = epoch_dir, outdir = outdir, debug=debug, show=show)
            if debug:
                name = epoch_dir.name
                print(f"✅[OK]: Successfully plotted and saved {name}")
        except Exception as e:            
            print(f"‼️[ERROR] per-epoch plot failed for {epoch_dir.name}: {e}")

    
    # 3. Multi-epoch overlay figure
    parent = root.name
    try:
        plot_all_epochs(
            root=root,
            outdir=outdir,
            debug=debug,
            show=show,
        )
        if debug:
            print(f"✅[OK]: Plotting all epochs together succeeded.")
    except Exception as e:
        print(f"‼️[ERROR]: plotting all-epochs failed: {e}", file=sys.stderr)
        return 2

    print(f"✅ Supernova processing completed, saved all outputs to {outdir}")
    return 0

def cmd_plot_lightcurve(args, debug=False) -> int:
    #from root given in path, extracts light curve and plots it:
    if getattr(args, "verbose", 0):
            debug = True
            print(f"\n\n--------DEBUG MODE ACTIVATED--------")
            print(f"Creating light curve...")
    root = Path(args.root).expanduser()
    # If no outdir specified → use root
    if args.outdir is None:
        outdir = root
        if debug:
            print(f"[info] No --outdir given → using root: {outdir}")
    else:
        outdir = Path(args.outdir).expanduser()

    outdir.mkdir(parents=True, exist_ok=True)
    
    if not root.is_dir():
        print(f"‼️[ERROR]: root not found: {root}", file=sys.stderr)
        return 2
    
    
    from single_sne.io.lightcurves import read_lightcurve
    lightcurve_df = read_lightcurve()


def cmd_xsh_tellurics(args, debug = False)-> int:
    from single_sne.io.xshooter import choose_vis_closest_to_nir
    if getattr(args, "verbose", 0):
            debug = True
            print(f"\n\n--------DEBUG MODE ACTIVATED--------")
            print(f"Creating tellurics...")
            
    if args.unbinned: 
        unbinned = True
    else:
        unbinned = False        
        
    if args.show: 
        show = True
    else:
        show = False   
    
    if args.snr:
        snr = True
    else:
        snr = False
        
    root = Path(args.root).expanduser()
    if args.outdir is None:
        outdir = root
        if debug:
            print(f"[info] No --outdir given → using root: {outdir}")
    else:
        outdir = Path(args.outdir).expanduser()

    outdir.mkdir(parents=True, exist_ok=True)
            
    sel = discover_merge1d_files(
        args.root,
        product_mode="TELL",
        prefer_end_products=True,
        allow_tmp=args.allow_tmp,
    )
    if not sel:
        print("ERROR: no MERGE1D files found")
        return 2
    from single_sne.io.xshooter import group_tellurics_by_star
    by_star = group_tellurics_by_star(sel)
    if debug: print(by_star)
    
    if debug:
        for star, arm_dict in by_star.items():
            print(f"[telluric groups] {star}: "
                f"{len(arm_dict.get('UVB', []))} UVB, "
                f"{len(arm_dict.get('VIS', []))} VIS, "
                f"{len(arm_dict.get('NIR', []))} NIR")
            
    for star, arms in by_star.items():
        by_star[star] = choose_vis_closest_to_nir(arms)
        
    if debug:
        for star, arm_dict in by_star.items():
            print(f"[telluric groups] {star} after selection: "
                f"{len(arm_dict.get('UVB', []))} UVB, "
                f"{len(arm_dict.get('VIS', []))} VIS, "
                f"{len(arm_dict.get('NIR', []))} NIR")
            
    # read present arms (nm + dimensionless → we assign your house flux unit)
    data = {}
    first_hdr = None
    for star in by_star:
        for arm in by_star[star]:
            for i in range(0, len(by_star[star][arm])):
                if debug: print(f"\nLength of arm: {len(by_star[star][arm])}");print(f"reading {by_star[star][arm][i]}")
                w_nm, f_native, err_native, hdr = read_primary_linear_wcs(by_star[star][arm][i], ext=0, debug = debug)
                print(f"Sanity check: wave: {min(w_nm)} to {max(w_nm)}, flux = {min(f_native)} to {max(f_native)}")
                if first_hdr is None:
                    first_hdr = hdr
                data[arm] = (w_nm, f_native, err_native)
                
        # metadata / output base
        obj, date_title, date_file = _get_object_date(first_hdr) if first_hdr else ("TARGET", "", "")

        base_out = args.outdir if args.outdir else (f"{star}_{date_file}_xshooter" if date_file else f"{star}_xshooter")
        # rebin per arm (nm grids)
        uvb = vis = nir = None
        wave_unit, flux_unit = INSTRUMENT_UNITS["XSHOOTER"]   # your chosen house unit

        uvb_unbinned, vis_unbinned, nir_unbinned = None, None, None
        
        if "UVB" in data:
            w, f, err  = data["UVB"]
            mask = np.isfinite(f) & np.isfinite(err) & (err != 0)
            w, f, err = w[mask], f[mask], err[mask]
            dw = args.dw_uvb
            wgrid, f_out, ferr_out = bin_spectrum(w, f, dw, wave_unit, flux_unit, flux_err= err)
            uvb = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit), ferr_out.to_value(flux_unit))
            if unbinned:   
                uvb_unbinned = (w, f, err)
            else: 
                uvb_unbinned = None

        if "VIS" in data:
            
            w, f, err  = data["VIS"]
            mask = np.isfinite(f) & np.isfinite(err) & (err != 0)
            w, f, err = w[mask], f[mask], err[mask]
            dw = args.dw_vis
            wgrid, f_out, ferr_out = bin_spectrum(w, f, dw, wave_unit, flux_unit, flux_err= err)
            vis = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit), ferr_out.to_value(flux_unit))
            if unbinned: vis_unbinned = (w, f, err)


        if "NIR" in data:
            w, f, err  = data["NIR"]
            mask = np.isfinite(f) & np.isfinite(err) & (err != 0)
            w, f, err = w[mask], f[mask], err[mask]
            dw = args.dw_nir
            wgrid, f_out, ferr_out = bin_spectrum(w, f, dw, wave_unit, flux_unit, flux_err= err)
            nir = (wgrid.to_value(wave_unit), f_out.to_value(flux_unit), ferr_out.to_value(flux_unit))
            if unbinned: nir_unbinned = (w, f, err)
        
        
        # merge
        uvb_vis_overlap = (args.uvb_vis_overlap[0] * u.nm,
                        args.uvb_vis_overlap[1] * u.nm)
        vis_nir_overlap = (args.vis_nir_overlap[0] * u.nm,
                        args.vis_nir_overlap[1] * u.nm)
        

        def comb_spectrum(uvb, vis, nir, uvb_vis_overlap, vis_nir_overlap, debug = debug):
            w_comb, f_comb, err_comb, s_vis, s_nir = merge_uvb_vis_nir(
                uvb=(uvb[0]*wave_unit, uvb[1]*flux_unit, uvb[2]*flux_unit ) if uvb else None,
                vis=(vis[0]*wave_unit, vis[1]*flux_unit, vis[2]*flux_unit) if vis else None,
                nir=(nir[0]*wave_unit, nir[1]*flux_unit, nir[2]*flux_unit) if nir else None,
                uvb_vis_overlap=uvb_vis_overlap,
                uvb_vis_edge=args.uvb_vis_edge * wave_unit,
                vis_nir_overlap=vis_nir_overlap,
                vis_nir_edge=args.vis_nir_edge * wave_unit,
                scale_stat=args.scale_stat,
                debug=debug,
            )
            return w_comb, f_comb, err_comb, s_vis, s_nir
        
        w_comb, f_comb, err_comb, s_vis, s_nir = comb_spectrum(
                uvb if uvb else None, 
                vis if vis else None, 
                nir if nir else None, 
                uvb_vis_overlap=uvb_vis_overlap,
                vis_nir_overlap=vis_nir_overlap,
                debug=debug,
            )
        
        
            

        # write ASCII (nm, Flam)
        out_dat = outdir / f"{base_out}.dat"
        outfile_base = outdir / base_out
        
        if debug: print(f"\n\n\n\n\n\n\n -----------------------\nOutdir:{outdir}\nOut_dat:", out_dat)
        
        if w_comb.size:
            w = w_comb.to_value(wave_unit)
            f = f_comb.to_value(flux_unit)
            err = err_comb.to_value(flux_unit)
            m = np.isfinite(w) & np.isfinite(f) & np.isfinite(err)
            w, f, err = w[m], f[m], err[m]
            order = np.argsort(w)
            w, f, err = w[order], f[order], err[order]
            with open(out_dat, "w") as fh:
                fh.write("# wavelength(nm)  flux(erg/s/cm2/Ang) flux_error(erg/s/cm2/Ang)\n")
                for wi, fi, erri in zip(w, f, err):
                    fh.write(f"{wi:.8f} {fi:.16e} {erri:.16e}\n")
            if debug:
                print(f"[OK] wrote {out_dat}")
                
        if unbinned:
            w_comb_unbinned, f_comb_unbinned, err_comb_unbinned, s_vis_unbinned, s_nir_unbinned = comb_spectrum(
                uvb_unbinned if uvb_unbinned else None, 
                vis_unbinned if vis_unbinned else None, 
                nir_unbinned if nir_unbinned else None, 
                uvb_vis_overlap=uvb_vis_overlap,
                vis_nir_overlap=vis_nir_overlap,
                debug=debug,
            )
            
            # write ASCII (nm, Flam)
            out_dat = outdir / f"{base_out}_unbinned.dat"
            outfile_base = outdir / base_out
            
            if w_comb_unbinned.size:
                w = w_comb_unbinned.to_value(wave_unit)
                f = f_comb_unbinned.to_value(flux_unit)
                err = err_comb_unbinned.to_value(flux_unit)
                m = np.isfinite(w) & np.isfinite(f) & np.isfinite(err)
                w, f, err = w[m], f[m], err[m]
                order = np.argsort(w)
                w, f, err = w[order], f[order], err[order]
                with open(out_dat, "w") as fh:
                    fh.write("# wavelength(nm)  flux(erg/s/cm2/Ang) flux_error(erg/s/cm2/Ang)\n")
                    for wi, fi, erri in zip(w, f, err):
                        fh.write(f"{wi:.8f} {fi:.11e} {erri:.16e}\n")
                if debug:
                    print(f"[OK] wrote {out_dat}")

        # plot
        if unbinned:
            plot_signal_to_noise_combined(
                uvb=(uvb_unbinned[0], uvb_unbinned[1]/uvb_unbinned[2]) if uvb else None,
                vis=(vis_unbinned[0], vis_unbinned[1]/vis_unbinned[2]) if vis else None,
                nir=(nir_unbinned[0], nir_unbinned[1]/nir_unbinned[2]) if nir else None,
                combined=(w_comb_unbinned.to_value(wave_unit), f_comb_unbinned.to_value(flux_unit)/err_comb_unbinned.to_value(flux_unit)) if w_comb_unbinned.size else None,
                title=f"{star} ({date_title}) - Signal to Noise Ratio - VLT/X-shooter" if date_title else f"{obj} - VLT/X-shooter",
                outfile_base=outfile_base,
                debug=debug,
                show = show
                )
        
        plot_xsh_arms_and_combined(
                uvb=(uvb[0], uvb[1], uvb[2]) if uvb else None,
                vis=(vis[0], vis[1]*s_vis, vis[2]*s_vis) if vis else None,
                nir=(nir[0], nir[1]*s_nir, nir[2]*s_nir) if nir else None,
                combined=(w_comb.to_value(wave_unit), f_comb.to_value(flux_unit), err_comb.to_value(flux_unit)) if w_comb.size else None,
                title=f"{star} ({date_title}) - VLT/X-shooter" if date_title else f"{obj} - VLT/X-shooter",
                outfile_base=outfile_base,
                debug=debug,
                show = show,
                snr=snr,
                )
        print(f"✅ Wrote merged spectrum data: {out_dat}, plot: {outfile_base}.png")

    return 0

