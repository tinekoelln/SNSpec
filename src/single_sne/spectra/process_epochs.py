from __future__ import annotations
from pathlib import Path
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.units import Quantity
import pandas as pd
from astropy.table import Table
from astropy.constants import c
# ---- import own helpers ----
from single_sne.spectra.stitching import stitch_arms, combine_salt_dir, combine_flamingos_dir
from single_sne.io.jwst import read_jwst_arrays # your existing JWST reader
from single_sne.io.xshooter import read_xshooter_dat
from single_sne.units import INSTRUMENT_UNITS
from single_sne.spectra.spectra import to_fnu_mjy, is_strictly_increasing, bin_spectrum
from single_sne.units import nm_to_angstrom, angstrom_to_micron, INSTRUMENT_UNITS
from single_sne.spectra.stitching import _scale_in_overlap




from pathlib import Path
from typing import Iterable


def _iter_files(d: Path, patterns: Iterable[str] | str = ("*.dat", "*.fits")):
    # Coerce a single string into a tuple of one pattern
    if isinstance(patterns, (str, Path)):
        patterns = (str(patterns),)
    for pat in patterns:
        for p in d.glob(pat):
            if p.is_file():
                yield p

def parts_to_dataframe(parts_subset, df = None, debug = False):
    """
    Append one (instrument, wavelength, flux[, flux_error]) tuple to a pandas DataFrame.

    parts_subset is expected to be either:
        (tag, w, f)                       → no flux error
        (tag, w, f, ferr)                 → with flux error
    All wavelength/flux arrays must have the same length.

    Returns a new DataFrame with columns:
        instrument, wavelength, flux, flux_error
    Missing flux_error values are filled with NaN.
    """
    if debug: 
        print(f"[parts_to_dataframe] Shape of existing dataframe: {df.shape}")
    
    tag = parts_subset[0]
    w   = parts_subset[1]
    f   = parts_subset[2]
    ferr = parts_subset[3] if len(parts_subset) > 3 else None
    n = len(w)
    if debug: print(len(w), len(f), ferr if ferr is not None else np.full(n, np.nan))

    df_new = pd.DataFrame({
        "instrument": np.full(n, tag, dtype=object),
        "wavelength": w,
        "flux": f,
        "flux_error": ferr if ferr is not None else np.full(n, np.nan),
    })

    if df is None:
        print(f"[parts_to_dataframe]: dataframe doesn't yet exist")
        return df_new.reset_index(drop=True)

    df_out = pd.concat([df, df_new], ignore_index=True)
    return df_out.reset_index(drop=True)

def process_epoch(
    epoch_dir: str | Path,
    outdir: str | Path,
    *,
    debug=False,
    show = False,
):
    """Walk epoch_dir, build stitched spectrum (converting to JWST units if present, else SALT/FLAMINGOS units), saves converted stitched spectrum to outdir."""
    epoch_dir = Path(epoch_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    parts = []
    jw = epoch_dir / "JWST"
    if jw.is_dir():
        jwst_units = True
        w_unit = INSTRUMENT_UNITS["JWST"][0]
        f_unit = INSTRUMENT_UNITS["JWST"][1]
    else:
        jwst_units = False
        w_unit = INSTRUMENT_UNITS["SALT"][0]
        f_unit = INSTRUMENT_UNITS["SALT"][1]
        
    parts_dataframe = pd.DataFrame({
        "instrument": pd.Series(dtype="object"),
        "wavelength": pd.Series(dtype="float64"),
        "flux":       pd.Series(dtype="float64"),
        "flux_error": pd.Series(dtype="float64"),  # always present (NaN when unknown)
        })
    
    telescopes_listed = []
    # 1) XSHOOTER
    xsh = epoch_dir / "XSHOOTER"
    if xsh.is_dir():
        telescopes_listed.append("XSHOOTER")
        for datafile in _iter_files(xsh, patterns =( "*.dat")):
            if debug: print(f"[DEBUG]: XSHOOTER files: {list(_iter_files(xsh, patterns =( "*.dat")))}")
            try:
                if debug: print(f"[DEBUG]: reading in datafile {datafile}")
                w_xsh, f_xsh, _ = read_xshooter_dat(datafile, debug=debug)  # Quantities
                if debug: print(f"  + XSHOOTER: {w_xsh.min():.2f}–{w_xsh.max():.2f}")
                w_xsh_ang = nm_to_angstrom(w_xsh)
                if debug: print(f"[DEBUG]: XSHOOTER wavelength converting to angstrom")
                
                #convert to target units:
                if jwst_units:
                    if debug: 
                        print(f"Begin conversion to JWST units:")
                    f_xsh_mjy, _ = to_fnu_mjy(w_xsh_ang, f_xsh, debug=debug)
                    w_xsh_micron = angstrom_to_micron(w_xsh_ang, debug=debug)
            
                    if debug: print(f"Converted XSHOOTER wavelength units to",w_xsh_micron.unit);print(f"Range of xshooter data: {min(w_xsh_micron)} to {max(w_xsh_micron)}")
                    parts.append(("XSHOOTER", w_xsh_micron, f_xsh_mjy))
                    parts_xsh = parts[-1]
                    parts_dataframe = parts_to_dataframe(parts_xsh, parts_dataframe)
                    if debug: print(parts_dataframe)
                    if debug: print(f"    converted XSHOOTER data to JWST units: {w_xsh_micron.unit}, {f_xsh_mjy.unit}");print(parts_dataframe["instrument"].unique())
                    
                else:
                    parts.append(("XSHOOTER", w_xsh_ang, f_xsh))
                    parts_xsh = parts[-1]
                    parts_dataframe = parts_to_dataframe(parts_dataframe, parts_xsh)
                    if debug: print(parts_dataframe["instrument"].unique())
            except Exception as e:
                if debug: print(f"  ! XSHOOTER failed: {e}")

    # 2) JWST
    jw = epoch_dir / "JWST"
    if jw.is_dir():
        telescopes_listed.append("JWST")
        if debug: print(f"Beginning JWST Handling")
        for datafile in _iter_files(jw, patterns = ("*.dat",)):
            if debug: print(f"[DEBUG]: JWST files: {list(_iter_files(jw, patterns = ("*.dat",)))}")
            try:
                if debug: print(f"[DEBUG] Getting JWST wavelengths and fluxes...")
                w_jw, f_jw, _ = read_jwst_arrays(datafile, debug=debug)       # Quantities
                parts.append(("JWST", w_jw, f_jw))
                parts_jwst = parts[-1]
                parts_dataframe = parts_to_dataframe(parts_jwst, parts_dataframe)
                if debug: print(f"  + JWST: {w_jw.min():.3f}–{w_jw.max():.3f}");print(parts_dataframe["instrument"].unique())
            except Exception as e:
                if debug: print(f"  ! JWST failed: {e}")

    # 3) FLAMINGOS
    fl = epoch_dir / "FLAMINGOS"
    if fl.is_dir():
        telescopes_listed.append("FLAMINGOS")
        
        try:
            tag, w_flam, f_flam, fe_flam = combine_flamingos_dir(fl, jwst_units=jwst_units, debug=debug)
            
            w_flam_binned, f_flam_binned, fe_flam_binned = bin_spectrum(w_flam, f_flam, bin_size = 10,  wave_unit = 'AA', flux_unit = INSTRUMENT_UNITS["FLAMINGOS2"][1], flux_err= fe_flam)
            tag_name = "FLAMINGOS"                   # or "JWST", "SALT", ...
            n   = len(w_flam_binned)                   # wave_b is your binned wavelength array
            tag = np.full(n, tag_name, dtype=object)
            # store error if you want; your `parts` list can ignore it if unused
        except Exception as e:
            print(f"  ! FLAMINGOS combine failed: {e}")
            
        #convert to target units:
        if jwst_units:
            if fe_flam_binned is not None:
                f_f_mjy, fe_f_mjy = to_fnu_mjy(w_flam_binned, f_flam_binned, fe_flam_binned, debug = debug)
                w_f_micron = angstrom_to_micron(w_flam_binned)
                '''if debug: 
                    mAB = -2.5 * np.log10((f_f_mjy.to(u.Jy)).value) + 8.90
                    print(f"Measured magnitude:", mAB)          
                    plt.plot(w_flam.to(u.um), (f_flam*w_flam**2/c.to(u.AA/u.s)).to(u.mJy), label = 'Unbinned')
                    plt.plot(w_flam_binned.to(u.um), (f_flam_binned*w_flam_binned**2/c.to(u.AA/u.s)).to(u.mJy), label = 'Binned')
                    plt.plot(w_f_micron, f_f_mjy, label = 'Converted')
                    plt.legend()
                    plt.show()'''
                    
                
                parts.append((tag, w_f_micron, f_f_mjy, fe_f_mjy))
                parts_flamingos = parts[-1]
                parts_dataframe = parts_to_dataframe(parts_flamingos, parts_dataframe)
            else:
                f_f_mjy, _ = to_fnu_mjy(w_flam_binned, f_flam_binned)
                w_f_micron = angstrom_to_micron(w_flam_binned)
                parts.append(tag, w_f_micron, f_f_mjy)
                parts_flamingos = parts[-1]
                parts_dataframe = parts_to_dataframe(parts_flamingos, parts_dataframe)
                
            if debug:  
                print(f"    converted FLAMINGOS data to JWST units: {w_f_micron.unit}, {f_f_mjy.unit}")
            
        else:
            parts.append(tag, w_flam_binned, f_flam_binned,fe_flam_binned)
            parts_flamingos = parts[-1]
            parts_dataframe = parts_to_dataframe(parts_flamingos, parts_dataframe)

    # 4) SALT
    sa = epoch_dir / "SALT"
    if sa.is_dir():
        telescopes_listed.append("SALT")
        try:
            tag, w_s, f_s = combine_salt_dir(sa, jwst_units=jwst_units, debug=debug)
            bin_spectrum(w_s, f_s, bin_size = 10, wave_unit = INSTRUMENT_UNITS["FLAMINGOS2"][0], flux_unit = INSTRUMENT_UNITS["FLAMINGOS2"][1])
            if debug: print(f"\n[SALT] Read in SALT data successfully.")
        except Exception as e:
            print(f"  ! SALT combine failed: {e}")
        #scale SALT elements to one another first if multiple present:
        
        #convert to target units:
        if jwst_units:
            f_s_mjy, _ = to_fnu_mjy(w_s, f_s)
            w_s_micron = angstrom_to_micron(w_s)
            parts.append((tag, w_s_micron, f_s_mjy))
            if debug: print(f"[SALT] Converted SALT data to JWST units, appended to parts")
            parts_salt = parts[-1]
            parts_dataframe = parts_to_dataframe(parts_salt, parts_dataframe, debug=debug)
            print(parts_dataframe.shape)
            if debug: print(f"    converted SALT data to JWST units: {w_s_micron.unit}, {f_s_mjy.unit}")
        else:
            parts.append(tag, w_s, f_s)
            parts_salt = parts[-1]
            parts_dataframe = parts_to_dataframe(parts_salt, parts_dataframe)
        

    #5. sort all data by wavelength
    if debug: print(f"Dataframe acquisition finished. Instruments:{parts_dataframe.instrument.unique()}");
    parts_dataframe = parts_dataframe.sort_values(by="wavelength").reset_index(drop=True)
    
    #6. determine if flux scaling is needed:
    groups = {inst: subdf for inst, subdf in parts_dataframe.groupby("instrument")}

    
    #Save unscaled dataframe:
    parent = epoch_dir.parent.name   # e.g. "SN2025cy"
    epoch  = epoch_dir.name    
    
    inst_order = []
    for p in epoch_dir.iterdir():
        if p.is_dir():
            inst_order.append(p.name.upper())
    inst_label = "+".join(inst_order)
    combined_spectrum_unscaled = outdir / f"{parent}_{epoch}_unscaled_{inst_label}.dat"
    with open(combined_spectrum_unscaled, "w") as fh:
        fh.write(f"# Combined spectrum for {parent} {epoch}\n")
        fh.write(f"# Instruments: {inst_label}\n")
        fh.write(f"# Columns: #instrument #wavelength[{w_unit}] #flux[{f_unit}] #flux_err[{f_unit}] (if present)\n")
        parts_dataframe.to_csv(fh, index=False, sep=" ", float_format="%.6e",header=False, lineterminator="\n")
    print(f"💾 Unscaled combined spectrum saved to {combined_spectrum_unscaled.name}")
    #loop over groups, determining if scaling is needed, no scaling if no overlap
    # 1) Order instruments by their min wavelength
    ordered = sorted(groups.items(), key=lambda kv: kv[1]["wavelength"].min())
    rev = ordered[::-1]

    # 2) Iterate adjacent pairs and check overlap
    # Start from rightmost as reference
    (inst_ref, df_ref) = rev[0]
    new_groups = {inst_ref: df_ref.copy()}

    # start with the rightmost as reference
    inst_ref, df_ref = rev[0]
    df_ref = df_ref.copy().sort_values("wavelength").reset_index(drop=True)
    new_groups[inst_ref] = df_ref

    for i in range(1, len(rev)):
        inst_left, df_left = rev[i]
        if debug:
            print(f"[process_epochs] Scaling {inst_left} → reference {inst_ref}")

        df_left = df_left.copy().sort_values("wavelength").reset_index(drop=True)
        df_left_scaled = df_left.copy()
        s = 1.0

        a_min, a_max = df_ref["wavelength"].min(), df_ref["wavelength"].max()
        b_min, b_max = df_left["wavelength"].min(), df_left["wavelength"].max()
        lo, hi = max(a_min, b_min), min(a_max, b_max)

        #Sanity check: Are lengths equal?
        if debug:
            print(f"Wavelength: {len(df_left['wavelength'])}")
            print(f"Flux: {len(df_left['flux'])}")
            if 'flux_error' in df_left.columns:
                print(f"Flux_err: {len(df_left['flux_error'])}")
            else:
                print("Flux_err: (none)")
        if hi > lo:
            # compute scale: LEFT → RIGHT(reference) in [lo, hi]
            s, _ = _scale_in_overlap(
                df_left["wavelength"].to_numpy(dtype=float),
                df_left["flux"].to_numpy(dtype=float),
                df_ref["wavelength"].to_numpy(dtype=float),
                df_ref["flux"].to_numpy(dtype=float),
                ov_lo=float(lo), ov_hi=float(hi),
                stat="median",
                scale_to_right=True,  # scale LEFT to match RIGHT
                debug=debug,
            )
            df_left_scaled["flux"] = df_left_scaled["flux"] * s
            if "flux_error" in df_left_scaled.columns:
                df_left_scaled["flux_error"] = df_left_scaled["flux_error"] * abs(s)

            if debug:
                print(f"[scaled] {inst_left} → {inst_ref} by ×{s:.4g} in [{lo:.3g}, {hi:.3g}]")

            # stitch by concatenation (you can also call your stitch_arms if desired)
            df_ref = (
                pd.concat([df_left_scaled, df_ref], ignore_index=True)
                .sort_values("wavelength")
                .reset_index(drop=True)
            )
        else:
            if debug:
                print(f"[no overlap] {inst_left} → {inst_ref}; appended without scaling")


            # Insert a NaN "gap" row so plots (plot/step) don't draw a connecting line
            if len(df_left) >= 2:
                step = df_left["wavelength"].iloc[-1] - df_left["wavelength"].iloc[-2]
                gap_wave = df_left["wavelength"].iloc[-1] + step
            else:
                gap_wave = np.nan
                
            gap_cols = {c: np.nan for c in df_ref.columns}
            gap_cols["instrument"] = np.nan
            gap_cols["wavelength"] = gap_wave
            gap_row = pd.DataFrame([gap_cols])
            if debug: print(gap_row)
            df_ref = (
                pd.concat([df_left_scaled, gap_row, df_ref], ignore_index=True)
                .sort_values("wavelength")
                .reset_index(drop=True)
            )

        new_groups[inst_left] = df_ref
        # update reference label to the newly combined block
        inst_ref = f"{inst_left}+{inst_ref}"

    # final combined dataframe (keep original instrument labels; order by λ, then instrument)
    df_scaled_all = (
        pd.concat(new_groups.values(), ignore_index=True)
        .sort_values(["wavelength", "instrument"])
        .reset_index(drop=True)
    )
    
    # final filename
    combined_spectrum = outdir / f"{parent}_{epoch}_combined_{inst_label}.dat"
    with open(combined_spectrum, "w") as fh:
        fh.write(f"# Combined spectrum for {parent} {epoch}\n")
        fh.write(f"# Instruments: {inst_label}\n")
        fh.write(f"# Columns: #instrument #wavelength[{w_unit}] #flux[{f_unit}] #flux_err[{f_unit}] (if present)\n")
        df_scaled_all.to_csv(fh, index=False, sep=" ", float_format="%.6e", header=False, na_rep = "NaN", lineterminator="\n")
    print(f"💾 Combined spectrum saved to {combined_spectrum.name}")
    
    
    print(f"✅Saved combined spectrum to {combined_spectrum}")
        
    #Save copy in epoch dir too:
    if outdir != epoch_dir:
        combined_spectrum_copy = epoch_dir / combined_spectrum.name
        with open(combined_spectrum_copy, "w") as fh:
            fh.write(f"# Combined spectrum for {parent} {epoch}\n")
            fh.write(f"# Instruments: {inst_label}\n")
            fh.write(f"# Columns: #instrument #wavelength[{w_unit}] #flux[{f_unit}] #flux_err[{f_unit}] (if present)\n")
            df_scaled_all.to_csv(fh, index=False, sep=" ", float_format="%.6e", header=False, lineterminator="\n")
        print(f"💾 Saved combined spectrum copy to  {combined_spectrum_copy.name}")
    
    if not parts:
        if debug: print(f"  (no data detected)")
        
    return df_scaled_all.wavelength, df_scaled_all.flux, df_scaled_all.flux_error
            
        

def process_all_epochs(root: str | Path, outdir: str | Path, *, debug=False):
    """
    Walk `root` (with Epoch1/, Epoch2/, ...) and build stitched spectra per epoch.
    Saves <epoch>_combined.dat, .png, .pdf into `outdir/<epoch>/`.
    Returns a dict: { "Epoch1": (wave, flux), ... } with astropy Quantities.
    """
    root = Path(root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    # epoch directories: Epoch1, Epoch2, Epoch3...
    for epoch_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("epoch")):
        ep = epoch_dir.name
        if debug: print(f"[Processing] {ep}")
        w, f, f_err = process_epoch(epoch_dir, outdir / ep, debug=debug)
        if debug: print(f"✅ Processed {ep}, saved to {outdir / ep}")
        results[ep] = (w, f, f_err)
    
    return results

    
