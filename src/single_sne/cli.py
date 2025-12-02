
# src/single_sne/cli.py
# Command-line interface for the project.
# Run as:  python -m single_sne.cli <subcommand> [options]
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="single_sne", description="Single SNe utilities (demo CLI)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv)")
    sub = p.add_subparsers(dest="command", required=True)

    # convert-fnu
    from .cli_handlers import cmd_convert_fnu
    pc = sub.add_parser("convert-fnu", help="Convert fλ (erg/s/cm²/Å) to fν (mJy)")
    pc.add_argument("--infile", required=True, help="Input 2-column ASCII file")
    pc.add_argument("--outfile", required=True, help="Output file to write")
    pc.set_defaults(func=cmd_convert_fnu)

    
    # bin_spectrum
    from .cli_handlers import cmd_bin_spectrum
    pb = sub.add_parser("bin-spectrum", help="Flux-conserving binning of a 2–3 col ASCII spectrum (wavelength, flux[, flux_err]).")

    pb.add_argument("--infile", nargs="?", default="-", help="Input 2–3 col ASCII (λ, flux[, σ]). Use '-' for stdin. Lines starting with '#' are ignored.")
    pb.add_argument("--outfile", nargs="?", default="-", help="Output 2–3 col ASCII. Use '-' for stdout.")

    pb.add_argument("--bin-size", type=float, required=True,help="Bin width in wave_unit (must be > 0).")
    pb.add_argument("--wave-unit", default="um",help='Wavelength unit, e.g. "um", "nm", "Angstrom".')
    pb.add_argument("--flux-unit", default="mJy",help='Flux unit, e.g. "mJy", "Jy", "erg/(s cm2 Angstrom)".')
    pb.add_argument("--out-wave-unit", default="um",help='Output wavelength unit, e.g. "um", "nm", "Angstrom".')
    pb.add_argument("--out-flux-unit", default="mJy",help='Output flux unit, e.g. "mJy", "Jy", "erg/(s cm2 Angstrom)".')
    pb.add_argument("--overwrite", action="store_true",help="Allow writing to an existing outfile.")
    pb.add_argument("--skip-header", type=int, default=0,help="Number of header lines to skip before data (default: 0).")

    pb.set_defaults(func=cmd_bin_spectrum)

    #read_jwst
    from .cli_handlers import cmd_read_jwst
    pj = sub.add_parser("read-jwst", help="Load JWST ASCII and echo as 2/3-col text")
    
    pj.add_argument("--infile", required=True)
    pj.add_argument("--outfile", required=True)
    pj.add_argument("--as-quantity", action="store_true")
    
    pj.set_defaults(func=cmd_read_jwst)
    
    # --- units subcommand
    from .cli_handlers import cmd_units
    sp = sub.add_parser("units", help="Show default wavelength/flux units for an instrument")
    sp.add_argument("-i", "--instrument", required=True,help="Instrument (e.g. JWST, XSHOOTER, FLAMINGOS-2)")
    sp.add_argument("--output", choices=["text", "latex"], default="text",help="Output format for units")
    sp.add_argument("--list-on-error", action="store_true",help="List supported instruments if the name is unknown")
    sp.set_defaults(func=cmd_units)

    # --- list-instruments subcommand
    from .cli_handlers import cmd_list_instruments
    sp2 = sub.add_parser("list-instruments", help="List supported instrument keys")
    sp2.set_defaults(func=cmd_list_instruments)

    #--- xsh-merge-plot
    from .cli_handlers import cmd_xsh_merge_plot
    sx = sub.add_parser("xsh-merge-plot", help="Discover, rebin, stitch, and plot X-shooter MERGE1D arms")
    
    sx.add_argument("--root", default=".", help="Search root directory")
    sx.add_argument("--outdir",default= None, help="Directory to write output files (default: current directory).")
    sx.add_argument("--product-mode", choices=["SCI", "TELL", "ANY"], default="SCI", help="Which MERGE1D products to use")
    sx.add_argument("--no-prefer-end", action="store_true", help="Do NOT prefer files under reflex_end_products")
    sx.add_argument("--allow-tmp", action="store_true", help="Include reflex_tmp_products/response directories")
    sx.add_argument("--clean", action="store_true", help="If set, pixels flagged by the QUAL mask in XSHOOTER FITS files will be filtered out.")
    sx.add_argument("--show", action="store_true", help="If set, plots will be shown as pop-up window.")
    sx.add_argument("--snr", action="store_true", help="If set, plots will have a subfigure showing the SNR")
    sx.add_argument("--unbinned", action="store_true", help="If set, will also save unbinned version of the spectrum")
    sx.add_argument("--etc", action="store_true", help="If set, will also calculate the SNR per spectral bin as provided by the ETC")

    # bin widths (nm)
    sx.add_argument("--dw-uvb", type=float, default=0.3, help="UVB bin width [nm]")
    sx.add_argument("--dw-vis", type=float, default=0.3, help="VIS bin width [nm]")
    sx.add_argument("--dw-nir", type=float, default=1.0, help="NIR bin width [nm]")

    # stitch windows/edges (nm)
    sx.add_argument("--uvb-vis-overlap", nargs=2, type=float, default=(550.0, 555.0), metavar=("LO", "HI"), help="Overlap window UVB↔VIS [nm]")
    sx.add_argument("--uvb-vis-edge", type=float, default=555.0, help="Handoff edge UVB→VIS [nm]")
    sx.add_argument("--vis-nir-overlap", nargs=2, type=float, default=(1010.0, 1020.0), metavar=("LO", "HI"), help="Overlap window VIS↔NIR [nm]")
    sx.add_argument("--vis-nir-edge", type=float, default=1019.0, help="Handoff edge VIS→NIR [nm]")
    sx.add_argument("--scale-stat", choices=["median", "mean"], default="median", help="Statistic for overlap scale")
    sx.add_argument("--out", default="", help="Output basename (default: from header)")
    # if you support overwriting the .dat/.pdf/.png, expose it:
    sx.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    
    
    #-----xsh-tell processing tellurics:
    from .cli_handlers import cmd_xsh_tellurics
    sx_tell = sub.add_parser(
    "xsh-tellurics",
    help="Merge and plot X-shooter MERGE1D telluric standards per object",
    )
    sx_tell.add_argument("--root", default=".", help="Search root directory")
    sx_tell.add_argument("--outdir", default=None, help="Output directory (default: root)")
    sx_tell.add_argument("--no-prefer-end", action="store_true",
                        help="Do NOT prefer files under reflex_end_products")
    sx_tell.add_argument("--allow-tmp", action="store_true",
                        help="Include reflex_tmp_products/response directories")
    sx_tell.add_argument("--show", action="store_true", help="If set, plots will be shown as pop-up window.")
    sx_tell.add_argument("--snr", action="store_true", help="If set, plots will have a subfigure showing the SNR")

    sx_tell.add_argument("--unbinned", action="store_true", help="If set, will also save unbinned version of the spectrum")
    
    
    # bin widths (nm)
    sx_tell.add_argument("--dw-uvb", type=float, default=0.3, help="UVB bin width [nm]")
    sx_tell.add_argument("--dw-vis", type=float, default=0.3, help="VIS bin width [nm]")
    sx_tell.add_argument("--dw-nir", type=float, default=1.0, help="NIR bin width [nm]")

    sx_tell.add_argument("--uvb-vis-overlap", nargs=2, type=float, default=(550.0, 555.0), metavar=("LO", "HI"), help="Overlap window UVB↔VIS [nm]")
    sx_tell.add_argument("--vis-nir-overlap", nargs=2, type=float,
                        default=(1010.0, 1020.0), metavar=("LO", "HI"))
    sx_tell.add_argument("--vis-nir-edge", type=float, default=1019.0)
    sx_tell.add_argument("--uvb-vis-edge", type=float, default=555.0, help="Handoff edge UVB→VIS [nm]")
    sx_tell.add_argument("--scale-stat", choices=["median", "mean"],
                        default="median")

    sx_tell.set_defaults(func=cmd_xsh_tellurics)
    
    # --- process-epochs
    from .cli_handlers import cmd_process_epochs
    s_ep = sub.add_parser(
        "process-epochs",
        help="Process all epochs under ROOT (stitch per-epoch), then plot all epochs together and individually."
    )
    s_ep.add_argument("--root", required=True, help="Parent directory containing Epoch* subfolders")
    s_ep.add_argument("--outdir", default = None, required=False, help="Directory to write outputs")
    s_ep.add_argument("--show", action="store_true", help="If set, plots will be shown as pop-up window.")

    s_ep.set_defaults(func=cmd_process_epochs)


    # wire handler
    sx.set_defaults(func=cmd_xsh_merge_plot)
    
    
    return p

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    #set log level from -v occurrences:
    level = logging.WARNING if args.verbose == 0 else logging.INFO if args.verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    from astropy import units as u
    if hasattr(args, "wave_unit"):
        try:
            u.Unit(args.wave_unit); u.Unit(args.flux_unit) if hasattr(args, "wave_unit") else None
            u.Unit(args.out_wave_unit); u.Unit(args.out_flux_unit) if hasattr(args, "out_flux_unit") else None
        except Exception as e:
            print(f"ERROR: bad unit: {e}", file=sys.stderr)
            return 2
        
        if args.bin_size <= 0:
            print("ERROR: --bin-size must be > 0", file=sys.stderr)
            return 2

    return args.func(args, debug=(args.verbose > 0))


if __name__ == "__main__":
    raise SystemExit(main())
