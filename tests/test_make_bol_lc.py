from pathlib import Path
import numpy as np
from datetime import datetime, UTC
from single_sne.pseudobol_lightcurve.aux import rd_lcbol_data, read_pbinfo, al_av, sort_filters_by_lambda, build_time_grid, build_time_grid_all, distmod_from_z_with_pecvel
from single_sne.pseudobol_lightcurve.make_bol_lc import mklcbol, _interp_mag_any, _interp_mag_linear_with_error
from single_sne.pseudobol_lightcurve.mklcbolinput import read_blackgem_csv, read_uvot_ab_txt
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])

# Base directory for test data (relative to THIS file)
DATA_DIR = Path(__file__).parent / "tests_data"


# assuming these are already defined as we discussed:
# - LightCurveHeader
# - FilterLightCurve
# - rd_lcbol_data

def test_rd_lcbol_data():
    hdr, lcdata = rd_lcbol_data(DATA_DIR / "sn2002bo_lcbolinput.dat")

    print("Header:")
    print(hdr)
    print()

    print(f"Number of filters: {len(lcdata)} (hdr.nfilt = {hdr.nfilt})")
    for lc in lcdata:
        print(f"{lc.filt:10s}  N_meas = {len(lc.time)}")



def test_pbinfo_and_extinction():
    pbinfo_path = DATA_DIR / "pbinfo.dat"  # adjust to your repo layout
    filt_meta = read_pbinfo(pbinfo_path)
    hdr, lcdata = rd_lcbol_data(DATA_DIR / "sn2002bo_lcbolinput_copy.dat")

    print("\nHeader:")
    print(hdr)

    #print("\nFilters in SN file:")
    #for lc in lcdata:
    #    print(f"  {lc.filt}  ({len(lc.time)} measurements)")

    # ---- 2) Load passband info ----
    pbinfo_file = Path(pbinfo_path)     # <-- adjust path
    pbinfo = read_pbinfo(pbinfo_file)

    #print("\nPassband info for these filters:")
    for lc in lcdata:
        fname = lc.filt
        assert fname in pbinfo, f"Filter {fname} missing in pbinfo.dat"

        pb = pbinfo[fname]

        print(
            f"{fname:10s}  λ_eff={pb.lambda_eff:8.1f} Å  "
            f"EW={pb.ew:7.1f} Å  zpt={pb.zpt:6.3f}"
        )

        # ---- 3) quick extinction sanity check ----
        al_host, al_host_err = al_av(
            pb.lambda_eff,
            r_v=hdr.rvhost,
            rverr=hdr.rvhosterr,
        )

        al_mw, al_mw_err = al_av(
            pb.lambda_eff,
            r_v=hdr.rvmw,
            rverr=hdr.rvmwerr,
        )

        #print(
        #    f"    Aλ/AV(host) = {al_host:5.3f} ± {al_host_err:5.3f}   "
        #    f"Aλ/AV(MW) = {al_mw:5.3f} ± {al_mw_err:5.3f}"
        #)
        
def test_sorted_filters_sn2002bo():
    infile = DATA_DIR / "sn2002bo_lcbolinput.dat"    
    pbinfo_file = DATA_DIR / "pbinfo.dat"

    hdr, lcdata = rd_lcbol_data(infile)
    pbinfo = read_pbinfo(pbinfo_file)

    lc_sorted, pb_sorted = sort_filters_by_lambda(lcdata, pbinfo)

    #print("\nFilters sorted by λ_eff:")
    #for lc, pb in zip(lc_sorted, pb_sorted):
    #    print(f"{lc.filt:10s}  λ_eff = {pb.lambda_eff:8.1f} Å")
        
        
    
    
def test_mklcbol_sn2002bo():
    """
    Run mklcbol on the sn2002bo test file and produce a plot of L_bol vs t.

    This is both a functional test (file is created, values finite) and a
    visual check (saved PNG) that you can open manually.
    """

    # ------------------------------------------------------------------
    # 1. Input files: adjust these to your repo layout
    # ------------------------------------------------------------------
    # Ideally: put the .dat files under tests/data/ and use relative paths.
    # For now I'm using the absolute paths you've been using:
    infile = Path(DATA_DIR / "sn2002bo_lcbolinput_copy.dat")
    pbinfo = Path(DATA_DIR / "pbinfo.dat")

    assert infile.exists(), f"Input file not found: {infile}"
    assert pbinfo.exists(), f"pbinfo file not found: {pbinfo}"
    tmp_path = Path(DATA_DIR)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Output bolometric LC file will be written into a temporary directory
    fout = tmp_path / "sn2002bo_lcbol_gauss.dat"

    # ------------------------------------------------------------------
    # 2. Run mklcbol
    # ------------------------------------------------------------------
    outpath = mklcbol(
        infile=infile,
        pbinfo=pbinfo,
        interpmeth="g",   # or "g" if you want to test GP interpolation
        batch=True,
        fout=fout,
    )

    assert outpath.exists(), "mklcbol did not create the output file"

    # ------------------------------------------------------------------
    # 3. Read bolometric LC: time, L_bol, L_bol_err
    # ------------------------------------------------------------------
    # The file has comment lines starting with '#', then:
    # time[d]    lbol[erg/s]      lbolerr[erg/s]
    data = np.loadtxt(outpath, comments="#")
    t, Lbol, Lbol_err = data.T

    # simple sanity checks
    assert np.all(np.isfinite(Lbol)), "Non-finite L_bol values"
    assert np.all(Lbol > 0), "L_bol should be positive"
    assert Lbol.max() > Lbol.min(), "L_bol should vary over time"
    data_idl = DATA_DIR / "sn2002bo_lcbol_UBVRI.dat"
    t_idl, Lbol_idl, Lbolerr_idl = np.loadtxt(
    data_idl,
    comments="#",
    unpack=True,      # <- THIS is the key
    usecols=(0, 1, 2) # optional, but makes it explicit
    )
    print(f"Length of IDL time array: {len(t_idl)}")
    # ------------------------------------------------------------------
    # 4. Plot L_bol(t) with error bars and save PNG
    # -----------------------------------------------------------------
    '''fig, ax = plt.subplots(
    3, 1,
    figsize=(7, 7),
    gridspec_kw={'height_ratios': [2, 1, 1]}
    )'''
    fig, ax = plt.subplots(1,1)
    ax.errorbar(t, Lbol, yerr=Lbol_err, fmt=".", capsize=0, label = "Python Output")
    ax.set_xlabel("Time [d]")
    ax.set_yscale('log')
    ax.set_ylim((1e42, 0.15e44))
    ax.set_ylabel(r"$L_{\rm bol}\ [\mathrm{erg\,s^{-1}}]$")
    ax.set_title("sn2002bo pseudo-bolometric light curve")
    ax.errorbar(t_idl, Lbol_idl, yerr = Lbolerr_idl, fmt=".", capsize=0, label = "IDL Output")
    ax.legend()
    '''ax[1].scatter(t, Lbol/Lbol_idl, label = r"$L_{py}/L_{IDL}$")
    ax[1].legend()
    ax[2].scatter(t, Lbol_err/Lbolerr_idl, label = r"$\sigma(L)_{py}/ \sigma(L)_{IDL}$")
    ax[2].legend()'''

    fig.tight_layout()
    png_path = tmp_path / "sn2002bo_lcbol_comp.png"
    fig.savefig(png_path, dpi = 500)
    
    

    print(f"\nBolometric LC written to: {outpath}")
    print(f"Plot saved to:           {png_path}")
    
    
def test_read_blackgem_csv():
    bg = read_blackgem_csv(DATA_DIR / "SN2025cy_BG_data.csv")
    print(bg.head())
    print("BG filters mapped to pbinfo:", bg["pb_name"].unique())
    
    uv = read_uvot_ab_txt(DATA_DIR / "UVOT_AB_mags.txt")
    print(uv.head())
    print("UVOT filters mapped to pbinfo:", uv["pb_name"].unique())
    
def test_build_mklcbol_input():
    from pathlib import Path
    from single_sne.pseudobol_lightcurve.mklcbolinput import (
        build_mklcbol_input,
        ExtinctionInfo,
    )

    bg_file = DATA_DIR / "SN2025cy_BG_data.csv"
    uvot_file = DATA_DIR / "UVOT_AB_mags.txt"
    
    #mu_2025cy, dmu_2025cy = distmod_from_z_with_pecvel(0.011, H0=73.0, sigma_v=300.0)
    
    mu_from_ned = 33.54
    sigma_from_ned = 0.62

    ext_2025cy = ExtinctionInfo(
    av_host=0.0,
    av_host_err=0.0,
    rv_host=3.1,
    rv_host_err=0.0,   # or 0.3 if you want to keep a generic error
    av_mw=0.443,
    av_mw_err=0.0,
    rv_mw=3.1,
    rv_mw_err=0.0,
    dist_mod=mu_from_ned,
    dist_mod_err=sigma_from_ned,
    )
    out = build_mklcbol_input(
        outfile=DATA_DIR /  "SN2025cy_lcbolinput.dat",
        sn_name="SN2025cy",
        bg_csv=bg_file,
        uvot_txt=uvot_file,
        extinction=ext_2025cy,
        ned_name=None,      # or "SN2025cy" once you wire the NED query
    )
    print("Wrote lcbol input:", out)
    
    
def test_bol_lc():
    from single_sne.pseudobol_lightcurve.make_bol_lc import mklcbol
    from pathlib import Path

    lcbol_input = Path(DATA_DIR / "SN2025cy_lcbolinput.dat")
    pbinfo_file = Path(DATA_DIR / "pbinfo_SN2025cy.dat")
    
    f_all = mklcbol(
        infile=lcbol_input,
        pbinfo=pbinfo_file,
        interpmeth="g",   # or "u"/"c"/"s" if you want
        bands=None,        # use ALL filters present
    )

    print("Bolometric LC (all filters):", f_all)
    
    bg_bands = ["u_BG", "q_BG", "i_BG"]

    f_bg = mklcbol(
        infile=lcbol_input,
        pbinfo=pbinfo_file,
        interpmeth="g",
        bands=bg_bands,
    )

    print("Bolometric LC (BlackGEM only):", f_bg)
    
    def read_lcbol(fname):
        t, L, dL = [], [], []
        with open(fname) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 3:
                    continue
                t.append(float(cols[0]))
                L.append(float(cols[1]))
                dL.append(float(cols[2]))
        return np.array(t), np.array(L), np.array(dL)

    t_all, L_all, dL_all = read_lcbol(f_all)
    t_bg,  L_bg,  dL_bg  = read_lcbol(f_bg)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title("Pseudobolometric light curve for SN2025cy")
    ax.errorbar(t_all, L_all, dL_all, fmt="o", ms=3, label="All filters")
    ax.errorbar(t_bg,  L_bg,  dL_bg,  fmt="s", ms=3, label="BlackGEM only")
    ax.set_yscale("log")
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$L_{\rm bol}$ [erg s$^{-1}$]")
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    fig.savefig(DATA_DIR / "SN2025cy_lcbol.pdf", dpi = 600)