import shutil
import matplotlib
matplotlib.use("Agg")  # safe for CI and scripts

import matplotlib.pyplot as plt
import scienceplots

def setup_science_style():
    # Check if `latex` is available on the system
    if shutil.which("latex") is not None:
        plt.style.use(["science"])           # full LaTeX
    else:
        plt.style.use(["science", "no-latex"])  # same look, no LaTeX