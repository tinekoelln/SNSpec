import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
import numpy as np
from single_sne.plotting.plot_helpers import setup_science_style  # wherever you put it
setup_science_style()

def create_edges(w_centers):
    # variable bin widths (dw can be array or scalar)
    edges = np.empty(len(w_centers) + 1)
    edges[1:-1] = 0.5*(w_centers[1:] + w_centers[:-1])
    # assume first/last width equals neighbor
    edges[0]  = w_centers[0]  - (edges[1]   - w_centers[0])
    edges[-1] = w_centers[-1] + (w_centers[-1] - edges[-2])
    return edges
    
def plot_stairs(ax, wave, flux, **kw):
    """
    Step-plot using wavelength bin edges (len(edges) = len(flux)+1).
    """
    wave_edges = create_edges(wave)
    import numpy as np
    we = np.asarray(wave_edges)
    f  = np.asarray(flux)
    m = np.isfinite(we)
    we = we[m]
    # stairs expects edges & values; ensures no diagonals
    ax.stairs(f, we, **kw)