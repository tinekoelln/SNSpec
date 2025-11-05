import numpy as np
from single_sne.spectra.stitching import _scale_in_overlap, stitch_arms

def test_scale_in_overlap_basic():
    # reference spectrum
    w1 = np.linspace(500, 600, 300)
    f1 = 1e-16 * (1 + 0.1*np.sin(w1/10))

    # second spectrum scaled by factor 1.5
    true_scale = 1.5
    w2 = np.linspace(550, 650, 300)
    f2 = np.interp(w2, w1, f1) * true_scale

    s_est, n_used = _scale_in_overlap(w1, f1, w2, f2, 560, 580)

    assert np.isclose(s_est * true_scale, 1.0, rtol=1e-3)
    assert n_used > 10



def test_stitch_basic_edge_join():
    import numpy as np, astropy.units as u
    w1 = np.linspace(500, 560, 61) * u.nm
    f1 = np.ones_like(w1.value) * 10 * (u.erg / (u.s * u.cm**2 * u.AA))
    w2 = np.linspace(540, 620, 81) * u.nm
    f2 = np.ones_like(w2.value) * 5 * (u.erg / (u.s * u.cm**2 * u.AA))

    w, f, s = stitch_arms(w1, f1, w2, f2, overlap=(545, 555)*u.nm, stitch_edge=555*u.nm)
    assert np.isclose(s, 2.0)
    assert np.all(w.to(u.nm).value[:-1] < w.to(u.nm).value[1:])
    # Left of edge equals left flux
    mask_left = (w <= 555*u.nm)
    assert np.allclose(f[mask_left].value, 10.0)
    # Right of edge equals scaled right flux
    mask_right = (w > 555*u.nm)
    assert np.allclose(f[mask_right].value, 10.0)
