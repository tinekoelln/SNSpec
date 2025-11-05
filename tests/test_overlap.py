import numpy as np
from single_sne.spectra.stitching import _scale_in_overlap

def test_simple_scale():
    w1 = np.linspace(500, 600, 50)
    f1 = np.ones_like(w1) * 10
    w2 = np.linspace(520, 620, 50)
    f2 = np.ones_like(w2) * 5

    s, n = _scale_in_overlap(w1, f1, w2, f2, 530, 560)
    assert np.isclose(s, 2.0)
    assert n > 0
