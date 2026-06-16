import numpy as np

from quint.chunking.similarities import activate_similarities


def test_no_tail_bias_on_flat_similarities():
    """A flat (all-equal) similarity matrix should activate to a flat vector.

    Diagonals to the right are zero-padded at the tail, so without normalization
    the last p_size-1 positions decayed toward zero and spuriously looked like
    chunk boundaries (issue #12, point 2).
    """
    activated = activate_similarities(np.ones((12, 12)), p_size=10)
    assert np.allclose(activated, activated[0]), activated


def test_handles_fewer_sentences_than_p_size():
    """Inputs with fewer than p_size sentences must not crash (p_size is clamped)."""
    for n in (1, 2, 3, 5):
        activated = activate_similarities(np.ones((n, n)), p_size=10)
        assert activated.shape == (n,)
