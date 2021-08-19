from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
np.random.seed(137)

from mrmustard import GaussianPlugin, Coherent, Thermal
gp = GaussianPlugin()


@given(st.integers(0, 10), st.floats(0.1, 5.0))
def test_vacuum_state(nmodes, hbar):
    cov, disp = gp.vacuum_state(nmodes, hbar)
    assert np.allclose(cov, np.eye(2 * nmodes) * hbar / 2)
    assert np.allclose(disp, np.zeros_like(disp))


@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_single(hbar, x, y):
    cov, disp = gp.coherent_state(x, y, hbar)
    assert np.allclose(cov, np.eye(2) * hbar / 2)
    assert np.allclose(disp, np.array([x, y]) * np.sqrt(2 * hbar))

# a test like test_coherent_state_single but with x and y being lists of a single float each
@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_list(hbar, x, y):
    cov, disp = gp.coherent_state([x], [y], hbar)
    assert np.allclose(cov, np.eye(2) * hbar / 2)
    assert np.allclose(disp, np.array([x, y]) * np.sqrt(2 * hbar))

# a test like test_coherent_state_list but with x and y being numpy arrays of length 1
@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_array(hbar, x, y):
    cov, disp = gp.coherent_state(np.array([x]), np.array([y]), hbar)
    assert np.allclose(cov, np.eye(2) * hbar / 2)
    assert np.allclose(disp, np.array([x, y]) * np.sqrt(2 * hbar))


# a hypothesis strategy to generate x and y as numpy arrays of the same length
# using hypothesis.extra.numpy.arrays. The length can vary between 2 and 10
@st.composite
def xy_arrays(draw):
    length = draw(st.integers(2, 10))
    x = draw(arrays(dtype=np.float, shape=(length,), elements=st.floats(-5.0, 5.0)))
    y = draw(arrays(dtype=np.float, shape=(length,), elements=st.floats(-5.0, 5.0)))
    return x, y


@given(hbar=st.floats(0.5, 2.0), xy=xy_arrays())
def test_coherent_state_multiple(hbar, xy):
    x, y = xy
    cov, disp = gp.coherent_state(x, y, hbar)
    assert np.allclose(cov, np.eye(2 * len(x)) * hbar / 2)
    assert np.allclose(disp, np.concatenate([x, y], axis=-1) * np.sqrt(2 * hbar))


@given(xy=xy_arrays())
def test_the_purity_of_a_pure_state(xy):
    x, y = xy
    state = Coherent(x, y)
    purity = gp.purity(state.cov, state.hbar)
    expected = 1.0
    assert np.isclose(purity, expected)


@given(nbar=st.floats(0.0, 3.0), hbar=st.floats(0.5, 2.0))
def test_the_purity_of_a_mixed_state(nbar, hbar):
    state = Thermal(nbar, hbar)
    purity = gp.purity(state.cov, state.hbar)
    expected = 1 / (2 * nbar + 1)
    assert np.isclose(purity, expected)
