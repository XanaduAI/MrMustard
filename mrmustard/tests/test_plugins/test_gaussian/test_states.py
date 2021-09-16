import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from mrmustard import *
from mrmustard.plugins import gaussian as gp


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


@given(hbar=st.floats(0.5, 2.0), r=st.floats(0.0, 10.0), phi=st.floats(0.0, 2 * np.pi), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_displaced_squeezed_state(hbar, r, phi, x, y):
    cov, means = gp.displaced_squeezed_state(r, phi, x, y, hbar)
    S = Sgate(modes=[0], r=r, phi=phi)
    D = Dgate(modes=[0], x=x, y=y)
    state = D(S(Vacuum(num_modes=1, hbar=hbar)))
    assert np.allclose(cov, state.cov, rtol=1e-3)
    assert np.allclose(means, state.means)


# a hypothesis strategy to generate x and y as numpy arrays of the same length
# using hypothesis.extra.numpy.arrays. The length can vary between 2 and 10
@st.composite
def xy_arrays(draw):
    length = draw(st.integers(2, 10))
    return draw(arrays(dtype=np.float, shape=(2, length), elements=st.floats(-5.0, 5.0)))


n = st.shared(st.integers(2, 10))
arr = arrays(dtype=np.float, shape=(n), elements=st.floats(-5.0, 5.0))


@given(hbar=st.floats(0.5, 2.0), x=arr, y=arr)
def test_coherent_state_multiple(hbar, x, y):
    cov, disp = gp.coherent_state(x, y, hbar)
    assert np.allclose(cov, np.eye(2 * len(x)) * hbar / 2)
    assert len(x) == len(y)
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
