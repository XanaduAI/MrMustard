import numpy as np
import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from mrmustard import *
from mrmustard.plugins import gaussian as gp
from mrmustard import Backend
from mrmustard.concrete.states import Gaussian


@given(st.integers(0, 10), st.floats(0.1, 5.0))
def test_vacuum_state(num_modes, hbar):
    cov, disp = gp.vacuum_cov(num_modes, hbar), gp.vacuum_means(num_modes, hbar)
    assert np.allclose(cov, np.eye(2 * num_modes) * hbar / 2)
    assert np.allclose(disp, np.zeros_like(disp))


@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_single(hbar, x, y):
    state = Coherent(x, y, hbar=hbar)
    assert np.allclose(state.cov, np.array([[hbar / 2, 0], [0, hbar / 2]]))
    assert np.allclose(state.means, np.array([x, y]) * np.sqrt(2 * hbar))


# a test like test_coherent_state_single but with x and y being lists of a single float each
@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_list(hbar, x, y):
    assert np.allclose(gp.displacement([x], [y], hbar), np.array([x, y]) * np.sqrt(2 * hbar))


# a test like test_coherent_state_list but with x and y being numpy arrays of length 1
@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_array(hbar, x, y):
    assert np.allclose(gp.displacement(np.array([x]), np.array([y]), hbar), np.array([x, y]) * np.sqrt(2 * hbar))


@given(hbar=st.floats(0.5, 2.0), r=st.floats(0.0, 10.0), phi=st.floats(0.0, 2 * np.pi), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_displaced_squeezed_state(hbar, r, phi, x, y):
    state = DisplacedSqueezed(r, phi, x, y, hbar)
    cov, means = state.cov, state.means
    S = Sgate(modes=[0], r=r, phi=phi)
    D = Dgate(modes=[0], x=x, y=y)
    state = D(S(Vacuum(num_modes=1, hbar=hbar)))
    assert np.allclose(cov, state.cov, rtol=1e-3)
    assert np.allclose(means, state.means)


@st.composite
def xy_arrays(draw):
    length = draw(st.integers(2, 10))
    return draw(arrays(dtype=np.float, shape=(2, length), elements=st.floats(-5.0, 5.0)))


n = st.shared(st.integers(2, 10))
arr = arrays(dtype=np.float, shape=(n), elements=st.floats(-5.0, 5.0))


@given(hbar=st.floats(0.5, 2.0), x=arr, y=arr)
def test_coherent_state_multiple(hbar, x, y):
    state = Coherent(x, y, hbar=hbar)
    assert np.allclose(state.cov, np.eye(2 * len(x)) * hbar / 2)
    assert len(x) == len(y)
    assert np.allclose(state.means, np.concatenate([x, y], axis=-1) * np.sqrt(2 * hbar))


@given(xy=xy_arrays())
def test_the_purity_of_a_pure_state(xy):
    x, y = xy
    state = Coherent(x, y)
    purity = gp.purity(state.cov, state._hbar)
    expected = 1.0
    assert np.isclose(purity, expected)


@given(nbar=st.floats(0.0, 3.0), hbar=st.floats(0.5, 2.0))
def test_the_purity_of_a_mixed_state(nbar, hbar):
    state = Thermal(nbar, hbar)
    purity = gp.purity(state.cov, state._hbar)
    expected = 1 / (2 * nbar + 1)
    assert np.isclose(purity, expected)


@given(r1=st.floats(0.0, 1.0), phi1=st.floats(0.0, 2 * np.pi), r2=st.floats(0.0, 1.0), phi2=st.floats(0.0, 2 * np.pi))
def test_join_two_states(r1, phi1, r2, phi2):
    S1 = Sgate(modes=[0], r=r1, phi=phi1)(Vacuum(num_modes=1))
    S2 = Sgate(modes=[0], r=r2, phi=phi2)(Vacuum(num_modes=1))
    S12 = Sgate(modes=[0, 1], r=[r1, r2], phi=[phi1, phi2])(Vacuum(num_modes=2))
    assert np.allclose((S1 & S2).cov, S12.cov)


@given(
    r1=st.floats(0.0, 1.0),
    phi1=st.floats(0.0, 2 * np.pi),
    r2=st.floats(0.0, 1.0),
    phi2=st.floats(0.0, 2 * np.pi),
    r3=st.floats(0.0, 1.0),
    phi3=st.floats(0.0, 2 * np.pi),
)
def test_join_three_states(r1, phi1, r2, phi2, r3, phi3):
    S1 = Sgate(modes=[0], r=r1, phi=phi1)(Vacuum(num_modes=1))
    S2 = Sgate(modes=[0], r=r2, phi=phi2)(Vacuum(num_modes=1))
    S3 = Sgate(modes=[0], r=r3, phi=phi3)(Vacuum(num_modes=1))
    S123 = Sgate(modes=[0, 1, 2], r=[r1, r2, r3], phi=[phi1, phi2, phi3])(Vacuum(num_modes=3))
    assert np.allclose((S1 & S2 & S3).cov, S123.cov)


def test_join_states_hbar_error():
    S1 = Sgate(modes=[0], r=1, phi=0)(Vacuum(num_modes=1, hbar=1))
    S2 = Sgate(modes=[0], r=1, phi=0)(Vacuum(num_modes=1, hbar=2.0))
    with pytest.raises(ValueError):
        S1 & S2


def test_coh_state_is_same_as_dgate_on_vacuum():
    state = Coherent(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    expected = Dgate(modes=[0, 1], x=[1.0, 0.0], y=[0.0, 1.0])(Vacuum(2))
    assert np.allclose(state.cov, expected.cov)
    assert np.allclose(state.means, expected.means)


def test_sq_state_is_same_as_sgate_on_vacuum():
    state = SqueezedVacuum(0.1, 0.2)
    expected = Sgate(modes=[0], r=0.1, phi=0.2)(Vacuum(1))
    assert np.allclose(state.cov, expected.cov)
    assert np.allclose(state.means, expected.means)


def test_dispsq_state_is_same_as_dsgate_on_vacuum():
    state = DisplacedSqueezed(0.3, 0.4, 0.1, 0.2)
    expected = Dgate(modes=[0], x=0.1, y=0.2)(Sgate(modes=[0], r=0.3, phi=0.4)(Vacuum(1)))
    assert np.allclose(state.cov, expected.cov)
    assert np.allclose(state.means, expected.means)


def test_state_getitem():
    a = Gaussian(2)
    b = Gaussian(2)
    assert a == (a & b)[0, 1]
    assert b == (a & b)[2, 3]
