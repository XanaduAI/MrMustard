# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mrmustard import math, settings
from mrmustard.lab.abstract import State
from mrmustard.lab.gates import Attenuator, Dgate, Ggate, Sgate
from mrmustard.lab.states import (
    Coherent,
    DisplacedSqueezed,
    Fock,
    Gaussian,
    SqueezedVacuum,
    Thermal,
    Vacuum,
)
from mrmustard.physics import gaussian as gp
from tests.random import angle, medium_float, n_mode_pure_state, nmodes, r

hbar0 = settings.HBAR

# pylint: disable=protected-access


@st.composite
def xy_arrays(draw):
    length = draw(st.integers(2, 10))
    return draw(
        arrays(dtype=np.float64, shape=(2, length), elements=st.floats(-5.0, 5.0))
    )


@given(nmodes, st.floats(0.1, 5.0))
def test_vacuum_state(nmodes, hbar):
    settings._hbar = hbar
    cov, disp = gp.vacuum_cov(nmodes), gp.vacuum_means(nmodes)
    assert np.allclose(cov, np.eye(2 * nmodes) * hbar / 2)
    assert np.allclose(disp, np.zeros_like(disp))

    # restoring hbar to its original value
    settings._hbar = hbar0


@given(x=medium_float, y=medium_float)
def test_coherent_state_single(x, y):
    state = Coherent(x, y)
    assert np.allclose(
        state.cov, np.array([[settings.HBAR / 2, 0], [0, settings.HBAR / 2]])
    )
    assert np.allclose(state.means, np.array([x, y]) * np.sqrt(2 * settings.HBAR))


@given(hbar=st.floats(0.5, 2.0), x=medium_float, y=medium_float)
def test_coherent_state_list(hbar, x, y):
    settings._hbar = hbar
    assert np.allclose(gp.displacement([x], [y]), np.array([x, y]) * np.sqrt(2 * hbar))

    # restoring hbar to its original value
    settings._hbar = hbar0


@given(hbar=st.floats(0.5, 2.0), x=medium_float, y=medium_float)
def test_coherent_state_array(hbar, x, y):
    settings._hbar = hbar
    assert np.allclose(
        gp.displacement(np.array([x]), np.array([y])),
        np.array([x, y]) * np.sqrt(2 * hbar),
    )

    # restoring hbar to its original value
    settings._hbar = hbar0


@given(xy=xy_arrays())
def test_coherent_state_multiple(xy):
    x, y = xy
    state = Coherent(x, y)
    assert np.allclose(state.cov, np.eye(2 * len(x)) * settings.HBAR / 2)
    assert len(x) == len(y)
    assert np.allclose(
        state.means, np.concatenate([x, y], axis=-1) * np.sqrt(2 * settings.HBAR)
    )

    # restoring hbar to its original value
    settings._hbar = hbar0


@given(state=n_mode_pure_state(num_modes=1))
def test_the_purity_of_a_pure_state(state):
    purity = gp.purity(state.cov)
    expected = 1.0
    assert np.isclose(purity, expected)


@given(nbar=st.floats(0.0, 3.0))
def test_the_purity_of_a_mixed_state(nbar):
    state = Thermal(nbar)
    purity = gp.purity(state.cov)
    expected = 1 / (2 * nbar + 1)
    assert np.isclose(purity, expected)


@given(r1=r, phi1=angle, r2=r, phi2=angle)
def test_join_two_states(r1, phi1, r2, phi2):
    """Test Sgate acts the same in parallel or individually for two states."""
    S1 = Vacuum(1) >> Sgate(r=r1, phi=phi1)
    S2 = Vacuum(1) >> Sgate(r=r2, phi=phi2)
    S12 = Vacuum(2) >> Sgate(r=[r1, r2], phi=[phi1, phi2])
    assert S1 & S2 == S12


@given(r1=r, phi1=angle, r2=r, phi2=angle, r3=r, phi3=angle)
def test_join_three_states(r1, phi1, r2, phi2, r3, phi3):
    """Test Sgate acts the same in parallel or individually for three states."""
    S1 = Vacuum(1) >> Sgate(r=r1, phi=phi1)
    S2 = Vacuum(1) >> Sgate(r=r2, phi=phi2)
    S3 = Vacuum(1) >> Sgate(r=r3, phi=phi3)
    S123 = Vacuum(3) >> Sgate(r=[r1, r2, r3], phi=[phi1, phi2, phi3])
    assert S123 == S1 & S2 & S3


@given(xy=xy_arrays())
def test_coh_state(xy):
    """Test coherent state preparation."""
    x, y = xy
    assert Vacuum(len(x)) >> Dgate(x, y) == Coherent(x, y)


@given(r=r, phi=angle)
def test_sq_state(r, phi):
    """Test squeezed vacuum preparation."""
    assert Vacuum(1) >> Sgate(r, phi) == SqueezedVacuum(r, phi)


@given(
    x=medium_float,
    y=medium_float,
    r=r,
    phi=angle,
)
def test_dispsq_state(x, y, r, phi):
    """Test displaced squeezed state."""
    assert Vacuum(1) >> Sgate(r, phi) >> Dgate(x, y) == DisplacedSqueezed(r, phi, x, y)


def test_get_modes():
    """Test get_modes returns the states as expected."""
    a = Gaussian(2)
    b = Gaussian(2)
    assert a == (a & b).get_modes([0, 1])
    assert b == (a & b).get_modes([2, 3])


def test_hbar():
    """Test cov matrix is linear in hbar."""
    g = Gaussian(2)
    p = g.purity
    settings._hbar = 1.234
    assert g.purity == p

    # restoring hbar to its original value
    settings._hbar = hbar0


def test_get_single_mode():
    """Test get_modes leaves a single-mode state untouched."""
    a = Gaussian(1)[1]
    assert a == a.get_modes([1])


def test_get_single_mode_fail():
    """Test get_modes leaves a single-mode state untouched."""
    a = Gaussian(1)[1]
    with pytest.raises(ValueError):
        a.get_modes([0])


def test_iter():
    """Test we can iterate individual modes in states."""
    a = Gaussian(1)
    b = Gaussian(2)
    c = Gaussian(1)
    for i, mode in enumerate(a & b & c):
        assert (a, b.get_modes(0), b.get_modes(1), c)[i] == mode


@given(m=st.integers(0, 3))
def test_modes_after_projection(m):
    """Test number of modes is correct after single projection."""
    a = Gaussian(4) << Fock(1)[m]
    assert np.allclose(a.modes, [k for k in range(4) if k != m])
    assert len(a.modes) == 3


@given(n=st.integers(0, 3), m=st.integers(0, 3))
def test_modes_after_double_projection(n, m):
    """Test number of modes is correct after double projection."""
    assume(n != m)
    a = Gaussian(4) >> Dgate(x=1.0)[0, 1, 2, 3] << Fock([1, 2])[n, m]
    assert np.allclose(a.modes, [k for k in range(4) if k != m and k != n])
    assert len(a.modes) == 2


def test_random_state_is_entangled():
    """Tests that a Gaussian state generated at random is entangled."""
    state = Vacuum(2) >> Ggate(num_modes=2)
    mat = state.cov
    assert np.allclose(gp.log_negativity(mat), 0.0)
    assert np.allclose(
        gp.log_negativity(gp.physical_partial_transpose(mat, [0, 1])), 0.0, atol=1e-7
    )
    N1 = gp.log_negativity(gp.physical_partial_transpose(mat, [0]))
    N2 = gp.log_negativity(gp.physical_partial_transpose(mat, [1]))

    assert N1 > 0
    assert N2 > 0
    assert np.allclose(N1, N2)


@given(modes=st.lists(st.integers(), min_size=2, max_size=5, unique=True))
def test_getitem_set_modes(modes):
    """Test that using `super().__getitem__` and `modes`
    kwarg correctly set the modes of the state."""

    cutoff = len(modes) + 1
    ket = np.zeros([cutoff] * len(modes), dtype=np.complex128)
    ket[1, 1] = 1.0 + 0.0j

    state1 = State(ket=ket)[modes]
    state2 = State(ket=ket, modes=modes)

    assert state1.modes == state2.modes


@pytest.mark.parametrize("pure", [True, False])
def test_concat_pure_states(pure):
    """Test that fock states concatenate correctly and are separable"""
    state1 = Fock(1, cutoffs=[15])
    state2 = Fock(4, cutoffs=[15])

    if not pure:
        state1 >>= Attenuator(transmissivity=0.95)
        state2 >>= Attenuator(transmissivity=0.9)

    psi = state1 & state2

    # test concatenated state
    psi_dm = math.transpose(
        math.tensordot(state1.dm(), state2.dm(), [[], []]), [0, 2, 1, 3]
    )
    assert np.allclose(psi.dm(), psi_dm)


@pytest.mark.parametrize("n", ([1, 0, 0], [1, 1, 0], [0, 0, 1]))
@pytest.mark.parametrize("cutoffs", ([2, 2, 2], [2, 3, 3], [3, 3, 2]))
def test_ket_from_pure_dm(n, cutoffs):
    # prepare a fock (pure) state
    fock_state = Fock(n=n, cutoffs=cutoffs)
    dm_fock = fock_state.dm()

    # initialize a new state from the density matrix
    # (no knowledge of the ket)
    test_state = State(dm=dm_fock)
    test_ket = test_state.ket()

    # check test state calculated the same ket as the original state
    assert np.allclose(test_ket, fock_state.ket())


def test_ket_from_pure_dm_new_cutoffs():
    "tests that the shape of the internal fock representation reflects the new cutoffs"
    state = Vacuum(1) >> Sgate(0.1) >> Dgate(0.1, 0.1)  # weak gaussian state
    state = State(dm=state.dm(cutoffs=[20]))  # assign pure dm directly
    assert list(state.ket(cutoffs=[5]).shape) == [5]  # shape should be [5]


def test_ket_probability():
    "Test that the probability of a ket is calculated correctly."
    state = State(ket=np.array([0.5, 0.5]))
    assert np.isclose(state.probability, 2 * 0.5**2)


def test_dm_probability():
    "Test that the probability of a density matrix is calculated correctly."
    state = State(dm=np.array([[0.4, 0.1], [0.1, 0.4]]))
    assert np.isclose(state.probability, 0.8)


def test_padding_ket():
    "Test that padding a ket works correctly."
    state = State(ket=SqueezedVacuum(r=1.0).ket(cutoffs=[20]))
    assert len(state.ket(cutoffs=[10])) == 10
    assert len(state._ket) == 20  # pylint: disable=protected-access


def test_padding_dm():
    "Test that padding a density matrix works correctly."
    state = State(dm=(SqueezedVacuum(r=1.0) >> Attenuator(0.6)).dm(cutoffs=[20]))
    assert tuple(int(c) for c in state.dm(cutoffs=[10]).shape) == (10, 10)
    assert tuple(int(c) for c in state._dm.shape) == (
        20,
        20,
    )  # pylint: disable=protected-access


def test_state_repr_small_prob():
    "test that small probabilities are displayed correctly"
    state = State(ket=np.array([0.0001, 0.0001]))
    table = state._repr_markdown_()  # pylint: disable=protected-access
    assert "2.000e-06 %" in table


def test_state_repr_big_prob():
    "test that big probabilities are displayed correctly"
    state = State(ket=np.array([0.5, 0.5]))
    table = state._repr_markdown_()  # pylint: disable=protected-access
    assert "50.000%" in table
