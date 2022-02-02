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
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from mrmustard.physics import gaussian as gp
from mrmustard.lab.states import *
from mrmustard.lab.gates import *
from mrmustard.lab.abstract import State
from mrmustard import settings
from tests import random


@st.composite
def xy_arrays(draw):
    length = draw(st.integers(2, 10))
    return draw(arrays(dtype=np.float, shape=(2, length), elements=st.floats(-5.0, 5.0)))


@st.composite
def rphi_arrays(draw):
    length = draw(st.integers(2, 10))
    r = arrays(dtype=np.float, shape=(2, length), elements=st.floats(0.0, 1.0))
    phi = arrays(dtype=np.float, shape=(2, length), elements=st.floats(0.0, 2 * np.pi))
    return r, phi


@given(st.integers(0, 10), st.floats(0.1, 5.0))
def test_vacuum_state(num_modes, hbar):
    cov, disp = gp.vacuum_cov(num_modes, hbar), gp.vacuum_means(num_modes, hbar)
    assert np.allclose(cov, np.eye(2 * num_modes) * hbar / 2)
    assert np.allclose(disp, np.zeros_like(disp))


@given(x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_single(x, y):
    state = Coherent(x, y)
    assert np.allclose(state.cov, np.array([[settings.HBAR / 2, 0], [0, settings.HBAR / 2]]))
    assert np.allclose(state.means, np.array([x, y]) * np.sqrt(2 * settings.HBAR))


@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_list(hbar, x, y):
    assert np.allclose(gp.displacement([x], [y], hbar), np.array([x, y]) * np.sqrt(2 * hbar))


@given(hbar=st.floats(0.5, 2.0), x=st.floats(-5.0, 5.0), y=st.floats(-5.0, 5.0))
def test_coherent_state_array(hbar, x, y):
    assert np.allclose(
        gp.displacement(np.array([x]), np.array([y]), hbar),
        np.array([x, y]) * np.sqrt(2 * hbar),
    )


@given(xy=xy_arrays())
def test_coherent_state_multiple(xy):
    x, y = xy
    state = Coherent(x, y)
    assert np.allclose(state.cov, np.eye(2 * len(x)) * settings.HBAR / 2)
    assert len(x) == len(y)
    assert np.allclose(state.means, np.concatenate([x, y], axis=-1) * np.sqrt(2 * settings.HBAR))


@given(xy=xy_arrays())
def test_the_purity_of_a_pure_state(xy):
    x, y = xy
    state = Coherent(x, y)
    purity = gp.purity(state.cov, settings.HBAR)
    expected = 1.0
    assert np.isclose(purity, expected)


@given(nbar=st.floats(0.0, 3.0))
def test_the_purity_of_a_mixed_state(nbar):
    state = Thermal(nbar)
    purity = gp.purity(state.cov, settings.HBAR)
    expected = 1 / (2 * nbar + 1)
    assert np.isclose(purity, expected)


@given(
    r1=st.floats(0.0, 1.0),
    phi1=st.floats(0.0, 2 * np.pi),
    r2=st.floats(0.0, 1.0),
    phi2=st.floats(0.0, 2 * np.pi),
)
def test_join_two_states(r1, phi1, r2, phi2):
    S1 = Vacuum(1) >> Sgate(r=r1, phi=phi1)
    S2 = Vacuum(1) >> Sgate(r=r2, phi=phi2)
    S12 = Vacuum(2) >> Sgate(r=[r1, r2], phi=[phi1, phi2])
    assert S1 & S2 == S12


@given(
    r1=st.floats(0.0, 1.0),
    phi1=st.floats(0.0, 2 * np.pi),
    r2=st.floats(0.0, 1.0),
    phi2=st.floats(0.0, 2 * np.pi),
    r3=st.floats(0.0, 1.0),
    phi3=st.floats(0.0, 2 * np.pi),
)
def test_join_three_states(r1, phi1, r2, phi2, r3, phi3):
    S1 = Vacuum(1) >> Sgate(r=r1, phi=phi1)
    S2 = Vacuum(1) >> Sgate(r=r2, phi=phi2)
    S3 = Vacuum(1) >> Sgate(r=r3, phi=phi3)
    S123 = Vacuum(3) >> Sgate(r=[r1, r2, r3], phi=[phi1, phi2, phi3])
    assert S123 == S1 & S2 & S3


@given(xy=xy_arrays())
def test_coh_state(xy):
    x, y = xy
    assert Vacuum(len(x)) >> Dgate(x, y) == Coherent(x, y)


@given(r=st.floats(0.0, 1.0), phi=st.floats(0.0, 2 * np.pi))
def test_sq_state(r, phi):
    assert Vacuum(1) >> Sgate(r, phi) == SqueezedVacuum(r, phi)


@given(
    x=st.floats(-1.0, 1.0),
    y=st.floats(-1.0, 1.0),
    r=st.floats(0.0, 1.0),
    phi=st.floats(0.0, 2 * np.pi),
)
def test_dispsq_state(x, y, r, phi):
    assert Vacuum(1) >> Sgate(r, phi) >> Dgate(x, y) == DisplacedSqueezed(r, phi, x, y)


def test_get_modes():
    a = Gaussian(2)
    b = Gaussian(2)
    assert a == (a & b).get_modes([0, 1])
    assert b == (a & b).get_modes([2, 3])


@given(m=st.integers(0, 3))
def test_modes_after_projection(m):
    a = Gaussian(4) << Fock(1)[m]
    assert np.allclose(a.modes, [k for k in range(4) if k != m])
    assert len(a.modes) == 3


@given(n=st.integers(0, 3), m=st.integers(0, 3))
def test_modes_after_double_projection(n, m):
    assume(n != m)
    a = Gaussian(4) << Fock([1, 2])[n, m]
    assert np.allclose(a.modes, [k for k in range(4) if k != m and k != n])
    assert len(a.modes) == 2


def test_random_state_is_entangled():
    """Tests that a Gaussian state generated at random is entangled"""
    state = Vacuum(2) >> Ggate(num_modes=2)
    mat = state.cov
    assert np.allclose(gp.log_negativity(mat, 2), 0.0)
    assert np.allclose(
        gp.log_negativity(gp.physical_partial_transpose(mat, [0, 1]), 2), 0.0, atol=1e-7
    )
    N1 = gp.log_negativity(gp.physical_partial_transpose(mat, [0]), 2)
    N2 = gp.log_negativity(gp.physical_partial_transpose(mat, [1]), 2)

    assert N1 > 0
    assert N2 > 0
    assert np.allclose(N1, N2)


@given(modes=st.lists(st.integers(), min_size=2, max_size=5, unique=True))
def test_getitem_set_modes(modes):
    """Test that using `State.__getitem__` and `modes` kwarg correctly set the modes of the state."""

    cutoff = len(modes) + 1
    ket = np.zeros([cutoff] * len(modes), dtype=np.complex128)
    ket[1, 1] = 1.0 + 0.0j

    state1 = State(ket=ket)[modes]
    state2 = State(ket=ket, modes=modes)

    assert state1.modes == state2.modes
