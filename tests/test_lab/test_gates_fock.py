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

import pytest
from hypothesis import given, strategies as st
import numpy as np
from thewalrus.fock_gradients import (
    squeezing,
    beamsplitter,
    two_mode_squeezing,
    mzgate,
)

from tests import random
from mrmustard.physics import fock
from mrmustard.lab.states import Fock, State, SqueezedVacuum, TMSV
from mrmustard.physics import fock
from mrmustard.lab.gates import (
    Dgate,
    Sgate,
    Pgate,
    Rgate,
    CZgate,
    CXgate,
    BSgate,
    MZgate,
    S2gate,
    Attenuator,
    Interferometer,
)


@given(state=random.pure_state(num_modes=1), xy=random.vector(2))
def test_Dgate_1mode(state, xy):
    x, y = xy
    state_out = state >> Dgate(x, y) >> Dgate(-x, -y)
    assert state_out == state


def test_attenuator_on_fock():
    "tests that attenuating a fock state makes it mixed"
    assert (Fock(10) >> Attenuator(0.5)).is_pure == False


@given(state=random.pure_state(num_modes=2), xxyy=random.vector(4))
def test_Dgate_2mode(state, xxyy):
    x1, x2, y1, y2 = xxyy
    state_out = state >> Dgate([x1, x2], [y1, y2]) >> Dgate([-x1, -x2], [-y1, -y2])
    assert state_out == state


@pytest.mark.parametrize("gate", [Sgate(r=1), Dgate(1.0, 1.0), Pgate(10), Rgate(np.pi / 2)])
def test_single_mode_fock_equals_gaussian(gate):
    """Test same state is obtained via fock representation or phase space
    for single mode circuits."""
    cutoffs = [30]
    gaussian_state = SqueezedVacuum(0.5) >> Attenuator(0.5)
    fock_state = State(dm=gaussian_state.dm(cutoffs))

    via_fock_space_dm = (fock_state >> gate).dm(cutoffs).numpy()
    via_phase_space_dm = (gaussian_state >> gate).dm(cutoffs).numpy()
    assert np.allclose(via_fock_space_dm, via_phase_space_dm)


@pytest.mark.parametrize(
    "gate", [BSgate(np.pi / 2), MZgate(np.pi / 2), CZgate(0.1), CXgate(0.1), S2gate(0.1)]
)
def test_two_mode_fock_equals_gaussian(gate):
    """Test same state is obtained via fock representation or phase space
    for two modes circuits."""
    cutoffs = [20, 20]
    gaussian_state = TMSV(0.1) >> BSgate(np.pi / 2) >> Attenuator(0.5)
    fock_state = State(dm=gaussian_state.dm(cutoffs))

    via_fock_space_dm = (fock_state >> gate).dm(cutoffs).numpy()
    via_phase_space_dm = (gaussian_state >> gate).dm(cutoffs).numpy()
    assert np.allclose(via_fock_space_dm, via_phase_space_dm)


@pytest.mark.parametrize(
    "cutoffs,x,y",
    [
        [[5], 0.3, 0.5],
        [[5], 0.0, 0.0],
        [[2, 2], [0.1, 0.1], [0.25, -0.2]],
        [[3, 3], [0.0, 0.0], [0.0, 0.0]],
        [[2, 5, 1], [0.1, 5.0, 1.0], [-0.3, 0.1, 0.0]],
        [[3, 3, 3, 3], [0.1, 0.2, 0.3, 0.4], [-0.5, -4, 3.1, 4.2]],
    ],
)
def test_fock_representation_displacement(cutoffs, x, y):
    """Tests that DGate returns the correct unitary."""

    # apply gate
    dgate = Dgate(x, y)
    Ud = dgate.U(cutoffs)

    # compare with the standard way of calculating
    # transformation unitaries using the Choi isomorphism
    X = np.eye(2 * len(cutoffs))
    Y = np.zeros((2 * len(cutoffs), 2 * len(cutoffs)))
    expected_Ud = fock.wigner_to_fock_U(X, dgate.XYd[-1], cutoffs * 2)

    assert np.allclose(Ud, expected_Ud, atol=1e-5)


@given(r=st.floats(min_value=0, max_value=2), phi=st.floats(min_value=0, max_value=2 * np.pi))
def test_fock_representation_squeezing(r, phi):
    S = Sgate(r=r, phi=phi)
    expected = squeezing(r=r, theta=phi, cutoff=20)
    assert np.allclose(expected, S.U(cutoffs=[20]), atol=1e-5)


@given(
    theta=st.floats(min_value=0, max_value=2 * np.pi),
    phi=st.floats(min_value=0, max_value=2 * np.pi),
)
def test_fock_representation_beamsplitter(theta, phi):
    BS = BSgate(theta=theta, phi=phi)
    expected = beamsplitter(theta=theta, phi=phi, cutoff=20)
    assert np.allclose(expected, BS.U(cutoffs=[20, 20]), atol=1e-5)


@given(r=st.floats(min_value=0, max_value=2), phi=st.floats(min_value=0, max_value=2 * np.pi))
def test_fock_representation_two_mode_squeezing(r, phi):
    S2 = S2gate(r=r, phi=phi)
    expected = two_mode_squeezing(r=r, theta=phi, cutoff=20)
    assert np.allclose(expected, S2.U(cutoffs=[20, 20]), atol=1e-5)


@given(
    phi_a=st.floats(min_value=0, max_value=2 * np.pi, allow_infinity=False, allow_nan=False),
    phi_b=st.floats(min_value=0, max_value=2 * np.pi, allow_infinity=False, allow_nan=False),
)
def test_fock_representation_mzgate(phi_a, phi_b):
    MZ = MZgate(phi_a=phi_a, phi_b=phi_b, internal=False)
    expected = mzgate(theta=phi_b, phi=phi_a, cutoff=20)
    assert np.allclose(expected, MZ.U(cutoffs=[20, 20]), atol=1e-5)


@pytest.mark.parametrize(
    "cutoffs,angles,modes",
    [
        [[5, 4, 3], [np.pi, np.pi / 2, np.pi / 4], None],
        [[3, 4], [np.pi / 3, np.pi / 2], [0, 1]],
        [[3], np.pi / 6, [0]],
    ],
)
def test_fock_representation_rgate(cutoffs, angles, modes):
    """Tests that DGate returns the correct unitary."""

    # apply gate
    rgate = Rgate(angles, modes=modes)
    R = rgate.U(cutoffs)

    # compare with the standard way of calculating
    # transformation unitaries using the Choi isomorphism
    Y = np.zeros((2 * len(cutoffs), 2 * len(cutoffs)))
    d = np.zeros(2 * len(cutoffs))
    expected_R = fock.wigner_to_fock_U(rgate.XYd[0], d, cutoffs * 2)
    assert np.allclose(R, expected_R, atol=1e-5)


def test_raise_interferometer_error():
    """test Interferometer raises an error when both `modes` and `num_modes` are given"""
    num_modes = 3
    modes = [0, 2]
    with pytest.raises(ValueError):
        Interferometer(num_modes=num_modes, modes=modes)
    modes = [2, 5, 6]
    with pytest.raises(ValueError):
        Interferometer(num_modes=num_modes, modes=modes)
