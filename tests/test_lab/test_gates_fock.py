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

from mrmustard import settings
from mrmustard.lab.gates import Dgate, Sgate, BSgate, MZgate, S2gate
from hypothesis import given, strategies as st
from mrmustard.physics import fock
from thewalrus.fock_gradients import (
    displacement,
    squeezing,
    beamsplitter,
    two_mode_squeezing,
    mzgate,
)
import numpy as np
from tests import random
import pytest


@given(state=random.pure_state(num_modes=1), xy=random.vector(2))
def test_Dgate_1mode(state, xy):
    x, y = xy
    state_out = state >> Dgate(x, y) >> Dgate(-x, -y)
    assert state_out == state


@given(state=random.pure_state(num_modes=2), xxyy=random.vector(4))
def test_Dgate_2mode(state, xxyy):
    x1, x2, y1, y2 = xxyy
    state_out = state >> Dgate([x1, x2], [y1, y2]) >> Dgate([-x1, -x2], [-y1, -y2])
    assert state_out == state


def test_1mode_fock_equals_gaussian():
    pass  # TODO: implement with weak states and gates
    # gate = Ggate(num_modes=1)  # too much squeezing probably
    # gstate = Gaussian(num_modes=1)  # too much squeezing probably
    # fstate = State(fock=gstate.ket(cutoffs=[40]))
    # via_phase_space = gate(gstate)
    # via_fock_space = gate(fstate)
    # assert via_phase_space == via_fock_space


def test_fock_representation_displacement():
    """Tests the correct construction of the single mode displacement operation.
    Since the displacement uses the walrus, this is the same test as in the walrus."""
    cutoff = 5
    alpha = 0.3 + 0.5 * 1j
    # This data is obtained by using qutip
    # np.array(displace(40,alpha).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [
                0.84366482 + 0.00000000e00j,
                -0.25309944 + 4.21832408e-01j,
                -0.09544978 - 1.78968334e-01j,
                0.06819609 + 3.44424719e-03j,
                -0.01109048 + 1.65323865e-02j,
            ],
            [
                0.25309944 + 4.21832408e-01j,
                0.55681878 + 0.00000000e00j,
                -0.29708743 + 4.95145724e-01j,
                -0.14658716 - 2.74850926e-01j,
                0.12479885 + 6.30297236e-03j,
            ],
            [
                -0.09544978 + 1.78968334e-01j,
                0.29708743 + 4.95145724e-01j,
                0.31873657 + 0.00000000e00j,
                -0.29777767 + 4.96296112e-01j,
                -0.18306015 - 3.43237787e-01j,
            ],
            [
                -0.06819609 + 3.44424719e-03j,
                -0.14658716 + 2.74850926e-01j,
                0.29777767 + 4.96296112e-01j,
                0.12389162 + 1.10385981e-17j,
                -0.27646677 + 4.60777945e-01j,
            ],
            [
                -0.01109048 - 1.65323865e-02j,
                -0.12479885 + 6.30297236e-03j,
                -0.18306015 + 3.43237787e-01j,
                0.27646677 + 4.60777945e-01j,
                -0.03277289 + 1.88440656e-17j,
            ],
        ]
    )
    D = Dgate(x=alpha.real, y=alpha.imag)
    assert np.allclose(expected, D.U(cutoffs=[cutoff]), atol=1e-5)


@pytest.mark.parametrize(
    "cutoffs,x,y",
    [
        [[5], 1.0, 1.0],
        [[5], 0.0, 0.0],
        [[2, 2], [0.1, 0.1], [0.25, -0.2]],
        [[2, 5, 1], [0.1, 5.0, 1.0], [-0.3, 0.1, 0.0]],
        [[3, 3, 3, 3], [0.1, 0.2, 0.3, 0.4], [-0.5, -4, 3.1, 4.2]],
        [[3, 3], [0.0, 0.0], [0.0, 0.0]],
    ],
)
# @pytest.mark.parametrize("cutoffs,x,y", [[[5], [10], [10]]])
def test_fock_representation_multimode_displacement(cutoffs, x, y):
    """Tests the correct construction of the multiple mode displacement operation."""

    # apply gate
    dgate = Dgate(x, y)
    Ud = dgate.U(cutoffs)

    choi_state = dgate.bell >> dgate
    expected_Ud = fock.fock_representation(
        choi_state.cov,
        choi_state.means,
        shape=cutoffs * 2,
        return_unitary=True,
        choi_r=settings.CHOI_R,
    )

    assert np.allclose(Ud, expected_Ud)


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
