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
from hypothesis import given
from hypothesis import strategies as st
from thewalrus.symplectic import (
    beam_splitter,
    expand,
    rotation,
    squeezing,
    two_mode_squeezing,
)

from mrmustard import math, settings
from mrmustard.lab import (
    Amplifier,
    Attenuator,
    BSgate,
    Coherent,
    CXgate,
    CZgate,
    Dgate,
    MZgate,
    Pgate,
    Rgate,
    S2gate,
    Sgate,
)
from mrmustard.lab.states import TMSV, Thermal, Vacuum
from mrmustard.physics.gaussian import controlled_X, controlled_Z


@given(r=st.floats(0, 2), phi=st.floats(0, 2 * np.pi))
def test_two_mode_squeezing(r, phi):
    """Tests that the two-mode squeezing operation is implemented correctly"""
    cov = (Vacuum(num_modes=2) >> S2gate(r=r, phi=phi)).cov * 2 / settings.HBAR
    S = two_mode_squeezing(r, phi)
    assert np.allclose(cov, S @ S.T, atol=1e-6)


@given(r=st.floats(0, 1), phi=st.floats(0, 1))
def test_Sgate(r, phi):
    """Tests the Sgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r_choi, phi=0.0)
    S = Sgate(r=r, phi=phi)[0]
    cov = (Vacuum(2) >> S2 >> S).cov * 2 / settings.HBAR
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(squeezing(r, phi), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(s=st.floats(0, 1))
def test_Pgate(s):
    """Tests the Pgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r_choi, phi=0.0)
    P = Pgate(shearing=s, modes=[0])
    cov = (Vacuum(2) >> S2 >> P).cov * 2 / settings.HBAR
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    P_expanded = expand(np.array([[1, 0], [s, 1]]), [0], 2)
    expected = P_expanded @ expected @ P_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(s=st.floats(0, 1))
def test_CXgate(s):
    """Tests the CXgate is implemented correctly by applying it on one half of a maximally entangled state"""
    s = 2
    r_choi = np.arcsinh(1.0)
    S2a = S2gate(r=r_choi, phi=0.0, modes=[0, 2])
    S2b = S2gate(r=r_choi, phi=0.0, modes=[1, 3])
    CX = CXgate(s=s, modes=[0, 1])
    cov = (Vacuum(4) >> S2a >> S2b >> CX).cov * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4
    )
    CX_expanded = expand(math.asnumpy(controlled_X(s)), [0, 1], 4)
    expected = CX_expanded @ expected @ CX_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(s=st.floats(0, 1))
def test_CZgate(s):
    """Tests the CXgate is implemented correctly by applying it on one half of a maximally entangled state"""
    s = 2
    r_choi = np.arcsinh(1.0)
    S2a = S2gate(r=r_choi, phi=0.0, modes=[0, 2])
    S2b = S2gate(r=r_choi, phi=0.0, modes=[1, 3])
    CZ = CZgate(s=s, modes=[0, 1])
    cov = (Vacuum(4) >> S2a >> S2b >> CZ).cov * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4
    )
    CZ_expanded = expand(math.asnumpy(controlled_Z(s)), [0, 1], 4)
    expected = CZ_expanded @ expected @ CZ_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(theta=st.floats(0, 2 * np.pi))
def test_Rgate(theta):
    """Tests the Rgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r_choi, phi=0.0)
    R = Rgate(angle=theta)
    cov = (Vacuum(2) >> S2 >> R).cov * 2 / settings.HBAR
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(rotation(theta), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(theta=st.floats(0, 2 * np.pi), phi=st.floats(0, 2 * np.pi))
def test_BSgate(theta, phi):
    """Tests the BSgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r_choi, phi=0.0)
    BS = BSgate(theta=theta, phi=phi)
    cov = ((Vacuum(4) >> S2[0, 2]) >> S2[1, 3] >> BS[0, 1]).cov * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4
    )
    S_expanded = expand(beam_splitter(theta, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(r=st.floats(0, 1), phi=st.floats(0, 2 * np.pi))
def test_S2gate(r, phi):
    """Tests the S2gate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r, phi=phi)
    bell = (TMSV(r_choi) & TMSV(r_choi)).get_modes([0, 2, 1, 3])
    cov = (bell[0, 1, 2, 3] >> S2[0, 1]).cov * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4
    )
    S_expanded = expand(two_mode_squeezing(r, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(phi_ex=st.floats(0, 2 * np.pi), phi_in=st.floats(0, 2 * np.pi))
def test_MZgate_external_tms(phi_ex, phi_in):
    """Tests the MZgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    bell = (TMSV(r_choi) & TMSV(r_choi)).get_modes([0, 2, 1, 3])
    MZ = MZgate(phi_a=phi_ex, phi_b=phi_in, internal=False)
    cov = (bell[0, 1, 2, 3] >> MZ[0, 1]).cov * 2 / settings.HBAR

    bell = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4
    )

    ex_expanded = expand(rotation(phi_ex), [0], 4)
    in_expanded = expand(rotation(phi_in), [0], 4)
    BS_expanded = expand(beam_splitter(np.pi / 4, np.pi / 2), [0, 1], 4)

    after_ex = ex_expanded @ bell @ ex_expanded.T
    after_BS1 = BS_expanded @ after_ex @ BS_expanded.T
    after_in = in_expanded @ after_BS1 @ in_expanded.T
    expected = BS_expanded @ after_in @ BS_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(phi_a=st.floats(0, 2 * np.pi), phi_b=st.floats(0, 2 * np.pi))
def test_MZgate_internal_tms(phi_a, phi_b):
    """Tests the MZgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    bell = (TMSV(r_choi) & TMSV(r_choi)).get_modes([0, 2, 1, 3])
    MZ = MZgate(phi_a=phi_a, phi_b=phi_b, internal=True)
    cov = (bell[0, 1, 2, 3] >> MZ[0, 1]).cov * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4
    )
    BS = beam_splitter(np.pi / 4, np.pi / 2)
    S_expanded = expand(BS, [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    S_expanded = expand(rotation(phi_a), [0], 4)
    expected = S_expanded @ expected @ S_expanded.T
    S_expanded = expand(rotation(phi_b), [1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    BS = beam_splitter(np.pi / 4, np.pi / 2)
    S_expanded = expand(BS, [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(g=st.floats(1, 3), x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_amplifier_on_coherent_is_thermal_coherent(g, x, y):
    """Tests that amplifying a coherent state is equivalent to preparing a thermal state displaced state"""
    assert Vacuum(1) >> Dgate(x, y) >> Amplifier(g) == Thermal(g - 1) >> Dgate(
        np.sqrt(g) * x, np.sqrt(g) * y
    )


@given(eta=st.floats(0.1, 0.9), x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_amplifier_attenuator_on_coherent_coherent(eta, x, y):
    """Tests that amplifying and the attenuating a coherent state is equivalent to preparing a thermal state displaced state"""
    assert Vacuum(1) >> Dgate(x, y) >> Amplifier(1 / eta) >> Attenuator(eta) == Thermal(
        ((1 / eta) - 1) * eta
    ) >> Dgate(x, y)


@given(x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_number_means(x, y):
    """Tests that the number means of a displaced state are correct"""
    assert np.allclose(Coherent(x, y).number_means, x * x + y * y)
