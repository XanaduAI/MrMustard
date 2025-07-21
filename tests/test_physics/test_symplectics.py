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
from thewalrus.symplectic import beam_splitter, expand, rotation, squeezing, two_mode_squeezing

from mrmustard import settings
from mrmustard.lab import Amplifier, Attenuator, BSgate, Dgate, MZgate, Pgate, Rgate, S2gate, Sgate
from mrmustard.lab.states import Thermal, TwoModeSqueezedVacuum, Vacuum


@given(r=st.floats(0, 2), phi=st.floats(0, 2 * np.pi))
def test_two_mode_squeezing(r, phi):
    """Tests that the two-mode squeezing operation is implemented correctly"""
    cov = (Vacuum((0, 1)) >> S2gate((0, 1), r=r, phi=phi)).phase_space(0)[0] * 2 / settings.HBAR
    S = two_mode_squeezing(r, phi)
    assert np.allclose(cov, S @ S.T, atol=1e-6)


@given(r=st.floats(0, 1), phi=st.floats(0, 1))
def test_Sgate(r, phi):
    """Tests the Sgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate((0, 1), r=r_choi, phi=0.0)
    S = Sgate(0, r=r, phi=phi)
    cov = (Vacuum((0, 1)) >> S2 >> S).phase_space(0)[0] * 2 / settings.HBAR
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(squeezing(r, phi), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(s=st.floats(0, 1))
def test_Pgate(s):
    """Tests the Pgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate((0, 1), r=r_choi, phi=0.0)
    P = Pgate(0, shearing=s)
    cov = (Vacuum((0, 1)) >> S2 >> P).phase_space(0)[0] * 2 / settings.HBAR
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    P_expanded = expand(np.array([[1, 0], [s, 1]]), [0], 2)
    expected = P_expanded @ expected @ P_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(theta=st.floats(0, 2 * np.pi))
def test_Rgate(theta):
    """Tests the Rgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate((0, 1), r=r_choi, phi=0.0)
    R = Rgate(0, theta=theta)
    cov = (Vacuum((0, 1)) >> S2 >> R).phase_space(0)[0] * 2 / settings.HBAR
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(rotation(theta), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(theta=st.floats(0, 2 * np.pi), phi=st.floats(0, 2 * np.pi))
def test_BSgate(theta, phi):
    """Tests the BSgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate((0, 1), r=r_choi, phi=0.0)
    BS = BSgate((0, 1), theta=theta, phi=phi)
    cov = (
        (Vacuum((0, 1, 2, 3)) >> S2.on([0, 2]) >> S2.on([1, 3]) >> BS.on([0, 1])).phase_space(0)[0]
        * 2
        / settings.HBAR
    )
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0),
        [1, 3],
        4,
    )
    S_expanded = expand(beam_splitter(theta, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(r=st.floats(0, 1), phi=st.floats(0, 2 * np.pi))
def test_S2gate(r, phi):
    """Tests the S2gate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate((0, 1), r=r, phi=phi)
    bell = TwoModeSqueezedVacuum((0, 2), r=r_choi) >> TwoModeSqueezedVacuum((1, 3), r=r_choi)
    cov = (bell >> S2).phase_space(0)[0] * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0),
        [1, 3],
        4,
    )
    S_expanded = expand(two_mode_squeezing(r, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(phi_ex=st.floats(0, 2 * np.pi), phi_in=st.floats(0, 2 * np.pi))
def test_MZgate_external_tms(phi_ex, phi_in):
    """Tests the MZgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    bell = TwoModeSqueezedVacuum((0, 2), r=r_choi) >> TwoModeSqueezedVacuum((1, 3), r=r_choi)
    MZ = MZgate((0, 1), phi_a=phi_ex, phi_b=phi_in, internal=False)
    cov = (bell >> MZ).phase_space(0)[0] * 2 / settings.HBAR

    bell = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0),
        [1, 3],
        4,
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
    bell = TwoModeSqueezedVacuum((0, 2), r=r_choi) >> TwoModeSqueezedVacuum((1, 3), r=r_choi)
    MZ = MZgate((0, 1), phi_a=phi_a, phi_b=phi_b, internal=True)
    cov = (bell >> MZ).phase_space(0)[0] * 2 / settings.HBAR
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(
        two_mode_squeezing(2 * r_choi, 0.0),
        [1, 3],
        4,
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
    """Tests that amplifying a coherent state is equivalent to preparing a thermal displaced state"""
    assert Vacuum(0) >> Dgate(0, x, y) >> Amplifier(0, g) == Thermal(0, g - 1) >> Dgate(
        0, np.sqrt(g) * x, np.sqrt(g) * y
    )


@given(eta=st.floats(0.1, 0.9), x=st.floats(-2, 2), y=st.floats(-2, 2))
def test_amplifier_attenuator_on_coherent_coherent(eta, x, y):
    """Tests that amplifying and the attenuating a coherent state is equivalent to preparing a thermal displaced state"""
    assert Vacuum(0) >> Dgate(0, x, y) >> Amplifier(0, 1 / eta) >> Attenuator(0, eta) == Thermal(
        0, ((1 / eta) - 1) * eta
    ) >> Dgate(0, x, y)
