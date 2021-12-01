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
from hypothesis import settings, given, strategies as st

from thewalrus.symplectic import two_mode_squeezing, squeezing, rotation, beam_splitter, expand
import numpy as np

from mrmustard.lab.gates import Sgate, BSgate, S2gate, Rgate, MZgate
from mrmustard.lab.states import Vacuum, TMSV


@given(r=st.floats(0, 2))
def test_two_mode_squeezing(r):
    """Tests that the two-mode squeezing operation is implemented correctly"""
    cov = (Vacuum(num_modes=2) >> S2gate(r=r, phi=0.0)).cov
    expected = two_mode_squeezing(2 * r, 0.0)
    assert np.allclose(cov, expected, atol=1e-6)


@given(r=st.floats(0, 1), phi=st.floats(0, 1))
def test_Sgate(r, phi):
    """Tests the Sgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r_choi, phi=0.0)
    S = Sgate(r=r, phi=phi)[0]
    cov = (Vacuum(2) >> S2 >> S).cov
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(squeezing(r, phi), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(theta=st.floats(0, 2 * np.pi))
def test_Rgate(theta):
    """Tests the Rgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r_choi, phi=0.0)
    R = Rgate(angle=theta)
    cov = (Vacuum(2) >> S2 >> R).cov
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
    cov = ((Vacuum(4) >> S2[0, 2]) >> S2[1, 3] >> BS[0, 1]).cov
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)
    S_expanded = expand(beam_splitter(theta, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(r=st.floats(0, 1), phi=st.floats(0, 2 * np.pi))
def test_S2gate(r, phi):
    """Tests the S2gate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(r=r, phi=phi)
    # bell = (TMSV(r_choi) & TMSV(r_choi)).get_modes([0, 2, 1, 3])
    cov = (S2.bell >> S2[0, 1]).cov
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)
    S_expanded = expand(two_mode_squeezing(r, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected, atol=1e-6)


@given(phi_ex=st.floats(0, 2 * np.pi), phi_in=st.floats(0, 2 * np.pi))
def test_MZgate_external_tms(phi_ex, phi_in):
    """Tests the MZgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    # bell = (TMSV(r_choi) & TMSV(r_choi)).get_modes([0, 2, 1, 3])
    MZ = MZgate(phi_a=phi_ex, phi_b=phi_in, internal=False)
    cov = (MZ.bell >> MZ[0, 1]).cov

    bell = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)

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
    # bell = (TMSV(r_choi) & TMSV(r_choi))[0, 2, 1, 3]
    MZ = MZgate(phi_a=phi_a, phi_b=phi_b, internal=True)
    cov = (MZ.bell >> MZ[0, 1]).cov
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)
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
