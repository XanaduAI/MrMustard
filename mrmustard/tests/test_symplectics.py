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
from mrmustard.lab.states import Vacuum


@given(r=st.floats(0, 2))
def test_two_mode_squeezing(r):
    """Tests that the two-mode squeezing operation is implemented correctly"""
    S2 = S2gate(modes=[0, 1], r=r, phi=0.0)
    cov = S2(Vacuum(num_modes=2)).cov
    expected = two_mode_squeezing(2 * r, 0.0)
    assert np.allclose(cov, expected)


@given(r=st.floats(0, 1), phi=st.floats(0, 1))
def test_Sgate(r, phi):
    """Tests the Sgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(modes=[0, 1], r=r_choi, phi=0.0)
    S = Sgate(modes=[0], r=r, phi=phi)
    cov = S(S2(Vacuum(num_modes=2))).cov
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(squeezing(r, phi), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected)


@given(theta=st.floats(0, 2 * np.pi))
def test_Rgate(theta):
    """Tests the Rgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2 = S2gate(modes=[0, 1], r=r_choi, phi=0.0)
    R = Rgate(modes=[0], angle=theta)
    cov = R(S2(Vacuum(num_modes=2))).cov
    expected = two_mode_squeezing(2 * r_choi, 0.0)
    S_expanded = expand(rotation(theta), [0], 2)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected)


@given(theta=st.floats(0, 2 * np.pi), phi=st.floats(0, 2 * np.pi))
def test_BSgate(theta, phi):
    """Tests the BSgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2a = S2gate(modes=[0, 2], r=r_choi, phi=0.0)
    S2b = S2gate(modes=[1, 3], r=r_choi, phi=0.0)
    BS = BSgate(modes=[0, 1], theta=theta, phi=phi)
    cov = BS(S2b(S2a(Vacuum(num_modes=4)))).cov
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)
    S_expanded = expand(beam_splitter(theta, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected)


@given(r=st.floats(0, 1), phi=st.floats(0, 2 * np.pi))
def test_S2gate(r, phi):
    """Tests the S2gate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2a = S2gate(modes=[0, 2], r=r_choi, phi=0.0)
    S2b = S2gate(modes=[1, 3], r=r_choi, phi=0.0)
    S2c = S2gate(modes=[0, 1], r=r, phi=phi)
    cov = S2c(S2b(S2a(Vacuum(num_modes=4)))).cov
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)
    S_expanded = expand(two_mode_squeezing(r, phi), [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected)


@given(phi_a=st.floats(0, 2 * np.pi), phi_b=st.floats(0, 2 * np.pi))
def test_MZgate_external_tms(phi_a, phi_b):
    """Tests the MZgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2a = S2gate(modes=[0, 2], r=r_choi, phi=0.0)
    S2b = S2gate(modes=[1, 3], r=r_choi, phi=0.0)
    MZ = MZgate(modes=[0, 1], phi_a=phi_a, phi_b=phi_b, internal=False)
    cov = MZ(S2b(S2a(Vacuum(num_modes=4)))).cov
    expected = expand(two_mode_squeezing(2 * r_choi, 0.0), [0, 2], 4) @ expand(two_mode_squeezing(2 * r_choi, 0.0), [1, 3], 4)
    S_expanded = expand(rotation(phi_a), [0], 4)
    expected = S_expanded @ expected @ S_expanded.T
    BS = beam_splitter(np.pi / 4, np.pi / 2)
    S_expanded = expand(BS, [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    S_expanded = expand(rotation(phi_b), [0], 4)
    expected = S_expanded @ expected @ S_expanded.T
    BS = beam_splitter(np.pi / 4, np.pi / 2)
    S_expanded = expand(BS, [0, 1], 4)
    expected = S_expanded @ expected @ S_expanded.T
    assert np.allclose(cov, expected)


@given(phi_a=st.floats(0, 2 * np.pi), phi_b=st.floats(0, 2 * np.pi))
def test_MZgate_internal_tms(phi_a, phi_b):
    """Tests the MZgate is implemented correctly by applying it on one half of a maximally entangled state"""
    r_choi = np.arcsinh(1.0)
    S2a = S2gate(modes=[0, 2], r=r_choi, phi=0.0)
    S2b = S2gate(modes=[1, 3], r=r_choi, phi=0.0)
    MZ = MZgate(modes=[0, 1], phi_a=phi_a, phi_b=phi_b, internal=True)
    cov = MZ(S2b(S2a(Vacuum(num_modes=4)))).cov
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
    assert np.allclose(cov, expected)
