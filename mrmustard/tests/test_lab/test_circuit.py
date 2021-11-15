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

from mrmustard.lab.circuit import Circuit
from mrmustard.lab.gates import Dgate, Sgate, BSgate, MZgate, S2gate
from hypothesis import given, strategies as st
from thewalrus.fock_gradients import displacement, squeezing, beamsplitter, two_mode_squeezing, mzgate
import numpy as np
from mrmustard.tests.random import single_mode_unitary


@given(gates=st.lists(single_mode_unitary(), min_size=1, max_size=5))
def test_gate_compositions(gates):
    pass  # TODO test that the gate composition is correct


@given(x=st.floats(min_value=-2, max_value=2), y=st.floats(min_value=-2, max_value=2))
def test_fock_representation_displacement(x, y):
    D = Dgate(x=x, y=y)
    expected = displacement(r=np.sqrt(x ** 2 + y ** 2), phi=np.arctan2(y, x), cutoff=20)
    assert np.allclose(expected, D.U(cutoffs=[20]))


@given(r=st.floats(min_value=0, max_value=2), phi=st.floats(min_value=0, max_value=2 * np.pi))
def test_fock_representation_squeezing(r, phi):
    S = Sgate(r=r, phi=phi)
    expected = squeezing(r=r, theta=phi, cutoff=20)
    assert np.allclose(expected, S.U(cutoffs=[20]))


@given(theta=st.floats(min_value=0, max_value=2 * np.pi), phi=st.floats(min_value=0, max_value=2 * np.pi))
def test_fock_representation_beamsplitter(theta, phi):
    BS = BSgate(theta=theta, phi=phi)
    expected = beamsplitter(theta=theta, phi=phi, cutoff=20)
    assert np.allclose(expected, BS.U(cutoffs=[20, 20]))


@given(r=st.floats(min_value=0, max_value=2), phi=st.floats(min_value=0, max_value=2 * np.pi))
def test_fock_representation_two_mode_squeezing(r, phi):
    S2 = S2gate(r=r, phi=phi)
    expected = two_mode_squeezing(r=r, theta=phi, cutoff=20)
    assert np.allclose(expected, S2.U(cutoffs=[20, 20]))


@given(phi_a=st.floats(min_value=0, max_value=2 * np.pi), phi_b=st.floats(min_value=0, max_value=2 * np.pi))
def test_fock_representation_mzgate(phi_a, phi_b):
    MZ = MZgate(phi_a=phi_a, phi_b=phi_b, internal=False)
    expected = mzgate(theta=phi_b, phi=phi_a, cutoff=20)
    assert np.allclose(expected, MZ.U(cutoffs=[20, 20]))
