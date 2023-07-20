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
from hypothesis import given
from hypothesis import strategies as st

from tests.test_lab.test_states import xy_arrays

from mrmustard.lab.representations.fock_ket import FockKet
from mrmustard.lab.representations.fock_dm import FockDM


@given(array=xy_arrays())
def test_purity_of_fock_ket_state(array):
    fockket = FockKet(array=array)
    assert np.allclose(fockket.purity, 1.0)


def test_purity_of_fock_dm_state():
    array = np.array([[0.5, 0.2],[-0.1, 1.5]])
    fockdm = FockDM(array=array)
    assert np.allclose(fockdm.purity, 0.615)


@given(array=xy_arrays())
def test_norm_of_fock_ket_state(array):
    fockket = FockKet(array=array)
    assert np.allclose(fockket.norm, np.abs(np.norm(array)))


def test_norm_of_fock_dm_state():
    array = np.array([[0.5, 0.2],[-0.1, 1.5]])
    fockdm = FockDM(array=array)
    assert np.allclose(fockdm.norm, 2.0)


@given(array=xy_arrays())
def test_probabilities_of_fock_ket_state(array):
    fockket = FockKet(array=array)
    assert np.allclose(fockket.probability, np.abs(array))


@given(array=xy_arrays())
def test_probabilities_of_fock_dm_state(array):
    fockdm = FockDM(array=array)
    assert np.allclose(fockdm.probability, np.diag(array))



# @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
# def test_number_means(x, y):
#     assert np.allclose(State(ket=Coherent(x, y).ket([80])).number_means, x * x + y * y)
#     assert np.allclose(State(dm=Coherent(x, y).dm([80])).number_means, x * x + y * y)


# @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
# def test_number_variances_coh(x, y):
#     assert np.allclose(fock.number_variances(Coherent(x, y).ket([80]), False)[0], x * x + y * y)
#     assert np.allclose(fock.number_variances(Coherent(x, y).dm([80]), True)[0], x * x + y * y)


# def test_number_variances_fock():
#     assert np.allclose(fock.number_variances(Fock(n=1).ket(), False), 0)
