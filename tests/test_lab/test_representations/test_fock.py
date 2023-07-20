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

from mrmustard.lab.representations.fock import Fock

@given(array = xy_arrays())
def test_number_means_of_fock_state(array):
    fock = Fock(array=array)
    expected = 1.0
    assert np.allclose(fock.number_means, expected)

@given(array = xy_arrays())
def test_number_variances_of_fock_state(array):
    fock = Fock(array=array)
    expected = 1.0
    assert np.allclose(fock.number_variances, expected)


class TestFockThrowErrors():

    def test_number_cov_with_error(self):
        self.assertRaises(NotImplementedError)

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
