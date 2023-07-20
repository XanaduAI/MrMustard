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

from mrmustard.lab.representations.fock_dm import FockDM


def test_purity_of_fock_dm_state():
    array = np.array([[0.5, 0.2], [-0.1, 1.5]])
    fockdm = FockDM(array=array)
    assert np.allclose(fockdm.purity, 0.615)


def test_norm_of_fock_dm_state():
    array = np.array([[0.5, 0.2], [-0.1, 1.5]])
    fockdm = FockDM(array=array)
    assert np.allclose(fockdm.norm, 2.0)


@given(array=xy_arrays())
def test_probabilities_of_fock_dm_state(array):
    fockdm = FockDM(array=array)
    assert np.allclose(fockdm.probability, np.diag(array))
