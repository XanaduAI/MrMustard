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

from tests.test_lab.test_states import xy_arrays
from tests.random import vector

from mrmustard.lab.representations.wavefunction_ket import WaveFunctionKet


@given(array=xy_arrays())
def test_purity_of_wavefunctionq_ket_state(array):
    qs = vector(array.shape[-1])
    wfket = WaveFunctionKet(qs=qs, quadrature_angle=0, array=array)
    assert np.allclose(wfket.purity, 1.0)


@given(array=xy_arrays())
def test_norm_of_wavefunctionq_ket_state(array):
    qs = vector(array.shape[-1])
    wfket = WaveFunctionKet(qs=qs, quadrature_angle=0, array=array)
    assert np.allclose(wfket.norm, np.abs(np.norm(array)))


@given(array=xy_arrays())
def test_probabilities_of_wavefunctionq_ket_state(array):
    qs = vector(array.shape[-1])
    wfket = WaveFunctionKet(qs=qs, array=array)
    assert np.allclose(wfket.probability, np.abs(array))
