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

from mrmustard.lab import Coherent
from mrmustard.lab.representations.fock_ket import FockKet


class TestFockKetInit:
    """Test class for the initialization of the FockKet class."""

    def test_init_fock_ket_with_all_arguments(self):
        "Test that the initialization of the fock ket class with all arguments."
        FockKet(array=np.random.uniform(3))

    def test_init_fock_ket_with_non_physical_state(self):
        "Test that the initialization of the fock ket class with non physical state."
        with self.assertRaises(ValueError):
            FockKet(array=np.array([100, 100, 100]))


class TestFockKetProperties:
    """Test class for each properties returns the correct values."""

    @given(array=xy_arrays())
    def test_purity_of_fock_ket_state(self, array):
        """Test that the purity of any fock ket class state is 1.0."""
        fockket = FockKet(array=array)
        assert np.allclose(fockket.purity, 1.0)

    @given(array=xy_arrays())
    def test_norm_of_fock_ket_state(self, array):
        """Test that the norm of the fock ket class is correct."""
        fockket = FockKet(array=array)
        assert np.allclose(fockket.norm, np.abs(np.norm(array)))

    @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
    def test_number_means_function_of_fock_ket_state_from_coherent_state(self, x, y):
        """Test that the number means is correct for a coherent state."""
        ket = Coherent(x, y).ket([80])
        fockket = FockKet(array=ket)
        expected = x**2 + y**2
        assert np.allclose(fockket.number_means(), expected)

    @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
    def test_number_variance_function_of_fock_ket_state_from_coherent_state(self, x, y):
        """Test that the number variance is correct for a coherent state."""
        ket = Coherent(x, y).ket([80])
        fockket = FockKet(array=ket)
        expected = x**2 + y**2
        assert np.allclose(fockket.number_variances()[0], expected)

    def test_number_variances_function_of_fock_ket_state_from_fock_state(self):
        """Test that the number variance is correct for a fock state."""
        ket = np.array([0, 1, 0, 0, 0])
        fockket = FockKet(array=ket)
        expected = 0
        assert np.allclose(fockket.number_variances(), expected)

    @given(array=xy_arrays())
    def test_probabilities_of_fock_ket_state(self, array):
        """Test that the probability of the wavefunction ket class is the absolute value of the fock array."""
        fockket = FockKet(array=array)
        assert np.allclose(fockket.probability, np.abs(array))


class TestFockKetThrowErrors:
    """Test class for all non-implement properties or methods of the class."""

    fockket = FockKet(array=np.random.uniform(3))

    def test_number_cov_with_error(self):
        """Test that number covariance method is not implement for FockKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.fockket.number_cov()
