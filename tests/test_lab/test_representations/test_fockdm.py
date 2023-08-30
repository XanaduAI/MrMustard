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
from mrmustard.lab.representations.fock_dm import FockDM


class TestFockDMInit:
    """Test class for the initialization of the FockDM class."""

    def test_init_fock_dm_with_all_arguments(self):
        "Test that the initialization of the fock dm class with all arguments."
        FockDM(array=np.random.uniform((3, 3)))

    def test_init_fock_dm_with_non_physical_state(self):
        "Test that the initialization of the fock dm class with non physical state."
        with self.assertRaises(ValueError):
            FockDM(array=np.array([[100, 100, 100], [100, 100, 100], [100, 100, 100]]))


class TestFockDMProperties:
    """Test class for each properties returns the correct values."""

    def test_purity_of_fock_dm_state(self):
        """Test that the purity of a fock density matrix is correct."""
        array = np.array([[0.5, 0.2], [-0.1, 1.5]])
        fockdm = FockDM(array=array)
        assert np.allclose(fockdm.purity, 0.615)

    def test_norm_of_fock_dm_state(self):
        """Test that the norm of the fock dm class is correct."""
        array = np.array([[0.5, 0.2], [-0.1, 1.5]])
        fockdm = FockDM(array=array)
        assert np.allclose(fockdm.norm, 2.0)

    @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
    def test_number_means_function_of_fock_dm_state_from_coherent_state(self, x, y):
        """Test that the number means is correct for a coherent state."""
        dm = Coherent(x, y).dm([80]).dm([80])
        fockdm = FockDM(array=dm)
        expected = x**2 + y**2
        assert np.allclose(fockdm.number_means(), expected)

    @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
    def test_number_variance_function_of_fock_dm_state_from_coherent_state(self, x, y):
        """Test that the number variance is correct for a coherent state."""
        dm = Coherent(x, y).dm([80])
        fockdm = FockDM(array=dm)
        expected = x**2 + y**2
        assert np.allclose(fockdm.number_variances()[0], expected)

    def test_number_variances_function_of_fock_dm_state_from_fock_state(self):
        """Test that the number variance is correct for a fock state."""
        dm = np.array([0, 1, 0, 0, 0])
        fockdm = FockDM(array=dm)
        expected = 0
        assert np.allclose(fockdm.number_variances(), expected)

    @given(array=xy_arrays())
    def test_probabilities_of_fock_dm_state(self, array):
        """Test that the probability of the fock dm class is the diagonal elements of the fock array."""
        fockdm = FockDM(array=array)
        assert np.allclose(fockdm.probability, np.diag(array))


class TestFockDMThrowErrors:
    """Test class for all non-implement properties or methods of the class."""

    fockdm = FockDM(array=np.random.uniform(3))

    def test_number_cov_with_error(self):
        """Test that number covariance method is not implement for FockDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.fockdm.number_cov()
