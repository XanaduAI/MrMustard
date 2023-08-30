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


class TestWaveFunctionKetInit:
    """Test class for the initialization of the WaveFunctionKet class."""

    def test_init_wavefunction_ket_with_all_arguments(self):
        "Test that the initialization of the wavefunction ket class with all arguments."
        WaveFunctionKet(points=np.random(3), quadrature_angle=0, wavefunction=np.random.uniform(3))

    def test_init_wavefunction_ket_with_non_physical_state(self):
        "Test that the initialization of the wavefunction ket class with non physical state."
        with self.assertRaises(ValueError):
            WaveFunctionKet(
                points=np.random(3), quadrature_angle=0, wavefunction=np.array([100, 100, 100])
            )


class TestWaveFunctionKetProperties:
    """Test class for each properties returns the correct values."""

    def test_purity_of_wavefunction_ket(self):
        """Test that the purity of any wavefunction ket class state is 1.0."""
        wfket = WaveFunctionKet(
            points=np.random(3), quadrature_angle=0, wavefunction=np.random.uniform(3)
        )
        assert wfket.purity, 1.0

    @given(array=xy_arrays())
    def test_norm_of_wavefunction_ket(self, array):
        """Test that the norm of the wavefunction ket class is correct."""
        points = vector(array.shape[-1])
        wfket = WaveFunctionKet(points=points, quadrature_angle=0, wavefunction=array)
        assert np.allclose(wfket.norm, np.abs(np.norm(array)))

    @given(array=xy_arrays())
    def test_probabilities_of_wavefunction_ket_state(self, array):
        """Test that the probability of the wavefunction ket class is the absolute value of the wavefunction."""
        points = vector(array.shape[-1])
        wfket = WaveFunctionKet(points=points, quadrature_angle=0, wavefunction=array)
        assert np.allclose(wfket.probability, np.abs(array))


class TestWaveFunctionKetThrowErrors:
    """Test class for all non-implement properties or methods of the class."""

    wfket = WaveFunctionKet(
        points=np.random(3), quadrature_angle=0, wavefunction=np.random.uniform(3)
    )

    def test_number_means_with_error(self):
        """Test that number means method is not implement for WaveFunctionKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.wfket.number_means()

    def test_number_cov_with_error(self):
        """Test that number covariance method is not implement for WaveFunctionKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.wfket.number_cov()

    def test_number_variance_with_error(self):
        """Test that number variance method is not implement for WaveFunctionKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.wfket.number_variances()
