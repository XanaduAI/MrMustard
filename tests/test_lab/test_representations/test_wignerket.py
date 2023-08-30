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

from mrmustard import settings
from mrmustard.physics import gaussian
from mrmustard.lab.representations.wigner_ket import WignerKet


class TestWignerKetInit:
    """Test class for the initialization of the WignerKet class."""

    def test_init_wigner_ket_with_2dsymp(self):
        "Test that the initialization of the WignerKet class works with 2d symplectic matrix."
        wignerket = WignerKet(
            symplectic=np.random.random((3, 3)), displacement=np.random.random(3), coeffs=1.0
        )
        assert wignerket.cov.shape == 3
        assert wignerket.cov.shape[0] == 1
        assert wignerket.means.shape == 2
        assert wignerket.means.shape[0] == 1

    def test_init_wigner_ket_with_covariancematrix_with_2dcov(self):
        "Test that the initialization of the WignerKet class works with the class method: from_covariance with a 2d covariance matrix."
        wignerket = WignerKet.from_covariance(
            cov=np.random.random((3, 3)), means=np.random.random(3)
        )
        assert wignerket.data.symplectic.shape == 3
        assert wignerket.data.symplectic.shape[0] == 1
        assert wignerket.data.displacement.shape == 2
        assert wignerket.data.displacement.shape[0] == 1

        assert wignerket.cov.shape == 3
        assert wignerket.cov.shape[0] == 1
        assert wignerket.means.shape == 2
        assert wignerket.means.shape[0] == 1

    def test_init_wigner_ket_with_covariancematrix_with_3dcov(self):
        "Test that the initialization of the WignerKet class works with the class method: from_covariance with a 3d covariance matrix."
        wignerket = WignerKet.from_covariance(
            cov=np.random.random((2, 3, 3)), means=np.random.random(2, 3)
        )
        assert wignerket.data.symplectic.shape == 3
        assert wignerket.data.symplectic.shape[0] == 1
        assert wignerket.data.displacement.shape == 2
        assert wignerket.data.displacement.shape[0] == 1


class TestWignerKetProperties:
    """Test class for each properties returns the correct values."""

    def test_purity_with_wigner_ket_state(self):
        """Test that the purity of any wigner ket class state is 1.0."""
        wignerket = WignerKet(
            symplectic=np.random.random((1, 3, 3)), displacement=np.random.random(1, 3), coeffs=1.0
        )
        assert wignerket.purity, 1.0

    @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
    def test_number_means_function_of_wigner_ket_state_from_coherent_state(self, x, y):
        """Test that the number means is correct for a coherent state."""
        wignerket = WignerKet.from_covariance(
            cov=gaussian.vacuum_cov(1, settings.HBAR),
            means=gaussian.displacement(x, y, settings.HBAR),
        )
        expected = x**2 + y**2
        assert np.allclose(wignerket.number_means(), expected)


class TestWignerKetThrowErrors:
    """Test class for all non-implement properties or methods of the class."""

    wignerket = WignerKet(
        symplectic=np.random.random((1, 3, 3)), displacement=np.random.random(1, 3), coeffs=1.0
    )

    def test_norm_with_error(self):
        """Test that the norm is not implement for WignerKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.wignerket.norm()

    def test_number_variance_with_error(self):
        """Test that the number variance is not implement for WignerKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.wignerket.number_variances()

    def test_probability_with_error(self):
        """Test that the probability method is not implement for WignerKet and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.wignerket.probability()
