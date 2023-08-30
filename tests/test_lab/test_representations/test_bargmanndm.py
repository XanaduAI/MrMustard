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

from mrmustard.lab.representations.bargmann_dm import BargmannDM


class TestBargmannDMInit:
    """Test class for the initialization of the BargmannDM class."""

    def test_init_bargmann_dm_with_2dA(self):
        "Test that the initialization of the BargmannDM class works with 2d A matrix."
        bargmanndm = BargmannDM(A=np.random.random((3, 3)), b=np.random.random(3), c=1.0)
        assert bargmanndm.data.A.shape == 3
        assert bargmanndm.data.A.shape[0] == 1
        assert bargmanndm.data.b.shape == 2
        assert bargmanndm.data.b.shape[0] == 1

    def test_init_bargmann_dm_with_3dA(self):
        "Test that the initialization of the BargmannDM class works with 3d A matrix."
        bargmanndm = BargmannDM(A=np.random.random((2, 3, 3)), b=np.random.random(2, 3), c=1.0)
        assert bargmanndm.data.A.shape == 3
        assert bargmanndm.data.A.shape[0] == 2
        assert bargmanndm.data.b.shape == 2
        assert bargmanndm.data.b.shape[0] == 2


class TestBargmannDMThrowErrors:
    """Test class for all non-implement properties or methods of the class."""

    bargmanndm = BargmannDM(A=np.random.random((3, 3)), b=np.random.random(3), c=1.0)

    def test_purity_with_error(self):
        """Test that the purity is not implement for BargmannDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.bargmanndm.purity()

    def test_norm_with_error(self):
        """Test that the norm is not implement for BargmannDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.bargmanndm.norm()

    def test_number_means_with_error(self):
        """Test that number means method is not implement for BargmannDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.bargmanndm.number_means()

    def test_number_cov_with_error(self):
        """Test that number covariance method is not implement for BargmannDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.bargmanndm.number_cov()

    def test_number_variance_with_error(self):
        """Test that number variance method is not implement for BargmannDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.bargmanndm.number_variances()

    def test_probability_with_error(self):
        """Test that the probability method is not implement for BargmannDM and returns an error."""
        with self.assertRaises(NotImplementedError):
            self.bargmanndm.probability()
