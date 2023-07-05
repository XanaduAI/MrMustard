# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" This class corresponds to test child class for the GaussianData class.

Unlike some of its -abstract- parent test classes, this class is meant to be run with pytest.

Check parents test classe-s for more details on the rationale.

The fixtures for PARAMS, DATA and OTHER must correspond to the concrete class being tested, here
GaussianData.
"""

import numpy as np
import pytest

from copy import deepcopy

from mrmustard.lab.representations.data.gaussian_data import GaussianData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory

#########   Instantiating class to test  #########

@pytest.fixture
def COV() -> Matrix:
    return np.eye(10) * 42

@pytest.fixture
def MEANS() -> Vector:
    return np.ones(10) * 42

@pytest.fixture
def COEFFS() -> Scalar:
    return 42

@pytest.fixture
def PARAMS(COV, MEANS, COEFFS) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {'cov': COV, 'means': MEANS, 'coeffs':COEFFS}
    return params_dict


@pytest.fixture()
def DATA(PARAMS) -> GaussianData:
    r"""Instance of the class that must be tested."""
    return general_factory(GaussianData, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> GaussianData:
    r"""Another instance of the class that must be tested."""
    return deepcopy(DATA)


class TestGaussianData(): #TODO, first import TestData, later TestMatVecData

    ####################  Init  ######################

    def test_defining_neither_cov_nor_mean_raises_ValueError(self, COEFFS):
        with pytest.raises(ValueError):
            _ = GaussianData(coeffs=COEFFS)

    def test_if_coeffs_is_undefined_it_is_equal_to_1(self, COV, MEANS):
        gaussian_data = GaussianData(cov=COV, means=MEANS)
        assert gaussian_data.coeffs == 1

    @pytest.mark.parametrize('x', [3])
    def test_if_cov_is_none_then_initialized_at_npeye_of_correct_shape(self, MEANS, COEFFS, x):
        comparison_mat = np.eye(x)
        means = np.ones(x)
        gaussian_data = GaussianData(means=means, coeffs=COEFFS)
        assert np.allclose(gaussian_data.cov, comparison_mat) == True

    def test_non_symplectic_covariance_raises_ValueError(self):
        pass

    #NOTE : these do not test edge cases where someone feeds just [0] as means, it only guarantees
    #  means is not empty. Do we want to secure edge cases?

    ##################  Negative  ####################
    # NOTE : tested in parent class

    ##################  Equality  ####################
    # NOTE : tested in parent class

    ##################  Addition  ####################
    # NOTE : tested in parent class

    ################  Subtraction  ###################
    # NOTE : tested in parent class

    #############  Scalar division  ##################
    # NOTE : tested in parent class

    ##############  Multiplication  ##################

    def test_if_given_scalar_mul_multiplies_coeffs(self):
        pass

    def test_if_given_scalar_mul_does_not_multiply_anything_else_than_coeffs(self):
        pass

    # TODO : test compute_mul_covs
    # TODO : test compute_mul_coeffs
    # TODO : test compute_mul_means

    ###############  Outer product  ##################
    # NOTE : not implemented so no test