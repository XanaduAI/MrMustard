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

The fixtures must correspond to the concrete class being tested, here GaussianData.

"""
import numpy as np
import operator as op
import pytest

from copy import deepcopy

from mrmustard.lab.representations.data.gaussian_data import GaussianData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory
from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData


#########   Instantiating class to test  #########
@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return GaussianData

@pytest.fixture
def COV() -> Matrix:
    r"""Some matrix for the object's parameterization."""
    return np.eye(10) * 42

@pytest.fixture
def MEANS() -> Vector:
    r"""Some vector for the object's parameterization."""
    return np.ones(10) * 42

@pytest.fixture
def COEFFS() -> Scalar:
    r"""Some scalar for the object's parameterization."""
    return 42

@pytest.fixture
def PARAMS(COV, MEANS, COEFFS) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {'cov': COV, 'means': MEANS, 'coeffs':COEFFS}
    return params_dict


@pytest.fixture()
def DATA(TYPE, PARAMS) -> GaussianData:
    r"""Instance of the class that must be tested."""
    return general_factory(TYPE, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> GaussianData:
    r"""Another instance of the class that must be tested."""
    return deepcopy(DATA)


class TestGaussianData(TestMatVecData):

    ####################  Init  ######################

    def test_defining_neither_cov_nor_mean_raises_ValueError(self, COEFFS):
        with pytest.raises(ValueError):
            GaussianData(coeffs=COEFFS)


    def test_if_coeffs_is_undefined_it_is_equal_to_1(self, COV, MEANS):
        gaussian_data = GaussianData(cov=COV, means=MEANS)
        assert gaussian_data.coeffs == 1


    @pytest.mark.parametrize('x', [0, 2, 10, 250])
    def test_if_2D_cov_is_none_then_initialized_at_npeye_of_correct_shape(self, COEFFS, x):
        comparison_covariance = np.eye(x)
        means = np.ones(x)
        gaussian_data = GaussianData(means=means, coeffs=COEFFS)
        assert np.allclose(gaussian_data.cov, comparison_covariance)


    @pytest.mark.parametrize('x', [0, 2, 10, 250])
    def test_if_1D_mean_is_none_then_initialized_at_npzeros_of_correct_shape(self, COEFFS, x):
        covariance = np.eye(x)
        comparison_means = np.zeros(x)
        gaussian_data = GaussianData(cov=covariance, coeffs=COEFFS)
        assert np.allclose(gaussian_data.means, comparison_means)


    def test_if_neither_means_nor_cov_is_defined_raises_ValueError(self, COEFFS):
        with pytest.raises(ValueError):
            GaussianData(coeffs=COEFFS)

    @pytest.mark.skip(reason="Currently not implemented")
    def test_non_symmetric_covariance_raises_ValueError(self, MEAN):
        with pytest.raises(ValueError):
            non_symmetric_mat = np.eye(10)
            non_symmetric_mat[0] += np.array(range(10))
            GaussianData(cov=non_symmetric_mat, means=MEAN)

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

    @pytest.mark.parametrize('x', [2, 7, 200])
    def test_if_given_scalar_mul_multiplies_coeffs_and_nothing_else(self, DATA, x):
        pre_op_data = deepcopy(DATA)
        multiplied_data = DATA * x
        assert multiplied_data.c == (pre_op_data.coeffs * x) #coeffs are multiplied
        assert np.allclose(multiplied_data.cov, pre_op_data.cov) # unaltered
        assert np.allclose(multiplied_data.means, pre_op_data.means) # unaltered

    # TODO : test compute_mul_covs
    # TODO : test compute_mul_coeffs
    # TODO : test compute_mul_means

    ###############  Outer product  ##################
    # NOTE : not implemented => not tested