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
from scipy.stats import multivariate_normal as mvg

from mrmustard.lab.representations.data.gaussian_data import GaussianData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory
from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData
from thewalrus.random import random_covariance
from mrmustard import settings


#########   Instantiating class to test  #########
@pytest.fixture
def D() -> int:
    """The dimension: matrices will be DxD and vectors will be D. D must be even."""
    return 4


@pytest.fixture
def N() -> int:
    """The number of elements in the batch."""
    return 3


@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return GaussianData


@pytest.fixture
def COV(N, D) -> Matrix:
    r"""Some batch of matrices for the object's parameterization."""
    c = settings.rng.normal(size=(N, D, D))
    c = c + np.transpose(c, (0, 2, 1))  # symmetrize
    c = np.einsum("bij,bkj->bik", c, np.conj(c))  # make positive semi-definite
    return c


@pytest.fixture
def MEANS(N, D) -> Vector:
    r"""Some batch of vectors for the object's parameterization."""
    return settings.rng.normal(size=(N, D))


@pytest.fixture
def COEFFS(N) -> Scalar:
    r"""Some batch of scalars for the object's parameterization."""
    return settings.rng.normal(size=N)


@pytest.fixture
def PARAMS(COV, MEANS, COEFFS) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {"cov": COV, "means": MEANS, "coeffs": COEFFS}
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

    @pytest.mark.parametrize("x", [2])
    def test_if_2D_cov_is_none_then_initialized_at_npeye_of_correct_shape(self, COEFFS, x):
        comparison_covariance = np.array([np.eye(x), np.eye(x)])
        means = np.array([np.ones(x), np.ones(x)])
        gaussian_data = GaussianData(means=means, coeffs=COEFFS)
        assert np.allclose(gaussian_data.cov, comparison_covariance)

    @pytest.mark.parametrize("x", [2, 10, 250])
    def test_if_1D_mean_is_none_then_initialized_at_npzeros_of_correct_shape(self, COEFFS, x):
        covariance = np.array([np.eye(x), np.eye(x)])
        comparison_means = np.array([np.zeros(x), np.zeros(x)])
        gaussian_data = GaussianData(cov=covariance, coeffs=COEFFS)
        assert np.allclose(gaussian_data.means, comparison_means)

    @pytest.mark.skip(reason="Currently not implemented")
    def test_non_symmetric_covariance_raises_ValueError(self, MEAN):
        with pytest.raises(ValueError):
            non_symmetric_mat = np.eye(10)
            non_symmetric_mat[0] += np.array(range(10))
            GaussianData(cov=[non_symmetric_mat], means=MEAN)

    # ##################  Negative  ####################
    # # NOTE : tested in parent class

    # ##################  Equality  ####################
    # # NOTE : tested in parent class

    # ##################  Addition  ####################
    # # NOTE : tested in parent class

    # ################  Subtraction  ###################
    # # NOTE : tested in parent class

    # #############  Scalar division  ##################
    # # NOTE : tested in parent class

    # ##############  Multiplication  ##################

    @pytest.mark.parametrize("c", [2, 7, 200])
    def test_if_given_scalar_mul_multiplies_coeffs_and_nothing_else(self, DATA, c):
        pre_op_data = deepcopy(DATA)
        multiplied_data = DATA * c
        assert np.allclose(multiplied_data.c, (pre_op_data.coeffs * c))  # coeffs are multiplied
        assert np.allclose(multiplied_data.cov, pre_op_data.cov)  # unaltered
        assert np.allclose(multiplied_data.means, pre_op_data.means)  # unaltered

    @pytest.mark.parametrize("operator", [op.add, op.mul, op.sub])
    def test_operating_on_two_gaussians_returns_a_gaussian_object(
        self, DATA, OTHER, TYPE, operator
    ):
        output = operator(DATA, OTHER)
        assert isinstance(output, TYPE)

    @pytest.mark.parametrize(
        "x", [np.array([-0.1, 0.2, -0.3, 0.4]), np.array([0.1, -0.2, 0.3, -0.4])]
    )
    def test_multiplication_is_correct(self, DATA, OTHER, D, N, x):
        our_res = (DATA * OTHER).value(x)

        scipy_res = 0.0
        for i in range(N):
            g1 = DATA.coeffs[i] * mvg.pdf(x, mean=DATA.means[i], cov=DATA.cov[i])
            for j in range(N):
                g2 = OTHER.coeffs[j] * mvg.pdf(x, mean=OTHER.means[j], cov=OTHER.cov[j])
                scipy_res += g1 * g2

        assert np.allclose(our_res, scipy_res)

    # ###############  Outer product  ##################
    # # NOTE : not implemented => not tested

    def _helper_full_gaussian_pdf(self, k, cov, means, x, c=1):
        return self._helper_gaussian_precoeff(k, cov) * self._helper_gaussian_exp(cov, means, x, c)

    @staticmethod
    def _helper_gaussian_precoeff(k, cov):
        pi_part = (2 * np.pi) ** (k / 2)
        det_part = np.sqrt(np.linalg.det(cov))
        return 1 / (pi_part * det_part)

    @staticmethod
    def _helper_gaussian_exp(cov, mean, x, c):
        coeff = -(1 / 2)
        precision = np.linalg.inv(cov)
        eta = x - mean
        pre_exponential = np.dot(np.dot(np.transpose(eta), precision), eta)
        exponential = np.exp(coeff * pre_exponential)
        return c * exponential

    @staticmethod
    def _helper_mul_covs(cov1, cov2):
        precision1 = np.linalg.inv(cov1)
        precision2 = np.linalg.inv(cov2)
        return np.linalg.inv(precision1 + precision2)

    @staticmethod
    def _helper_mul_means(new_cov, cov1, cov2, mean1, mean2):
        precision1 = np.linalg.inv(cov1)
        precision2 = np.linalg.inv(cov2)
        eta1 = np.dot(precision1, mean1)
        eta2 = np.dot(precision2, mean2)
        etas = eta1 + eta2
        return np.dot(new_cov, etas)

    def _helper_mul_alpha(self, k, cov1, cov2, mean1, mean2, c=1):
        joint_cov = cov1 + cov2
        return self._helper_full_gaussian_pdf(k=k, cov=joint_cov, means=mean2, x=mean1, c=c)

    def _helper_full_gaussian_mul(self, k, cov1, cov2, mean1, mean2, c=1):
        new_cov = self._helper_mul_covs(cov1, cov2)
        new_mean = self._helper_mul_means(new_cov, cov1, cov2, mean1, mean2)
        new_coeff = self._helper_mul_alpha(k, cov1, cov2, mean1, mean2, c=c)
        return new_cov, new_mean, new_coeff
