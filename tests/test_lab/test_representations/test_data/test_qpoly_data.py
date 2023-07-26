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
""" This class corresponds to test child class for the MatVecData class.

Unlike some of its -abstract- parent test classes, this class is meant to be run with pytest.

Check parents test classe-s for more details on the rationale.

The fixtures must correspond to the concrete class being tested, here QPolyData.
"""

import numpy as np
import pytest

from copy import deepcopy
from thewalrus.random import random_covariance

from mrmustard.lab.representations.data.qpoly_data import QPolyData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory
# from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData

np.random.seed(42)
D = 10 #dimension, matrix will be DxD while means will be D
N = 3 #number of elements in the batch

#########   Instantiating class to test  #########
@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return QPolyData


@pytest.fixture
def A() -> Matrix:
    r"""Some matrix for the object's parameterization."""
    return [random_covariance(D) for _ in range(N)]#np.random.rand(N,D,D)


@pytest.fixture
def B() -> Vector:
    r"""Some vector for the object's parameterization."""
    return np.random.rand(N,D)


@pytest.fixture
def C() -> Scalar:
    r"""Some scalar for the object's parameterization."""
    return np.random.rand(N, 1)


@pytest.fixture
def PARAMS(A, B, C) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {"A": A, "b": B, "c": C}
    return params_dict


@pytest.fixture()
def DATA(TYPE, PARAMS) -> QPolyData:
    r"""Instance of the class that must be tested."""
    return general_factory(TYPE, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> QPolyData:
    r"""Another instance of the class that must be tested."""
    return deepcopy(DATA)


class TestQPolyData(): #TODO re-add inheritance
    ####################  Init  ######################
    def test_non_symmetric_matrix_raises_ValueError(self, B, C):
        non_symmetric_mat = np.eye(10)  # TODO factory method for this
        non_symmetric_mat[0] += np.array(range(10))
        with pytest.raises(ValueError):
            QPolyData([non_symmetric_mat], B, C)

    def non_real_matrix_raises_ValueError(self, B, C):
        non_symmetric_mat = np.eye(10)  # TODO factory method for this
        non_symmetric_mat[0] += np.array(range(10))
        non_symmetric_complex_mat = 1j * non_symmetric_mat
        with pytest.raises(ValueError):
            QPolyData([non_symmetric_complex_mat], B, C)

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

    ###############  Multiplication  #################
    # def test_object_mul_result_has_correct_number_of_A_and_b_elements(self,DATA, OTHER):
    #     post_op_result = DATA * OTHER
    #     nb_combined_mats = post_op_result.A.shape[0]
    #     nb_combined_vectors = post_op_result.b.shape[0]
    #     nb_combined_constants = post_op_result.c.shape[0]
    #     assert nb_combined_mats == nb_combined_vectors == nb_combined_constants == N*N


    # @pytest.mark.parametrize("x", [2, 7, 100])
    # def test_object_mul_adds_matrices_and_vectors_and_multiplies_coeffs_over_cartesian_prod(
    #     self, DATA, A, B, C, TYPE, x
    # ):
    #     other_a = deepcopy(A) * x
    #     other_b = deepcopy(B) * x
    #     other_c = deepcopy(C) * x
    #     other_params = {"A": other_a, "b": other_b, "c": other_c}
    #     other_data = general_factory(TYPE, **other_params)
    #     result_data_object = DATA * other_data
    #     assert np.allclose(other_a + DATA.A, result_data_object.A)
    #     assert np.allclose(other_b + DATA.b, result_data_object.b)
    #     assert np.allclose(other_c * DATA.c, result_data_object.c)

    ###############  Outer product  ##################
    # NOTE : not implemented => not tested
