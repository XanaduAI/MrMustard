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
import operator as op
import pytest

from copy import deepcopy
from functools import reduce

from mrmustard.lab.representations.data.qpoly_data import QPolyData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory
# from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData

np.random.seed(42)

#########   Instantiating class to test  #########
@pytest.fixture
def D():
    """The dimension: matrices will be DxD and vectors will be D."""
    return 5

@pytest.fixture
def N():
    """The number of elements in the batch."""
    return 3

@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return QPolyData

@pytest.fixture
def A(D,N) -> Matrix:
    r"""Some batch of matrices for the object's parameterization."""
    As = []
    for _ in range(N):
        A = np.random.normal(size=(D,D)) + 1j*np.random.normal(size=(D,D))
        A = A + A.T  # symmetrize A
        As.append(A)
    return As


@pytest.fixture
def B(D,N) -> Vector:
    r"""Some batch of vectors for the object's parameterization."""
    bs = []
    for _ in range(N):
        b = np.random.normal(size=D) + 1j*np.random.normal(size=D)
        bs.append(b)
    return bs


@pytest.fixture
def C(N) -> Scalar:
    r"""Some batch of scalars for the object's parameterization."""
    cs = []
    for _ in range(N):
        c = np.random.normal() + 1j*np.random.normal()
        cs.append(c)
    return cs


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


@pytest.fixture()
def X(D) -> QPolyData:
    r"""Generates a random complex X-vector of the correct dimension."""
    return np.random.normal(size=D) + 1j*np.random.normal(size=D)
            
class TestQPolyData(): #TODO re-add inheritance
    ####################  Init  ######################
    def test_non_symmetric_matrix_raises_ValueError(self, B, C):
        non_symmetric_mat = np.eye(10)  # TODO factory method for this
        non_symmetric_mat[0] += np.array(range(10))
        with pytest.raises(ValueError):
            QPolyData([non_symmetric_mat], B, C)

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
    def test_object_mul_result_has_correct_number_of_A_and_b_elements(self,DATA, OTHER, N):
        post_op_result = DATA * OTHER
        nb_combined_mats = post_op_result.A.shape[0]
        nb_combined_vectors = post_op_result.b.shape[0]
        nb_combined_constants = post_op_result.c.shape[0]
        assert nb_combined_mats == nb_combined_vectors == nb_combined_constants == N*N

    @pytest.mark.parametrize("operator", [op.add, op.mul])
    @pytest.mark.parametrize("Y", [np.ones(2), np.ones(5), np.ones(100)*42])
    @pytest.mark.parametrize("Z", [np.ones(5), np.ones(2), np.ones(100)*0.42])
    def test_commutative_operations_on_cartesian_product_correct_irrespective_of_nb_elements(self,
                                                                                             DATA,
                                                                                             operator,
                                                                                             Y,
                                                                                             Z):
        res_manual = [operator(y,z) for y in Y for z in Z]
        res = DATA._operate_on_all_combinations(Y,Z, operator)
        assert np.allclose(res_manual, res)

    
    def test_result_of_qpoly_objects_multiplication_is_correct(self, DATA, OTHER, X):
        result_data = DATA * OTHER
        manual_operation = (TestQPolyData.helper_exp_qpoly_batched(DATA.A, DATA.b, DATA.c, X) 
                            * TestQPolyData.helper_exp_qpoly_batched(OTHER.A, OTHER.b, OTHER.c, X))
        our_operation = TestQPolyData.helper_exp_qpoly_batched(result_data.A, 
                                                               result_data.b, 
                                                               result_data.c, 
                                                               X)
        assert np.isclose(our_operation, manual_operation)

    @staticmethod
    def helper_exp_qpoly(A,b,c,x):
            r"""Returns a coefficiented exponential evaluated at x."""
            return c * np.exp( (x @ A @ x)/2 + (x @ b))
    
    @staticmethod
    def helper_exp_qpoly_batched(A:list, b:list, c:list, x):
        r"""Returns the sum of coefficiented exponentials evaluated at x."""
        return sum(TestQPolyData.helper_exp_qpoly(Ai,bi,ci,x) for Ai,bi,ci in zip(A,b,c))