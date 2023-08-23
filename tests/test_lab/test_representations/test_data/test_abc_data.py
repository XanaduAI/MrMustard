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

The fixtures must correspond to the concrete class being tested, here ABCData.
"""

import numpy as np
import operator as op
import pytest

from copy import deepcopy

from mrmustard.lab.representations.data.abc_data import ABCData
from mrmustard.typing import Batch, Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory
from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData

np.random.seed(42)


#########   Instantiating class to test  #########
@pytest.fixture
def D() -> int:
    """The dimension: matrices will be DxD and vectors will be D."""
    return 5


@pytest.fixture
def N() -> int:
    """The number of elements in the batch."""
    return 3


@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return ABCData


@pytest.fixture
def A(D, N) -> Batch[Matrix]:
    r"""Some batch of matrices for the object's parameterization."""
    A = np.random.normal(size=(N, D, D)) + 1j * np.random.normal(size=(N, D, D))
    return A + A.transpose((0, 2, 1))  # symmetrize As


@pytest.fixture
def B(D, N) -> Batch[Vector]:
    r"""Some batch of vectors for the object's parameterization."""
    return np.random.normal(size=(N, D)) + 1j * np.random.normal(size=(N, D))


@pytest.fixture
def C(N) -> Batch[Scalar]:
    r"""Some batch of scalars for the object's parameterization."""
    return np.random.normal(size=N) + 1j * np.random.normal(size=N)


@pytest.fixture
def PARAMS(A, B, C) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {"A": A, "b": B, "c": C}
    return params_dict


@pytest.fixture()
def DATA(TYPE, PARAMS) -> ABCData:
    r"""Instance of the class that must be tested."""
    return general_factory(TYPE, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> ABCData:
    r"""Another instance of the class that must be tested."""
    return deepcopy(DATA)


@pytest.fixture()
def X(D) -> ABCData:
    r"""Generates a random complex X-vector of the correct dimension."""
    return np.random.normal(size=D) + 1j * np.random.normal(size=D)


class TestABCData(TestMatVecData):  # TODO re-add inheritance
    ####################  Init  ######################
    def test_non_symmetric_matrix_raises_ValueError(self, B, C):
        non_symmetric_mat = np.eye(10)  # TODO factory method for this
        non_symmetric_mat[0] += np.array(range(10))
        with pytest.raises(ValueError):
            ABCData([non_symmetric_mat], B, C)

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
    def test_object_mul_result_has_correct_number_of_A_and_b_elements(self, DATA, OTHER, N):
        post_op_result = DATA * OTHER
        nb_combined_mats = post_op_result.A.shape[0]
        nb_combined_vectors = post_op_result.b.shape[0]
        nb_combined_constants = post_op_result.c.shape[0]
        assert nb_combined_mats == nb_combined_vectors == nb_combined_constants == N * N

    def test_result_of_abc_objects_multiplication_is_correct(self, DATA, OTHER, X):
        our_operation = (DATA * OTHER).value(X)
        manual_operation = DATA.value(X) * OTHER.value(X)
        assert np.isclose(our_operation, manual_operation)
