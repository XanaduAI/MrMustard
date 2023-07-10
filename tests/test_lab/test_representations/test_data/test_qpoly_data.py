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

import numpy as np
import pytest
from copy import deepcopy
from mrmustard.lab.representations.data.qpoly_data import QPolyData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory

#from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData

#########   Instantiating class to test  #########
@pytest.fixture
def TYPE():
    return QPolyData

@pytest.fixture
def A() -> Matrix:
    return np.eye(10) * 42

@pytest.fixture
def B() -> Vector:
    return np.ones(10) * 42

@pytest.fixture
def C() -> Scalar:
    return 42

@pytest.fixture
def PARAMS(A, B, C) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {'A': A, 'b': B, 'c':C}
    return params_dict


@pytest.fixture()
def DATA(TYPE, PARAMS) -> QPolyData:
    r"""Instance of the class that must be tested."""
    return general_factory(TYPE, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> QPolyData:
    r"""Another instance of the class that must be tested."""
    return deepcopy(DATA)

class TestQPolyData(): #TestMatVecData
    
    ####################  Init  ######################
    def test_non_symmetric_matrix_raises_ValueError(self, B, C):
        non_symmetric_mat = np.eye(10) #TODO factory method for this
        non_symmetric_mat[0] += np.array(range(10))
        with pytest.raises(ValueError):
            QPolyData(non_symmetric_mat, B, C)


    def non_real_matrix_raises_ValueError(self, B, C):
        non_symmetric_mat = np.eye(10) #TODO factory method for this
        non_symmetric_mat[0] += np.array(range(10))
        non_symmetric_complex_mat = 1j * non_symmetric_mat
        with pytest.raises(ValueError):
            QPolyData(non_symmetric_complex_mat, B, C)


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
    # @pytest.mark.parametrize('x', [2])
    # def test_object_mul_adds_matrices_element_wise_and_multiplies_coeffs(self, DATA, A, B, C, TYPE, x):
    #     other_a = deepcopy(A) * x
    #     other_b = deepcopy(B) * x
    #     other_c = deepcopy(C) * x
    #     other_params = {'A': other_a, 'b': other_b, 'c': other_c}
    #     other_data = general_factory(TYPE, **other_params)
    #     result_data_object = DATA * other_data

    #     assert (np.allclose(other_a + DATA.A, result_data_object))

        # manual_result_a = other_a + DATA.A
        # manual_result_b = other_b + DATA.b
        # manual_result_c = other_c + DATA.c
        # results_params = {'A': manual_result_a, 'b': manual_result_b, 'c': manual_result_c}
        # manually_created_result_object = general_factory(TYPE, **results_params)

        # object_computation_result = DATA * other_data
        
        # assert manually_created_result_object == object_computation_result


    def test_object_mul_adds_vectors_element_wise(self):
        pass

    ###############  Outer product  ##################
    # NOTE : not implemented => not tested

   