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
import operator as op
import pytest
from copy import deepcopy
from tests.test_lab.test_representations.test_data.mock_data import (MockData, 
                                                                     MockCommonAttributesObject,
                                                                     MockNoCommonAttributesObject)
from tests.test_lab.test_representations.test_data.test_data import TestData
from mrmustard.lab.representations.data.array_data import ArrayData
from tools_for_tests import factory


#########   Instantiating class to test  #########

@pytest.fixture
def PARAMS() -> dict:
    r""" Parameters for the class instance which is created. """
    params_dict = {'array' : np.ones(10)}
    return params_dict


@pytest.fixture()
def DATA(PARAMS) -> ArrayData:
    r""" Instance of the class that must be tested. """
    return factory(ArrayData, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> ArrayData:
    r""" Another instance of the class that must be tested. """
    return deepcopy(DATA)



class TestArrayData(TestData):      

    ##################  Negative  ####################
    def test_negative_returns_new_object_with_element_wise_negative_of_array(self, DATA):
        new_data = deepcopy(DATA)
        neg_data = - new_data #neg of the object
        broadcast_neg_array = - DATA.array #manual broadcast of neg
        assert np.allclose(neg_data.array, broadcast_neg_array)


    ##################  Equality  ####################
    # NOTE: tested in parent class


    # #############  Arity-2 operations  ################
    # # Addition, subtraction and multiplication all go here
    # @pytest.mark.parametrize("other", [MockData(array=np.ones(10))])
    # @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])
    # def test_arity2_operation_returns_element_wise_operation_on_array(self, DATA, other, operator):
    #     res_from_object_op = operator(DATA, other)
    #     res_from_manual_op = operator(DATA.array, other.array)
    #     assert np.allclose(res_from_object_op.array, res_from_manual_op)
    
    ##################  Addition  ####################
    # NOTE: tested above

    ################  Subtraction  ###################
    # NOTE: tested above
    
    # #############  Scalar division  ##################
    @pytest.mark.parametrize("x", [2])
    def test_truediv_returns_new_object_with_element_wise_division_performed(self, DATA, x):
        res_from_object_op = DATA / x
        res_from_manual_op = DATA.array / x
        assert np.allclose(res_from_object_op.array, res_from_manual_op)


    ###############  Outer product  ##################
    #TODO : write tests