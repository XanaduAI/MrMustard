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
from tools_for_tests import factory


#########   Instantiating class to test  #########

@pytest.fixture
def PARAMS() -> dict:
    r""" Parameters for the class instance which is created.

    Returns:
        A dict with the parameter names as keys and their associated values.
    
    """
    params_list = ['mat', 'vec', 'coeffs', 'array', 'cutoffs']
    return dict.fromkeys(params_list)


@pytest.fixture()
def DATA(PARAMS) -> MockData:
    r""" Instance of the class that must be tested.
    
    Note that this fixture must be modified to match each child class in the subsequent tests.
    """
    return factory(MockData, **PARAMS)


class TestData():

    #########   Common to different methods  #########
    def test_original_data_object_is_left_untouched_after_applying_negation(self, DATA):
        pre_op_data = deepcopy(DATA)
        _ = - DATA
        #iterate over all elements in the 
        assert DATA == pre_op_data


    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.eq, op.and_])
    @pytest.mark.parametrize("other", [MockData()])
    def test_original_data_object_is_left_untouched_after_applying_operation_of_arity_two(self,
                                                                                          DATA,
                                                                                          other, 
                                                                                          operator):
        pre_op_data = deepcopy(DATA)
        _ = operator(pre_op_data, other)
        assert DATA == DATA


    @pytest.mark.parametrize("other", [MockData(), MockCommonAttributesObject(), deepcopy(DATA)])
    def test_truediv_raises_TypeError_if_divisor_is_not_scalar(self, DATA, other):
        with pytest.raises(TypeError):
            DATA / other


    @pytest.mark.parametrize("other", [MockNoCommonAttributesObject()])
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv, op.eq, op.and_])
    def test_algebraic_op_raises_TypeError_if_other_object_has_different_attributes(self, DATA, 
                                                                                     other,
                                                                                     operator):
        with pytest.raises(TypeError):
            operator(DATA, other)


    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])
    @pytest.mark.parametrize("other", [MockData()])
    def test_new_object_created_by_arity2_operation_has_same_attribute_shapes_as_old_object(self, DATA,
                                                                                  other,
                                                                                  operator):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        for k in DATA.__dict__.keys():
            new_data = operator(DATA, other)
            try: # works for all numpy array attributes
                assert getattr(DATA, k).shape == getattr(new_data, k).shape
            except AttributeError: # works for scalar attributes
                pass


    @pytest.mark.parametrize("operator", [op.neg])
    def test_new_object_created_by_negation_has_same_attribute_shapes_as_old_object(self, DATA, operator):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        for k in DATA.__dict__.keys():
            new_data = operator(DATA)
            try: # numpy array attributes
                assert getattr(DATA, k).shape == getattr(new_data, k).shape
            except AttributeError: # scalar attributes
                pass               

    
    ####################  Init  ######################

    def test_when_arguments_given_attribute_values_match_them(self):
        pass #TODO : code this


    ##################  Equality  ####################
    #@pytest.mark.parametrize("other", [deepcopy(DATA)])
    def test_when_all_attributes_are_equal_objects_are_equal(self, DATA):
        other = deepcopy(DATA) # TODO: why this needs to be inside and not in parameterize?!?
        for k in DATA.__dict__.keys():
            getattr(other, k)
            assert getattr(DATA, k) == getattr(other, k)