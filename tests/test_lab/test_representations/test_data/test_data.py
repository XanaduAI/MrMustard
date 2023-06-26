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
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from tests.test_lab.test_representations.test_data.mock_data import (MockData, 
                                                                     MockCommonAttributesObject,
                                                                     MockNoCommonAttributesObject)
from tools_for_tests import everything_except, factory


#########   Instantiating class to test  #########
PARAMS_LIST = ['mat', 'vec', 'coeffs', 'array', 'cutoffs']

@pytest.fixture(params=PARAMS_LIST)
def DATA(params):
    r""" Instance of the class that must be tested.
    
    Note that this fixture must be modified to each child class in the subsequent tests.
    """
    params_dict = dict.fromkeys(params)
    return factory(MockData, **params_dict)


class TestData():

    #########   Common to different methods  #########
    @pytest.mark.parametrize("operator", [op.neg])
    def test_data_object_is_left_untouched_after_applying_negation(self, DATA, operator):
        pre_op_data = deepcopy(DATA)
        _ = operator(pre_op_data)
        assert DATA == pre_op_data


    @given(other = st.from_type(MockData))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv, op.eq, op.and_])
    def test_data_object_is_left_untouched_after_applying_operation_of_arity_two(self, 
                                                                                 DATA,
                                                                                 other, 
                                                                                 operator):
        pre_op_data = deepcopy(DATA)
        _ = operator(pre_op_data, other)
        assert DATA == DATA


    @given(other = everything_except( (int, float, complex) ))
    def test_truediv_raises_TypeError_if_divisor_is_not_scalar(self, DATA, other):
        with pytest.raises(TypeError):
            DATA / other


    @given(other = st.from_type(MockNoCommonAttributesObject))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv, op.eq, op.and_])
    def test_algebraic_op_raises_TypeError_if_other_object_has_different_attributes(self, 
                                                                                     DATA,
                                                                                     other,
                                                                                     operator):
        with pytest.raises(TypeError):
            operator(DATA, other)


    @given(other = st.from_type(MockData))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_new_object_created_by_arity2_operation_has_same_attribute_shapes_as_old_object(self,
                                                                                  DATA,
                                                                                  other,
                                                                                  operator):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        for k in DATA.__dict__.keys():
            new_data = operator(DATA, other)
            try: # works for all numpy array attributes
                assert getattr(DATA, k).shape == getattr(new_data, k).shape
            except AttributeError: # works for scalar attributes
                assert getattr(DATA, k) == getattr(new_data, k)


    @given(other = st.from_type(MockData))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_new_object_created_by_negation_has_same_attribute_shapes_as_old_object(self,
                                                                                    DATA,
                                                                                    operator):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        for k in DATA.__dict__.keys():
            new_data = operator(DATA)
            try: # numpy array attributes
                assert getattr(DATA, k).shape == getattr(new_data, k).shape
            except AttributeError: # scalar attributes
                pass               

    
    ####################  Init  ######################

    def test_when_arguments_given_attribute_values_match_them(self, DATA):
        pass


    ##################  Equality  ####################
    @given(other = deepcopy(DATA))
    def test_when_all_attributes_are_equal_objects_are_equal(self, DATA, other):
        for k in DATA.__dict__.keys():
            assert getattr(DATA, k) == getattr(other, k)