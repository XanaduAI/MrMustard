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
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from tests.test_lab.test_representations.test_data.mock_data import (MockData, 
                                                                     MockCommonAttributesObject,
                                                                     MockNoCommonAttributesObject)
from tools_for_tests import everything_except

DATA = MockData()

class TestData():

    #########   Common to different methods  #########
    @given(other = everything_except( (int, float, complex) ))
    def test_truediv_raises_TypeError_if_divisor_is_not_scalar(self, other):
        with pytest.raises(TypeError):
            DATA / other


    @given(other = st.from_type(MockNoCommonAttributesObject))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv, op.eq, op.and_])
    def test_algebraic_op_raises_TypeError_if_other_object_has_different_attributes(self, 
                                                                                     other,
                                                                                     operator):
        with pytest.raises(TypeError):
            operator(DATA, other)


    @given(other = st.from_type(MockData))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv, op.eq, op.and_])
    def test_data_object_is_left_untouched_after_applying_operation_of_arity_two(self, 
                                                                                 other, 
                                                                                 operator):
        
        _ = operator(DATA, other)
        assert DATA == DATA

    
    @pytest.mark.parametrize("operator", [op.neg])
    def test_data_object_is_left_untouched_after_applying_negation(self, operator):
        _ = operator(DATA)
        assert DATA == DATA


    @given(other = st.from_type(MockData))
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul, op.truediv])
    def test_new_object_created_by_method_has_same_attribute_shapes_as_old_object(self,
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

    


    ####################  Init  ######################

    def test_when_arguments_given_attribute_values_match_them(self):
        pass


    ##################  Equality  ####################
    def test_when_all_attributes_are_equal_objects_are_equal(self):
        # NOTE : see https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
        # to iterate over attributes 
        pass