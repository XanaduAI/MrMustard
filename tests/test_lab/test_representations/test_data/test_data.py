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
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from tests.test_lab.test_representations.test_data.mock_data import MockData, MockNoCommonAttributeObject
from tools_for_tests import everything_except

DATA = MockData()

class TestData():

    #########   Common to different methods  #########
    @given(x=everything_except( (int, float, complex) ))
    def test_truediv_raises_TypeError_if_other_is_not_scalar(self, x):
        with pytest.raises(TypeError):
            DATA / x

    def test_TypeError_is_raised_if_other_object_doesnt_have_same_attributes(self):
        # eq, add, sub, mul, and
        pass

    def test_new_object_created_by_method_has_same_attribute_shapes_as_old_object(self):
        # neg, add, sub, truediv, rmul, mul, simplify
        # for each object checked, check all its attributes and check that the other has the same ones
        pass


    ####################  Init  ######################

    def test_when_arguments_given_attribute_values_match_them(self):
        pass


    ##################  Equality  ####################
    def test_when_all_attributes_are_equal_objects_are_equal(self):
        # NOTE : see https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
        # to iterate over attributes 
        pass
