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



class TestData():

    #########   Common to different methods  #########

    def test_TypeError_is_raised_if_other_is_not_scalar():
        # truediv
        pass

    def test_TypeError_is_raised_if_other_object_doesnt_have_same_attributes():
        # eq, add, sub, mul, and
        pass

    def test_new_object_created_by_method_has_same_attribute_shapes_as_old_object():
        # neg, add, sub, truediv, rmul, mul, simplify
        # for each object checked, check all its attributes and check that the other has the same ones
        pass


    ####################  Init  ######################

    def test_when_arguments_given_attribute_values_match_them():
        pass


    ##################  Equality  ####################
     def test_when_all_attributes_are_equal_objects_are_equal():
        # NOTE : see https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
        # to iterate over attributes 
        pass
