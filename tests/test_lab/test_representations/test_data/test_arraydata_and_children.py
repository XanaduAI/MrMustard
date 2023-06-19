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
from tests.test_lab.test_representations.test_data.test_data import TestData



class TestArrayDataAlgebra(TestData):

    
    #########   Common to different methods  #########


    ####################  Init  ######################

    def test_cutoffs_are_correct_for_array_dimensions_2D_to_4D():
        pass


    ##################  Negative  ####################

    def test_negative_returns_new_object_with_element_wise_negative_of_array():
        pass

    
    ##################  Equality  ####################

    # NOTE: tested in parent class

    
    ##################  Addition  ####################

    def test_add_returns_new_object_with_element_wise_addition_performed():
        pass


    ################  Subtraction  ###################

    def test_sub_returns_new_object_with_element_wise_subtraction_performed():
        pass
    

    #############  Scalar division  ##################

    def test_truediv_returns_new_object_with_element_wise_division_performed():
        pass


    ###########  Scalar multiplication  ##############

    def test_mul_returns_new_object_with_element_wise_multiplication_performed():
        pass


    ###############  Outer product  ##################
    
    #TODO : write tests for and



class TestWavefunctionArrayData(TestArrayDataAlgebra):

    #########   Common to different methods  #########

    def test_ValueError_is_raised_if_qs_are_different():
        # test on add, sub, mul, and
        pass

    def test_qs_for_new_objects_are_same_as_initial_qs():
        # test on add, sub, 
        pass


    ####################  Init  ######################

    def test_qs_attribute_is_same_as_given_in_arguments():
        pass
    
    ##################  Equality  ####################

    def test_eq_returns_false_if_array_same_but_qs_different():
        pass

    ###########  Object multiplication  ##############

    def test_mul_returns_new_object_with_array_being_element_wise_mul_of_the_two_objects():
        pass

    ###############  Outer product  ##################

    # TODO : test and
    
    #################### Other #######################

    def test_qs_is_same_returns_true_when_same_qs():
        pass

    def test_qs_is_same_returns_false_when_different_qs():
        pass

    def test_qs_is_same_raises_Typeerror_if_other_has_no_qs():
        pass