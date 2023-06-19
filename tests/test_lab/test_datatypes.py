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


class TestArrayDataAlgebra():

    ##################################################
    #########   Common to different methods  #########
    ##################################################
    def test_new_object_created_by_method_has_same_shape_as_old_object():
        # neg, add, sub, truediv, mul, simplify
        pass

    def test_TypeError_is_raised_if_other_object_has_no_array_attribute():
        # eq, add, sub, truediv, mul, and
        pass

    def test_TypeError_is_raised_if_other_is_not_scalar():
        # truediv, mul
        pass


    ##################################################
    ####################  Init  ######################
    ##################################################
    
    def test_array_given_as_init_param_is_same_as_attribute():
        pass

    def test_cutoffs_are_correct_for_array_dimensions_2D_to_4D():
        pass


    ##################################################
    ##################  Negative  ####################
    ##################################################

    def test_negative_returns_new_object_with_element_wise_negative_of_array():
        pass

    ##################################################
    ##################  Equality  ####################
    ##################################################

    def test_eq_with_same_arrays_returns_true_for_2D_to_4D():
        # test for at least 2-3-4 dimensional arrays
        pass

    def test_eq_with_different_arrays_returns_false_for_2D_to_4D():
        # test for at least 2-3-4 dimensional arrays
        pass

    ##################################################
    ##################  Addition  ####################
    ##################################################

    def test_add_returns_new_object_with_element_wise_addition_performed():
        pass

    ##################################################
    ################  Subtraction  ###################
    ##################################################

    def test_sub_returns_new_object_with_element_wise_subtraction_performed():
        pass


    ##################################################
    #############  Scalar division  ##################
    ##################################################

    def test_truediv_returns_new_object_with_element_wise_division_performed():
        pass

    ##################################################
    ###########  Scalar multiplication  ##############
    ##################################################

    def test_mul_returns_new_object_with_element_wise_multiplication_performed():
        pass

    ##################################################
    ###############  Outer product  ##################
    ##################################################

    #TODO : write tests for and


    

class TestMatVecDataAlgebra():
    pass

class TestWavefunctionArrayData(TestArrayDataAlgebra):
    pass

class TestGaussianDataAlgebra(TestMatVecDataAlgebra):
    pass

class TestQPolyDataAlgebra(TestMatVecDataAlgebra):
    pass

class TestSymplecticdata(TestMatVecDataAlgebra):
    pass







def test_polyQ_from_gaussian_from_polyQ_is_same_as_polyQ():
    pass

def test_gaussian_from_polyQ_from_gaussian_is_same_as_gaussian():
    pass

def test_operating_on_wavefunctions_with_different_qs_raises_ValueError():
    pass


