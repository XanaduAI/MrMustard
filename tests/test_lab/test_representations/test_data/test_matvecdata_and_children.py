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



class TestMatVecDataAlgebra(TestData):
    
    #########   Common to different methods  #########


    ####################  Init  ######################

    def if_coeffs_not_given_they_are_equal_to_1():
        pass

    ##################  Negative  ####################

    def test_negative_returns_new_object_with_same_matrix():
        pass

    def test_negative_returns_new_object_with_same_vector():
        pass

    def test_negative_returns_new_object_withnew_coeffs_with_element_wise_neg():
        pass


    ##################  Equality  ####################

    def test_with_at_least_one_different_among_matrix_vector_coeffs_returns_false():
        pass


    ##################  Addition  ####################
    # TODO: more complex tests of the general concat case!

    def test_if_mat_and_vec_are_same_add_only_modifies_coeffs():
        pass

    def test_if_mat_and_vec_are_same_add_keeps_them_same_in_new_object():
        pass


    ################  Subtraction  ###################
    # NOTE : indirectly via add and neg

    #############  Scalar division  ##################

    def test_truediv__returns_new_object_which_leaves_mat_and_vec_untouched():
        pass

    def test_truediv_returns_new_object_with_coeffs_element_wise_divided():
        pass

    ###########  Scalar multiplication  ##############
    # NOTE : part of mul, tested in children classes

    ###############  Outer product  ##################
    # TODO: write tests


class TestGaussianDataAlgebra(TestMatVecDataAlgebra):

    #########   Common to different methods  #########
    ####################  Init  ######################
    ##################  Negative  ####################
    ##################  Equality  ####################
    ##################  Addition  ####################
    ################  Subtraction  ###################
    #############  Scalar division  ##################
    ###########  Scalar multiplication  ##############
    ###############  Outer product  ##################


class TestQPolyDataAlgebra(TestMatVecDataAlgebra):
    
    #########   Common to different methods  #########
    ####################  Init  ######################
    ##################  Negative  ####################
    ##################  Equality  ####################
    ##################  Addition  ####################
    ################  Subtraction  ###################
    #############  Scalar division  ##################
    ###########  Scalar multiplication  ##############
    ###############  Outer product  ##################


class TestSymplecticdata(TestMatVecDataAlgebra):
    
    #########   Common to different methods  #########
    ####################  Init  ######################
    ##################  Negative  ####################
    ##################  Equality  ####################
    ##################  Addition  ####################
    ################  Subtraction  ###################
    #############  Scalar division  ##################
    ###########  Scalar multiplication  ##############
    ###############  Outer product  ##################