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

    def test_negative_returns_new_object_with_same_matrix_and_vector():
        pass

    def test_negative_returns_new_object_with_with_element_wise_neg_on_coeffs():
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
    # NOTE : tested via add and neg

    #############  Scalar division  ##################

    def test_truediv_returns_new_object_with_same_mat_and_vec():
        pass

    def test_truediv_returns_new_object_with_coeffs_element_wise_divided():
        pass


    ###############  Multiplication  ##################

    def test_new_object_resulting_from_mul_has_same_shape():
        pass

    def test_object_mul_when_matrix_and_vector_are_same_coeffs_get_multiplied():
        pass

    def test_object_mul_when_matrix_and_vector_are_same_only_coeffs_get_multiplied():
        pass

    def test_scalar_mul_multiplies_coeffs():
        pass

    def test_scalar_mul_only_multiplies_coeffs():
        pass


    ###############  Outer product  ##################
    # TODO: write tests




class TestGaussianDataAlgebra(TestMatVecDataAlgebra):

    ####################  Init  ######################

    def test_defining_neither_cov_nor_mean_raises_ValueError():
        pass

    def test_if_coeffs_is_undefined_it_is_equal_to_1():
        pass

    def test_if_cov_is_none_then_initialized_at_npeye_of_correct_shape():
        pass

    def test_non_symplectic_covariance_raises_ValueError():
        pass

    #NOTE : these do not test edge cases where someone feeds just [0] as means, it only guarantees
    #  means is not empty. Do we want to secure edge cases?

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

    ##############  Multiplication  ##################

    def test_if_given_scalar_mul_multiplies_coeffs():
        pass

    def test_if_given_scalar_mul_does_not_multiply_anything_else_than_coeffs():
        pass

    # TODO : test compute_mul_covs
    # TODO : test compute_mul_coeffs
    # TODO : test compute_mul_means

    ###############  Outer product  ##################
    # NOTE : not implemented so no test

    



class TestQPolyDataAlgebra(TestMatVecDataAlgebra):
    
    ####################  Init  ######################

    def test_non_symmetric_matrix_raises_ValueError():
        pass

    def non_real_matrix_raises_ValueError():
        pass

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

    def test_object_mul_adds_matrices_element_wise():
        pass

    def test_object_mul_adds_vectors_element_wise():
        pass

    ###############  Outer product  ##################
    # NOTE : not implemented so no test





class TestSymplecticdata(TestMatVecDataAlgebra):
    
    ####################  Init  ######################

    def test_init_without_coeffs_has_coeffs_equal_to_1():
        pass

    def test_init_with_a_non_symplectic_matrix_raises_ValueError():
        pass

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
    
    def test_mul_raises_TypeError_with_object():
        pass

    def test_mul_with_scalar_multiplies_coeffs():
        pass

    def test_mul_with_scalar_only_multiplies_coeffs():
        pass

    ###############  Outer product  ##################
    # NOTE : not implemented so no test