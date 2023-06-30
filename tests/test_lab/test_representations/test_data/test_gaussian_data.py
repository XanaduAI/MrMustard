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

class TestGaussianData(TestMatVecDataAlgebra):

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