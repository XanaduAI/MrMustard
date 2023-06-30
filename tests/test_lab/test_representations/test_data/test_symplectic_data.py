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

class TestSymplecticData(TestMatVecDataAlgebra):
    
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