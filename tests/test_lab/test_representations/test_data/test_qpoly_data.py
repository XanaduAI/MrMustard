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