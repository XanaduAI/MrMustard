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
""" This module contains helper functions used in the tests for representations.
"""

import numpy as np

all = [ 'helper_coeffs_are_computed_correctly',
       'helper_mat_vec_unchanged_computed_coeffs_are_correct']

def helper_coeffs_are_computed_correctly(new_data_object, old_data_object, operator, x
                                              ) -> None:
    r""" Helper assert function which ensures the coefficients are computed correctly.

    Based on the given operator and a scalar, this test ensures that the coefficients are 
    applied the element-wise operation.

    Args:
        new_data_object: the data object that was created by the operation
        old_data_object: the initial data object before operation
        operator:        the operator which should be applied, either + or *
        x:               the item by which to multiply or add the coefficients

    Returns:
        None (performs the assert)
    """
    manually_computed_coeffs = operator(old_data_object.coeffs, x)
    assert np.allclose(new_data_object.coeffs, manually_computed_coeffs)

def helper_mat_vec_unchanged_computed_coeffs_are_correct(
                                                              new_data_object, 
                                                              old_data_object, 
                                                              operator,
                                                              x
                                                              ) -> None:
    r""" Ensures the matrix and vector remian unchanged while the coefficients are updated.

    Args:
        new_data_object: the data object that was created by the operation
        old_data_object: the initial data object before operation
        operator:        the operator which should be applied, either + or *
        x:               the item by which to multiply or add the coefficients
    """
    helper_coeffs_are_computed_correctly(new_data_object, old_data_object, operator, x)
    assert np.allclose(new_data_object.mat, old_data_object.mat)
    assert np.allclose(new_data_object.vec, old_data_object.vec)







