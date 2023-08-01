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
""" This class corresponds to test child class for the MatVecData class.

Unlike some of its -abstract- parent test classes, this class is meant to be run with pytest.

Check parents test classe-s for more details on the rationale.

The fixtures must correspond to the concrete class being tested, here SymplecticData.
"""

import numpy as np
import operator as op
import pytest

from copy import deepcopy

from mrmustard.lab.representations.data.symplectic_data import SymplecticData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.utils.misc_tools import general_factory
from tests.test_lab.test_representations.test_data.test_matvec_data import TestMatVecData
from tests.test_lab.test_representations.test_data.tools_for_tests import (
    helper_mat_vec_unchanged_computed_coeffs_are_correct,
)


#########   Instantiating class to test  #########
@pytest.fixture
def D() -> int:
    """The dimension: matrices will be DxD and vectors will be D."""
    return 4

@pytest.fixture
def N() -> int:
    """The number of elements in the batch."""
    return 3

@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return SymplecticData


@pytest.fixture
def SYMPLECTIC(N) -> Matrix: #TODO: generator for symplectic matrices
    r"""A symplectic matrix used for object parameterization.
    Taken from https://mathworld.wolfram.com/SymplecticGroup.html
    """
    mats = []
    for _ in range(N):
        m = np.array([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ])
        mats.append(m)
    return np.array(mats)


@pytest.fixture
def DISPLACEMENT(N,D) -> Vector:
    r"""Some vector for the object's parameterization."""
    return np.array([np.random.normal(size=D) + 1j*np.random.normal(size=D) for _ in range(N)])


@pytest.fixture
def COEFFS(N) -> Scalar:
    r"""Some scalar for the object's parameterization."""
    return np.array([np.random.normal() + 1j*np.random.normal() for _ in range(N)])


@pytest.fixture
def PARAMS(SYMPLECTIC, DISPLACEMENT, COEFFS) -> dict:
    r"""Parameters for the class instance which is created."""
    params_dict = {"symplectic": SYMPLECTIC, "displacement": DISPLACEMENT, "coeffs": COEFFS}
    return params_dict


@pytest.fixture()
def DATA(TYPE, PARAMS) -> SymplecticData:
    r"""Instance of the class that must be tested."""
    return general_factory(TYPE, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> SymplecticData:
    r"""Another instance of the class that must be tested."""
    return deepcopy(DATA)


class TestSymplecticData(TestMatVecData):
    ####################  Init  ######################
    def test_init_with_a_non_symplectic_matrix_raises_ValueError(self, DISPLACEMENT, COEFFS):
        non_symplectic_mat = np.eye(10)  # TODO factory method for this
        non_symplectic_mat[0] += np.array(range(10))
        with pytest.raises(ValueError):
            SymplecticData([non_symplectic_mat], DISPLACEMENT, COEFFS)

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

    def test_mul_raises_TypeError_with_object(self, DATA, OTHER):
        with pytest.raises(TypeError):
            OTHER * DATA

    @pytest.mark.parametrize("x", [0, 2, 10, 2500])
    def test_mul_with_scalar_multiplies_coeffs_and_leaves_mat_and_vec_unaltered(self, DATA, x):
        pre_op_data = deepcopy(DATA)
        post_op_data = DATA * x
        helper_mat_vec_unchanged_computed_coeffs_are_correct(post_op_data, pre_op_data, op.mul, x)

    ###############  Outer product  ##################
    # NOTE : not implemented => not tested
