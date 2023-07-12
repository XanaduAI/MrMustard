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

The fixtures must correspond to the concrete class being tested, here WavefunctionArrayData.
"""

import numpy as np
import operator as op
import pytest

from copy import deepcopy

from mock_data import MockNoCommonAttributesObject
from mrmustard.lab.representations.data.wavefunctionarray_data import (
    WavefunctionArrayData,
)
from mrmustard.utils.misc_tools import general_factory
from tests.test_lab.test_representations.test_data.test_array_data import TestArrayData


#########   Instantiating class to test  #########
@pytest.fixture
def TYPE():
    r"""Type of the object under test."""
    return WavefunctionArrayData


@pytest.fixture
def PARAMS() -> dict:
    r"""Parameters for the class instance which is created."""
    params = {"qs": np.ones(10), "array": np.ones(10)}
    return params


@pytest.fixture()
def DATA(PARAMS) -> WavefunctionArrayData:
    r"""Instance of the class that must be tested."""
    return general_factory(WavefunctionArrayData, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> WavefunctionArrayData:
    r"""Alternative instance of the class to be tested."""
    return deepcopy(DATA)


class TestWavefunctionArrayData(TestArrayData):
    r"""Class for tests of the WavefunctionArrayData class, inherits from parent tests."""

    #########   Common to different methods  #########
    @pytest.mark.parametrize(
        "other",
        [
            WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10)),
            WavefunctionArrayData(qs=np.zeros(10), array=np.zeros(10)),
            WavefunctionArrayData(qs=np.zeros(9), array=np.ones(10)),
            WavefunctionArrayData(qs=np.zeros(8), array=np.zeros(10)),
            WavefunctionArrayData(qs=np.eye(8), array=np.zeros(10)),
        ],
    )
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])
    def test_different_value_or_shape_of_qs_raises_ValueError(self, DATA, other, operator):
        with pytest.raises(ValueError):
            operator(DATA, other)

    @pytest.mark.parametrize("other", [WavefunctionArrayData(qs=np.ones(10), array=np.ones(10))])
    @pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])  # op.and_
    def test_qs_for_new_objects_are_same_as_initial_qs_after_arity2_operation(
        self, DATA, other, operator
    ):
        new_obj = operator(DATA, other)
        assert np.allclose(DATA.qs, new_obj.qs)

    def test_qs_for_new_objects_are_same_as_initial_qs_after_negation(self, DATA):
        new_obj = -DATA
        assert np.allclose(DATA.qs, new_obj.qs)

    ####################  Init  ######################
    # NOTE : tested in parent

    ##################  Equality  ####################
    @pytest.mark.parametrize(
        "other, truth_val",
        [
            (WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10)), False),
            (WavefunctionArrayData(qs=np.zeros(10), array=np.zeros(10)), False),
        ],
    )
    def test_eq_returns_false_if_qs_different_irrespective_of_array(self, DATA, other, truth_val):
        assert (DATA == other) == truth_val

    ###########  Object multiplication  ##############
    # NOTE : tested in parent

    ###############  Outer product  ##################
    # NOTE : not implemented => not tested

    #################### Other #######################
    @pytest.mark.parametrize(
        "other, truth_val",
        [
            (WavefunctionArrayData(qs=np.ones(10), array=np.zeros(10)), True),
            (WavefunctionArrayData(qs=np.ones(10), array=np.ones(10)), True),
            (WavefunctionArrayData(qs=np.zeros(10), array=np.zeros(10)), False),
            (WavefunctionArrayData(qs=np.zeros(10), array=np.ones(10)), False),
        ],
    )
    def test_qs_is_same_returns_true_when_same_qs_and_false_when_diff_qs_irrespective_of_array(
        self, DATA, other, truth_val
    ):
        assert DATA._qs_is_same(other) == truth_val

    @pytest.mark.parametrize("other", [MockNoCommonAttributesObject()])
    def test_qs_is_same_raises_TypeError_if_other_has_no_qs(self, DATA, other):
        with pytest.raises(TypeError):
            DATA._qs_is_same(other)
