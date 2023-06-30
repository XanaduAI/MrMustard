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
""" This class corresponds to the abstract parent class for all Data objects.

This file is *NOT* meant to be run on its own with pytest but meant to be inherited by children 
test classes which will be run with pytest.

Test inheritance - why?
- - - - - - - - - - - -
Just like standard inheritance, test inheritance allows us to test for properties that are common 
throughout generations without having to use parameterization over types.
It reinforces the *Single responsibility* principle from SOLID by allowing one test to be in charge
 of testing a single behaviour/state element and not behave as a factory at the same time.
It promotes the Open-Closed principle from SOLID by making the implementation of new testing 
classes seamless. When creating a new Data class NewData which inherits from Data, then the
correspodning test class TestNewData will inherit from TestData and guarantee that everything which
held true for the parents holds true for the child.
Fianlly, Test Driven development (TDD) benefits from test inheritance as a test class can easily be
created for any new class.

Test inheritance - how?
- - - - - - - - - - - -
In order to allow for test class inheritance, a few adjustments are necessary:

1) The PARAMS, DATA and OTHER fixtures must be redefined in each child test file and adapted to 
match the specific class of the child.

2) We must accept that the instance created by OTHER is by default a deepcopy of the instance
created by the DATA fixture. Developpers are welcome to code their own versions of the OTHER 
fixture and we encourage them to do so whenever the need arises.
We however advocate against using the `mark.parametrize` fixture for instances of `other` since it 
breaks when inheriting the test. With `mark.parametrize`, the class instance will be of the type
defined in the file where the test was written, blocking resolution sequence.
"""

import operator as op
import pytest

from copy import deepcopy

from mrmustard.utils.misc_tools import general_factory
from tests.test_lab.test_representations.test_data.mock_data import (
    MockData,
    MockCommonAttributesObject,
    MockNoCommonAttributesObject,
)


#########   Instantiating class to test  #########


@pytest.fixture
def PARAMS() -> dict:
    r"""Parameters for the class instance which is created, here all are None."""
    params_list = ["mat", "vec", "coeffs", "array", "cutoffs"]
    return dict.fromkeys(params_list)


@pytest.fixture()
def DATA(PARAMS) -> MockData:
    r"""Instance of the class that must be tested, here the class is a Mock."""
    return general_factory(MockData, **PARAMS)


@pytest.fixture()
def OTHER(DATA) -> MockData:
    r"""Another instance of the class that must be tested, here again, the class is a Mock."""
    return deepcopy(DATA)


class TestData:
    r"""Parent class for testing all children of the Data class.

    Here only the behaviours common to all children are tested.
    """

    #########   Common to different methods  #########
    def test_original_data_object_is_left_untouched_after_applying_negation(self, DATA):
        pre_op_data_control = deepcopy(DATA)
        _ = -DATA
        assert DATA == pre_op_data_control

    @pytest.mark.parametrize("operator", [op.add, op.sub, op.eq, op.and_])
    def test_original_data_object_is_left_untouched_after_applying_operation_of_arity_two(
        self, DATA, OTHER, operator
    ):
        pre_op_data_control = deepcopy(DATA)
        _ = operator(DATA, OTHER)
        assert DATA == pre_op_data_control

    @pytest.mark.parametrize(
        "other", [MockData(), MockCommonAttributesObject(), deepcopy(DATA)]
    )
    def test_truediv_raises_TypeError_if_divisor_is_not_scalar(self, DATA, other):
        with pytest.raises(TypeError):
            DATA / other

    @pytest.mark.parametrize("other", [MockNoCommonAttributesObject()])
    @pytest.mark.parametrize(
        "operator", [op.add, op.sub, op.mul, op.truediv, op.eq, op.and_]
    )
    def test_algebraic_op_raises_TypeError_if_other_object_has_different_attributes(
        self, DATA, other, operator
    ):
        with pytest.raises(TypeError):
            operator(DATA, other)

    @pytest.mark.parametrize("operator", [op.add, op.sub])
    def test_new_object_created_by_arity2_operation_has_same_attribute_shapes_as_old_object(
        self, DATA, OTHER, operator
    ):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        for k in DATA.__dict__.keys():
            new_data = operator(DATA, OTHER)
            try:  # works for all numpy array attributes
                assert getattr(DATA, k).shape == getattr(new_data, k).shape
            except AttributeError:  # works for scalar attributes
                pass

    @pytest.mark.parametrize("operator", [op.neg])
    def test_new_object_created_by_negation_has_same_attribute_shapes_as_old_object(
        self, DATA, operator
    ):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        for k in DATA.__dict__.keys():
            new_data = operator(DATA)
            try:  # numpy array attributes
                assert getattr(DATA, k).shape == getattr(new_data, k).shape
            except AttributeError:  # scalar attributes
                pass

    ##################  Equality  ####################
    def test_when_all_attributes_are_equal_objects_are_equal(self, DATA):
        # NOTE: are we ok with try/except blocks in tests?
        # NOTE: are we ok with for loops in tests?
        other = deepcopy(DATA)
        for k in DATA.__dict__.keys():
            getattr(other, k)
            try:  # non-array, non-list attributes
                assert getattr(DATA, k) == getattr(other, k)
            except ValueError:
                assert all(getattr(DATA, k) == getattr(other, k))
        assert DATA == other
