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

import pytest
from copy import deepcopy
from tests.mock_classes.general_mocks import *
from mrmustard.utils.misc_tools import *


###############################################################################
##########################   duck_type_checker   ##############################
@pytest.mark.parametrize("cls_instance", [MockAnimal(), MockFruit()])
def test_dtc_returns_true_when_given_same_object(cls_instance):
    other_instance = deepcopy(cls_instance)
    assert duck_type_checker(cls_instance, other_instance) == True


@pytest.mark.parametrize("obj_a, obj_b", [(MockAnimal(), MockAnimal()), (MockFruit(), MockFruit())])
def test_dtc_returns_true_when_given_different_object_of_same_type(obj_a, obj_b):
    assert duck_type_checker(obj_a, obj_b)


@pytest.mark.parametrize(
    "obj_type_a, obj_type_b", [(MockAnimal(), MockFruit()), (MockFruit(), MockAnimal())]
)
def test_dtc_returns_same_bool_irrelevant_of_object_order(obj_type_a, obj_type_b):
    assert duck_type_checker(obj_type_a, obj_type_b) == duck_type_checker(obj_type_b, obj_type_a)


@pytest.mark.parametrize(
    "obj_type_a, obj_type_b, truth_val",
    [
        (MockAnimal(), MockFruit(), False),
        (MockAnimal(), MockAnimal(), True),
        (MockFruit(), MockFruit(), True),
        (MockFruit(), MockAnimal(), False),
    ],
)
def test_dtc_returns_correct_bool_with_objects_of_different_or_same_type(
    obj_type_a, obj_type_b, truth_val
):
    assert duck_type_checker(obj_type_a, obj_type_b) == truth_val


@pytest.mark.parametrize("x, y", [(1, 1.0), (1.0, "s"), ("s", 2.0), (True, 1.0), (MockAnimal(), 1)])
def test_dtc_raises_TypeError_when_given_objects_without_dicts(x, y):
    with pytest.raises(TypeError):
        duck_type_checker(x, y)


##########################   general_factory   ##############################
@pytest.mark.parametrize(
    "cls, args, kwargs",
    [
        (MockAnimal, (1954, False), {"colour": "red"}),
        (MockFruit, (), {}),
        (MockNoDefaultParams, (), {"a": 1954, "b": "no"}),
        (MockNoDefaultParams, (1954, "yes"), {}),
    ],
)
def test_gfactory_returns_instance_of_correct_class(cls, args, kwargs):
    assert isinstance(general_factory(cls, *args, **kwargs), cls) == True


@pytest.mark.parametrize(
    "cls, args, kwargs",
    [
        (MockAnimal, (1954, False, "red", "extra_arg"), {}),
        (MockNoDefaultParams, (), {"a": 1954, "b": "no", "c": False}),
        (MockNoDefaultParams, (1954, "yes", True), {}),
        (MockNoDefaultParams, (), {}),
    ],
)
def test_gfactory_raises_TypeError_when_unexpected_number_or_wrong_args_given(cls, args, kwargs):
    with pytest.raises(TypeError):
        general_factory(cls, *args, **kwargs)
