# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from hypothesis import given, strategies as st
import pytest

from mrmustard.utils.parametrized import Parametrized
from mrmustard.math import Math
from mrmustard.utils.parameter import Constant, Orthogonal, Euclidian, Symplectic, Trainable

math = Math()


def from_python_types():
    relevant_python_types = [int, float, complex, list, tuple, dict, str, set]
    chosen_type = random.choice(relevant_python_types)
    return st.from_type(chosen_type)


@given(kwargs=st.dictionaries(st.from_regex("[a-zA-Z]+"), from_python_types(), min_size=1))
def test_attribute_assignment(kwargs):
    """Test that arguments are assigned as attributes of the class."""
    parametrized = Parametrized(**kwargs)
    keys = kwargs.keys()

    instance_attributes = parametrized.__dict__

    # assert arguments are assigned as attributes of the class
    expected_attribute_names = [f"_{key}" for key in keys]
    instance_attribute_names = list(instance_attributes.keys())
    assert all(attribute in instance_attribute_names for attribute in expected_attribute_names)

    # assert attributes of the class are assigned the correct value
    assert all(instance_attributes[f"_{key}"] == value for key, value in kwargs.items())


@given(kwargs=st.dictionaries(st.from_regex("[a-zA-Z]+"), from_python_types(), min_size=1))
def test_attribute_type_assignment(kwargs):
    """Test that attributes of the class that do not belong
    to the backend are assigned the correct type.
    """
    parametrized = Parametrized(**kwargs)

    instance_attributes = parametrized.__dict__
    assert all(type(instance_attributes[f"_{key}"]) is type(value) for key, value in kwargs.items())


@pytest.mark.parametrize("trainable_class", (Euclidian, Orthogonal, Symplectic))
@pytest.mark.parametrize("bounds", [None, (0, 10)])
def test_attribute_from_backend_type_assignment(trainable_class, bounds):
    """Test that arguments that are trainable get defined on the backend,
    are assigned correctly as attributes of the Parametrized instance
    and are the correct type of trainable instance.
    """

    name = f"{trainable_class.__name__}_tensor".lower()
    value = 5
    kwargs = {
        name: value,
        f"{name}_trainable": True,
        f"{name}_bounds": bounds,
    }

    parametrized = Parametrized(**kwargs)
    attrib = getattr(parametrized, f"_{name}")

    assert isinstance(attrib, trainable_class)
    assert isinstance(attrib, Trainable)
    assert math.from_backend(attrib.value)
    assert attrib.name == name


def test_attribute_from_backend_constant_assignment():
    """Test that arguments that are NOT trainable get defined on the backend,
    are assigned correctly as attributes of the Parametrized instance
    and are instances of :class:`Constant`.
    """

    name = "constant_tensor"
    value = math.new_constant(5, name)
    kwargs = {name: value, f"{name}_trainable": False}

    parametrized = Parametrized(**kwargs)
    attrib = getattr(parametrized, f"_{name}")

    assert isinstance(attrib, Constant)
    assert math.from_backend(attrib.value)
    assert attrib.name == name


def test_get_parameters():
    """Test that the `get_trainable_parameters` property returns the correct
    set of parameters"""

    kwargs = {
        "regular_attribute": "just_a_string",
        "constant_attribute": math.new_constant(1, "constant_attribute"),
        "symplectic_attribute": math.new_variable(2, None, "symplectic_attribute"),
        "symplectic_attribute_trainable": True,
        "euclidian_attribute": math.new_variable(3, None, "euclidian_attribute"),
        "euclidian_attribute_trainable": True,
        "orthogonal_attribute": math.new_variable(4, None, "orthogonal_attribute"),
        "orthogonal_attribute_trainable": True,
    }
    parametrized = Parametrized(**kwargs)

    trainable_params = parametrized.trainable_parameters
    assert len(trainable_params) == 3
    assert all(isinstance(param, Trainable) for param in trainable_params)

    constant_params = parametrized.constant_parameters
    assert len(constant_params) == 1
    assert all(isinstance(param, Constant) for param in constant_params)
