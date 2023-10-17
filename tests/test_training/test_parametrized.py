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

"""Tests for the parametrized class."""

import pytest

from mrmustard.training import Parametrized
from mrmustard.lab.circuit import Circuit
from mrmustard.lab.gates import BSgate, S2gate
from mrmustard.training.parameter import (
    Constant,
    Unitary,
    Orthogonal,
    Euclidean,
    Symplectic,
    Trainable,
)
import mrmustard.math as math


@pytest.mark.parametrize("kwargs", [{"a": 5}, {"b": 4.5}])
def test_attribute_assignment(kwargs):
    """Test that arguments are converted into Trainable or Constant and
    assigned as attributes of the class."""
    parametrized = Parametrized(**kwargs)

    instance_attributes = parametrized.__dict__

    for name in kwargs.keys():
        attrib = instance_attributes[f"{name}"]
        assert isinstance(attrib, (Trainable, Constant))
        assert instance_attributes[f"{name}"].name == name


@pytest.mark.parametrize("trainable_class", (Euclidean, Orthogonal, Symplectic, Unitary))
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
    attrib = getattr(parametrized, f"{name}")

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
    attrib = getattr(parametrized, f"{name}")

    assert isinstance(attrib, Constant)
    assert math.from_backend(attrib.value)
    assert attrib.name == name


def test_get_parameters():
    """Test that the `get_trainable_parameters` and `get_constant_parameters` properties
    return the correct set of parameters"""

    kwargs = {
        "numeric_attribute": 2,
        "constant_attribute": math.new_constant(1, "constant_attribute"),
        "symplectic_attribute": math.new_variable(2, None, "symplectic_attribute"),
        "symplectic_attribute_trainable": True,
        "euclidian_attribute": math.new_variable(3, None, "euclidian_attribute"),
        "euclidian_attribute_trainable": True,
        "orthogonal_attribute": math.new_variable(4, None, "orthogonal_attribute"),
        "orthogonal_attribute_trainable": True,
        "unitary_attribute": math.new_variable(4, None, "unitary_attribute"),
        "unitary_attribute_trainable": True,
    }
    parametrized = Parametrized(**kwargs)

    trainable_params = parametrized.trainable_parameters
    assert len(trainable_params) == 4
    assert all(isinstance(param, Trainable) for param in trainable_params)

    constant_params = parametrized.constant_parameters
    assert len(constant_params) == 2
    assert all(isinstance(param, Constant) for param in constant_params)

    trainable_params = parametrized.traverse_trainables(owner_tag="foo")
    assert len(trainable_params) == 4
    assert all(isinstance(param, Trainable) for param in trainable_params.values())
    assert all(tag.startswith("foo") for tag in trainable_params)
    assert all(tag.split("/")[1] in kwargs for tag in trainable_params)

    constant_params = parametrized.traverse_constants()
    assert len(constant_params) == 2
    assert all(isinstance(param, Constant) for param in constant_params.values())
    assert all(tag.startswith("Parametrized") for tag in constant_params)
    assert all(tag.split("/")[1] in kwargs for tag in constant_params)


def test_get_nested_parameters():
    """Test that nested Parametrized objects (e.g. a circuit) return all the trainable
    and constant parameters via `get_trainable_parameters` and `get_constant_parameters`
    properties"""

    s2 = S2gate(r=0.0, phi=0.0, r_trainable=False, phi_trainable=True)
    bs = BSgate(
        theta=0.0,
        phi=0.0,
        theta_trainable=True,
        phi_trainable=False,
    )
    circ = Circuit([s2, bs])

    trainables = circ.trainable_parameters
    constants = circ.constant_parameters
    assert len(trainables) == 2
    assert len(constants) == 2

    assert (s2.phi in trainables) and (bs.theta in trainables)
    assert (s2.r in constants) and (bs.phi in constants)

    trainables = circ.traverse_trainables()
    constants = circ.traverse_constants("Device")
    assert len(trainables) == 2
    assert len(constants) == 2
    assert all(tag.startswith("Circuit/_ops[") for tag in trainables)
    assert all(tag.startswith("Device/_ops[") for tag in constants)

    assert (s2.phi in trainables.values()) and (bs.theta in trainables.values())
    assert (s2.r in constants.values()) and (bs.phi in constants.values())
