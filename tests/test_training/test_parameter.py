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

"""Tests for the parameter class."""

import numpy as np
import pytest

from mrmustard.training.parameter import (
    create_parameter,
    Constant,
    Orthogonal,
    Unitary,
    Euclidean,
    Symplectic,
    Trainable,
)
from mrmustard.math import Math

math = Math()


@pytest.mark.parametrize("from_backend", [True, False])
def test_create_constant(from_backend):
    """Checks if the factory function `create_parameter`
    returns an instance of the Constant class when args
    are not trainable."""

    value = np.random.rand(*np.random.randint(5, size=5))
    name = "constant_tensor"
    if from_backend:
        value = math.new_constant(value, name)

    param = create_parameter(value, name, is_trainable=False)

    assert isinstance(param, Constant)
    assert math.from_backend(param.value)
    assert param.name == name


@pytest.mark.parametrize("trainable_class", (Euclidean, Orthogonal, Symplectic, Unitary))
@pytest.mark.parametrize("from_backend", [True, False])
@pytest.mark.parametrize("bounds", [None, (0, 10)])
def test_create_trainable(trainable_class, from_backend, bounds):
    """Checks if the factory function `create_parameter`
    returns an instance of the Euclidean/Orthogonal/Symplectic/Unitary class when args
    are trainable."""

    value = 5
    name = f"{trainable_class.__name__}_tensor".lower()
    if from_backend:
        value = math.new_variable(value, bounds, name)

    param = create_parameter(value, name, is_trainable=True, bounds=bounds)

    assert isinstance(param, trainable_class)
    assert isinstance(param, Trainable)
    assert math.from_backend(param.value)
    assert param.name == name
