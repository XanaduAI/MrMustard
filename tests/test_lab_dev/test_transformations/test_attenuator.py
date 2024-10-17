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

"""Tests for the ``Attenuator`` class."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import Attenuator


class TestAttenuator:
    r"""
    Tests for the ``Attenuator`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    transmissivity = [[0.1], 0.1, [0.1, 0.2]]

    @pytest.mark.parametrize("modes,transmissivity", zip(modes, transmissivity))
    def test_init(self, modes, transmissivity):
        gate = Attenuator(modes, transmissivity)

        assert gate.name == "Att"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="transmissivity"):
            Attenuator(modes=[0, 1], transmissivity=[0.2, 0.3, 0.4])

    def test_representation(self):
        rep1 = Attenuator(modes=[0], transmissivity=0.1).ansatz
        e = 0.31622777
        assert math.allclose(rep1.A, [[[0, e, 0, 0], [e, 0, 0, 0.9], [0, 0, 0, e], [0, 0.9, e, 0]]])
        assert math.allclose(rep1.b, np.zeros((1, 4)))
        assert math.allclose(rep1.c, [1.0])

    def test_trainable_parameters(self):
        gate1 = Attenuator([0], 0.1)
        gate2 = Attenuator(
            [0], 0.1, transmissivity_trainable=True, transmissivity_bounds=(-0.2, 0.2)
        )

        with pytest.raises(AttributeError):
            gate1.transmissivity.value = 0.3

        gate2.transmissivity.value = 0.2
        assert gate2.transmissivity.value == 0.2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Attenuator(modes=[0], transmissivity=[0.1, 0.2]).ansatz
