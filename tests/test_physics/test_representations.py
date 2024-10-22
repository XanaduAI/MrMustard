# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Representation class."""

# pylint: disable=missing-function-docstring

import pytest

from mrmustard.physics.representations import Representation, RepEnum
from mrmustard.physics.wires import Wires
from mrmustard.physics.ansatz import PolyExpAnsatz

from ..random import Abc_triple


class TestRepresentation:
    r"""
    Tests for the Representation class.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init(self, triple):
        empty_rep = Representation()
        assert empty_rep.ansatz is None
        assert empty_rep.wires == Wires()
        assert empty_rep._idx_reps == {}

        ansatz = PolyExpAnsatz(*triple)
        wires = Wires()
        rep = Representation(ansatz, wires)
        assert rep.ansatz == ansatz
        assert rep.wires == wires
        assert rep._idx_reps == dict.fromkeys(
            wires.indices, (RepEnum.from_ansatz(ansatz), None, tuple())
        )
