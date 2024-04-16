# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Tests for circuit components utils. """

# pylint: disable=fixme, missing-function-docstring, protected-access, pointless-statement

import pytest

from mrmustard.physics.triples import identity_Abc
from mrmustard.physics.representations import Bargmann
from mrmustard.lab_dev.circuit_components_utils import TraceOut
from mrmustard.lab_dev.states import Coherent
from mrmustard.lab_dev.transformations import Dgate, Attenuator, Unitary
from mrmustard.lab_dev.wires import Wires


class TestTraceOut:
    r"""
    Tests ``TraceOut`` objects.
    """

    @pytest.mark.parametrize("modes", [[0], [1, 2], [3, 4, 5]])
    def test_init(self, modes):
        tr = TraceOut(modes)

        assert tr.name == "Tr"
        assert tr.wires == Wires(modes_in_bra=set(modes), modes_in_ket=set(modes))
        assert tr.representation == Bargmann(*identity_Abc(len(modes)))

    def test_trace_out_states(self):
        assert Coherent([0, 1, 2], x=1) >> TraceOut([0]) == Coherent([1, 2], x=1).dm()
        assert Coherent([0, 1, 2], x=1) >> TraceOut([1, 2]) == Coherent([0], x=1).dm()
