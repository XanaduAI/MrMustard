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

"""Tests for trace out."""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.lab.circuit_components_utils import TraceOut
from mrmustard.lab.states import Coherent, Ket
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.triples import identity_Abc
from mrmustard.physics.wires import Wires


class TestTraceOut:
    r"""
    Tests ``TraceOut`` objects.
    """

    @pytest.mark.parametrize("modes", [(0,), (1, 2), (3, 4, 5)])
    def test_init(self, modes):
        tr = TraceOut(modes)

        assert tr.name == "Tr"
        assert tr.wires == Wires(modes_in_bra=set(modes), modes_in_ket=set(modes))
        assert tr.ansatz == PolyExpAnsatz(*identity_Abc(len(modes)))

    def test_trace_out_bargmann_states(self):
        state = Coherent(0, x=1) >> Coherent(1, x=1) >> Coherent(2, x=1)

        assert state >> TraceOut(0) == (Coherent(1, x=1) >> Coherent(2, x=1)).dm()
        assert state >> TraceOut((1, 2)) == Coherent(0, x=1).dm()

        trace = state >> TraceOut((0, 1, 2))
        assert np.isclose(trace, 1.0)

    def test_trace_out_complex(self):
        cc = CircuitComponent.from_bargmann(
            (
                np.array([[0.1 + 0.2j, 0.3 + 0.4j], [0.3 + 0.4j, 0.5 - 0.6j]]),
                np.array([0.7 + 0.8j, -0.9 + 0.10j]),
                0.11 - 0.12j,
            ),
            modes_out_ket=(0,),
            modes_out_bra=(0,),
        )
        assert (cc >> TraceOut(0)).dtype == math.complex128

    def test_trace_out_fock_states(self):
        state = (Coherent(0, x=1) >> Coherent(1, x=1) >> Coherent(2, x=1)).to_fock(10)
        assert state >> TraceOut(0) == (Coherent(1, x=1) >> Coherent(2, x=1)).to_fock(7).dm()
        assert state >> TraceOut((1, 2)) == Coherent(0, x=1).to_fock(7).dm()

        no_state = state >> TraceOut((0, 1, 2))
        assert np.isclose(no_state, 1.0)

    def test_trace_out_with_batch(self):
        state = Ket.from_fock([0], settings.rng.random((2, 3, 4)), batch_dims=2)
        assert (state >> TraceOut(0)).shape == (2, 3)
