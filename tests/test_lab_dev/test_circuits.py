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

"""Tests for the ``Circuit`` class."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.circuits import Circuit
from mrmustard.lab_dev.states import Vacuum, Number
from mrmustard.lab_dev.transformations import BSgate, Dgate, Sgate


class TestCircuit:
    r"""
    Tests for the ``Circuit`` class.
    """

    def test_init(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])
        assert circ.components == [vac, s01, bs01, bs12]

    def test_eq(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])

        assert Circuit([vac, s01]) == Circuit([vac, s01])
        assert Circuit([vac, s01]) != Circuit([vac, s01, bs01])

    def test_len(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])

        assert len(Circuit([vac])) == 1
        assert len(Circuit([vac, s01])) == 2
        assert len(Circuit([vac, s01, s01])) == 3

    def test_get_item(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])
        assert circ.components[0] == vac
        assert circ.components[1] == s01
        assert circ.components[2] == bs01
        assert circ.components[3] == bs12

    def test_rshift(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ1 = Circuit([vac]) >> s01
        circ2 = Circuit([bs01, bs12])

        assert circ1 >> circ2 == Circuit([vac, s01, bs01, bs12])

    def test_repr(self):
        vac01 = Vacuum([0, 1])
        vac2 = Vacuum([2])
        vac012 = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1], r = [0, 1], phi=[2, 3])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])
        n12 = Number([0, 1], n=3, cutoff=4)
        n2 = Number([2], n=3, cutoff=4)
        cc = CircuitComponent.from_attributes("my_cc", bs01.representation, bs01.wires)

        assert repr(Circuit([])) == ""

        circ1 = Circuit([vac012])
        draw1 = ""
        draw1 += "\nmode 0:     Vac"
        draw1 += "\nmode 1:     Vac"
        draw1 += "\nmode 2:     Vac"
        assert repr(circ1) == draw1 + "\n\n"

        circ2 = Circuit([vac012, s01, bs01, bs12, cc, n12.dual])
        draw2 = ""
        draw2 += "\nmode 0:     Vac──Sgate(0,2)──╭•──────────────────────────────────my_cc──N"
        draw2 += "\nmode 1:     Vac──Sgate(1,3)──╰BSgate(0.0,0.0)──╭•────────────────my_cc──N"
        draw2 += "\nmode 2:     Vac────────────────────────────────╰BSgate(0.0,0.0)──────────"
        assert repr(circ2) == draw2 + "\n\n"

        circ3 = Circuit([bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01])
        draw3 = ""
        draw3 += "\nmode 0:   ──╭•────────────────╭•────────────────╭•────────────────╭•────────────── ---"
        draw3 += "\nmode 1:   ──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0) ---"
        draw3 += "\n\n"
        draw3 += "\nmode 0:   --- ──╭•────────────────╭•────────────────╭•────────────────╭•────────────── ---"
        draw3 += "\nmode 1:   --- ──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0) ---"
        draw3 += "\n\n"
        draw3 += "\nmode 0:   --- ──╭•────────────────╭•────────────────╭•──────────────"
        draw3 += "\nmode 1:   --- ──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0)──╰BSgate(0.0,0.0)"
        assert repr(circ3) == draw3 + "\n\n"

        circ4 = Circuit([vac01, s01, vac2, bs01, bs12, n2.dual, cc, n12.dual])
        draw4 = ""
        draw4 += "\nmode 0:     Vac──Sgate(0,2)──╭•──────────────────────────────────my_cc──N"
        draw4 += "\nmode 1:     Vac──Sgate(1,3)──╰BSgate(0.0,0.0)──╭•────────────────my_cc──N"
        draw4 += "\nmode 2:          Vac───────────────────────────╰BSgate(0.0,0.0)──N       "
        assert repr(circ4) == draw4 + "\n\n"

    def test_repr_issue_334(self):
        r"""
        Tests the bug reported in GH issue #334.
        """
        circ1 = Circuit([Sgate([0, 1], [1, -1])])
        draw1 = ""
        draw1 += "\nmode 0:   ──Sgate(1,0.0)─"
        draw1 += "\nmode 1:   ──Sgate(-1,0.0)"
        draw1 += "\n\n"
        assert repr(circ1) == draw1
