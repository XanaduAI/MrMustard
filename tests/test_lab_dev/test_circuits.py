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

import pytest

from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.circuits import Circuit
from mrmustard.lab_dev.states import Vacuum, Number, Coherent, SqueezedVacuum
from mrmustard.lab_dev.transformations import (
    BSgate,
    Sgate,
    Dgate,
    Attenuator,
)
from mrmustard import settings
from mrmustard.utils.serialize import load


class TestCircuit:
    r"""
    Tests for the ``Circuit`` class.
    """

    def test_init(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ1 = Circuit([vac, s01, bs01, bs12])
        assert circ1.components == [vac, s01, bs01, bs12]
        assert circ1.path == []

        circ2 = Circuit() >> vac >> s01 >> bs01 >> bs12
        assert circ2.components == [vac, s01, bs01, bs12]
        assert circ2.path == []

    def test_propagate_shapes(self):
        MAX = settings.AUTOSHAPE_MAX
        circ = Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)])
        assert [op.auto_shape() for op in circ] == [(5,), (MAX, MAX)]
        circ.propagate_shapes()
        assert [op.auto_shape() for op in circ] == [(5,), (MAX, 5)]

        circ = Circuit([SqueezedVacuum([0, 1], r=[0.5, -0.5]), BSgate([0, 1], 0.9)])
        assert [op.auto_shape() for op in circ] == [(6, 6), (MAX, MAX, MAX, MAX)]
        circ.propagate_shapes()
        assert [op.auto_shape() for op in circ] == [(6, 6), (12, 12, 6, 6)]

    def test_make_path(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])

        circ.make_path("l2r")
        assert circ.path == [(0, 1), (0, 2), (0, 3)]

        circ.make_path("r2l")
        assert circ.path == [(2, 3), (1, 2), (0, 1)]

        with pytest.raises(ValueError):
            circ.make_path("my_strategy")

    def test_lookup_path(self, capfd):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])
        circ.lookup_path()
        out1, _ = capfd.readouterr()
        exp1 = "\n"
        exp1 += "→ index: 0\n"
        exp1 += "mode 0:     ◖Vac◗\n"
        exp1 += "mode 1:     ◖Vac◗\n"
        exp1 += "mode 2:     ◖Vac◗\n\n\n"
        exp1 += "→ index: 1\n"
        exp1 += "mode 0:   ──S(0.0,0.0)\n"
        exp1 += "mode 1:   ──S(0.0,0.0)\n\n\n"
        exp1 += "→ index: 2\n"
        exp1 += "mode 0:   ──╭•──────────\n"
        exp1 += "mode 1:   ──╰BS(0.0,0.0)\n\n\n"
        exp1 += "→ index: 3\n"
        exp1 += "mode 1:   ──╭•──────────\n"
        exp1 += "mode 2:   ──╰BS(0.0,0.0)\n\n\n\n"
        assert out1 == exp1

        circ.path += [(0, 1)]
        circ.lookup_path()
        out2, _ = capfd.readouterr()
        exp2 = "\n"
        exp2 += "→ index: 0\n"
        exp2 += "mode 0:     ◖Vac◗──S(0.0,0.0)\n"
        exp2 += "mode 1:     ◖Vac◗──S(0.0,0.0)\n"
        exp2 += "mode 2:     ◖Vac◗────────────\n\n\n"
        exp2 += "→ index: 2\n"
        exp2 += "mode 0:   ──╭•──────────\n"
        exp2 += "mode 1:   ──╰BS(0.0,0.0)\n\n\n"
        exp2 += "→ index: 3\n"
        exp2 += "mode 1:   ──╭•──────────\n"
        exp2 += "mode 2:   ──╰BS(0.0,0.0)\n\n\n\n"
        assert out2 == exp2

    @pytest.mark.parametrize("path", [[(0, 1), (2, 3)], [(0, 1), (2, 3), (0, 2), (0, 4), (0, 5)]])
    def test_path(self, path):
        vac12 = Vacuum([1, 2])
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.2)
        d12 = Dgate([1, 2], x=0.1, y=[0.1, 0.2])
        a1 = Attenuator([1], transmissivity=0.8)
        n12 = Number([1, 2], n=1).dual

        circuit = Circuit([vac12, d1, d2, d12, a1, n12])
        circuit.path = path

        assert circuit.path == path

    def test_path_errors(self):
        vac12 = Vacuum([1, 2])
        d1 = Dgate([1], x=0.1, y=0.1)
        d12 = Dgate([1, 2], x=0.1, y=[0.1, 0.2])

        circuit1 = Circuit([vac12, vac12])
        with pytest.raises(ValueError, match="Dangling wires cannot be overwritten."):
            circuit1.path += [(0, 1)]

        circuit2 = Circuit([vac12.adjoint, vac12.adjoint])
        with pytest.raises(ValueError, match="Dangling wires cannot be overwritten."):
            circuit2.path += [(0, 1)]

        circuit3 = Circuit([vac12, d1, d12])
        with pytest.raises(ValueError, match="is invalid."):
            circuit3.path += [(0, 2)]

        circuit4 = Circuit([vac12.adjoint, d1.adjoint, d12.adjoint])
        with pytest.raises(ValueError, match="is invalid."):
            circuit4.path += [(0, 2)]

    def test_graph(self):
        vac12 = Vacuum([1, 2])
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.2)
        d12 = Dgate([1, 2], x=0.1, y=[0.1, 0.2])
        a1 = Attenuator([1], transmissivity=0.8)
        n12 = Number([1, 2], n=1).dual

        circuit = Circuit([vac12, d1, d2, d12, a1, n12])
        assert not circuit._graph  # pylint: disable=protected-access

        circuit.make_path("l2r")
        graph1 = circuit._graph  # pylint: disable=protected-access

        c0 = circuit.components[0]
        c1 = circuit.components[1]
        assert graph1[c0.wires.output.ket[1].ids[0]] == c1.wires.input.ket[1].ids[0]

        circuit.make_path("r2l")
        graph2 = circuit._graph  # pylint: disable=protected-access
        assert graph1 == graph2

        circuit.path = [(0, 1), (2, 3)]
        graph3 = circuit._graph  # pylint: disable=protected-access
        assert graph1 == graph3

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
        vac1 = Vacuum([1])
        vac2 = Vacuum([2])
        vac012 = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1], r=[0.0, 1.0], phi=[2.0, 3.0])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])
        n12 = Number([0, 1], n=3)
        n2 = Number([2], n=3)
        cc = CircuitComponent._from_attributes(
            bs01.representation, bs01.wires, "my_cc"
        )  # pylint: disable=protected-access

        assert repr(Circuit()) == ""

        circ1 = Circuit([vac012])
        r1 = ""
        r1 += "\nmode 0:     ◖Vac◗"
        r1 += "\nmode 1:     ◖Vac◗"
        r1 += "\nmode 2:     ◖Vac◗"
        assert repr(circ1) == r1 + "\n\n"

        circ2 = Circuit([vac012, s01, bs01, bs12, cc, n12.dual])
        r2 = ""
        r2 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•──────────────────────────CC──|3)="
        r2 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╭•────────────CC──|3)="
        r2 += "\nmode 2:     ◖Vac◗────────────────────────────╰BS(0.0,0.0)──────────"
        assert repr(circ2) == r2 + "\n\n"

        circ3 = Circuit([bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01, bs01])
        r3 = ""
        r3 += "\nmode 0:   ──╭•────────────╭•────────────╭•────────────╭•────────────╭•────────────╭•────────── ---"
        r3 += "\nmode 1:   ──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0) ---"
        r3 += "\n\n"
        r3 += (
            "\nmode 0:   --- ──╭•────────────╭•────────────╭•────────────╭•────────────╭•──────────"
        )
        r3 += (
            "\nmode 1:   --- ──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0)──╰BS(0.0,0.0)"
        )
        assert repr(circ3) == r3 + "\n\n"

        circ4 = Circuit([vac01, s01, vac2, bs01, bs12, n2.dual, cc, n12.dual])
        r4 = ""
        r4 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•──────────────────────────CC────|3)="
        r4 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╭•────────────CC────|3)="
        r4 += "\nmode 2:            ◖Vac◗─────────────────────╰BS(0.0,0.0)──|3)=      "
        assert repr(circ4) == r4 + "\n\n"

        circ5 = Circuit() >> vac1 >> bs01 >> vac1.dual >> vac1 >> bs01 >> vac1.dual
        r5 = ""
        r5 += "\nmode 0:          ──╭•───────────────────────────╭•──────────────────"
        r5 += "\nmode 1:     ◖Vac◗──╰BS(0.0,0.0)──|Vac)=  ◖Vac◗──╰BS(0.0,0.0)──|Vac)="
        assert repr(circ5) == r5 + "\n\n"

    def test_repr_issue_334(self):
        r"""
        Tests the bug reported in GH issue #334.
        """
        circ1 = Circuit([Sgate([0, 1], [1.0, -1.0], [2.0, -2.0])])
        r1 = ""
        r1 += "\nmode 0:   ──S(1.0,2.0)──"
        r1 += "\nmode 1:   ──S(-1.0,-2.0)"
        r1 += "\n\n"
        assert repr(circ1) == r1

    def test_serialize_makes_zip(self, tmpdir):
        """Test that serialize makes a JSON and a zip."""
        settings.CACHE_DIR = tmpdir
        circ = Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)])
        path = circ.serialize()
        assert list(path.parent.glob("*")) == [path]
        assert path.suffix == ".zip"

        assert load(path) == circ
        assert list(path.parent.glob("*")) == [path]

    def test_serialize_custom_name(self, tmpdir):
        """Test that circuits can be serialized with custom names."""
        settings.CACHE_DIR = tmpdir
        circ = Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)])
        path = circ.serialize(filestem="custom_name")
        assert path.name == "custom_name.zip"

    def test_path_is_loaded(self, tmpdir):
        """Test that circuit paths are saved if already evaluated."""
        settings.CACHE_DIR = tmpdir
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])
        assert not load(circ.serialize())._path

        circ.make_path()
        assert circ._path

        assert load(circ.serialize())._path == circ.path
