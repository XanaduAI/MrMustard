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

import pytest

from mrmustard import settings
from mrmustard.lab import (
    Attenuator,
    BSgate,
    Circuit,
    CircuitComponent,
    Coherent,
    Dgate,
    Number,
    Sgate,
    SqueezedVacuum,
    Vacuum,
)
from mrmustard.utils.serialize import load


class TestCircuit:
    r"""
    Tests for the ``Circuit`` class.
    """

    def test_init(self):
        vac = Vacuum((0, 1, 2))
        sgate = Sgate(0)
        bs01 = BSgate((0, 1))
        bs12 = BSgate((1, 2))

        circ1 = Circuit([vac, sgate, sgate.on(1), bs01, bs12])
        assert circ1.components == [vac, sgate, sgate.on(1), bs01, bs12]
        assert circ1.path == [(0, 1), (0, 2), (0, 3), (0, 4)]

        circ2 = Circuit() >> vac >> sgate >> sgate.on(1) >> bs01 >> bs12
        assert circ2.components == [vac, sgate, sgate.on(1), bs01, bs12]
        assert circ2.path == [(0, 1), (0, 2), (0, 3), (0, 4)]

    def test_lookup_path(self, capfd):
        vac = Vacuum((0, 1, 2))
        sgate = Sgate(0)
        bs01 = BSgate((0, 1))
        bs12 = BSgate((1, 2))

        circ = Circuit([vac, sgate, bs01, bs12])
        circ.check_contraction(0)
        out1, _ = capfd.readouterr()
        exp1 = "\n"
        exp1 += "→ index: 0\n"
        exp1 += "mode 0:     ◖Vac◗\n"
        exp1 += "mode 1:     ◖Vac◗\n"
        exp1 += "mode 2:     ◖Vac◗\n\n\n"
        exp1 += "→ index: 1\n"
        exp1 += "mode 0:   ──S(0.0,0.0)\n\n\n"
        exp1 += "→ index: 2\n"
        exp1 += "mode 0:   ──╭•──────────\n"
        exp1 += "mode 1:   ──╰BS(0.0,0.0)\n\n\n"
        exp1 += "→ index: 3\n"
        exp1 += "mode 1:   ──╭•──────────\n"
        exp1 += "mode 2:   ──╰BS(0.0,0.0)\n\n\n\n"
        assert out1 == exp1

        circ.path += [(0, 1)]
        circ.check_contraction(1)
        out2, _ = capfd.readouterr()
        exp2 = "\n"
        exp2 += "→ index: 0\n"
        exp2 += "mode 0:     ◖Vac◗──S(0.0,0.0)\n"
        exp2 += "mode 1:     ◖Vac◗────────────\n"
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
        vac12 = Vacuum((1, 2))
        d1 = Dgate(1, x=0.1, y=0.1)
        d2 = Dgate(2, x=0.1, y=0.2)
        a1 = Attenuator(1, transmissivity=0.8)
        numstate = Number(1, n=1).dual

        circuit = Circuit([vac12, d1, d2, a1, numstate, numstate.on(2)])
        circuit.path = path

        assert circuit.path == path

    def test_eq(self):
        vac = Vacuum((0, 1, 2))
        sgate = Sgate(0)
        bs01 = BSgate((0, 1))

        assert Circuit([vac, sgate, sgate.on(1)]) == Circuit([vac, sgate, sgate.on(1)])
        assert Circuit([vac, sgate, sgate.on(1)]) != Circuit([vac, sgate, sgate.on(1), bs01])

    def test_len(self):
        vac = Vacuum((0, 1, 2))
        sgate = Sgate(0)

        assert len(Circuit([vac])) == 1
        assert len(Circuit([vac, sgate, sgate.on(1)])) == 3
        assert len(Circuit([vac, sgate, sgate.on(1), sgate, sgate.on(2)])) == 5

    def test_get_item(self):
        vac = Vacuum((0, 1, 2))
        sgate = Sgate(0)
        bs01 = BSgate((0, 1))
        bs12 = BSgate((1, 2))

        circ = Circuit([vac, sgate, sgate.on(1), bs01, bs12])
        assert circ.components[0] == vac
        assert circ.components[1] == sgate
        assert circ.components[2] == sgate.on(1)
        assert circ.components[3] == bs01
        assert circ.components[4] == bs12

    def test_rshift(self):
        vac = Vacuum((0, 1, 2))
        sgate = Sgate(0)
        bs01 = BSgate((0, 1))
        bs12 = BSgate((1, 2))

        circ1 = Circuit([vac]) >> sgate >> sgate.on(1)
        circ2 = Circuit([bs01, bs12])

        assert circ1 >> circ2 == Circuit([vac, sgate, sgate.on(1), bs01, bs12])

    def test_repr(self):
        vac01 = Vacuum((0, 1))
        vac1 = Vacuum(1)
        vac2 = Vacuum(2)
        vac012 = Vacuum((0, 1, 2))
        s0 = Sgate(0, r=0.0, phi=2.0)
        s1 = Sgate(1, r=1.0, phi=3.0)
        bs01 = BSgate((0, 1))
        bs12 = BSgate((1, 2))
        n0 = Number(0, n=3)
        n1 = Number(1, n=3)
        n2 = Number(2, n=3)
        cc = CircuitComponent(bs01.ansatz, bs01.wires, "my_cc")

        assert repr(Circuit()) == ""

        circ1 = Circuit([vac012])
        r1 = ""
        r1 += "\nmode 0:     ◖Vac◗"
        r1 += "\nmode 1:     ◖Vac◗"
        r1 += "\nmode 2:     ◖Vac◗"
        assert repr(circ1) == r1 + "\n\n"

        circ2 = Circuit([vac012, s0, s1, bs01, bs12, cc, n0.dual, n1.dual])
        r2 = ""
        r2 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•──────────────────────────CC──|3)=(3,4)"
        r2 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╭•────────────CC──|3)=(3,4)"
        r2 += "\nmode 2:     ◖Vac◗────────────────────────────╰BS(0.0,0.0)───────────────"
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

        circ4 = Circuit([vac01, s0, s1, vac2, bs01, bs12, n2.dual, cc, n0.dual, n1.dual])
        r4 = ""
        r4 += "\nmode 0:     ◖Vac◗──S(0.0,2.0)──╭•──────────────────────────CC─────────|3)=(3,4)"
        r4 += "\nmode 1:     ◖Vac◗──S(1.0,3.0)──╰BS(0.0,0.0)──╭•────────────CC─────────|3)=(3,4)"
        r4 += "\nmode 2:            ◖Vac◗─────────────────────╰BS(0.0,0.0)──|3)=(3,4)           "
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
        circ1 = Circuit([Sgate(0, 1.0, 2.0), Sgate(1, -1.0, -2.0)])
        r1 = ""
        r1 += "\nmode 0:   ──S(1.0,2.0)──"
        r1 += "\nmode 1:   ──S(-1.0,-2.0)"
        r1 += "\n\n"
        assert repr(circ1) == r1

    def test_optimize_path(self):
        "tests the optimize method"
        # contracting the last two first is better
        circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Coherent(0, x=1.0).dual])
        circ.optimize(with_BF_heuristic=True)  # with default heuristics
        assert circ.path == [(1, 2), (0, 1)]

        circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Coherent(0, x=1.0).dual])
        circ.optimize(with_BF_heuristic=False)  # without the BF heuristic
        assert circ.path == [(1, 2), (0, 1)]

        circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Coherent(0, x=1.0).dual])
        circ.optimize(n_init=1, verbose=False)
        assert circ.path == [(1, 2), (0, 1)]

    def test_wrong_path(self):
        "tests an exception is raised if contract is called with a wrond path"
        circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Dgate(0, x=1.0)])
        circ.path = [(0, 3)]
        with pytest.raises(ValueError):
            circ.contract()

    def test_contract(self):
        "tests the contract method"
        circ = Circuit([Number(0, n=15), Sgate(0, r=1.0), Dgate(0, x=1.0)])
        assert circ.contract() == Number(0, n=15) >> Sgate(0, r=1.0) >> Dgate(0, x=1.0)

    def test_serialize_makes_zip(self, tmpdir):
        """Test that serialize makes a JSON and a zip."""
        settings.CACHE_DIR = tmpdir
        circ = Circuit([Coherent(0, x=1.0), Dgate(0, 0.1)])
        path = circ.serialize()
        assert list(path.parent.glob("*")) == [path]
        assert path.suffix == ".zip"

        assert load(path) == circ
        assert list(path.parent.glob("*")) == [path]

    def test_serialize_custom_name(self, tmpdir):
        """Test that circuits can be serialized with custom names."""
        settings.CACHE_DIR = tmpdir
        circ = Circuit([Coherent(0, x=1.0), Dgate(0, 0.1)])
        path = circ.serialize(filestem="custom_name")
        assert path.name == "custom_name.zip"

    def test_path_is_loaded(self, tmpdir):
        """Test that circuit paths are saved if already evaluated."""
        settings.CACHE_DIR = tmpdir
        vac = Vacuum(0)
        S0 = Sgate(0)
        s0 = SqueezedVacuum(1)
        bs01 = BSgate((0, 1))
        c0 = Coherent(0).dual
        c1 = Coherent(1).dual

        circ = Circuit([vac, S0, s0, bs01, c0, c1])
        base_path = circ.path
        assert load(circ.serialize()).path == base_path

        circ.optimize()
        opt_path = circ.path
        assert opt_path != base_path
        assert load(circ.serialize()).path == opt_path
