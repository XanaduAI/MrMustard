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

"""Tests for Wires class."""

# pylint: disable=missing-function-docstring

from unittest.mock import patch

from ipywidgets import HTML
import pytest

from mrmustard.lab_dev.wires import Wires


class TestWires:
    r"""
    Tests for the Wires class.
    """

    def test_init(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8}, {9}, {10})
        assert w.args == ({0, 1, 2}, {3, 4, 5}, {6, 7}, {8}, {9}, {10})

    def test_ids(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        assert w.ids == [w.id + i for i in range(9)]

    def test_ids_with_subsets(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8}, {9, 10}, {11})

        assert w.input.ids == [w.ids[3], w.ids[4], w.ids[5], w.ids[8], w.ids[11]]
        assert w.output.ids == [
            w.ids[0],
            w.ids[1],
            w.ids[2],
            w.ids[6],
            w.ids[7],
            w.ids[9],
            w.ids[10],
        ]
        assert w.bra.ids == [w.ids[0], w.ids[1], w.ids[2], w.ids[3], w.ids[4], w.ids[5]]
        assert w.ket.ids == [w.ids[6], w.ids[7], w.ids[8]]
        assert w.quantum.ids == [
            w.ids[0],
            w.ids[1],
            w.ids[2],
            w.ids[3],
            w.ids[4],
            w.ids[5],
            w.ids[6],
            w.ids[7],
            w.ids[8],
        ]
        assert w.classical.ids == [w.ids[9], w.ids[10], w.ids[11]]

        assert w.output.bra.ids == [w.ids[0], w.ids[1], w.ids[2]]
        assert w.input.bra.ids == [w.ids[3], w.ids[4], w.ids[5]]

    def test_indices(self):
        w = Wires({0, 10, 20}, {30, 40, 50}, {60, 70}, {80})
        assert w.indices == (0, 1, 2, 3, 4, 5, 6, 7, 8)

    def test_indices_with_subsets(self):
        w = Wires({0, 10, 20}, {30, 40, 50}, {60, 70}, {80})

        assert w.output.indices == (0, 1, 2, 6, 7)
        assert w.bra.indices == (0, 1, 2, 3, 4, 5)
        assert w.input.indices == (3, 4, 5, 8)
        assert w.ket.indices == (6, 7, 8)

        assert w.output.bra.indices == (0, 1, 2)
        assert w.input.bra.indices == (3, 4, 5)
        assert w.output.ket.indices == (6, 7)
        assert w.input.ket.indices == (8,)

    def test_wire_subsets(self):
        w = Wires({0}, {1}, {2}, {3})
        assert w.output.bra.modes == {0}
        assert w.input.bra.modes == {1}
        assert w.output.ket.modes == {2}
        assert w.input.ket.modes == {3}

    def test_index_dicts(self):
        w = Wires({0, 2, 1}, {6, 7, 8}, {3, 4}, {4}, {5}, {9})
        d = [{0: 0, 1: 1, 2: 2}, {6: 3, 7: 4, 8: 5}, {3: 6, 4: 7}, {4: 8}, {5: 9}, {9: 10}]

        assert w.index_dicts == d
        assert w.input.index_dicts == d
        assert w.input.bra.index_dicts == d

    def test_ids_dicts(self):
        w = Wires({0, 2, 1}, {6, 7, 8}, {3, 4}, {4}, {5}, {9})
        d = [
            {0: w.id, 1: w.id + 1, 2: w.id + 2},
            {6: w.id + 3, 7: w.id + 4, 8: w.id + 5},
            {3: w.id + 6, 4: w.id + 7},
            {4: w.id + 8},
            {5: w.id + 9},
            {9: w.id + 10},
        ]

        assert w.ids_dicts == d
        assert w.input.ids_dicts == d
        assert w.input.bra.ids_dicts == d

    def test_adjoint(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        w_adj = w.adjoint
        assert w.input.ket.modes == w_adj.input.bra.modes
        assert w.output.ket.modes == w_adj.output.bra.modes
        assert w.input.bra.modes == w_adj.input.ket.modes
        assert w.output.bra.modes == w_adj.output.ket.modes

    def test_dual(self):
        w = Wires({0, 1, 2}, {3, 4, 5}, {6, 7}, {8})
        w_d = w.dual
        assert w.input.ket.modes == w_d.output.ket.modes
        assert w.output.ket.modes == w_d.input.ket.modes
        assert w.input.bra.modes == w_d.output.bra.modes
        assert w.output.bra.modes == w_d.input.bra.modes

    def test_add(self):
        w1 = Wires({0}, {0, 1}, {2}, {3})
        w2 = Wires({1}, {2}, {3}, {4})
        w12 = Wires({0, 1}, {0, 1, 2}, {2, 3}, {3, 4})

        assert (w1 + w2).modes == w12.modes

    def test_add_error(self):
        w1 = Wires({0}, {1}, {2}, {3})
        w2 = Wires({0}, {2}, {3}, {4})
        with pytest.raises(Exception):
            w1 + w2  # pylint: disable=pointless-statement

    def test_bool(self):
        assert Wires({0})
        assert not Wires({0}).input

    def test_getitem(self):
        w = Wires({0, 1}, {0, 2})

        w0 = Wires({0}, {0})
        assert w[0] == w0
        assert w._mode_cache == {(0,): w0}  # pylint: disable=protected-access

        w1 = Wires({1})
        assert w[1] == w1
        assert w._mode_cache == {(0,): w0, (1,): w1}  # pylint: disable=protected-access

        w2 = Wires(set(), {2})
        assert w[2] == w2
        assert w._mode_cache == {  # pylint: disable=protected-access
            (0,): w0,
            (1,): w1,
            (2,): w2,
        }

        assert w[0].indices == (0, 2)
        assert w[1].indices == (1,)
        assert w[2].indices == (3,)

    def test_eq_neq(self):
        w1 = Wires({0, 1}, {2, 3}, {4, 5}, {6, 7})
        w2 = Wires({0, 1}, {2, 3}, {4, 5}, {6, 7})
        w3 = Wires(set(), {2, 3}, {4, 5}, {6, 7})
        w4 = Wires({0, 1}, set(), {4, 5}, {6, 7})
        w5 = Wires({0, 1}, {2, 3}, set(), {6, 7})
        w6 = Wires({0, 1}, {2, 3}, {4, 5}, set())

        assert w1 == w2
        assert w1 != w3
        assert w1 != w4
        assert w1 != w5
        assert w1 != w6

    def test_matmul(self):
        # contracts 1,1 on bra side
        # contracts 3,3 and 13,13 on ket side
        # contracts 17,17 on classical
        u = Wires({1, 5}, {2, 6, 15}, {3, 7, 13}, {4, 8}, {16, 17}, {18})
        v = Wires({0, 9, 14}, {1, 10}, {2, 11}, {13, 3, 12}, {19}, {17})
        new_wires, perm = u @ v
        assert new_wires.args == (
            {0, 5, 9, 14},
            {2, 6, 10, 15},
            {2, 7, 11},
            {4, 8, 12},
            {16, 19},
            {18},
        )
        assert perm == [9, 0, 10, 11, 1, 2, 12, 3, 13, 4, 14, 5, 6, 15, 7, 16, 8]

    def test_matmul_keeps_ids(self):
        U = Wires(set(), set(), {0}, {0})
        psi = Wires(set(), set(), {0}, set())
        assert (psi @ U)[0].ids[0] == U.ids[0]

    def test_matmul_error(self):
        u = Wires(set(), set(), {0}, set())  # only output wire
        v = Wires(set(), set(), {0}, set())  # only output wire
        with pytest.raises(ValueError):
            u @ v  # pylint: disable=pointless-statement

    @patch("mrmustard.lab_dev.wires.display")
    def test_ipython_repr(self, mock_display):
        """Test the IPython repr function."""
        wires = Wires({0}, {}, {3}, {3, 4})
        wires._ipython_display_()  # pylint:disable=protected-access
        [widget] = mock_display.call_args.args
        assert isinstance(widget, HTML)
